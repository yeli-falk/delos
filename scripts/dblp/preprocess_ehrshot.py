"""One-time preprocessor for EHRshot raw CSV → per-patient visit sequences.

The raw EHRSHOT_ASSETS/data/ehrshot.csv is ~41M rows. We don't want to
re-parse it every time the dataset is instantiated, so we do it ONCE and
write a single pickle that the dataset class loads in seconds.

Per-patient temporal event graph schema:
  * Nodes  = top-K medical concept codes, by global frequency (training
             cohort only, to avoid test-set leakage into the vocabulary).
  * Snap   = one visit (visit_occurrence row group). Visits sorted by start.
  * Edge   = co-occurrence within visit (clique of the visit's code-bag).

Filters applied:
  * Drop omop_table in {note, person, death, visit_occurrence, visit_detail}
    -- these are not medical-event codes.
  * Drop code prefixes in {Visit, Domain, CARE_SITE, Medicare Specialty,
    CMS Place of Service, Gender, Race, Ethnicity} -- bookkeeping codes
    that always co-occur and would dominate the top-K vocabulary.
  * Drop patients with fewer than `--min_visits` valid visits (default 4,
    enough for history_size=3 + num_preds=1).

Output pickle schema:
  {
    'code_vocab':      list[str]    length K, code strings (e.g. 'LOINC/...').
    'code_vocab_type': np.int32     (K,)  vocabulary id (LOINC=0, SNOMED=1, ...).
    'code_freq':       np.int64     (K,)  global event count for that code.
    'patient_ids':     list[str]    length P.
    'patient_split':   list[str]    length P, 'train'/'val'/'test'.
    'patient_visits':  dict[pid -> list[(ts_min, code_idx_list)]],
                        sorted by ts; ts is minutes since 1970.
    'pmi_topk_adj':    np.bool      (K, K) symmetric mask. For each code i,
                        the (up to) `--pmi_topk` partners j with highest PMI
                        across the training cohort. Visit-time adjacency in
                        the dataset class is the intersection of this static
                        skeleton with the (active outer active) mask, so
                        edges only appear for clinically meaningful pairs
                        that are both present at the current visit. Set to
                        all-True (i.e. clique fallback) when --pmi_topk=0.
  }
"""
from __future__ import annotations

import argparse
import datetime as dt
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# --- vocabularies and table filters ---------------------------------------
KEEP_OMOP_TABLES = {
    "measurement",
    "observation",
    "drug_exposure",
    "procedure_occurrence",
    "condition_occurrence",
    "device_exposure",
}
DROP_CODE_PREFIXES = {
    "Visit", "Domain", "CARE_SITE", "Medicare Specialty",
    "CMS Place of Service", "Gender", "Race", "Ethnicity",
}
# Stable indexing for vocabulary type. Anything else gets the 'other' bucket.
VOCAB_TYPE_ID = {
    "LOINC": 0, "SNOMED": 1, "RxNorm": 2, "CPT4": 3,
    "ICD10PCS": 4, "ICD9Proc": 5, "RxNorm Extension": 6,
    "Cancer Modifier": 7,
}
OTHER_TYPE_ID = 8


def _epoch_minutes(ts_str: str) -> int:
    """'2015-09-07 09:50:00' -> integer minutes since epoch. NaN -> 0."""
    if not isinstance(ts_str, str) or not ts_str:
        return 0
    try:
        d = dt.datetime.fromisoformat(ts_str)
    except ValueError:
        # Some rows have 'T' separator; isoformat handles both since 3.11.
        d = dt.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    return int(d.timestamp() // 60)


def parse_splits(splits_csv: Path) -> dict:
    """Load EHRshot's official train/val/test split mapping."""
    df = pd.read_csv(splits_csv)
    return dict(zip(df["omop_person_id"].astype(str), df["split"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="dataset/ehrshot/ehrshot.csv",
                   help="Path to raw ehrshot.csv (~41M rows).")
    p.add_argument("--splits", default="dataset/ehrshot/person_id_map.csv")
    p.add_argument("--out", default="dataset/ehrshot/ehrshot_cohort.pkl",
                   help="Output pickle path.")
    p.add_argument("--top_k_codes", type=int, default=500,
                   help="Vocabulary size: keep this many most-frequent codes.")
    p.add_argument("--min_visits", type=int, default=4,
                   help="Drop patients with fewer than this many valid visits.")
    p.add_argument("--min_year", type=int, default=2010,
                   help="Drop events earlier than Jan 1 of this year (heavy "
                        "tail before 2010 in EHRshot).")
    p.add_argument("--chunksize", type=int, default=1_000_000,
                   help="Rows per pandas chunk; lower if RAM-constrained.")
    p.add_argument("--max_patients", type=int, default=0,
                   help="If >0, keep only top-N patients by event count "
                        "(for smoke tests). 0 = keep all.")
    p.add_argument("--pmi_topk", type=int, default=10,
                   help="For each code i, keep edges only to its top-K "
                        "highest-PMI partners across the training cohort. "
                        "0 disables PMI filtering (visit-clique adjacency, "
                        "which makes the graph layer redundant with the "
                        "active mask).")
    p.add_argument("--pmi_min_pair_count", type=int, default=3,
                   help="Minimum joint count for a (i, j) pair to be eligible "
                        "for the top-K filter. Filters out single-coincidence "
                        "pairs that get inflated PMI from low marginals.")
    args = p.parse_args()

    csv_path = Path(args.csv).expanduser()
    splits_path = Path(args.splits).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ehrshot] csv={csv_path}  size={csv_path.stat().st_size / 1e9:.2f} GB")
    print(f"[ehrshot] splits={splits_path}")
    print(f"[ehrshot] top_k_codes={args.top_k_codes}  min_visits={args.min_visits}"
          f"  min_year={args.min_year}", flush=True)

    pid_to_split = parse_splits(splits_path)
    print(f"[ehrshot] split map: {len(pid_to_split)} patients in person_id_map.csv")

    # First pass: tally code frequency over training patients only, so the
    # vocabulary doesn't leak signal from val/test.
    print("\n[pass 1/2] computing code frequencies on training cohort...",
          flush=True)
    train_pids = {p for p, s in pid_to_split.items() if s == "train"}
    code_counts: Counter = Counter()
    chunks_seen = 0
    rows_seen = 0
    cutoff = f"{args.min_year}-01-01"
    for chunk in pd.read_csv(
            csv_path,
            usecols=["patient_id", "start", "code", "visit_id", "omop_table"],
            chunksize=args.chunksize,
            dtype={"patient_id": str, "code": str, "omop_table": str,
                   "visit_id": str, "start": str}):
        chunks_seen += 1
        rows_seen += len(chunk)
        chunk = chunk[chunk["omop_table"].isin(KEEP_OMOP_TABLES)]
        chunk = chunk[chunk["start"] >= cutoff]
        if chunk.empty: continue
        # Drop bookkeeping vocabularies.
        prefix = chunk["code"].str.split("/").str[0]
        chunk = chunk[~prefix.isin(DROP_CODE_PREFIXES)]
        chunk = chunk[chunk["patient_id"].isin(train_pids)]
        if chunk.empty: continue
        code_counts.update(chunk["code"].values.tolist())
        if chunks_seen % 5 == 0:
            print(f"  chunk {chunks_seen}: {rows_seen:,} rows seen, "
                  f"{len(code_counts):,} unique codes", flush=True)
    print(f"[pass 1/2] done. {rows_seen:,} rows scanned, "
          f"{len(code_counts):,} unique training codes.")

    # Pick top-K and freeze the vocabulary.
    top_k = code_counts.most_common(args.top_k_codes)
    code_vocab = [c for c, _ in top_k]
    code_to_idx = {c: i for i, c in enumerate(code_vocab)}
    code_freq = np.array([n for _, n in top_k], dtype=np.int64)
    code_type = np.array(
        [VOCAB_TYPE_ID.get(c.split("/")[0], OTHER_TYPE_ID) for c in code_vocab],
        dtype=np.int32,
    )
    print(f"[vocab] kept top {len(code_vocab)} codes  "
          f"(min_freq={top_k[-1][1]:,}, max_freq={top_k[0][1]:,})")

    # Second pass: per-patient per-visit code aggregation over ALL split
    # patients (val/test included, but only with codes in the train vocab).
    print("\n[pass 2/2] building per-patient visit sequences...", flush=True)
    # patient_id -> visit_id -> {'ts': int (first event ts in visit),
    #                            'codes': set[int]}
    patient_visits: dict = defaultdict(lambda: defaultdict(
        lambda: {"ts": None, "codes": set()}))
    chunks_seen = 0
    rows_kept = 0
    for chunk in pd.read_csv(
            csv_path,
            usecols=["patient_id", "start", "code", "visit_id", "omop_table"],
            chunksize=args.chunksize,
            dtype={"patient_id": str, "code": str, "omop_table": str,
                   "visit_id": str, "start": str}):
        chunks_seen += 1
        chunk = chunk[chunk["omop_table"].isin(KEEP_OMOP_TABLES)]
        chunk = chunk[chunk["start"] >= cutoff]
        chunk = chunk[chunk["patient_id"].isin(pid_to_split)]
        chunk = chunk[chunk["code"].isin(code_to_idx)]
        chunk = chunk[chunk["visit_id"].notna() & (chunk["visit_id"] != "")]
        if chunk.empty: continue
        rows_kept += len(chunk)
        chunk["code_idx"] = chunk["code"].map(code_to_idx).astype(np.int32)
        chunk["ts_min"] = chunk["start"].map(_epoch_minutes).astype(np.int64)
        for pid, vid, code_idx, ts in zip(
                chunk["patient_id"].values,
                chunk["visit_id"].values,
                chunk["code_idx"].values,
                chunk["ts_min"].values):
            v = patient_visits[pid][vid]
            v["codes"].add(int(code_idx))
            t = int(ts)
            if v["ts"] is None or t < v["ts"]:
                v["ts"] = t
        if chunks_seen % 5 == 0:
            print(f"  chunk {chunks_seen}: {rows_kept:,} rows kept, "
                  f"{len(patient_visits):,} patients so far", flush=True)
    print(f"[pass 2/2] done. {rows_kept:,} rows aggregated, "
          f"{len(patient_visits):,} patients have at least one event.")

    # Convert to sorted (ts, sorted_code_list) tuples per patient. Drop
    # patients with too few visits.
    visits_final: dict = {}
    n_dropped_too_few = 0
    for pid, visits in patient_visits.items():
        seq = [(v["ts"], sorted(v["codes"])) for v in visits.values()
               if v["ts"] is not None and v["codes"]]
        seq.sort(key=lambda x: x[0])
        if len(seq) < args.min_visits:
            n_dropped_too_few += 1
            continue
        visits_final[pid] = seq

    print(f"[filter] dropped {n_dropped_too_few:,} patients "
          f"(< {args.min_visits} visits); kept {len(visits_final):,}")

    # --- Static co-occurrence skeleton (PMI top-K per code) ---
    # Without this filter the per-visit adjacency would be a full clique over
    # all active codes, making the graph layer redundant with the active mask
    # (a GCN over a clique is just a mean-pool). Here we compute population-
    # level pointwise mutual information from training visits and keep, for
    # each code i, the K partners with highest PMI. The per-visit adjacency
    # in the dataset class is then the intersection of this skeleton with
    # (active outer active), so the encoder sees a sparse, structurally
    # meaningful graph instead of a clique.
    N = len(code_vocab)
    if args.pmi_topk > 0:
        print(f"\n[pmi] computing PMI top-{args.pmi_topk} skeleton on "
              f"training cohort", flush=True)
        code_visit_count = np.zeros(N, dtype=np.int64)
        pair_count = np.zeros((N, N), dtype=np.int64)
        n_train_visits = 0
        for pid, seq in visits_final.items():
            if pid_to_split.get(pid) != "train":
                continue
            for _, codes in seq:
                if len(codes) < 2:
                    code_visit_count[np.asarray(codes, dtype=np.int64)] += 1
                    n_train_visits += 1
                    continue
                arr = np.asarray(codes, dtype=np.int64)
                code_visit_count[arr] += 1
                pair_count[np.ix_(arr, arr)] += 1
                n_train_visits += 1
        np.fill_diagonal(pair_count, 0)
        print(f"[pmi] {n_train_visits:,} training visits scanned, "
              f"{int(pair_count.sum() // 2):,} unordered co-occurrence events")

        # Min-pair-count filter to avoid PMI inflation from one-off coincidences.
        pair_count[pair_count < args.pmi_min_pair_count] = 0

        eps = 1e-9
        c_i = code_visit_count.astype(np.float64)
        N_v = max(n_train_visits, 1)
        # PMI(i, j) = log( N * c_ij / (c_i * c_j) ); -inf when pair count is 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_pmi = np.log(
                (pair_count.astype(np.float64) * N_v + eps) /
                (c_i[:, None] * c_i[None, :] + eps)
            )
        log_pmi[pair_count == 0] = -np.inf

        # Top-K per row, then symmetrise. argpartition is O(N) per row.
        K = min(args.pmi_topk, N - 1)
        adj = np.zeros((N, N), dtype=bool)
        for i in range(N):
            row = log_pmi[i]
            # Negative for descending: largest K via argpartition.
            idx = np.argpartition(-row, K)[:K]
            # Drop any partners that are -inf (i.e. truly zero co-occurrence).
            idx = idx[np.isfinite(row[idx])]
            adj[i, idx] = True
        adj = adj | adj.T            # symmetric edge
        np.fill_diagonal(adj, False) # no self-loops
        pmi_topk_adj = adj
        print(f"[pmi] skeleton: K={K}, edges={int(adj.sum() // 2):,}, "
              f"density={adj.sum() / (N * N):.4f}, "
              f"mean_degree={adj.sum(1).mean():.1f}")
    else:
        pmi_topk_adj = np.ones((N, N), dtype=bool)
        np.fill_diagonal(pmi_topk_adj, False)
        print(f"\n[pmi] disabled (--pmi_topk=0); using clique fallback "
              f"adjacency (graph layer reduces to active-mask mean-pool)")

    if args.max_patients > 0 and len(visits_final) > args.max_patients:
        # Smoke-test mode: keep top-N patients by visit count.
        pids_sorted = sorted(visits_final, key=lambda p: len(visits_final[p]),
                             reverse=True)[:args.max_patients]
        visits_final = {p: visits_final[p] for p in pids_sorted}
        print(f"[smoke] truncated to top {args.max_patients} patients "
              f"by visit count")

    pids_ordered = sorted(visits_final.keys())
    splits_ordered = [pid_to_split[p] for p in pids_ordered]
    n_per_split = Counter(splits_ordered)
    print(f"[final] {len(pids_ordered):,} patients in cohort: "
          f"train={n_per_split.get('train', 0)} "
          f"val={n_per_split.get('val', 0)} "
          f"test={n_per_split.get('test', 0)}")

    visit_lens = [len(visits_final[p]) for p in pids_ordered]
    print(f"[final] visits/patient: min={min(visit_lens)} "
          f"median={int(np.median(visit_lens))} "
          f"mean={np.mean(visit_lens):.1f} "
          f"max={max(visit_lens)}")
    codes_per_visit = [len(c) for p in pids_ordered for _, c in visits_final[p]]
    print(f"[final] codes/visit: min={min(codes_per_visit)} "
          f"median={int(np.median(codes_per_visit))} "
          f"mean={np.mean(codes_per_visit):.1f} "
          f"max={max(codes_per_visit)}")

    payload = {
        "code_vocab": code_vocab,
        "code_vocab_type": code_type,
        "code_freq": code_freq,
        "patient_ids": pids_ordered,
        "patient_split": splits_ordered,
        "patient_visits": visits_final,
        "pmi_topk_adj": pmi_topk_adj,
        "args": vars(args),
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[done] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
