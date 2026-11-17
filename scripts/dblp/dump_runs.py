"""Dump a flat table of all runs matching a glob, one row per run.

For each `<TAG>_cjepa/` directory found, also tries to read the matching
`<TAG>_diagnostic/diagnostic.json` so the row carries both the link-prediction
numbers and the representation diagnostics in one place.

Usage:
    # everything
    python scripts/dblp/dump_runs.py

    # just the K-sweep
    python scripts/dblp/dump_runs.py --pattern 'outputs/reddit_body_k*_cjepa'

    # multi-seed
    python scripts/dblp/dump_runs.py --pattern 'outputs/reddit_body_seed*_cjepa'

    # markdown / csv / json
    python scripts/dblp/dump_runs.py --fmt md   > docs/results.md
    python scripts/dblp/dump_runs.py --fmt csv  > docs/results.csv
    python scripts/dblp/dump_runs.py --fmt json > docs/results.json

    # mean +/- std across a seed sweep
    python scripts/dblp/dump_runs.py --pattern 'outputs/reddit_body_seed*_cjepa' --aggregate
"""
from __future__ import annotations

import argparse
import csv
import glob as _glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _read_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_run(delos_dir: Path) -> dict:
    """Pull a single flat row from a <tag>_cjepa directory + its diagnostic."""
    tag = delos_dir.name[:-len("_cjepa")] if delos_dir.name.endswith("_cjepa") \
        else delos_dir.name
    diag_dir = delos_dir.parent / f"{tag}_diagnostic"

    # Prefer the extended metrics file (runs rescored with AUC + Hits@1 by
    # rescore_link_metrics.py); fall back to test_metrics.json otherwise.
    test = (_read_json(delos_dir / "test_metrics_v2.json")
            or _read_json(delos_dir / "test_metrics.json") or {})
    diag = _read_json(diag_dir / "diagnostic.json") or {}
    args = _read_json(delos_dir / "args.json") or {}
    block = _read_json(delos_dir / "block_probe.json") or {}
    state = _read_json(delos_dir / "state_probe.json") or {}

    # Link-prediction metrics are {"all": {h: {...}}, "new": {h: {...}}}; a
    # flat {h: {...}} file is also accepted.
    def _horizon0(block):
        if not block:
            return {}
        if "0" in block:
            return block["0"]
        if 0 in block:
            return block[0]
        return next(iter(block.values()), {})

    if "all" in test or "new" in test:
        all0 = _horizon0(test.get("all", {}))
        new0 = _horizon0(test.get("new", {}))
    else:
        all0 = _horizon0(test)
        new0 = {}

    enc = diag.get("encoder", {}) or {}
    jep = diag.get("jepa", {}) or {}
    row = {
        "tag": tag,
        "MRR_all": all0.get("MRR"),
        "Hits@1_all": all0.get("Hits@1"),
        "Hits@10_all": all0.get("Hits@10"),
        "AUC_all": all0.get("AUC"),
        "MRR_new": new0.get("MRR"),
        "Hits@1_new": new0.get("Hits@1"),
        "Hits@10_new": new0.get("Hits@10"),
        "AUC_new": new0.get("AUC"),
        "enc_eff_rank": _safe_get(enc, "effective_rank", "eff_rank"),
        "jepa_eff_rank": _safe_get(jep, "effective_rank", "eff_rank"),
        "enc_top1_sv": _safe_get(enc, "effective_rank", "top10_energy", default=[None])[0],
        "jepa_top1_sv": _safe_get(jep, "effective_rank", "top10_energy", default=[None])[0],
        "cka": diag.get("cka_enc_pred"),
        "enc_probe_degree": _safe_get(enc, "probe_degree", "r2_test"),
        "jepa_probe_degree": _safe_get(jep, "probe_degree", "r2_test"),
        "enc_probe_kcore": _safe_get(enc, "probe_kcore", "r2_test"),
        "jepa_probe_kcore": _safe_get(jep, "probe_kcore", "r2_test"),
        "data_source": args.get("data_source"),
        "seed": args.get("seed"),
        "enc_block_acc": _safe_get(block, "encoder", "acc_test"),
        "jepa_block_acc": _safe_get(block, "jepa", "acc_test"),
        "enc_block_f1": _safe_get(block, "encoder", "macro_f1_test"),
        "jepa_block_f1": _safe_get(block, "jepa", "macro_f1_test"),
        "block_n_classes": block.get("n_classes_seen"),
        "enc_state_auroc": _safe_get(state, "encoder", "auroc_test"),
        "jepa_state_auroc": _safe_get(state, "jepa", "auroc_test"),
        "deg_state_auroc": _safe_get(state, "degree_baseline", "auroc_test"),
        # Fair copy-last baseline + AUPRC columns.
        "last_state_auroc": _safe_get(state, "encoder_last", "auroc_test"),
        "enc_state_ap": _safe_get(state, "encoder", "auprc_test"),
        "last_state_ap": _safe_get(state, "encoder_last", "auprc_test"),
        "jepa_state_ap": _safe_get(state, "jepa", "auprc_test"),
        # Onset subset (rows not yet positive at the previous snapshot) —
        # the split where label-persistence carries no signal.
        "last_onset_ap": _safe_get(state, "encoder_last", "subset_onset",
                                   "auprc_test"),
        "jepa_onset_ap": _safe_get(state, "jepa", "subset_onset",
                                   "auprc_test"),
        # Activity-trend control: probe on the degree-at-each-history-bin
        # vector. The forecast must beat this to show learned dynamics.
        "deghist_state_auroc": _safe_get(state, "degree_history_baseline",
                                         "auroc_test"),
        "deghist_onset_ap": _safe_get(state, "degree_history_baseline",
                                      "subset_onset", "auprc_test"),
        "state_n_pos_test": state.get("n_pos_test"),
    }
    # Per-horizon curve (multi-horizon runs only; blank at NPRED=1). For each
    # forecast distance k, the fair comparison is jepa vs copy-last AUROC —
    # copy-last reuses an increasingly stale embedding as k grows, so the gap
    # should widen with k if the predictor models real dynamics.
    for k in range(3):
        lk = _safe_get(state, "encoder_last", f"subset_h{k}", "auroc_test")
        jk = _safe_get(state, "jepa", f"subset_h{k}", "auroc_test")
        row[f"last_h{k}_auroc"] = lk
        row[f"jepa_h{k}_auroc"] = jk
        if lk is not None and jk is not None:
            row[f"gap_h{k}"] = jk - lk
    # Derived columns
    if row["jepa_eff_rank"] is not None and row["enc_eff_rank"] is not None:
        row["rank_gap"] = row["jepa_eff_rank"] - row["enc_eff_rank"]
    if row["MRR_new"] is not None and row["MRR_all"] not in (None, 0):
        row["new_over_all"] = row["MRR_new"] / row["MRR_all"]
    if row["jepa_block_acc"] is not None and row["enc_block_acc"] is not None:
        row["block_acc_gap"] = row["jepa_block_acc"] - row["enc_block_acc"]
    if row["jepa_state_auroc"] is not None and row["enc_state_auroc"] is not None:
        row["state_auroc_gap"] = row["jepa_state_auroc"] - row["enc_state_auroc"]
    # The headline number: forecast vs the fair copy-last baseline.
    if row["jepa_state_auroc"] is not None and row["last_state_auroc"] is not None:
        row["state_gap_vs_last"] = row["jepa_state_auroc"] - row["last_state_auroc"]
    return row


COLS = (
    "tag", "seed",
    "MRR_all", "MRR_new", "new_over_all",
    "Hits@1_all", "Hits@1_new",
    "AUC_all", "AUC_new",
    "enc_eff_rank", "jepa_eff_rank", "rank_gap",
    "cka",
    "enc_probe_degree", "jepa_probe_degree",
    "enc_probe_kcore", "jepa_probe_kcore",
    "enc_block_acc", "jepa_block_acc", "block_acc_gap", "block_n_classes",
    "enc_state_auroc", "last_state_auroc", "jepa_state_auroc",
    "deg_state_auroc", "deghist_state_auroc",
    "state_auroc_gap", "state_gap_vs_last",
    "enc_state_ap", "last_state_ap", "jepa_state_ap",
    "last_onset_ap", "jepa_onset_ap", "deghist_onset_ap",
    # Per-horizon curve (multi-horizon runs; blank at NPRED=1).
    "last_h0_auroc", "jepa_h0_auroc", "gap_h0",
    "last_h1_auroc", "jepa_h1_auroc", "gap_h1",
    "last_h2_auroc", "jepa_h2_auroc", "gap_h2",
    "state_n_pos_test",
)


def _fmt(v):
    if v is None: return ""
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)


def print_console(rows):
    widths = {c: max(len(c), max((len(_fmt(r.get(c))) for r in rows), default=0))
              for c in COLS}
    line = "  ".join(c.ljust(widths[c]) for c in COLS)
    print(line)
    print("  ".join("-" * widths[c] for c in COLS))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in COLS))


def print_md(rows):
    print("| " + " | ".join(COLS) + " |")
    print("|" + "|".join("---" for _ in COLS) + "|")
    for r in rows:
        print("| " + " | ".join(_fmt(r.get(c)) for c in COLS) + " |")


def print_csv(rows):
    w = csv.DictWriter(sys.stdout, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c) for c in COLS})


def print_json(rows):
    print(json.dumps(rows, indent=2))


def aggregate_rows(rows):
    """Collapse rows that share a base tag into one row of mean ± std."""
    if not rows:
        return rows
    # Strip trailing _seed\d+ / _s\d+ from each tag to find the base.
    def base(tag):
        return re.sub(r"_s(eed)?\d+$", "", tag)
    groups = defaultdict(list)
    for r in rows:
        groups[base(r["tag"])].append(r)

    numeric_cols = [c for c in COLS if c not in ("tag", "seed")]
    out = []
    for tag, grp in sorted(groups.items()):
        row = {"tag": tag, "seed": f"n={len(grp)}"}
        for c in numeric_cols:
            vals = [r.get(c) for r in grp if isinstance(r.get(c), (int, float))]
            if not vals:
                continue
            mu = mean(vals)
            sd = stdev(vals) if len(vals) > 1 else 0.0
            row[c] = f"{mu:.4f}±{sd:.4f}"
        out.append(row)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="outputs/*_cjepa",
                   help="Glob for the *_cjepa directories.")
    p.add_argument("--fmt", choices=["table", "md", "csv", "json"], default="table")
    p.add_argument("--aggregate", action="store_true",
                   help="Group runs that share a base tag (strip trailing "
                        "_seedNN / _sNN) and report mean ± std per column. "
                        "Use for multi-seed sweeps.")
    args = p.parse_args()

    dirs = sorted(Path(p) for p in _glob.glob(args.pattern))
    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        print(f"[dump_runs] no matches for {args.pattern!r}", file=sys.stderr)
        sys.exit(1)

    rows = [load_run(d) for d in dirs]
    if args.aggregate:
        rows = aggregate_rows(rows)
    elif args.fmt == "table":
        rows.sort(key=lambda r: r["tag"])

    {"table": print_console, "md": print_md,
     "csv": print_csv, "json": print_json}[args.fmt](rows)


if __name__ == "__main__":
    main()
