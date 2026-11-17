"""TTGBench dynamics audit, read straight from dataset/ttgbench.zip.

TTGBench ships user-item interaction STREAMS (ml_<name>.csv: u, i, ts, label,
idx) with one coarse category label per interaction (<name>_labels.json: an
int or a list of ints per event, indexing <name>_unique_labels.json). The
label is the ITEM's category, so it is static per item; the dynamic node
state a forecaster could target is a USER's dominant consumed category per
time bin, built exactly like CHASE Music's "dominant listening genre per
month". This script builds that label and reports the same layer-1 / layer-4
numbers as scripts/analysis/dataset_dynamics.py (persistence floor, switch
rate, recurrence, Jaccard of the user-item snapshots) so the sets can be
ranked against MOOC / CHASE / BTS before anyone writes a loader.

Binning: --bins B equal-frequency over events (JODIE's rule; B=0 = calendar
--months M per bin). A user is labelled in a bin when it has >= --min_events
interactions there; the label is the most frequent category (multi-label
events count once per category; ties -> lowest id). Items are never labelled.

  python scripts/analysis/ttgbench_dynamics.py                 # all labelled sets
  python scripts/analysis/ttgbench_dynamics.py --datasets FOOD IMDB --bins 30
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import zipfile

import numpy as np
import pandas as pd

LABELLED = ["FOOD", "IMDB", "Beeradvocate", "Ratebeer", "Amazon-Kindle"]


def load(z, d):
    df = pd.read_csv(z.open(f"{d}/{d}/ml_{d}.csv"), usecols=["u", "i", "ts"])
    names = json.load(z.open(f"{d}/{d}/{d}_unique_labels.json"))
    raw = json.load(z.open(f"{d}/{d}/{d}_labels.json"))
    if len(raw) != len(df):
        raise ValueError(f"{d}: {len(raw)} labels for {len(df)} events")
    # explode to (event, category) pairs
    ev, cat = [], []
    for e, x in enumerate(raw):
        if isinstance(x, int):
            ev.append(e); cat.append(x)
        else:
            for c in x:
                ev.append(e); cat.append(int(c))
    pairs = pd.DataFrame({"e": np.asarray(ev, np.int64), "c": np.asarray(cat, np.int64)})
    return df, pairs, names


def assign_bins(ts, bins, months):
    if bins > 0:
        order = np.argsort(ts, kind="stable")
        b = np.empty(len(ts), np.int64)
        b[order] = np.minimum((np.arange(len(ts)) * bins) // len(ts), bins - 1)
        return b
    t = pd.to_datetime(ts, unit="s")
    m = (t.year - t.year.min()) * 12 + (t.month - 1)
    return (m // months).to_numpy()


def audit(df, pairs, names, bins, months, min_events, max_lag, splits=(0.7, 0.1, 0.2)):
    C = len(names)
    b = assign_bins(df.ts.values, bins, months)
    T = int(b.max()) + 1
    users = np.unique(df.u.values); u_idx = {u: k for k, u in enumerate(users)}
    N = len(users)
    ue = np.array([u_idx[u] for u in df.u.values])
    # events per (bin, user)
    ev_cnt = np.zeros((T, N), np.int64)
    np.add.at(ev_cnt, (b, ue), 1)
    # category votes per (bin, user, cat)
    votes = np.zeros((T, N, C), np.int64)
    np.add.at(votes, (b[pairs.e.values], ue[pairs.e.values], pairs.c.values), 1)
    labels = np.where(ev_cnt >= min_events, votes.argmax(-1), -1)
    v = labels >= 0
    both = v[1:] & v[:-1]
    same = (labels[1:] == labels[:-1]) & both
    ch = (labels[1:] != labels[:-1]) & both
    n_tr = int(T * splits[0]); n_va = int(T * splits[1]); te0 = n_tr + n_va - 1
    n_ch = ch.sum(0); ever = v.any(0)
    lags = {k: float((((labels[k:] == labels[:-k]) & v[k:] & v[:-k]).sum()
                      / max((v[k:] & v[:-k]).sum(), 1))) for k in range(1, min(max_lag, T - 1) + 1)}
    # structure: user-item edge sets per bin
    key = df.u.values.astype(np.int64) * (df.i.max() + 1) + df.i.values
    E = [np.unique(key[b == t]) for t in range(T)]
    jac = []
    new = []; gone = []
    for t in range(1, T):
        inter = np.intersect1d(E[t - 1], E[t], assume_unique=True).size
        uni = E[t - 1].size + E[t].size - inter
        jac.append(inter / uni if uni else np.nan)
        new.append(1 - inter / max(E[t].size, 1)); gone.append(1 - inter / max(E[t - 1].size, 1))
    all_e = np.unique(key).size
    # users active per bin / labelled per bin
    act = ev_cnt > 0
    tr = labels[:n_tr][v[:n_tr]]
    freq = np.bincount(tr, minlength=C) / max(len(tr), 1) if len(tr) else np.zeros(C)
    return dict(
        T=T, N=N, C=C, items=int(df.i.nunique()), events=int(len(df)),
        labelled_share=float(v.mean()),
        labelled_users_share=float(ever.mean()),
        users_with_ge_3_labelled_bins=int((v.sum(0) >= 3).sum()),
        both_pairs=int(both.sum()),
        persistence=float(same.sum() / max(both.sum(), 1)),
        persistence_test=float(same[te0:].sum() / max(both[te0:].sum(), 1)),
        switch_rate=float(ch.sum() / max(both.sum(), 1)),
        nodes_1plus=float((n_ch >= 1)[ever].mean()), nodes_2plus=float((n_ch >= 2)[ever].mean()),
        recurring=float((n_ch >= 2).sum() / max((n_ch >= 1).sum(), 1)),
        persistence_at_lag=lags, train_class_freq=freq.round(3).tolist(),
        majority_share=float(freq.max()) if len(tr) else np.nan,
        jaccard1=float(np.nanmean(jac)), edges_new=float(np.mean(new)), edges_gone=float(np.mean(gone)),
        kept_share=float(np.mean([e.size for e in E]) / max(all_e, 1)),
        active_per_bin=(int(act.sum(1).min()), int(act.sum(1).max())),
        labelled_per_bin=(int(v.sum(1).min()), int(v.sum(1).max())),
        splits=(n_tr, n_va, T - n_tr - n_va))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default="dataset/ttgbench.zip")
    ap.add_argument("--datasets", nargs="+", default=LABELLED)
    ap.add_argument("--bins", type=int, default=30, help="equal-frequency bins (0 = calendar months)")
    ap.add_argument("--months", type=int, default=1, help="months per bin when --bins 0")
    ap.add_argument("--min_events", type=int, default=2, help="events a user needs in a bin to be labelled")
    ap.add_argument("--max_lag", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    z = zipfile.ZipFile(a.zip)
    rows = {}
    for d in a.datasets:
        df, pairs, names = load(z, d)
        t0, t1 = (dt.datetime.utcfromtimestamp(float(x)).date() for x in (df.ts.min(), df.ts.max()))
        r = audit(df, pairs, names, a.bins, a.months, a.min_events, a.max_lag)
        rows[d] = r
        print(f"\n=== {d}: events={r['events']:,} users={r['N']:,} items={r['items']:,} "
              f"classes={r['C']} span {t0}..{t1}  bins={r['T']} ({'/'.join(map(str, r['splits']))})")
        print(f"  labels    labelled cells={r['labelled_share']:.3f}  users ever labelled={r['labelled_users_share']:.2f} "
              f"(>=3 labelled bins: {r['users_with_ge_3_labelled_bins']:,})  consecutive labelled pairs={r['both_pairs']:,}")
        print(f"            train class freq={r['train_class_freq']}  majority={r['majority_share']:.3f}")
        print(f"            persistence={r['persistence']:.3f} (test {r['persistence_test']:.3f})  switch rate={r['switch_rate']:.3f}  "
              f"users 1+={r['nodes_1plus']:.2f} 2+={r['nodes_2plus']:.2f}  recurring={r['recurring']:.2f}")
        print("            persistence at lag k: " + "  ".join(f"{k}:{v:.3f}" for k, v in r["persistence_at_lag"].items()))
        print(f"  structure Jaccard={r['jaccard1']:.3f}  new={r['edges_new']:.3f} gone={r['edges_gone']:.3f}  "
              f"share of all-time edges/bin={r['kept_share']:.3f}")
        print(f"  coverage  active users/bin {r['active_per_bin'][0]:,}-{r['active_per_bin'][1]:,}  "
              f"labelled users/bin {r['labelled_per_bin'][0]:,}-{r['labelled_per_bin'][1]:,}")
        del df, pairs
    cols = ["dataset", "bins", "users", "C", "lab%", "pairs", "majority", "persist(te)", "switch%", "users1+%", "recur%", "jac1", "kept%"]
    tab = [[d, r["T"], f"{r['N']:,}", r["C"], f"{100 * r['labelled_share']:.1f}", f"{r['both_pairs']:,}",
            f"{r['majority_share']:.2f}", f"{r['persistence_test']:.3f}", f"{100 * r['switch_rate']:.1f}",
            f"{100 * r['nodes_1plus']:.0f}", f"{100 * r['recurring']:.0f}", f"{r['jaccard1']:.3f}",
            f"{100 * r['kept_share']:.1f}"] for d, r in rows.items()]
    w = [max(len(str(x)) for x in col) for col in zip(cols, *tab)]
    print("\n" + "  ".join(c.rjust(n) for c, n in zip(cols, w)))
    for t in tab:
        print("  ".join(str(x).rjust(n) for x, n in zip(t, w)))
    print("\n  Labels = a user's dominant consumed category per bin (>= min_events events). pairs = consecutive "
          "labelled (bin, user) cells the persistence floor is\n  measured on; majority = largest train class "
          "share (the constant-prediction floor); jac1 / kept% on the user-item edge sets per bin.")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1, default=float)


if __name__ == "__main__":
    main()
