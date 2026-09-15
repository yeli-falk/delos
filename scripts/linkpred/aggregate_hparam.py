"""Rank the link-prediction hyperparameter-search cells.

Third of the three sweep aggregators. Scans a root written by
``scripts/linkpred/slurm_hparam_search.sh``; every cell there scored the
VALIDATION split, so ranking on it does not touch test.

Reads either writer, as the final aggregator does:
  JEPA cells   test_metrics.json   (train_dyg_jepa --probe_split val)
  baselines    analysis_val.json   (analyze_jepa --probe_split val)

Ranked on MRR over the ``new`` subset by default. That choice matters: the
``all`` subset is dominated by edges already present in the history window,
which a model scores well by echoing the current graph, so ranking on it
largely ranks how well a method reproduces the input. ``new`` is edge
FORMATION, which is what the predictor is supposed to contribute.
``--subset all`` / ``--metric Hits@10`` override.

Usage:
    python scripts/linkpred/aggregate_hparam.py --root outputs/lp_hparam/mooc_predictor --emit_flags
    python scripts/linkpred/aggregate_hparam.py --root outputs/lp_hparam/mooc_variants --by_arm --emit_flags
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path


def _load(p: Path):
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def read_cell(d: Path):
    """The {"all":..,"new":..} block from whichever writer produced this cell."""
    b = _load(d / "test_metrics.json")
    if b:
        return b
    b = _load(d / "analysis_val.json")
    if b:
        lp = b.get("link_pred") or {}
        for src in ("encoder", "jepa", "supervised", "random"):
            if lp.get(src):
                return lp[src]
    return None


def score(block, subset, metric):
    sub = (block or {}).get(subset)
    if not isinstance(sub, dict):
        return None
    vals = [m[metric] for m in sub.values()
            if isinstance(m, dict) and metric in m
            and m[metric] is not None
            and not (isinstance(m[metric], float) and math.isnan(m[metric]))]
    return st.mean(vals) if vals else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--metric", default="MRR")
    p.add_argument("--subset", choices=["all", "new"], default="new")
    p.add_argument("--by_arm", action="store_true")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--csv", default=None)
    p.add_argument("--flag_prefix", default=None)
    p.add_argument("--emit_flags", action="store_true")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist.")

    rows = []
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        if d.name.startswith("_"):
            continue
        blk = read_cell(d)
        if blk is None:
            print(f"[skip] {d.name}: no val metrics (unfinished or failed)")
            continue
        s_new, s_all = score(blk, "new", args.metric), score(blk, "all", args.metric)
        key = s_new if args.subset == "new" else s_all
        if key is None:
            print(f"[skip] {d.name}: no {args.metric} on subset '{args.subset}'")
            continue
        hp = ""
        f = d / "hp_args.txt"
        if f.exists():
            hp = f.read_text().strip()
        rows.append({"cell": d.name, "arm": d.name.split("_", 1)[0],
                     "score": key, "new": s_new, "all": s_all, "hp": hp})
    if not rows:
        raise SystemExit(f"No finished cells under {root}.")
    rows.sort(key=lambda r: r["score"], reverse=True)     # HIGHER is better

    print(f"\n{len(rows)} cells under {root}")
    print(f"(ranked on VAL {args.metric} / subset '{args.subset}', HIGHER is better)")
    print(f"{'rank':<5}{'cell':<50}{'new':>10}{'all':>10}")
    print("-" * 76)
    def _f(v): return f"{v:.4f}" if v is not None else "    --  "
    for i, r in enumerate(rows[:args.top], 1):
        print(f"{i:<5}{r['cell'][:48]:<50}{_f(r['new']):>10}{_f(r['all']):>10}")

    # A cell whose `new` score is at or below a 1/(1+negatives) floor has not
    # learned edge formation, however it ranks on `all`.
    print(f"\nSpread on '{args.subset}': "
          f"{rows[-1]['score']:.4f} .. {rows[0]['score']:.4f} "
          f"(range {rows[0]['score'] - rows[-1]['score']:.4f})")

    winners = {}
    if args.by_arm:
        print(f"\n{'arm':<12}{'best cell':<50}{'score':>10}")
        print("-" * 72)
        for arm in sorted({r["arm"] for r in rows}):
            best = max((r for r in rows if r["arm"] == arm), key=lambda r: r["score"])
            winners[arm] = best
            print(f"{arm:<12}{best['cell'][:48]:<50}{best['score']:>10.4f}")
    else:
        winners["best"] = rows[0]

    if args.emit_flags:
        prefix = args.flag_prefix or (
            "BASELINE_HP" if root.name.endswith("_baselines") else "VARIANT_HP")
        print("\n# --- paste into scripts/linkpred/slurm_all_methods.sh ---")
        for arm, r in winners.items():
            flags = r["hp"].split("flags=", 1)[-1].split("base=")[0].strip()
            if args.by_arm:
                print(f'{prefix}_{arm}="{flags}"')
            else:
                print(f'BASE_HP="{flags}"')

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\n[saved] {args.csv}")


if __name__ == "__main__":
    main()
