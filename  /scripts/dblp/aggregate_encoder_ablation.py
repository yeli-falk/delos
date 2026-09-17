#!/usr/bin/env python3
"""Aggregate the encoder ablation into a mean±std table.

Walks the ablation output tree written by ``slurm_encoder_ablation.sh``, reads
every ``test_metrics.json`` (plus its sibling ``args.json`` for the run's
data_source / encoder_kind / seed), and prints one table per dataset with rows
= encoders and columns = primary metrics, each cell ``mean ± std`` over seeds.
The best encoder per column is marked with ``*``.

Metric files come in two shapes:
  * link prediction  -> {"all": {tp: {MRR, Hits@k, AUC}}, "new": {...}}
  * regression       -> {horizon: {MAE, RMSE, MAPE}}
Both are reduced by averaging each metric over horizons.

Usage:
    python scripts/dblp/aggregate_encoder_ablation.py [ROOT]
    python scripts/dblp/aggregate_encoder_ablation.py outputs/encoder_ablation --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Which metrics to report per task, and whether higher is better.
LINK_METRICS = [("MRR", True), ("Hits@10", True), ("AUC", True)]
REG_METRICS = [("MAE", False), ("RMSE", False)]


def reduce_metrics(blob: dict):
    """Collapse a test_metrics.json into {metric_name: scalar}, averaged over
    horizons. Returns (task, {metric: value})."""
    if "all" in blob or "new" in blob:                      # link prediction
        per_h = blob.get("all", {})
        out = {}
        for name, _ in LINK_METRICS:
            vals = [m[name] for m in per_h.values() if name in m]
            if vals:
                out[name] = statistics.mean(vals)
        return "link_pred", out
    # regression: top-level keys are horizons
    out = {}
    for name, _ in REG_METRICS:
        vals = [m[name] for m in blob.values()
                if isinstance(m, dict) and name in m]
        if vals:
            out[name] = statistics.mean(vals)
    return "regression", out


def load_run(metrics_path: Path):
    """Return (data_source, encoder_kind, seed, task, {metric: val}) or None."""
    try:
        blob = json.loads(metrics_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    task, reduced = reduce_metrics(blob)
    if not reduced:
        return None

    # Prefer args.json (authoritative); fall back to the path layout
    # outputs/encoder_ablation/<dataset>/<enc>_s<seed>/jepa/.
    args_path = metrics_path.parent / "args.json"
    data_source = encoder = None
    seed = None
    if args_path.exists():
        try:
            a = json.loads(args_path.read_text())
            data_source = a.get("data_source")
            encoder = a.get("encoder_kind")
            seed = a.get("seed")
        except (json.JSONDecodeError, OSError):
            pass
    if data_source is None or encoder is None:
        run_dir = metrics_path.parent.parent          # <enc>_s<seed>
        data_source = data_source or run_dir.parent.name
        stem = run_dir.name
        if "_s" in stem:
            encoder = encoder or stem.rsplit("_s", 1)[0]
            if seed is None:
                try:
                    seed = int(stem.rsplit("_s", 1)[1])
                except ValueError:
                    pass
    return data_source, encoder, seed, task, reduced


def fmt_cell(vals, best: bool) -> str:
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    star = "*" if best else " "
    return f"{mean:7.4f} ± {std:6.4f}{star}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="outputs/encoder_ablation",
                    help="Ablation output root (default: outputs/encoder_ablation).")
    ap.add_argument("--csv", default=None, help="Also write a long-format CSV here.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"[error] root not found: {root}")

    # (data_source, encoder) -> metric -> [per-seed values]
    table = defaultdict(lambda: defaultdict(list))
    task_of = {}
    seeds_seen = defaultdict(set)
    csv_rows = []

    for mp in sorted(root.rglob("test_metrics.json")):
        rec = load_run(mp)
        if rec is None:
            continue
        ds, enc, seed, task, reduced = rec
        task_of[ds] = task
        seeds_seen[(ds, enc)].add(seed)
        for name, val in reduced.items():
            table[(ds, enc)][name].append(val)
            csv_rows.append({"dataset": ds, "encoder": enc, "seed": seed,
                             "task": task, "metric": name, "value": val})

    if not table:
        raise SystemExit(f"[error] no test_metrics.json found under {root}")

    datasets = sorted({ds for ds, _ in table})
    for ds in datasets:
        task = task_of[ds]
        metrics = LINK_METRICS if task == "link_pred" else REG_METRICS
        encoders = sorted({e for d, e in table if d == ds})

        print(f"\n=== {ds}  ({task}, mean ± std over seeds; * = best) ===")
        header = f"{'encoder':<10} " + " ".join(
            f"{name:>17}" for name, _ in metrics) + "   n_seeds"
        print(header)
        print("-" * len(header))

        # Find best encoder per metric.
        best_enc = {}
        for name, higher in metrics:
            means = {e: statistics.mean(table[(ds, e)][name])
                     for e in encoders if table[(ds, e)].get(name)}
            if means:
                best_enc[name] = (max if higher else min)(means, key=means.get)

        for e in encoders:
            cells = []
            for name, _ in metrics:
                vals = table[(ds, e)].get(name)
                if vals:
                    cells.append(fmt_cell(vals, best_enc.get(name) == e))
                else:
                    cells.append(f"{'n/a':>17}")
            n = len(seeds_seen[(ds, e)])
            print(f"{e:<10} " + " ".join(f"{c:>17}" for c in cells) + f"   {n:>3}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "encoder", "seed",
                                              "task", "metric", "value"])
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n[csv] wrote {len(csv_rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
