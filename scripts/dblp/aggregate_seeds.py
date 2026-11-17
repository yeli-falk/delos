"""Aggregate multi-seed link-prediction results into a mean +/- std table.

Reads outputs/dblp_cjepa_s*/test_metrics.json and
outputs/dblp_ablation_probe_s*/test_metrics.json and prints a paper-ready
ablation row pair.

Usage:
    python scripts/dblp/aggregate_seeds.py
    python scripts/dblp/aggregate_seeds.py --pattern "outputs/dblp_*_s*"
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev



METRICS = ("MRR", "Hits@1", "Hits@3", "Hits@10")


def load_metrics(path: Path) -> dict:
    """Return per-metric values averaged across horizons (this matches what
    we report in the paper -- single number per metric)."""
    with open(path) as f:
        data = json.load(f)
    out = {}
    for k in METRICS:
        vals = [m.get(k) for m in data.values() if k in m]
        vals = [v for v in vals if v is not None]
        if vals:
            out[k] = mean(vals)
    return out


def aggregate_runs(dirs):
    """Return per-metric (mean, std, n) across all matching runs."""
    per_metric = defaultdict(list)
    for d in dirs:
        f = d / "test_metrics.json"
        if not f.exists():
            print(f"  [skip] {f}: missing")
            continue
        m = load_metrics(f)
        for k, v in m.items():
            per_metric[k].append(v)
    out = {}
    for k, vs in per_metric.items():
        out[k] = (mean(vs), stdev(vs) if len(vs) >= 2 else 0.0, len(vs))
    return out


def fmt_row(label: str, agg: dict) -> str:
    cells = []
    for k in METRICS:
        if k in agg:
            mu, sd, n = agg[k]
            cells.append(f"{mu:.3f} ± {sd:.3f}")
        else:
            cells.append("---")
    n = max((v[2] for v in agg.values()), default=0)
    return f"  {label:<35}  " + "  ".join(f"{c:>16}" for c in cells) + f"  n={n}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cjepa_glob", default="outputs/dblp_cjepa_s*",
                   help="Directory glob for C-JEPA seeded runs.")
    p.add_argument("--probe_glob", default="outputs/dblp_ablation_probe_s*",
                   help="Directory glob for pretrained-probe seeded runs.")
    args = p.parse_args()

    cjepa_dirs = sorted(Path().glob(args.cjepa_glob))
    probe_dirs = sorted(Path().glob(args.probe_glob))

    print(f"[found] {len(cjepa_dirs)} C-JEPA runs, {len(probe_dirs)} probe runs")
    print()
    print(f"  {'method':<35}  " + "  ".join(f"{m:>16}" for m in METRICS))
    print("  " + "-" * 105)

    cjepa = aggregate_runs(cjepa_dirs)
    probe = aggregate_runs(probe_dirs)
    print(fmt_row("Pretrained encoder + probe", probe))
    print(fmt_row("C-JEPA (ours)", cjepa))
    print()

    # Also dump as JSON for downstream use.
    out = {
        "C-JEPA": {k: {"mean": v[0], "std": v[1], "n": v[2]} for k, v in cjepa.items()},
        "pretrained_probe": {k: {"mean": v[0], "std": v[1], "n": v[2]} for k, v in probe.items()},
    }
    with open("outputs/seed_aggregate.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] outputs/seed_aggregate.json")

    # LaTeX-ready table (mean +/- std)
    print("\n--- LaTeX rows (paste into your ablation table) ---")
    def latex_cells(agg):
        cells = []
        for k in METRICS:
            if k in agg:
                mu, sd, _ = agg[k]
                cells.append(f"${mu:.3f} \\pm {sd:.3f}$")
            else:
                cells.append("---")
        return cells
    if probe:
        print("Pretrained encoder $+$ probe   & " + "  & ".join(latex_cells(probe)) + " \\\\")
    if cjepa:
        print("\\textbf{C-JEPA (ours)}        & " + "  & ".join(latex_cells(cjepa)) + " \\\\")


if __name__ == "__main__":
    main()
