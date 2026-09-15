#!/usr/bin/env python
"""Collect and rank the DyG-JEPA hyperparameter-search runs.

Scans a sweep root (default: ``outputs/dyg_hparam``) for per-cell run
directories produced by ``scripts/dblp/slurm_hparam_search.sh``. Each cell
holds:

  * ``test_metrics.json`` -- downstream metrics written at the end of training
  * ``best.ckpt``         -- carries the selection-criterion ``val_loss``

Runs are ranked by validation loss (lower is better) so the winning
configuration is unambiguous and not chosen on the test set. The matching
test metrics are printed alongside for convenience.

Usage:
    python scripts/dblp/aggregate_hparam_search.py [SWEEP_ROOT] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _val_loss(run_dir: Path) -> float | None:
    """Read val_loss from best.ckpt without importing torch if avoidable."""
    ckpt = run_dir / "best.ckpt"
    if not ckpt.exists():
        return None
    try:
        import torch
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        return blob.get("val_loss")
    except Exception as e:  # pragma: no cover - diagnostic path
        print(f"[warn] could not read {ckpt}: {e}", file=sys.stderr)
        return None


def collect(root: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics_path = run_dir / "test_metrics.json"
        if not metrics_path.exists():
            print(f"[skip] no test_metrics.json in {run_dir.name} "
                  f"(unfinished or failed)", file=sys.stderr)
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        rows.append({
            "run": run_dir.name,
            "val_loss": _val_loss(run_dir),
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="outputs/dyg_hparam",
                    help="sweep root containing per-cell run directories")
    ap.add_argument("--csv", default=None, help="optional path to dump a CSV")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"[error] sweep root not found: {root}")

    rows = collect(root)
    if not rows:
        sys.exit(f"[error] no finished runs under {root}")

    # Rank by val_loss (lower better); runs missing it sink to the bottom.
    rows.sort(key=lambda r: (r.get("val_loss") is None,
                             r.get("val_loss", float("inf"))))

    cols = ["run", "val_loss"] + sorted(
        {k for r in rows for k in r if k not in ("run", "val_loss")})

    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))

    best = rows[0]
    print(f"\n[best] {best['run']}  (val_loss="
          f"{_fmt(best.get('val_loss'))})")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"[csv] wrote {args.csv}")


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    main()
