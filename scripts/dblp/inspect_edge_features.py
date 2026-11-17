"""Inspect the scale of the cached edge features.

Loads the cached LIWC features for JODIE Reddit and SNAP Reddit-Body and
reports per-dim statistics, plus the pre-aggregation magnitudes. Features
with a huge native scale make an unnormalised MSE pretraining loss unstable,
because a few dimensions dominate it.

Usage:
    python scripts/dblp/inspect_edge_features.py
    python scripts/dblp/inspect_edge_features.py --path dataset/reddit/soc-redditHyperlinks-body.feat.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


CANDIDATES = [
    "dataset/reddit/soc-redditHyperlinks-body.feat.npz",
    "dataset/reddit/soc-redditHyperlinks-title.feat.npz",
    "dataset/jodie/reddit.feat.npz",
    "dataset/jodie/wikipedia.feat.npz",
]


def _summarise(features: np.ndarray, name: str) -> None:
    """Print a per-dim summary of an (E, F) edge-feature matrix."""
    print(f"\n=== {name}  shape={features.shape}  dtype={features.dtype} ===")
    if features.size == 0:
        print("  (empty)"); return

    # Overall stats
    print(f"  GLOBAL    min={features.min():+.4g}   "
          f"max={features.max():+.4g}   "
          f"mean={features.mean():+.4g}   "
          f"std={features.std():.4g}")

    # Per-dim stats — show the most extreme dims to expose scale variance
    per_dim_std = features.std(axis=0)
    per_dim_max = np.abs(features).max(axis=0)
    print(f"  per-dim std:  min={per_dim_std.min():.4g}  "
          f"median={np.median(per_dim_std):.4g}  "
          f"max={per_dim_std.max():.4g}")
    print(f"  per-dim |max|: min={per_dim_max.min():.4g}  "
          f"median={np.median(per_dim_max):.4g}  "
          f"max={per_dim_max.max():.4g}")

    # The crucial signal: if any dim has std >> 1, the MSE loss is dominated
    # by that dim. Anything above 10 is suspicious; above 100 is the bug.
    big = (per_dim_std > 10).sum()
    huge = (per_dim_std > 100).sum()
    if huge > 0:
        print(f"  ⚠  {huge}/{features.shape[1]} dims have std > 100  "
              "(this is the optimiser-instability hypothesis confirmed)")
    elif big > 0:
        print(f"  ⚠  {big}/{features.shape[1]} dims have std > 10")
    else:
        print(f"  ok: all per-dim stds <= 10")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=None,
                   help="A specific .feat.npz to inspect. If absent, walks "
                        "the standard candidates list.")
    args = p.parse_args()

    paths = [Path(args.path)] if args.path else [Path(p) for p in CANDIDATES]
    found_any = False
    for path in paths:
        if not path.exists():
            print(f"[skip] {path} (not found)")
            continue
        found_any = True
        d = np.load(path)
        if "features" not in d.files:
            print(f"[skip] {path} (no 'features' key — old cache?)")
            continue
        _summarise(d["features"], str(path))
    if not found_any:
        print("No .feat.npz files found. Make sure --with_edge_features has "
              "been used at least once to populate the caches.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
