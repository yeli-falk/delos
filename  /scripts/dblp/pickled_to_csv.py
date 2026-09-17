"""Convert preprocessed temporal-graph snapshots into our `(src, dst, year)` CSV.

Many DTDG papers (DySAT, EvolveGCN, ROLAND, ...) release their preprocessed
datasets as a pickled list of `networkx.Graph` objects, one per snapshot.
This script flattens that format to a CSV the existing
`DBLPDynamicGraphDataset(data_path=...)` already consumes.

Usage:
    python scripts/dblp/pickled_to_csv.py \
        --input dataset/dblp/graphs.pkl \
        --output dataset/dblp/edges.csv \
        --year_start 2000

The `--input` file must unpickle to a list whose elements are either:
  * a `networkx.Graph`/`MultiGraph`/`DiGraph`, in which case `.edges()` is used;
  * a 2-tuple `(num_nodes, edge_list)` (numpy or python list of (src, dst));
  * a torch/numpy array of shape `(num_edges, 2)`.

If your data uses a different layout, adapt the `_iter_edges` function below.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Iterable, Tuple


def _iter_edges(snapshot) -> Iterable[Tuple[int, int]]:
    # networkx graph?
    if hasattr(snapshot, "edges"):
        for u, v in snapshot.edges():
            yield int(u), int(v)
        return
    # (num_nodes, edge_list)?
    if isinstance(snapshot, tuple) and len(snapshot) == 2:
        for u, v in snapshot[1]:
            yield int(u), int(v)
        return
    # (num_edges, 2) array-like?
    try:
        import numpy as np
        arr = np.asarray(snapshot)
        if arr.ndim == 2 and arr.shape[1] == 2:
            for u, v in arr.tolist():
                yield int(u), int(v)
            return
    except Exception:
        pass
    raise TypeError(
        f"Unrecognised snapshot type: {type(snapshot)}. "
        "Patch `_iter_edges` to handle your format."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Pickle file: list of per-snapshot graphs.")
    p.add_argument("--output", required=True,
                   help="Output CSV with header-less rows: src,dst,year")
    p.add_argument("--year_start", type=int, default=2000,
                   help="Year assigned to the first snapshot (rest follow +1).")
    p.add_argument("--undirected", action="store_true",
                   help="If set, also emit (dst, src) for each edge.")
    p.add_argument("--drop_self_loops", action="store_true", default=True)
    args = p.parse_args()

    in_path = Path(args.input).expanduser()
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "rb") as f:
        snapshots = pickle.load(f)
    if not isinstance(snapshots, list):
        raise ValueError(
            f"Top-level object in {in_path} must be a list of snapshots, "
            f"got {type(snapshots)}."
        )
    print(f"[load] {len(snapshots)} snapshots from {in_path}")

    total = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        for t, snap in enumerate(snapshots):
            year = args.year_start + t
            for u, v in _iter_edges(snap):
                if args.drop_self_loops and u == v:
                    continue
                w.writerow([u, v, year])
                total += 1
                if args.undirected:
                    w.writerow([v, u, year])
                    total += 1
    print(f"[done] wrote {total:,} edges to {out_path}")
    print(f"       year range: {args.year_start} .. {args.year_start + len(snapshots) - 1}")


if __name__ == "__main__":
    main()
