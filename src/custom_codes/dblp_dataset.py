"""DBLP-style discrete-time dynamic graph dataset.

Each sample is a sliding window over yearly snapshots of a co-authorship
graph: authors as nodes, edges if two authors share a paper in that year.
Node identity is stable across years; the edge set evolves; new authors
may appear (modelled via `active`).

Tensor shapes per item:
    x       : (T, N, F)        node features (degree, ego-features, ...)
    adj     : (T, N, N) bool   per-timestep adjacency
    active  : (T, N) bool       node existence
    times   : (T,)              year offsets

The real data is an edge-list dump with one `(src, dst, year)` row per line.
The sliding-window logic lives in the universal :class:`DTDGDataset`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from src.custom_codes.dtdg.canonical import SnapshotSequence
from src.custom_codes.dtdg.dataset import DTDGDataset
from src.custom_codes.dtdg.features import (  # noqa: F401  (back-compat re-exports)
    aggregate_edge_features_to_nodes,
    build_graph_features as _features_from_history,
)


def _load_real_edges(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a `(src, dst, year)` CSV-ish edge list -> (adj, active) arrays.

    Authors are deduplicated and assigned dense integer ids by first appearance.
    Years are densified to consecutive integers from min(year). Multi-edges
    collapse to a single bool edge per year.
    """
    rows = np.loadtxt(path, delimiter=",", dtype=str, ndmin=2)
    src_raw = rows[:, 0]
    dst_raw = rows[:, 1]
    year_raw = rows[:, 2].astype(int)

    authors = {}
    def aid(s):
        if s not in authors:
            authors[s] = len(authors)
        return authors[s]

    src = np.fromiter((aid(s) for s in src_raw), dtype=int)
    dst = np.fromiter((aid(s) for s in dst_raw), dtype=int)
    year = year_raw - year_raw.min()
    N = max(src.max(), dst.max()) + 1
    T = int(year.max() + 1)

    adj = np.zeros((T, N, N), dtype=bool)
    adj[year, src, dst] = True
    adj[year, dst, src] = True
    np.fill_diagonal(adj.reshape(T, -1)[:, ::N + 1].reshape(T, N), False)

    deg_per_year = adj.sum(-1)
    active = deg_per_year > 0
    # Once an author publishes, treat them as persistent so JEPA slot identity
    # is preserved. (They may have zero edges in subsequent years.)
    active = np.maximum.accumulate(active, axis=0)
    return adj, active


def build_dblp_snapshots(
    data_path: Optional[str],
    feat_dim: int = 8,
    seed: int = 42,
) -> SnapshotSequence:
    """Parse DBLP into a canonical SnapshotSequence."""
    if data_path is None:
        raise ValueError("data_path must be provided for DBLP")
    else:
        adj, active = _load_real_edges(Path(data_path).expanduser())

    feats = _features_from_history(adj, active, feat_dim=feat_dim, seed=seed)
    return SnapshotSequence(adj=adj, active=active, feats=feats)


class DBLPDynamicGraphDataset(Dataset):
    """Sliding-window dataset over a yearly co-authorship graph.

    Thin wrapper over the universal :class:`DTDGDataset`.

    Args:
      data_path:   path to a CSV-like edge list with `src,dst,year` rows;
      split:       "train" | "val" | "test", chronological 70/10/20.
      history_size, num_preds: window lengths (years).
    """

    SPLITS: Tuple[float, float, float] = (0.7, 0.1, 0.2)

    def __init__(
        self,
        data_path: Optional[str],
        split: str = "train",
        history_size: int = 3,
        num_preds: int = 1,
        feat_dim: int = 8,
        stride: int = 1,
        seed: int = 42,
    ):
        assert split in ("train", "val", "test")
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.window = history_size + num_preds
        self.stride = stride
        self.feat_dim = feat_dim

        seq = build_dblp_snapshots(
            data_path, feat_dim=feat_dim, seed=seed,
        )
        self.num_nodes = seq.num_nodes
        self.node_labels = seq.node_labels
        self._ds = DTDGDataset.from_global(
            seq, split=split, history_size=history_size, num_preds=num_preds,
            stride=stride, splits=self.SPLITS,
        )
        self.starts = self._ds.starts

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]

    @property
    def feature_dim(self) -> int:
        return self.feat_dim
