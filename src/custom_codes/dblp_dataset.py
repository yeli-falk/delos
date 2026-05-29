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

This file ships both a synthetic generator (for smoke tests) and a hook to
load a real DBLP edge-list dump (one `(src, dst, year)` row per line).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _make_synthetic_dblp(
    num_nodes: int = 100,
    num_years: int = 20,
    avg_new_authors_per_year: int = 4,
    p_edge_existing: float = 0.02,
    preferential_attachment: float = 0.5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random growing co-authorship graph.

    Returns:
      adj    : (T, N, N) bool, symmetric, no self-loops
      active : (T, N) bool, monotone over time (authors don't leave)
    """
    rng = np.random.RandomState(seed)
    adj = np.zeros((num_years, num_nodes, num_nodes), dtype=bool)
    active = np.zeros((num_years, num_nodes), dtype=bool)

    # Initial seed authors at year 0.
    n_init = max(10, num_nodes // 8)
    active[0, :n_init] = True

    for t in range(num_years):
        if t > 0:
            active[t] = active[t - 1].copy()
            n_new = min(
                rng.poisson(avg_new_authors_per_year),
                num_nodes - int(active[t].sum()),
            )
            inactive_idx = np.flatnonzero(~active[t])
            if n_new > 0 and inactive_idx.size > 0:
                newcomers = rng.choice(
                    inactive_idx, size=min(n_new, inactive_idx.size), replace=False,
                )
                active[t, newcomers] = True

        node_ids = np.flatnonzero(active[t])
        n_now = node_ids.size
        # Carry over a fraction of last year's edges (collaboration inertia).
        if t > 0:
            keep = rng.rand(num_nodes, num_nodes) < 0.4
            adj[t] = adj[t - 1] & keep

        # Sample new edges this year. Probability scales with concurrent degree
        # to mimic preferential attachment.
        if n_now >= 2:
            deg = adj[t, node_ids][:, node_ids].sum(-1) + 1
            p_node = deg / deg.sum()
            n_new_edges = int(p_edge_existing * n_now * (n_now - 1) / 2)
            for _ in range(n_new_edges):
                i, j = rng.choice(
                    node_ids, size=2, replace=False,
                    p=(p_node * preferential_attachment
                       + (1 - preferential_attachment) / n_now),
                )
                if i == j:
                    continue
                adj[t, i, j] = True
                adj[t, j, i] = True
    return adj, active


def _features_from_history(adj: np.ndarray, active: np.ndarray,
                           feat_dim: int = 8, seed: int = 0) -> np.ndarray:
    """Cheap per-(t, n) feature vector built from local graph stats.

      [degree, log(1+degree), is_new_this_year, mean_neighbour_degree,
       clustering_coef_proxy, year_index/T, sin(2pi*t/T), cos(2pi*t/T)]
    """
    T, N, _ = adj.shape
    feats = np.zeros((T, N, feat_dim), dtype=np.float32)
    deg = adj.sum(-1).astype(np.float32)
    for t in range(T):
        deg_t = deg[t]
        feats[t, :, 0] = deg_t / max(N, 1)
        feats[t, :, 1] = np.log1p(deg_t)
        feats[t, :, 2] = (active[t].astype(np.float32)
                          - (active[t - 1].astype(np.float32) if t > 0 else 0.0))
        # Mean neighbour degree (z-scored).
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_nb_deg = np.where(
                deg_t > 0, (adj[t] @ deg_t) / np.maximum(deg_t, 1), 0.0,
            )
        feats[t, :, 3] = mean_nb_deg / max(N, 1)
        feats[t, :, 4] = (deg_t > 0).astype(np.float32)
        feats[t, :, 5] = t / max(T - 1, 1)
        feats[t, :, 6] = np.sin(2 * np.pi * t / max(T, 1))
        feats[t, :, 7] = np.cos(2 * np.pi * t / max(T, 1))
    feats *= active[..., None].astype(np.float32)
    return feats


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


class DBLPDynamicGraphDataset(Dataset):
    """Sliding-window dataset over a yearly co-authorship graph.

    Args:
      data_path:   path to a CSV-like edge list with `src,dst,year` rows;
                   pass `None` together with `synthetic=True` to use a toy
                   graph.
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
        synthetic: bool = False,
        synthetic_nodes: int = 100,
        synthetic_years: int = 30,
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

        if synthetic or data_path is None:
            adj, active = _make_synthetic_dblp(
                num_nodes=synthetic_nodes, num_years=synthetic_years, seed=seed,
            )
        else:
            adj, active = _load_real_edges(Path(data_path).expanduser())

        self.num_nodes = adj.shape[1]
        self.feat_dim = feat_dim
        T_total = adj.shape[0]
        train_end = int(T_total * self.SPLITS[0])
        val_end = train_end + int(T_total * self.SPLITS[1])

        feats = _features_from_history(adj, active, feat_dim=feat_dim, seed=seed)

        if split == "train":
            lo, hi = 0, train_end
        elif split == "val":
            lo, hi = train_end, val_end
        else:
            lo, hi = val_end, T_total

        self.adj = adj[lo:hi]
        self.active = active[lo:hi]
        self.feats = feats[lo:hi]
        self.year_offset = lo

        max_start = self.adj.shape[0] - self.window
        if max_start < 0:
            raise ValueError(
                f"Split '{split}' has only {self.adj.shape[0]} years but "
                f"window={self.window}."
            )
        self.starts = np.arange(0, max_start + 1, stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = int(self.starts[idx])
        end = start + self.window
        adj_w = torch.from_numpy(self.adj[start:end])           # (T, N, N) bool
        active_w = torch.from_numpy(self.active[start:end])     # (T, N) bool
        x_w = torch.from_numpy(self.feats[start:end])           # (T, N, F)
        return {
            "x_hist": x_w[: self.history_size],
            "x_fut":  x_w[self.history_size :],
            "adj":    adj_w,
            "active": active_w,
            "year":   start + self.year_offset,
        }

    @property
    def feature_dim(self) -> int:
        return self.feat_dim
