"""England COVID dataset for DELOS (dynamic-topology node regression).

England COVID is the time-varying-graph node-regression benchmark from
PyTorch Geometric Temporal (`EnglandCovidDatasetLoader`). 129 NUTS3 regions
are the nodes; each of the 61 daily snapshots carries a different directed,
weighted mobility graph; the per-node signal is the daily reported COVID-19
case count. The task is to forecast each region's future case count from the
recent window of cases and the evolving mobility topology.

Each `__getitem__` returns the per-snapshot `adj` / `active` tensors and the
raw regression signal, so a regression probe can be fit and evaluated.

The canonical `england_covid.json` (the file PyG Temporal downloads) is
fetched once and cached under `data_dir`, then parsed with numpy; no
`torch_geometric_temporal` dependency is required unless `use_pyg=True`.

Tensor shapes per item:
    x_hist  : (T_hist, N, F)        node features (signal + degree + time)
    x_fut   : (T_pred, N, F)        future node features (for target encoding)
    adj     : (T, N, N) bool        per-snapshot mobility adjacency
    active  : (T, N) bool           node existence (all True here)
    signal  : (T, N) float32        raw case counts (un-normalised, for MAE)
    year    : int                   window start index (timeline position)
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset

from src.custom_codes.dtdg.canonical import SnapshotSequence
from src.custom_codes.dtdg.dataset import DTDGDataset

ENGLAND_COVID_URL = (
    "https://raw.githubusercontent.com/benedekrozemberczki/"
    "pytorch_geometric_temporal/master/dataset/england_covid.json"
)
NUM_NODES = 129
NUM_FEATURES = 4  # [signal_norm, degree_norm, sin(2pi t/T), cos(2pi t/T)]


# ---------------------------------------------------------------------------
# Raw loading: cached JSON download (default) or torch_geometric_temporal
# ---------------------------------------------------------------------------
def _load_raw_json(data_dir: Path, url: str) -> dict:
    """Return the parsed england_covid.json, downloading + caching if absent."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "england_covid.json"
    if not cache.exists():
        print(f"[england_covid] downloading -> {cache}", flush=True)
        urllib.request.urlretrieve(url, cache)
    with open(cache, "r") as f:
        return json.load(f)


def _raw_to_arrays(raw: dict) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Parse the JSON into (signal, edge_lists).

    Returns:
      signal     : (T, N) float32 raw case counts
      edge_lists : length-T list of (E_t, 2) int arrays (directed, raw ids)
    """
    T = int(raw["time_periods"])
    signal = np.asarray([np.asarray(row, dtype=np.float32) for row in raw["y"]],
                        dtype=np.float32)                       # (T, N)
    if signal.shape[1] != NUM_NODES:
        raise ValueError(
            f"expected {NUM_NODES} nodes, got {signal.shape[1]} from england_covid.json")
    ei = raw["edge_mapping"]["edge_index"]
    edge_lists = [np.asarray(ei[str(t)], dtype=np.int64).reshape(-1, 2)
                  for t in range(T)]
    return signal, edge_lists


def _load_via_pyg() -> Tuple[np.ndarray, List[np.ndarray]]:
    """Optional path: build the same (signal, edge_lists) from PyG Temporal."""
    from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader

    loader = EnglandCovidDatasetLoader()
    raw = loader._dataset
    return _raw_to_arrays(raw)


# ---------------------------------------------------------------------------
# Canonical snapshot construction
# ---------------------------------------------------------------------------
def _edges_to_adjacency(edge_lists: List[np.ndarray], num_nodes: int) -> np.ndarray:
    """(T, N, N) bool symmetric adjacency, self-loops removed.

    The mobility graph is directed; the DELOS spacetime attention consumes a
    boolean connectivity mask, so the adjacency is symmetrised (an edge in
    either direction is a spatial neighbour) and self-loops are dropped
    (`build_graph_features` and `normalize_adjacency` both assume no
    self-loop on the input adj).
    """
    T = len(edge_lists)
    adj = np.zeros((T, num_nodes, num_nodes), dtype=bool)
    for t, e in enumerate(edge_lists):
        if e.size == 0:
            continue
        src, dst = e[:, 0], e[:, 1]
        adj[t, src, dst] = True
        adj[t, dst, src] = True
    eye = np.eye(num_nodes, dtype=bool)
    adj[:, eye] = False
    return adj


def build_england_covid_snapshots(
    data_dir: str,
    url: str = ENGLAND_COVID_URL,
    use_pyg: bool = False,
    splits: Tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> Tuple[SnapshotSequence, np.ndarray, float, float]:
    """Parse England COVID into a canonical SnapshotSequence + raw signal.

    Returns (seq, raw_signal, mean, std). `mean`/`std` are computed over the
    chronological train portion only and applied to channel 0 of `seq.feats`.
    """
    if use_pyg:
        signal, edge_lists = _load_via_pyg()
    else:
        signal, edge_lists = _raw_to_arrays(_load_raw_json(Path(data_dir).expanduser(), url))

    T, N = signal.shape
    adj = _edges_to_adjacency(edge_lists, N)                    # (T, N, N) bool
    active = np.ones((T, N), dtype=bool)                        # regions always present

    # Train-only normalisation of the signal (channel 0).
    train_end = int(T * splits[0])
    train_sig = signal[:train_end]
    mean = float(train_sig.mean())
    std = float(train_sig.std() + 1e-6)
    signal_norm = (signal - mean) / std

    deg = adj.sum(-1).astype(np.float32)                        # (T, N)
    t_idx = np.arange(T, dtype=np.float32)
    sin_t = np.sin(2 * np.pi * t_idx / max(T, 1))
    cos_t = np.cos(2 * np.pi * t_idx / max(T, 1))

    feats = np.zeros((T, N, NUM_FEATURES), dtype=np.float32)
    feats[..., 0] = signal_norm
    feats[..., 1] = deg / max(N, 1)
    feats[..., 2] = sin_t[:, None]
    feats[..., 3] = cos_t[:, None]

    seq = SnapshotSequence(adj=adj, active=active, feats=feats)
    return seq, signal.astype(np.float32), mean, std


# ---------------------------------------------------------------------------
# Sliding-window dataset
# ---------------------------------------------------------------------------
class EnglandCovidDataset(Dataset):
    """Dynamic-graph node-regression dataset over England COVID.

    Thin parser: builds a `SnapshotSequence`, delegates windowing/splitting to
    the universal `DTDGDataset`, and augments each item with the raw `signal`
    slice needed for the regression probe/eval.

    Args:
      data_dir:     folder for the cached england_covid.json (auto-downloaded).
      split:        "train" | "val" | "test"  (chronological 70/10/20).
      history_size, num_preds: window lengths (days).
      use_pyg:      load via torch_geometric_temporal instead of the JSON cache.
    """

    SPLITS: Tuple[float, float, float] = (0.7, 0.1, 0.2)

    def __init__(
        self,
        data_dir: str = "dataset/england_covid",
        split: str = "train",
        history_size: int = 8,
        num_preds: int = 3,
        stride: int = 1,
        url: str = ENGLAND_COVID_URL,
        use_pyg: bool = False,
        seed: int = 42,
        split_mode: str = "chronological",
    ):
        assert split in ("train", "val", "test")
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.window = history_size + num_preds

        seq, raw_signal, mean, std = build_england_covid_snapshots(
            data_dir, url=url, use_pyg=use_pyg, splits=self.SPLITS,
        )
        self.raw_signal = raw_signal           # (T, N) un-normalised counts
        self.mean = mean
        self.std = std
        self.num_nodes = seq.num_nodes
        self.feat_dim = seq.feat_dim
        self.node_labels = seq.node_labels
        self._ds = DTDGDataset.from_global(
            seq, split=split, history_size=history_size, num_preds=num_preds,
            stride=stride, splits=self.SPLITS,
            split_mode=split_mode, split_seed=seed,
        )
        self.starts = self._ds.starts

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        item = self._ds[idx]
        start = int(self.starts[idx])
        end = start + self.window
        # Raw signal over the full window; the probe slices the future steps.
        item["signal"] = self.raw_signal[start:end].copy()      # (T_total, N)
        return item

    @property
    def feature_dim(self) -> int:
        return self.feat_dim
