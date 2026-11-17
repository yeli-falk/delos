"""CLDG-format dynamic graphs with STATIC per-node class labels (DBLP / Patent).

The release format of CLDG (Xu et al., ICDE'23, github.com/yimingxu24/CLDG)
and the follow-up multi-view work that ships the same folders. Two of them
were dropped into ``dataset/`` as Google-Drive zips
(``dblp-<stamp>.zip`` / ``patent-<stamp>.zip``); unpack with
``scripts/dblp/extract_cldg_datasets.sh`` to::

    dataset/cldg/<name>/<name>.txt        src dst t          t normalised to [0, 1]
    dataset/cldg/<name>/node2label.txt    node label         one row per node
    dataset/cldg/<name>/feature.txt       128 floats         one row per node
    (label.txt is node2label sorted by node; MVC Features/ are precomputed
     embeddings of another method -- neither is used here.)

What the two graphs are (measured on the files, 2026-09-24):

    name    nodes   edges    distinct t  classes   notes
    dblp    28,085  236,894      27        10      yearly co-authorship, undirected
                                                   (stored one direction, some
                                                   multi-edges); t is the year
                                                   index / 26, so the 27 distinct
                                                   values ARE the snapshots
    patent  12,214   41,916     891         6      citation graph, directed; the
                                                   timeline is extremely skewed
                                                   (83% of edges in the last 10%
                                                   of the axis; the 10%-quantile
                                                   of t is 0.87)
    (arXivAI from the same release was tried and dropped: 69,854 nodes, 27
     yearly timestamps with 99% of its 699k edges in the last 6 years, so the
     chronological train period holds ~4.7k edges. Not a temporal benchmark.)

``feature.txt`` is the transformer sinusoidal positional encoding of the NODE
INDEX (row i = [sin(i / 10000^(2k/128)), cos(...)]): a time-invariant identity
code, no content. It is what CLDG feeds its encoder, and it is what lets a GCN
latent say *which* neighbours a node has -- without it the shared 8-dim graph
statistics carry almost nothing about a research area or a patent class.
On by default (``with_file_features``), switchable off for an ablation.

Both label sets are one class per node for the whole timeline, so the probe
task is static multiclass node classification (``run_block_probe.py --task
multiclass --label_source dataset``), scored with a NODE-level train/test
split -- the same node with the same label must never sit on both sides.

Snapshotting choices (all env-overridable through ``_dataset_config.sh``):

  * dblp:   ``num_bins=0`` -> one snapshot per distinct timestamp (27 years,
            the dataset's own granularity). Every year is well populated
            (856 .. 26,104 edges).
  * patent: ``num_bins=30, binning='equal_freq'`` -> quantile bins with
            ~1,400 edges each. Equal-width bins would put 2 edges in the
            first snapshot and 15,158 in the 29th.
  * top_k:  the DTDG pipeline stores a DENSE (T, N, N) adjacency, so DBLP at
            28k nodes (21 GB per timeline, 12 GB per window on the GPU) needs
            the same top-K-by-activity subset the Reddit hyperlink graph uses.
            Labels and file features are indexed down to the kept nodes; the
            bin edges are always computed on the FULL stream so the snapshot
            grid does not depend on K.

Only the CLDG-specific parsing lives here; binning primitives, graph
features, windowing and splitting come from the universal DTDG pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

from src.custom_codes.dtdg.binning import (
    equal_frequency_bins,
    equal_width_bins,
    native_bins,
    scatter_snapshots,
    top_k_combined,
)
from src.custom_codes.dtdg.canonical import SnapshotSequence
from src.custom_codes.dtdg.dataset import DTDGDataset
from src.custom_codes.dtdg.features import build_graph_features


CLDG_NAMES = ("dblp", "patent")
BINNINGS = ("equal_width", "equal_freq")


def _fresh(cache: Path, source: Path) -> bool:
    return cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime


def _parse_edges(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``src dst t`` rows -> (src, dst, t) int64/int64/float64. Cached as .npz."""
    cache = path.with_suffix(".npz")
    if _fresh(cache, path):
        d = np.load(cache)
        return d["src"], d["dst"], d["t"]
    arr = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if arr.shape[1] < 3:
        raise ValueError(f"{path}: expected 'src dst t' rows, got {arr.shape[1]} columns")
    src = arr[:, 0].astype(np.int64)
    dst = arr[:, 1].astype(np.int64)
    t = arr[:, 2].astype(np.float64)
    np.savez(cache, src=src, dst=dst, t=t)
    return src, dst, t


def _parse_labels(path: Path, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """``node label`` rows -> (labels (N,) int64 in 0..C-1 or -1, raw class ids).

    Class ids are densified in sorted order (patent's 1..6 -> 0..5). Nodes
    absent from the file get -1, which every probe treats as "no label".
    """
    arr = np.loadtxt(path, dtype=np.int64, ndmin=2)
    node, raw = arr[:, 0], arr[:, 1]
    if node.max() >= num_nodes:
        raise ValueError(f"{path}: node id {node.max()} >= num_nodes {num_nodes}")
    classes = np.unique(raw)
    labels = np.full(num_nodes, -1, dtype=np.int64)
    labels[node] = np.searchsorted(classes, raw)
    return labels, classes


def _parse_features(path: Path, num_nodes: int) -> np.ndarray:
    """Whitespace-separated floats, one row per node -> (N, F) float32. Cached
    as .npy (the 40 MB DBLP file takes ~15 s through loadtxt)."""
    cache = path.with_suffix(".npy")
    if _fresh(cache, path):
        feats = np.load(cache)
    else:
        try:
            import pandas as pd
            feats = pd.read_csv(path, sep=r"\s+", header=None,
                                dtype=np.float32).to_numpy()
        except ImportError:
            feats = np.loadtxt(path, dtype=np.float32, ndmin=2)
        feats = np.ascontiguousarray(feats, dtype=np.float32)
        np.save(cache, feats)
    if feats.shape[0] != num_nodes:
        raise ValueError(f"{path}: {feats.shape[0]} rows for {num_nodes} nodes")
    return feats


def bin_cldg_stream(
    src: np.ndarray, dst: np.ndarray, t: np.ndarray, *,
    num_bins: int, binning: str, top_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Event stream -> (adj (T,N,N), active (T,N), keep (N,) raw ids, bin_idx).

    ``num_bins <= 0`` = one snapshot per distinct timestamp of the full
    stream; otherwise ``binning`` picks equal-width or equal-frequency
    (quantile) bins, with edges computed on the FULL stream. ``top_k > 0``
    keeps the K most active nodes (combined src+dst count) and the events
    between them; ``keep`` maps dense index -> raw node id.
    """
    if binning not in BINNINGS:
        raise ValueError(f"binning must be one of {BINNINGS}, got {binning!r}")
    n_all = int(max(src.max(), dst.max()) + 1)
    if 0 < top_k < n_all:
        keep = np.sort(top_k_combined(src, dst, top_k))
    else:
        keep = np.arange(n_all, dtype=np.int64)
    mask = np.isin(src, keep) & np.isin(dst, keep)
    s_k, d_k, t_k = src[mask], dst[mask], t[mask]
    if t_k.size == 0:
        raise ValueError("no events left after the top-K node filter; raise top_k")
    s_idx = np.searchsorted(keep, s_k)
    d_idx = np.searchsorted(keep, d_k)

    if num_bins <= 0:
        bin_idx, T = native_bins(t_k, reference=t)
    elif binning == "equal_freq":
        bin_idx, T = equal_frequency_bins(t_k, num_bins, reference=t), num_bins
    else:
        bin_idx, T = equal_width_bins(t_k, num_bins, reference=t), num_bins

    adj, active = scatter_snapshots(s_idx, d_idx, bin_idx, len(keep), T)
    return adj, active, keep, bin_idx


def build_cldg_snapshots(
    name: str,
    data_dir: str = "dataset/cldg",
    num_bins: int = 30,
    binning: str = "equal_freq",
    top_k: int = 0,
    with_file_features: bool = True,
    feat_dim: int = 8,
    seed: int = 42,
) -> Tuple[SnapshotSequence, dict]:
    """Parse one CLDG-format folder into a canonical SnapshotSequence.

    Returns ``(seq, meta)``; ``meta`` carries the kept raw node ids
    (``node_index``), the raw class ids (``classes``) and the per-bin edge
    counts for the log line.
    """
    if name not in CLDG_NAMES:
        raise ValueError(f"unknown CLDG dataset {name!r}; expected one of {CLDG_NAMES}")
    folder = Path(data_dir).expanduser() / name
    edge_path = folder / f"{name}.txt"
    if not edge_path.exists():
        raise FileNotFoundError(
            f"{edge_path} not found. Unpack the CLDG zips with\n"
            f"  bash scripts/dblp/extract_cldg_datasets.sh\n"
            f"(expects dataset/{name}-*.zip, the Google-Drive export).")

    src, dst, t = _parse_edges(edge_path)
    n_all = int(max(src.max(), dst.max()) + 1)
    adj, active, keep, bin_idx = bin_cldg_stream(
        src, dst, t, num_bins=num_bins, binning=binning, top_k=top_k)
    T, N = adj.shape[0], adj.shape[1]

    labels_all, classes = _parse_labels(folder / "node2label.txt", n_all)
    node_labels = labels_all[keep]

    feats = build_graph_features(adj, active, feat_dim=feat_dim, seed=seed)
    if with_file_features:
        static = _parse_features(folder / "feature.txt", n_all)[keep]      # (N, Fs)
        # Time-invariant identity code, replicated per snapshot and zeroed on
        # inactive (bin, node) cells like the graph statistics, so "absent"
        # stays distinguishable from "present".
        static_t = np.broadcast_to(static[None], (T,) + static.shape)
        static_t = static_t * active[..., None].astype(np.float32)
        feats = np.concatenate([feats, static_t.astype(np.float32)], axis=-1)

    per_bin = adj.sum(axis=(1, 2)) // 2
    n_events = int((np.isin(src, keep) & np.isin(dst, keep)).sum())
    print(f"[cldg:{name}] nodes {N}/{n_all}  events {n_events}/{len(src)} "
          f"({100 * n_events / len(src):.1f}%)  T={T} "
          f"({'native' if num_bins <= 0 else binning})  "
          f"edges/bin min={int(per_bin.min())} median={int(np.median(per_bin))} "
          f"max={int(per_bin.max())}  active/bin median={int(np.median(active.sum(1)))}  "
          f"classes={len(classes)}  feat_dim={feats.shape[-1]}", flush=True)
    if (per_bin == 0).any():
        print(f"[cldg:{name}] WARNING: {int((per_bin == 0).sum())} of {T} snapshots "
              f"have no edges -- consider fewer bins or equal_freq binning",
              flush=True)

    seq = SnapshotSequence(adj=adj, active=active, feats=feats,
                           node_labels=node_labels)
    meta = {"node_index": keep, "classes": classes, "edges_per_bin": per_bin,
            "num_nodes_total": n_all}
    return seq, meta


class CLDGFormatDynamicGraphDataset(Dataset):
    """CLDG-format DBLP / Patent as DTDG snapshots with static node labels.

    Same return signature as the other loaders (thin wrapper over
    :class:`DTDGDataset`), plus:

      node_labels   (N,) int64 in 0..C-1 (-1 = unlabelled), for
                    ``run_block_probe.py --task multiclass --label_source dataset``
      num_classes   C
      node_index    (N,) raw node ids of the kept subset

    Args:
      name:        "dblp" | "patent" (folder under ``data_dir``).
      num_bins:    snapshots; 0 = one per distinct timestamp.
      binning:     "equal_width" | "equal_freq" (ignored when num_bins == 0).
      top_k:       keep the K most active nodes (0 = all).
      with_file_features: append the 128-dim ``feature.txt`` rows.
      split:       "train" | "val" | "test", chronological 70/10/20 by default.
    """

    SPLITS: Tuple[float, float, float] = (0.7, 0.1, 0.2)

    def __init__(
        self,
        name: str,
        data_dir: str = "dataset/cldg",
        split: str = "train",
        history_size: int = 3,
        num_preds: int = 1,
        num_bins: int = 30,
        binning: str = "equal_freq",
        top_k: int = 0,
        with_file_features: bool = True,
        feat_dim: int = 8,
        stride: int = 1,
        seed: int = 42,
        split_mode: str = "chronological",
    ):
        assert split in ("train", "val", "test")
        self.split = split
        self.name = name
        self.history_size = history_size
        self.num_preds = num_preds
        self.window = history_size + num_preds
        self.stride = stride
        self.with_file_features = with_file_features

        seq, meta = build_cldg_snapshots(
            name, data_dir=data_dir, num_bins=num_bins, binning=binning,
            top_k=top_k, with_file_features=with_file_features,
            feat_dim=feat_dim, seed=seed,
        )
        self.num_nodes = seq.num_nodes
        self.feat_dim = seq.feat_dim
        self.node_labels = seq.node_labels
        self.node_labels_dyn = None                 # labels are static
        self.state_label_mode = "static"
        self.num_classes = int(len(meta["classes"]))
        self.node_index = meta["node_index"]
        self.classes = meta["classes"]
        self._ds = DTDGDataset.from_global(
            seq, split=split, history_size=history_size, num_preds=num_preds,
            stride=stride, splits=self.SPLITS,
            split_mode=split_mode, split_seed=seed,
        )
        self.starts = self._ds.starts

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]

    @property
    def feature_dim(self) -> int:
        return self.feat_dim
