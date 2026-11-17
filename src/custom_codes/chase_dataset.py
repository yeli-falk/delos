"""CHASE dynamic-label benchmarks (Kim et al., NeurIPS'26): DBLP-S, GossipCop, Music.

Snapshot graphs whose node labels change over time by construction
("Chasing Label Shifters", arXiv 2610.17582). They are released on the
Hugging Face Hub as ``jhkim611/CHASE-datasets`` (CC BY-NC 4.0;
``bash scripts/dblp/fetch_chase.sh`` -> ``dataset/chase/``).

The three graphs:

    name       nodes   snapshots  split      classes  label                      shifters
    dblp_s     34,717  14 yearly  9 / 2 / 3     2     main research area that   2.6 %
                                                      year (networks vs ML)
    gossipcop   6,091  37 weekly  26 / 5 / 6    2     >50 % of the articles the  3.5 %
                                                      user shared that week
                                                      are fake
    music         878  24 monthly 15 / 4 / 5    5     dominant genre listened   18.8 %
                                                      to that month

"shifters" = share of nodes active at t-1 and t whose label differs, i.e.
1 - the label-persistence floor. Node sets grow over time: a node is inactive
before its first appearance and the release stores label 0 there, so inactive
(t, node) cells are relabelled -1 (no label) here.

Per-snapshot node features come with the release (``feats`` (K, N, F)):
768-dim BERT [CLS] embeddings of the papers / articles of that period for
DBLP-S and GossipCop, 779-dim listening descriptors for Music. Edges are
stored in both directions with a weight (shared papers / articles / tracks);
the pipeline's bool snapshot keeps the support only.

Pipeline choices (all env-overridable through ``_dataset_config.sh``):

  * Splits are the paper's snapshot counts (above), not the generic 70/10/20.
    Windows are target-based as everywhere else (a test window's history may
    reach into the validation snapshots).
  * ``top_k``: 0 = the full graph (the launchers' default). The dense
    (T, N, N) snapshots make 34,717-node DBLP-S heavy, so the launchers run
    it at BATCH 1; ``top_k`` > 0 keeps the K most active nodes (sum of degrees
    over the timeline), with labels and features following.
  * Features: the 8 graph statistics + (``with_node_features``) the release's
    vectors, z-scored per dimension on the active rows of the train snapshots
    (``feat_norm``) and optionally reduced with a PCA fitted on the same rows
    (``feat_pca`` > 0). Zeroed on inactive cells. ``with_node_features=False``
    is the topology-only ablation.
  * Labels: ``node_labels_dyn`` (T, N) int64 with -1 = inactive, read by
    ``run_block_probe.py --task binary`` (DBLP-S, GossipCop) and
    ``--task multiclass`` (Music, 5 classes; the loader sets
    ``dyn_label_task`` so the probe takes its dynamic multiclass path).

The pickles hold only numpy arrays. They are still untrusted downloads, so
they are read through an unpickler that refuses every global except the
numpy array reconstructors; a parsed, top-K-reduced copy is cached as .npz
next to the pickle so the large DBLP-S file is loaded once.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from torch.utils.data import Dataset

from src.custom_codes.bts_flights_dataset import structure_audit
from src.custom_codes.dtdg.binning import scatter_snapshots
from src.custom_codes.dtdg.canonical import SnapshotSequence
from src.custom_codes.dtdg.dataset import DTDGDataset
from src.custom_codes.dtdg.features import build_graph_features


CHASE_NAMES = ("dblp_s", "gossipcop", "music")
CHASE_FILES = {"dblp_s": "DBLP-S.pkl", "gossipcop": "GossipCop.pkl", "music": "Music.pkl"}
# Train / val / test snapshot counts of the paper (Table 1).
CHASE_SPLITS = {"dblp_s": (9, 2, 3), "gossipcop": (26, 5, 6), "music": (15, 4, 5)}

_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Refuses every global except the numpy array reconstructors, so a
    tampered download cannot run code at load time."""

    def find_class(self, module, name):
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"refusing to load global {module}.{name} from a CHASE pickle "
            f"(only numpy arrays are expected)")


def load_chase_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        d = _NumpyOnlyUnpickler(f).load()
    for key in ("edge_indices", "edge_weights", "feats", "labels", "mask"):
        if key not in d:
            raise ValueError(f"{path}: missing key {key!r}; keys = {list(d)}")
    return d


def _fresh(cache: Path, source: Path) -> bool:
    return cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime


def _cache_path(pkl: Path, top_k: int) -> Path:
    tag = f"top{top_k}" if top_k > 0 else "full"
    return pkl.with_name(f"{pkl.stem}_{tag}.npz")


def parse_chase(pkl: Path, top_k: int = 0) -> Dict[str, np.ndarray]:
    """Pickle -> dict(adj (T,N,N) bool, active (T,N) bool, labels (T,N) int64
    with -1 on inactive cells, node_feats (T,N,F) float32, keep (N,) raw ids,
    n_all). ``top_k > 0`` keeps the K most active nodes (sum of degrees over
    the timeline). Cached as .npz keyed by K."""
    cache = _cache_path(pkl, top_k)
    if _fresh(cache, pkl):
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    raw = load_chase_pickle(pkl)
    labels_all = np.asarray(raw["labels"]).astype(np.int64)      # (T, N_all)
    mask_all = np.asarray(raw["mask"]).astype(bool)
    feats_all = np.asarray(raw["feats"], dtype=np.float32)       # (T, N_all, F)
    T, n_all = labels_all.shape
    edges = [np.asarray(e, dtype=np.int64).reshape(2, -1) for e in raw["edge_indices"]]
    if len(edges) != T:
        raise ValueError(f"{pkl}: {len(edges)} edge lists for {T} label rows")

    src = np.concatenate([e[0] for e in edges]) if edges else np.zeros(0, np.int64)
    dst = np.concatenate([e[1] for e in edges]) if edges else np.zeros(0, np.int64)
    if 0 < top_k < n_all:
        deg = np.bincount(src, minlength=n_all) + np.bincount(dst, minlength=n_all)
        keep = np.sort(np.argsort(-deg, kind="stable")[:top_k]).astype(np.int64)
    else:
        keep = np.arange(n_all, dtype=np.int64)
    N = len(keep)
    remap = np.full(n_all, -1, dtype=np.int64)
    remap[keep] = np.arange(N)

    bin_idx = np.concatenate([np.full(e.shape[1], t, dtype=np.int64)
                              for t, e in enumerate(edges)]) if edges else np.zeros(0, np.int64)
    s_k, d_k = remap[src], remap[dst]
    sel = (s_k >= 0) & (d_k >= 0)
    # Slice the subset out first and drop the raw release before the dense
    # snapshots are allocated, to bound peak memory.
    mask_k = mask_all[:, keep]
    labels = np.where(mask_k, labels_all[:, keep], -1).astype(np.int64)
    node_feats = np.ascontiguousarray(feats_all[:, keep])
    del raw, feats_all, labels_all, mask_all, edges, src, dst
    adj, active_edges = scatter_snapshots(s_k[sel], d_k[sel], bin_idx[sel], N, T)
    # A node is active when the release says so (it may have a label and
    # features without an edge in that snapshot).
    active = mask_k | active_edges

    out = dict(adj=adj, active=active, labels=labels, node_feats=node_feats,
               keep=keep, n_all=np.int64(n_all))
    np.savez(cache, **out)
    return out


def normalise_node_features(x: np.ndarray, fit_rows: np.ndarray, zscore: bool = True,
                            pca: int = 0, eps: float = 1e-6) -> np.ndarray:
    """(T, N, F) -> float32: z-score per dimension on ``fit_rows`` (a (T, N)
    bool mask of the rows the statistics may see), then an optional PCA to
    ``pca`` dims fitted on the same rows (always centred; scaled only when
    ``zscore``)."""
    ref = x[fit_rows].astype(np.float64)
    if ref.shape[0] == 0:
        raise ValueError("no rows to fit the feature normalisation on")
    mu = ref.mean(0)
    sd = ref.std(0)
    sd = np.where(sd > eps, sd, 1.0) if zscore else np.ones_like(sd)
    z = (x.astype(np.float64) - mu) / sd
    if 0 < pca < z.shape[-1]:
        zr = (ref - mu) / sd
        # Right singular vectors of the (rows, F) train matrix = PCA axes.
        _, _, vt = np.linalg.svd(zr, full_matrices=False)
        z = z @ vt[:pca].T
    elif not zscore:
        z = x.astype(np.float64)
    return z.astype(np.float32)


def chase_label_audit(labels: np.ndarray, n_tr: int, n_va: int) -> dict:
    """Label churn from the (T, N) matrix alone (-1 = no label): class
    frequencies on the train snapshots, label-persistence accuracy
    label(t) == label(t-1) over cells valid at both steps (overall and on the
    test snapshots), the switch rate, and the share of nodes whose label
    changes at least once / twice."""
    v = labels >= 0
    both = v[1:] & v[:-1]
    same = (labels[1:] == labels[:-1]) & both
    ch = (labels[1:] != labels[:-1]) & both
    n_ch = ch.sum(0)
    C = int(labels.max()) + 1 if v.any() else 0
    tr = labels[:n_tr][v[:n_tr]]
    te_lo = n_tr + n_va - 1                      # transitions INTO test snapshots
    both_te, same_te = both[te_lo:], same[te_lo:]
    return {
        "valid_rows": int(v.sum()), "cells": int(v.size), "n_classes": C,
        "train_class_freq": (np.bincount(tr, minlength=C) / max(len(tr), 1)).round(4).tolist(),
        "persistence_acc": float(same.sum() / max(both.sum(), 1)),
        "persistence_acc_test": float(same_te.sum() / max(both_te.sum(), 1)),
        "switch_rate": float(ch.sum() / max(both.sum(), 1)),
        "nodes_1plus_changes": float((n_ch >= 1).mean()),
        "nodes_2plus_changes": float((n_ch >= 2).mean()),
        "active_per_snapshot": v.sum(1).tolist(),
    }


# One parse per process: the three splits are built as three objects.
_PARSED: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}


def _parsed(pkl: Path, top_k: int) -> Dict[str, np.ndarray]:
    key = (str(pkl.resolve()), int(top_k))
    if key not in _PARSED:
        _PARSED[key] = parse_chase(pkl, top_k=top_k)
    return _PARSED[key]


class CHASEDataset(Dataset):
    """DBLP-S / GossipCop / Music as DTDG snapshots with PER-SNAPSHOT node
    labels. Exposes the dynamic-label contract of the JODIE / BTS / taxi
    loaders (``node_labels_dyn`` (T, N) with -1 invalid, ``state_label_mode``,
    ``n_classes``) plus ``dyn_label_task`` ("binary" | "multiclass") so
    ``run_block_probe.py`` picks the matching state probe. Splits are the
    paper's snapshot counts (:data:`CHASE_SPLITS`).

    Args:
      name:        "dblp_s" | "gossipcop" | "music" (file under ``data_dir``).
      top_k:       keep the K most active nodes (0 = all, the default).
      with_node_features: append the release's per-snapshot vectors.
      feat_norm:   z-score them on the train snapshots' active rows.
      feat_pca:    reduce them to this many PCA dims (0 = keep all).
    """

    def __init__(
        self,
        name: str,
        data_dir: str = "dataset/chase",
        split: str = "train",
        history_size: int = 3,
        num_preds: int = 1,
        top_k: int = 0,
        with_node_features: bool = True,
        feat_norm: bool = True,
        feat_pca: int = 0,
        feat_dim: int = 8,
        stride: int = 1,
        seed: int = 42,
        split_mode: str = "chronological",
    ):
        assert split in ("train", "val", "test")
        if name not in CHASE_NAMES:
            raise ValueError(f"unknown CHASE dataset {name!r}; expected one of {CHASE_NAMES}")
        self.split = split
        self.name = name
        self.history_size = history_size
        self.num_preds = num_preds
        self.window = history_size + num_preds
        self.stride = stride
        self.with_node_features = with_node_features

        pkl = Path(data_dir).expanduser() / CHASE_FILES[name]
        if not pkl.exists():
            raise FileNotFoundError(
                f"{pkl} not found. Download the release with\n"
                f"  bash scripts/dblp/fetch_chase.sh\n"
                f"(Hugging Face dataset jhkim611/CHASE-datasets, ~3.9 GB).")
        d = _parsed(pkl, top_k)
        adj, active, labels = d["adj"], d["active"], d["labels"]
        T, N = labels.shape
        n_tr, n_va, n_te = CHASE_SPLITS[name]
        if n_tr + n_va + n_te != T:
            raise ValueError(f"{pkl}: {T} snapshots, expected {n_tr + n_va + n_te}")
        # Fractions that int(T * f) maps back to the paper's counts exactly.
        self.SPLITS = ((n_tr + 0.5) / T, (n_va + 0.5) / T, 1.0 - (n_tr + n_va + 1.0) / T)
        self.split_counts = (n_tr, n_va, n_te)

        feats = build_graph_features(adj, active, feat_dim=feat_dim, seed=seed)
        if with_node_features:
            nf = d["node_feats"]
            if feat_norm or feat_pca > 0:
                fit_rows = np.zeros_like(active)
                fit_rows[:n_tr] = active[:n_tr]
                nf = normalise_node_features(nf, fit_rows, zscore=feat_norm, pca=feat_pca)
            nf = nf * active[..., None].astype(np.float32)
            feats = np.concatenate([feats, nf.astype(np.float32)], axis=-1)

        self.num_nodes = N
        self.feat_dim = feats.shape[-1]
        self.node_index = d["keep"]
        self.num_nodes_total = int(d["n_all"])
        self.node_labels_dyn = labels
        self.n_classes = int(labels.max()) + 1
        self.num_classes = self.n_classes
        self.dyn_label_task = "binary" if self.n_classes == 2 else "multiclass"
        self.state_label_mode = f"chase:{name}"
        # Static fallback (majority label per node) for the SnapshotSequence.
        maj = np.zeros(N, dtype=np.int64)
        for i in range(N):
            col = labels[:, i]; col = col[col >= 0]
            maj[i] = int(np.bincount(col).argmax()) if col.size else 0
        self.node_labels = maj
        self.audit = chase_label_audit(labels, n_tr, n_va)
        self.structure = structure_audit(adj)

        seq = SnapshotSequence(adj=adj, active=active, feats=feats, node_labels=maj)
        self._ds = DTDGDataset.from_global(
            seq, split=split, history_size=history_size, num_preds=num_preds,
            stride=stride, splits=self.SPLITS,
            split_mode=split_mode, split_seed=seed,
        )
        self.starts = self._ds.starts
        a = self.audit
        print(f"[chase:{name}] split={split} windows={len(self.starts)} N={N}/{self.num_nodes_total} "
              f"T={T} (paper split {n_tr}/{n_va}/{n_te}) feat_dim={self.feat_dim} "
              f"classes={self.n_classes} task={self.dyn_label_task}", flush=True)
        if split == "train":
            print(f"[chase:{name}] label audit: labelled cells={a['valid_rows']}/{a['cells']} "
                  f"train class freq={a['train_class_freq']} "
                  f"persistence_acc={a['persistence_acc']:.3f} (test snapshots {a['persistence_acc_test']:.3f}) "
                  f"switch_rate={a['switch_rate']:.3f} nodes_with_1+_changes={a['nodes_1plus_changes']:.2f} "
                  f"2+={a['nodes_2plus_changes']:.2f}", flush=True)
            st = self.structure
            print(f"[chase:{name}] structure audit: consecutive-snapshot Jaccard={st['jaccard']:.3f} "
                  f"edges new vs previous={st['new_share']:.3f} gone={st['gone_share']:.3f} "
                  f"share of all-time edges per snapshot={st['ever_share']:.3f}", flush=True)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]

    @property
    def feature_dim(self) -> int:
        return self.feat_dim
