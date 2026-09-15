"""The universal sliding-window DTDG dataset.

:class:`DTDGDataset` replaces the three near-identical sliding-window classes
(`DBLPDynamicGraphDataset`, `JODIEDynamicGraphDataset`,
`RedditHyperlinkDynamicGraphDataset`) and the per-patient
`EHRshotPerPatientDataset`. Per-dataset files now build the canonical
representation and call one of two constructors:

  - :meth:`DTDGDataset.from_global`     -- one shared timeline; chronological
        target-based train/val/test split; windows may borrow history from the
        previous split's tail. Used by DBLP / JODIE / Reddit.
  - :meth:`DTDGDataset.from_per_entity`  -- many independent short sequences
        (e.g. one per patient); windows slide within each entity and are
        concatenated into one flat list. Used by EHRshot.

Both emit the same canonical batch dict::

    {x_hist:(Th,N,F), x_fut:(Tp,N,F), adj:(T,N,N) bool, active:(T,N) bool,
     year:int}

with ``node_labels`` exposed as an attribute (static per-node) rather than per
window, matching the previous loaders.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.custom_codes.dtdg.canonical import SnapshotSequence

SPLITS_DEFAULT: Tuple[float, float, float] = (0.7, 0.1, 0.2)


def global_window_starts(
    num_steps: int, split: str, history_size: int, num_preds: int,
    stride: int = 1, splits: Tuple[float, float, float] = SPLITS_DEFAULT,
    split_mode: str = "chronological", seed: int = 0,
) -> np.ndarray:
    """Target-based window starts for a single shared timeline.

    ``split_mode='chronological'`` (default): each split owns a contiguous range
    of *target* steps; a window's history may reach into the previous split's
    tail. Test is strictly later than train -> this is the temporal-OOD setting.
    This is the OGB-MAG / TGB / EvolveGCN convention.

    ``split_mode='iid'``: the SAME window set, but partitioned by a seeded
    permutation of *non-overlapping* windows instead of by time — the control
    with no temporal shift. Windows are laid down at stride=window (so train and
    test windows never share a timestep, avoiding leakage) and then randomly
    assigned to splits. Use it to test whether a method's advantage is
    temporal-OOD-specific: compare its margin here vs the chronological split.
    """
    window = history_size + num_preds
    if split_mode == "iid":
        # Non-overlapping windows so random split assignment can't leak a
        # timestep across train/test.
        all_starts = np.arange(0, num_steps - window + 1, window, dtype=np.int64)
        if all_starts.size == 0:
            raise ValueError(
                f"iid split: timeline too short (num_steps={num_steps}, "
                f"window={window}) for even one non-overlapping window.")
        rng = np.random.RandomState(seed)
        perm = rng.permutation(all_starts.size)
        n_tr = int(round(all_starts.size * splits[0]))
        n_va = int(round(all_starts.size * splits[1]))
        if split == "train":
            sel = perm[:n_tr]
        elif split == "val":
            sel = perm[n_tr:n_tr + n_va]
        else:
            sel = perm[n_tr + n_va:]
        return np.sort(all_starts[sel])

    train_end = int(num_steps * splits[0])
    val_end = train_end + int(num_steps * splits[1])
    if split == "train":
        target_lo, target_hi = history_size, train_end
    elif split == "val":
        target_lo, target_hi = train_end, val_end
    else:
        target_lo, target_hi = val_end, num_steps

    start_lo = max(0, target_lo - history_size)
    start_hi = min(num_steps - window, target_hi - history_size - num_preds)
    if start_hi < start_lo:
        raise ValueError(
            f"Split '{split}' has no valid windows: "
            f"target range [{target_lo}, {target_hi}), "
            f"requires start in [{start_lo}, {start_hi}]. "
            f"Either reduce history_size+num_preds (window={window}) "
            f"or lengthen the timeseries (num_steps={num_steps})."
        )
    return np.arange(start_lo, start_hi + 1, stride, dtype=np.int64)


def per_entity_window_starts(
    lengths: Sequence[int], window: int, stride: int,
    max_windows_per_entity: int, split: str,
) -> np.ndarray:
    """Flatten per-entity sliding windows into a single (M, 2) start list.

    Returns rows of ``(entity_idx, t_start)``. Entities shorter than ``window``
    contribute nothing. If an entity yields more than ``max_windows_per_entity``
    windows they are subsampled: evenly (deterministic) for ``train``, last-K
    (recent) for ``val``/``test`` — matching the EHRshot loader.
    """
    starts: List[Tuple[int, int]] = []
    for e_idx, T_p in enumerate(lengths):
        if T_p < window:
            continue
        n_windows = (T_p - window) // stride + 1
        window_starts = np.arange(0, n_windows * stride, stride)
        if max_windows_per_entity > 0 and len(window_starts) > max_windows_per_entity:
            if split == "train":
                sel = np.linspace(0, len(window_starts) - 1,
                                  max_windows_per_entity, dtype=np.int64)
            else:
                sel = np.arange(len(window_starts) - max_windows_per_entity,
                                len(window_starts))
            window_starts = window_starts[sel]
        starts.extend([(e_idx, int(s)) for s in window_starts])
    if not starts:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(starts, dtype=np.int64)


class DTDGDataset(Dataset):
    """Universal discrete-time dynamic-graph dataset (see module docstring)."""

    def __init__(self):  # use the from_* constructors
        raise RuntimeError(
            "Construct via DTDGDataset.from_global or DTDGDataset.from_per_entity")

    # ------------------------------------------------------------------
    # Global single-timeline construction (DBLP / JODIE / Reddit)
    # ------------------------------------------------------------------
    @classmethod
    def from_global(
        cls,
        seq: SnapshotSequence,
        split: str = "train",
        history_size: int = 3,
        num_preds: int = 1,
        stride: int = 1,
        splits: Tuple[float, float, float] = SPLITS_DEFAULT,
        year_offset: int = 0,
        split_mode: str = "chronological",
        split_seed: int = 0,
    ) -> "DTDGDataset":
        assert split in ("train", "val", "test")
        self = cls.__new__(cls)
        self._mode = "global"
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.window = history_size + num_preds
        self.stride = stride
        self.num_nodes = seq.num_nodes
        self.feat_dim = seq.feat_dim
        self.adj = seq.adj
        self.active = seq.active
        self.feats = seq.feats
        self.node_labels = seq.node_labels
        self.year_offset = year_offset
        self.starts = global_window_starts(
            seq.num_steps, split, history_size, num_preds, stride, splits,
            split_mode=split_mode, seed=split_seed)
        return self

    # ------------------------------------------------------------------
    # Per-entity construction (EHRshot)
    # ------------------------------------------------------------------
    @classmethod
    def from_per_entity(
        cls,
        lengths: Sequence[int],
        window_builder: Callable[[int, int], dict],
        history_size: int,
        num_preds: int,
        num_nodes: int,
        feat_dim: int,
        split: str = "train",
        stride: int = 1,
        max_windows_per_entity: int = 16,
        node_labels: Optional[np.ndarray] = None,
        year_offset: int = 0,
    ) -> "DTDGDataset":
        """``window_builder(entity_idx, t_start) -> canonical dict``.

        The parser supplies ``window_builder`` (which knows how to assemble
        adjacency + features for a window of one entity); this class owns only
        the flat window indexing and subsampling.
        """
        assert split in ("train", "val", "test")
        self = cls.__new__(cls)
        self._mode = "per_entity"
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.window = history_size + num_preds
        self.stride = stride
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self._window_builder = window_builder
        self.node_labels = node_labels
        self.year_offset = year_offset
        self.starts = per_entity_window_starts(
            lengths, self.window, stride, max_windows_per_entity, split)
        return self

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.starts.shape[0]) if self._mode == "per_entity" \
            else len(self.starts)

    def __getitem__(self, idx: int):
        if self._mode == "per_entity":
            e_idx, t_start = self.starts[idx]
            return self._window_builder(int(e_idx), int(t_start))

        start = int(self.starts[idx])
        end = start + self.window
        adj_w = torch.from_numpy(self.adj[start:end])           # (T, N, N) bool
        active_w = torch.from_numpy(self.active[start:end])     # (T, N) bool
        x_w = torch.from_numpy(self.feats[start:end])           # (T, N, F)
        return {
            "x_hist": x_w[: self.history_size],
            "x_fut":  x_w[self.history_size:],
            "adj":    adj_w,
            "active": active_w,
            "year":   start + self.year_offset,
        }

    @property
    def feature_dim(self) -> int:
        return self.feat_dim
