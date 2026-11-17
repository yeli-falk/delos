"""Long-history plumbing and graph supports, shared by STD-MAE and STEP.

Both pre-training baselines read TWO windows per sample:

  * the short forecast history the rest of this repo already yields
    (``x_hist``, 12 steps on METR-LA / 8 days on England COVID), and
  * a much longer window ending at the same instant (``long_history``), which
    is the only thing their masked autoencoders ever see. STD-MAE's paper uses
    3 days = 864 steps on METR-LA, STEP's 1 week = 2016.

This repo's regression datasets are per-split sliding-window views and carry no
long window, so :func:`build_long_history_datasets` reconstructs the global
signal timeline once, works out where each split's windows sit on it, and wraps
each dataset in a :class:`LongHistoryView` that adds the long slice.

Living here rather than inside one baseline is deliberate: both arms must see
byte-identical long windows, or a difference between their rows could come from
the input geometry rather than the method.

Reaching back across a split boundary is deliberate and is not leakage: the long
window is strictly PAST input, exactly as in the reference (BasicTS keeps one
contiguous array and lets a val/test window's long history run back into earlier
data). The forecast TARGET always comes from the split's own window. When the
timeline runs out — the first windows of the whole dataset — the missing steps
are left-padded with the raw 0.0 that already encodes "no reading" in METR-LA,
so the MAE loss's ``null_val=0.0`` drops them instead of fitting them.

The graph side is here too: Graph WaveNet consumes DCRNN-style
``doubletransition`` supports (row-normalised A and Aᵀ). METR-LA has one static
graph; England COVID's mobility graph changes every snapshot, so its supports
are built per batch from the window's own history adjacency. (STEP replaces
these with a graph it learns, so only STD-MAE uses them.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.custom_codes.dataset_factory import build_dynamic_dataset

SPLIT_ORDER = ("train", "val", "test")


# -- global timeline reconstruction --
def _parquet_timeline(ds, split: str) -> np.ndarray:
    """Rebuild a contiguous (T, N) timeline from a pre-windowed split.

    The HF METR-LA mirror stores one row per (sensor, window) at stride 1, so
    window ``i`` and window ``i+1`` overlap in all but one step and step 0 of
    each window walks the timeline. Verified rather than assumed: a stride != 1
    or an unsorted file would silently produce a timeline that is not the data.
    """
    w = np.asarray(ds.raw_window_speeds, dtype=np.float32)       # (n, T, N)
    if w.shape[0] > 1:
        probe = slice(0, min(w.shape[0] - 1, 512))
        if not np.allclose(w[probe, 1], w[1:, 0][probe]):
            raise SystemExit(
                f"STD-MAE: the '{split}' windows are not consecutive stride-1 "
                f"windows over one timeline, so the long history cannot be "
                f"reconstructed from them. Re-download METR-LA "
                f"(scripts/metr_la/download_metr_la.sh) or use the DCRNN "
                f"metr-la.h5 layout, which carries the timeline directly.")
    return w[:, 0]                                               # (n, N)


def build_signal_timeline(datasets: Dict[str, Dataset]
                          ) -> Tuple[np.ndarray, Dict[str, int], int]:
    """``(timeline (T_global, N) raw, {split: window-start offset}, train_len)``.

    ``global_start(split, i) == offsets[split] + datasets[split].starts[i]`` is
    the index on ``timeline`` of the first HISTORY step of window ``i``.

    ``train_len`` is how much of the timeline belongs to the train split. STEP's
    graph-structure learner reads that whole span as a per-node feature, so it
    has to know exactly where train ends — reading one step past it would leak
    held-out signal into the learned adjacency.
    """
    order = [s for s in SPLIT_ORDER if s in datasets]
    if not order:
        raise SystemExit("STD-MAE: no datasets to build a timeline from.")
    ref = datasets[order[0]]

    # (1) Dynamic-topology datasets (England COVID) keep the whole timeline and
    #     index it with global starts, so there is nothing to stitch.
    raw = getattr(ref, "raw_signal", None)
    if raw is not None:
        raw = np.asarray(raw, dtype=np.float32)
        # Matches DTDGDataset.global_window_starts' own train boundary.
        splits = getattr(ref, "SPLITS", (0.7, 0.1, 0.2))
        return raw, {s: 0 for s in order}, int(raw.shape[0] * splits[0])

    layout = getattr(ref, "layout", None)
    if layout is None:
        raise SystemExit(
            f"STD-MAE needs a contiguous per-node signal timeline to build its "
            f"long history window. {type(ref).__name__} exposes neither a "
            f"global '.raw_signal' nor METR-LA's '.layout'. Supported: "
            f"--data_source england_covid / metr_la.")
    if order != list(SPLIT_ORDER[:len(order)]):
        # A per-split dataset gives no clue where its segment sits globally, so
        # the offsets only come out right if it is handed an unbroken prefix of
        # the splits (train, then val, then test).
        raise SystemExit(
            f"STD-MAE on a '{layout}' dataset needs an unbroken prefix of "
            f"{SPLIT_ORDER} to locate each split on the global timeline; got "
            f"{order}.")

    chunks: List[np.ndarray] = []
    for i, s in enumerate(order):
        ds = datasets[s]
        if layout in ("h5", "synthetic"):
            # `.raw_speeds` IS this split's contiguous slice of the timeline.
            chunk = np.asarray(ds.raw_speeds, dtype=np.float32)
        elif layout == "hf_parquet":
            chunk = _parquet_timeline(ds, s)
            if i == len(order) - 1:
                # The last window's remaining steps close out the timeline.
                tail = np.asarray(ds.raw_window_speeds[-1, 1:], dtype=np.float32)
                chunk = np.concatenate([chunk, tail], axis=0)
        else:
            raise SystemExit(f"STD-MAE: unsupported dataset layout {layout!r}.")
        chunks.append(chunk)

    offsets, acc = {}, 0
    for s, c in zip(order, chunks):
        offsets[s] = acc
        acc += int(c.shape[0])
    # The train chunk IS the train region of the timeline for these layouts.
    return np.concatenate(chunks, axis=0), offsets, int(chunks[0].shape[0])


# -- the dataset wrapper --
class LongHistoryView(Dataset):
    """A regression dataset plus ``long_history`` ``(L_long, N, 1)``.

    Every other attribute (``mean``, ``std``, ``adj``, ``num_nodes``,
    ``feature_dim``, ``signed_target``, ...) is delegated to the wrapped
    dataset, so ``probes.graph_inputs`` and the shared metric helpers treat
    this exactly like the dataset every other arm scores.
    """

    def __init__(self, base: Dataset, timeline: np.ndarray, offset: int,
                 long_len: int, history_size: int):
        self.base = base
        self.timeline = timeline
        self.offset = int(offset)
        self.long_len = int(long_len)
        self.history_size = int(history_size)
        starts = np.asarray(base.starts, dtype=np.int64)
        ends = self.offset + starts + self.history_size
        self.n_padded = int((ends - self.long_len < 0).sum())

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = dict(self.base[idx])
        end = self.offset + int(self.base.starts[idx]) + self.history_size
        start = end - self.long_len
        if start >= 0:
            seg = self.timeline[start:end]
        else:
            seg = np.zeros((self.long_len, self.timeline.shape[1]), np.float32)
            if end > 0:
                seg[-end:] = self.timeline[:end]
        seg = (seg - self.base.mean) / self.base.std
        item["long_history"] = torch.from_numpy(
            np.ascontiguousarray(seg, dtype=np.float32)).unsqueeze(-1)
        return item

    def __getattr__(self, name: str):
        # Only reached when normal lookup fails. Guard the dunder/partially-
        # constructed cases so pickling to DataLoader workers still works.
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            base = object.__getattribute__(self, "base")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(base, name)


@dataclass
class TimelineInfo:
    """What the pre-training baselines need about the reconstructed timeline.

    ``timeline`` is RAW (un-normalised) so callers can apply the train
    normalisation themselves; ``mean``/``std`` are the train statistics every
    other arm in the table also normalises with.
    """
    timeline: np.ndarray
    offsets: Dict[str, int]
    train_length: int
    mean: float
    std: float

    def normalised_train_region(self) -> np.ndarray:
        """``(train_length, N)`` train-normalised signal — STEP's node features."""
        return (self.timeline[:self.train_length] - self.mean) / self.std


def build_long_history_datasets(args, long_len: int,
                                splits: Sequence[str] = SPLIT_ORDER
                                ) -> Tuple[Dict[str, LongHistoryView], TimelineInfo]:
    """Build the splits, share the train normalisation, add the long window.

    Builds ``splits`` plus every split chronologically before them: the global
    timeline cannot be located from one split alone (see
    :func:`build_signal_timeline`). Returns the wrapped datasets and a
    :class:`TimelineInfo` describing the timeline they were cut from.
    """
    # Locating a split on the global timeline needs every split BEFORE it (the
    # offsets are cumulative), but nothing after: the long window only reaches
    # backwards. So a pre-training run that asks for train+val never pays for
    # pivoting the test parquet.
    last = max(SPLIT_ORDER.index(s) for s in splits)
    need = list(SPLIT_ORDER[:last + 1])
    base = {s: build_dynamic_dataset(args, s) for s in need}
    # Same rule as every other arm: val/test are normalised with TRAIN stats.
    for s in need[1:]:
        base[s].mean, base[s].std = base["train"].mean, base["train"].std

    timeline, offsets, train_length = build_signal_timeline(base)
    if long_len > timeline.shape[0]:
        raise SystemExit(
            f"--long_history {long_len} exceeds the whole timeline "
            f"({timeline.shape[0]} steps) for --data_source {args.data_source}. "
            f"Every long window would be padding. Lower --long_history (and "
            f"--patch_size with it).")

    out = {}
    for s in splits:
        view = LongHistoryView(base[s], timeline, offsets[s], long_len,
                               args.history_size)
        pad_pct = 100.0 * view.n_padded / max(len(view), 1)
        print(f"[data] {s:<5} windows={len(view):>6}  long_history={long_len}"
              f"  left-padded={view.n_padded} ({pad_pct:.1f}%)", flush=True)
        out[s] = view
    print(f"[data] timeline={timeline.shape[0]} steps x {timeline.shape[1]} "
          f"nodes  train_region={train_length}  offsets={offsets}", flush=True)
    return out, TimelineInfo(timeline, offsets, train_length,
                             base["train"].mean, base["train"].std)


# -- Graph WaveNet supports --
def _row_normalise(a: torch.Tensor) -> torch.Tensor:
    """``D_r^-1 A`` with all-zero rows left at zero (BasicTS's ``asym_adj``)."""
    rowsum = a.sum(-1, keepdim=True)
    return a / rowsum.clamp(min=1e-12) * (rowsum > 0)


def static_supports(adj: np.ndarray) -> List[torch.Tensor]:
    """DCRNN ``doubletransition`` supports for a fixed graph: ``[P, Pᵀ]``."""
    a = torch.as_tensor(np.asarray(adj, dtype=np.float32))
    return [_row_normalise(a), _row_normalise(a.transpose(0, 1))]


def batch_supports(adj_hist: torch.Tensor) -> List[torch.Tensor]:
    """Per-window ``doubletransition`` supports for a dynamic graph.

    ``adj_hist`` is ``(B, T_hist, N, N)``. Averaging over the history window
    turns the snapshot sequence into one weighted graph (an edge's weight is
    how often it was present), which is the natural static-graph reduction of
    the window Graph WaveNet is being given. Self-loops are added because the
    METR-LA adjacency the reference normalises has them and England COVID's
    parser strips them.
    """
    a = adj_hist.float().mean(dim=1)                             # (B, N, N)
    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    a = a + eye
    return [_row_normalise(a), _row_normalise(a.transpose(-1, -2))]
