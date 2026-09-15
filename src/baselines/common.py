"""Shared plumbing for the encoder-only baselines (CLDG / DDGCL / IDOL / MaskDGNN).

These baselines differ ONLY in their self-supervised objective. Everything that
could otherwise make their numbers incomparable lives here, so it is written
once and matches DyG-JEPA:

  * :func:`split_snapshots` — chronological train / val snapshot ranges derived
    from the dataset's own ``SPLITS``. The train range is exactly the span
    ``global_window_starts`` hands DyG-JEPA's train windows, and the val range
    is exactly its val targets — no split arithmetic is duplicated per baseline.
  * :class:`BestValTracker` — best-val-loss checkpoint selection plus the same
    early-stopping rule as ``train_dyg_jepa._update_best`` (an improvement must
    beat the running best by more than ``min_delta``).

Without these the baselines selected on *train* loss (or simply kept the final
epoch) while DyG-JEPA selected on val — an asymmetry that flatters the JEPA
independently of representation quality.
"""
from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np
import torch.nn as nn

SPLITS_FALLBACK: Tuple[float, float, float] = (0.7, 0.1, 0.2)

# (adj (T,N,N) bool, active (T,N) bool, feats (T,N,F) float32)
Snapshots = Tuple[np.ndarray, np.ndarray, np.ndarray]


class ConstantAdjacency:
    """A (T, N, N) snapshot adjacency for a graph that never changes.

    METR-LA is fixed-topology: one 207x207 adjacency shared by all ~24k 5-minute
    snapshots. The baselines index it as if it were a per-snapshot stack, and
    every way of giving them a real stack is wrong here:

    * materialising it costs ~1 GB at bool precision, per split;
    * ``np.broadcast_to`` is a free view but CLDG reduces over a span
      (``adj[s:e].any(axis=0)``), and numpy walks all ``e-s`` zero-stride
      snapshots to do it -- ~250M element-ops per view, thousands of times per
      run.

    So this exposes the same tiny surface the baselines actually use
    (``.shape``, ``len``, integer and slice indexing, ``.any(axis=0)``) and
    answers each in O(1) by returning the single underlying matrix. That is
    what lets the contrastive baselines run at METR-LA's native 5-minute
    resolution -- the same snapshots DyG-JEPA sees -- rather than a subsampled
    timeline that would make their columns incomparable.
    """

    def __init__(self, adj: np.ndarray, length: int):
        self._adj = adj
        self._T = int(length)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self._T,) + self._adj.shape

    @property
    def dtype(self):
        return self._adj.dtype

    def __len__(self) -> int:
        return self._T

    def __getitem__(self, key):
        if isinstance(key, slice):
            n = len(range(*key.indices(self._T)))
            return ConstantAdjacency(self._adj, n)
        idx = int(key)
        if not -self._T <= idx < self._T:
            raise IndexError(f"snapshot {idx} out of range for T={self._T}")
        return self._adj

    def any(self, axis=0):
        # OR-reducing identical snapshots is the snapshot itself.
        if axis != 0:
            raise ValueError("ConstantAdjacency.any only reduces over time")
        return self._adj

    def mean(self, axis=0):
        return self._adj


def _fixed_topology_snapshots(ds, method: str) -> Optional[Snapshots]:
    """Snapshots for a fixed-topology dataset (METR-LA), or None if not one.

    Such a dataset holds its split's contiguous feature timeline in ``.feats``
    (T, N, F) and one shared graph in ``.adj`` (N, N). ``active`` marks the
    readings that exist -- a missing loop-detector sample is stored as 0.0, the
    same entries every published METR-LA metric masks out.
    """
    feats = getattr(ds, "feats", None)
    adj = getattr(ds, "adj", None)
    if feats is None or adj is None or np.asarray(adj).ndim != 2:
        return None
    feats = np.asarray(feats, dtype=np.float32)
    raw = getattr(ds, "raw_speeds", None)
    # Channel 0 is the normalised signal, so "missing" cannot be read off it
    # directly; use the raw series when the dataset kept one.
    active = (np.asarray(raw) > 0 if raw is not None
              else np.ones(feats.shape[:2], dtype=bool))
    return (ConstantAdjacency(np.asarray(adj).astype(bool), feats.shape[0]),
            active.astype(bool), feats)


def split_snapshots(ds, method: str, args=None) -> Tuple[Snapshots, Snapshots]:
    """Return ``(train_snapshots, val_snapshots)`` for the dataset behind ``ds``.

    Two dataset families, one contract:

    * **global DTDG** (dblp / jodie_* / reddit_* / england_covid) -- the wrapper
      holds the whole timeline in ``ds._ds``, so both ranges are sliced out of
      it here. Train is ``[0, train_end)`` and val is ``[train_end, val_end)``
      — the same chronological boundaries ``global_window_starts`` uses, so no
      baseline ever trains on a snapshot DyG-JEPA holds out (or vice versa).
    * **fixed topology** (metr_la) -- the wrapper is already per-split and
      carries no global view, so the val range comes from building the val
      split through ``args``. Passing ``args`` is therefore required for those
      datasets; without it there is no way to reach the same val snapshots
      DyG-JEPA selects on, and silently re-splitting the *train* range instead
      would give the baselines a different selection signal than the JEPA arms.
    """
    inner = getattr(ds, "_ds", None)
    if inner is None or getattr(inner, "_mode", None) != "global":
        fixed = _fixed_topology_snapshots(ds, method)
        if fixed is None:
            raise SystemExit(
                f"{method} supports global dynamic-graph datasets "
                "(--data_source dblp / jodie_* / reddit_* / england_covid) and "
                "fixed-topology signal datasets (--data_source metr_la). The "
                "chosen dataset exposes neither a global snapshot sequence nor "
                "a (.feats, .adj) pair.")
        if args is None:
            raise SystemExit(
                f"{method}: {type(ds).__name__} is per-split, so building the "
                f"val snapshots needs the CLI args. This is a call-site bug: "
                f"pass split_snapshots(train_ds, '{method}', args).")
        from src.custom_codes.dataset_factory import build_dynamic_dataset
        val_fixed = _fixed_topology_snapshots(
            build_dynamic_dataset(args, split="val"), method)
        return fixed, val_fixed

    splits = getattr(ds, "SPLITS", SPLITS_FALLBACK)
    T = int(inner.adj.shape[0])
    train_end = max(1, int(T * splits[0]))
    val_end = min(T, train_end + int(T * splits[1]))
    if val_end <= train_end:
        raise SystemExit(
            f"{method}: the val split is empty (T={T}, train_end={train_end}, "
            f"splits={splits}). Model selection needs at least one val "
            f"snapshot — bin the timeline into more steps.")
    train = (inner.adj[:train_end], inner.active[:train_end],
             inner.feats[:train_end])
    val = (inner.adj[train_end:val_end], inner.active[train_end:val_end],
           inner.feats[train_end:val_end])
    return train, val


class BestValTracker:
    """Best-val-loss model selection with optional early stopping.

    Mirrors ``train_dyg_jepa._update_best`` exactly: an epoch counts as an
    improvement only if its val loss drops more than ``min_delta`` below the
    running best; ``patience`` non-improving epochs in a row stop training
    (``patience <= 0`` disables stopping but still tracks the best state).
    """

    def __init__(self, patience: int = 0, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.no_improve = 0
        self._state: Optional[dict] = None

    def update(self, val_loss: float, module: nn.Module) -> Tuple[bool, bool]:
        """Record ``val_loss``; snapshot ``module`` if it improved.

        Returns ``(improved, stop)``.
        """
        improved = val_loss < self.best - self.min_delta
        if improved:
            self.best = val_loss
            self.no_improve = 0
            self._state = copy.deepcopy(module.state_dict())
        else:
            self.no_improve += 1
        stop = self.patience > 0 and self.no_improve >= self.patience
        return improved, stop

    def restore(self, module: nn.Module) -> bool:
        """Load the best-val state back into ``module``. False if never set."""
        if self._state is None:
            return False
        module.load_state_dict(self._state)
        return True
