"""Canonical intermediate representations for the DTDG pipeline.

Per-dataset parsers convert raw files into one of these, then hand off to
:class:`~src.custom_codes.dtdg.dataset.DTDGDataset`. Keeping the contract in
dataclasses means the universal dataset never has to know which dataset it is
serving.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EdgeStream:
    """A continuous-time dynamic graph as a flat event stream.

    ``src``/``dst`` are dense (or raw, pre-factorised) node ids; ``t`` the
    event timestamp. ``edge_feat`` optionally carries per-event features
    (e.g. JODIE 172-dim LIWC); ``edge_label`` optional per-event labels.
    Binned into a :class:`SnapshotSequence` via :mod:`.binning`.
    """
    src: np.ndarray
    dst: np.ndarray
    t: np.ndarray
    edge_feat: Optional[np.ndarray] = None
    edge_label: Optional[np.ndarray] = None


@dataclass
class GraphPayload:
    """Optional side information attached to a dataset.

    ``node_feat_static`` are time-invariant per-node descriptors (e.g. EHRshot
    vocab one-hot). ``node_labels`` / ``edge_labels`` are downstream targets.
    ``meta`` is a free-form dict for dataset-specific extras (vocab lists, etc.).
    """
    node_feat_static: Optional[np.ndarray] = None
    node_labels: Optional[np.ndarray] = None
    edge_labels: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


@dataclass
class SnapshotSequence:
    """A discretised discrete-time dynamic graph over the full timeline.

    Shapes:
      adj    : (T, N, N) bool   per-snapshot adjacency (symmetric, no self-loop)
      active : (T, N) bool      node existence per snapshot
      feats  : (T, N, F) float32 per-(t, node) features fed to the encoder
    Optional ``node_labels`` are static per-node targets exposed for probing.
    """
    adj: np.ndarray
    active: np.ndarray
    feats: np.ndarray
    node_labels: Optional[np.ndarray] = None

    @property
    def num_steps(self) -> int:
        return int(self.adj.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.adj.shape[1])

    @property
    def feat_dim(self) -> int:
        return int(self.feats.shape[-1])
