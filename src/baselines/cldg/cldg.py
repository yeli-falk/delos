"""CLDG: Contrastive Learning on Dynamic Graphs (Xu et al., ICDE'23) — baseline.

Self-supervised temporal contrastive learning. The idea ("temporal translation
invariance"): the same node, viewed through two different temporal sub-spans of
the dynamic graph, should map to nearby embeddings; different nodes should be
pushed apart. Concretely:

  1. sample ``views`` temporal sub-spans of the (training) snapshot sequence,
  2. aggregate each span into one static graph + node features,
  3. encode each with a SHARED GNN -> per-node embeddings, project, L2-normalise,
  4. restrict to nodes present in ALL views (the positive correspondence),
  5. symmetric InfoNCE (NT-Xent, tau=0.07) with positives on the diagonal.

Only the encoder is kept; it is probed with the same link-prediction /
node-classification probe as DyG-JEPA for an apples-to-apples comparison.

This is faithful to the reference (https://github.com/yimingxu24/CLDG) in
*method* — symmetric NT-Xent over node-intersection positives, random /
sequential temporal view sampling, tau=0.07 — but reimplemented on this
project's dense-adjacency snapshots so it needs no DGL, and it consumes the
pipeline's node features (so the encoder input matches DyG-JEPA exactly).
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.world_models.graph_jepa_encoder import normalize_adjacency


class ProjectionHead(nn.Module):
    """Two-layer projection applied to encoder output before the contrastive
    loss (SimCLR-style; discarded after pretraining)."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def sampling_layer(snapshots: int, views: int, span: float, strategy: str,
                   rng) -> List[float]:
    """Return ``views`` view start-times in snapshot-index units.

    Mirrors CLDG's ``utils.sampling_layer``; each view spans ``span/snapshots``.
      - ``random``:     uniform starts in ``[0, span*(snapshots-1)/snapshots)``.
      - ``sequential``: pick ``views`` of the ``snapshots`` evenly-spaced starts.
    ``rng`` is a ``random.Random`` for reproducibility.
    """
    if strategy == "random":
        hi = span * (snapshots - 1) / snapshots
        return [rng.uniform(0.0, hi) for _ in range(views)]
    if strategy == "sequential":
        starts = [span * i / snapshots for i in range(snapshots)]
        if views > snapshots:
            raise ValueError(
                f"views ({views}) exceeds snapshots ({snapshots}) for the "
                f"sequential strategy")
        return rng.sample(starts, views)
    raise ValueError(f"unknown sampling strategy {strategy!r}")


def view_arrays(adj: np.ndarray, active: np.ndarray, feats: np.ndarray,
                start: float, length: float
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate the snapshots covered by one view ``[start, start+length)``.

    Returns (adj_v (N,N) bool, active_v (N,) bool, feats_v (N,F) float32):
    adjacency/active are OR-reduced over the span; features are mean-pooled.
    """
    T = adj.shape[0]
    s = int(math.floor(start))
    e = min(T, int(math.ceil(start + length)))
    e = max(e, s + 1)                              # at least one snapshot
    s = min(s, T - 1)
    adj_v = adj[s:e].any(axis=0)
    active_v = active[s:e].any(axis=0)
    feats_v = feats[s:e].mean(axis=0).astype(np.float32)
    return adj_v, active_v, feats_v


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.07
                 ) -> torch.Tensor:
    """Symmetric InfoNCE over aligned node embeddings (positives = diagonal).

    ``z1``/``z2`` are (M, D) L2-normalised embeddings of the SAME M nodes seen
    through two views. Matches CLDG: ``(CE(z1 z2^T / tau) + CE(z2 z1^T / tau))/2``.
    """
    device = z1.device
    logits12 = (z1 @ z2.t()) / tau
    logits21 = (z2 @ z1.t()) / tau
    labels = torch.arange(z1.shape[0], device=device)
    return 0.5 * (F.cross_entropy(logits12, labels)
                  + F.cross_entropy(logits21, labels))


class CLDG(nn.Module):
    """Shared GNN encoder + projection head for temporal contrastive learning."""

    def __init__(self, encoder: nn.Module, latent_dim: int, tau: float = 0.07):
        super().__init__()
        self.encoder = encoder
        self.proj = ProjectionHead(latent_dim)
        self.tau = tau

    def embed_view(self, feats_v: np.ndarray, adj_v: np.ndarray,
                   device) -> torch.Tensor:
        """Encode one aggregated view -> (N, D) projected, L2-normalised."""
        x = torch.from_numpy(feats_v).float().to(device)[None, None]   # (1,1,N,F)
        adj = torch.from_numpy(np.ascontiguousarray(adj_v)).float().to(device)
        z = self.encoder(x, normalize_adjacency(adj))                  # (1,1,N,D)
        z = self.proj(z[0, 0])                                         # (N,D)
        return F.normalize(z, dim=-1)
