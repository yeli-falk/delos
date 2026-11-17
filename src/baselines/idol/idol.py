"""IDOL: Topology-monitorable Contrastive Learning on Dynamic Graphs
(Zhu et al., KDD'24) — baseline.

Two pillars of the reference (https://github.com/ZulunZhu/dynamic-contrastive-learning):

  1. **InstantGNN dynamic PPR propagation.** Node features are smoothed by a
     personalised-PageRank propagation that is incrementally maintained as the
     graph evolves. On this project's dense per-snapshot adjacency the converged
     result is the APPNP propagation realised by the ``ppr_gcn`` encoder
     (see ``delos_encoder.PPRGCNEncoder``) — so the propagation is part of
     the encoder and applies identically at pretrain and probe time.

  2. **GGD group discrimination** (Graph Group Discrimination, Zheng et al.,
     NeurIPS'22) — the contrastive objective. Instead of node-pairwise InfoNCE,
     GGD frames self-supervision as a *group* binary-classification task: encode
     the real graph (the positive group, label 1) and a feature-corrupted copy
     (the negative group, label 0), summarise each node embedding into a scalar
     score via a projection MLP, and separate the two groups with a single
     ``BCEWithLogitsLoss``. This is far cheaper than InfoNCE (no NxN similarity
     matrix) and is the objective the reference trains with.

Only the encoder is kept; it is probed with the SAME link-prediction /
node-classification probe as DELOS for an apples-to-apples comparison
(``src.train.analyze_jepa --skip_diagnostic --link_sources encoder``). Faithful to the
reference in *method* — PPR-propagated features + GGD group discrimination with
feature-permutation corruption — but reimplemented on this project's dense
snapshots so it needs no Cython/C++ build and consumes the pipeline's node
features (so the encoder input matches DELOS exactly).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.world_models.delos_encoder import normalize_adjacency


class ProjectionMLP(nn.Module):
    """Stack of square linear layers; output is summed to a per-node score.

    Mirrors GGD's ``self.mlp`` (a ``ModuleList`` of ``Linear(n_hidden,
    n_hidden)``) applied before the per-node score reduction.
    """

    def __init__(self, dim: int, layers: int = 1):
        super().__init__()
        self.lins = nn.ModuleList(nn.Linear(dim, dim) for _ in range(max(layers, 1)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for lin in self.lins:
            z = lin(z)
        return z


def ggd_group_loss(score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
    """Group-discrimination BCE: positives -> 1, negatives -> 0.

    ``score_pos``/``score_neg`` are (M,) per-node scalar scores for the real and
    corrupted groups. Matches GGD: ``BCEWithLogitsLoss(cat([s_pos, s_neg]),
    cat([ones, zeros]))``.
    """
    logits = torch.cat([score_pos, score_neg], dim=0)
    labels = torch.cat([torch.ones_like(score_pos),
                        torch.zeros_like(score_neg)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)


class GGD(nn.Module):
    """Shared encoder + projection head for group-discrimination pretraining."""

    def __init__(self, encoder: nn.Module, latent_dim: int, proj_layers: int = 1):
        super().__init__()
        self.encoder = encoder
        self.proj = ProjectionMLP(latent_dim, proj_layers)

    def _encode(self, feats: np.ndarray, adj_norm: torch.Tensor,
                device) -> torch.Tensor:
        """Encode one (N, F) feature matrix on a fixed graph -> (N, D)."""
        x = torch.from_numpy(np.ascontiguousarray(feats)).float().to(device)
        z = self.encoder(x[None, None], adj_norm)               # (1, 1, N, D)
        return z[0, 0]                                          # (N, D)

    def group_scores(self, feats: np.ndarray, adj_v: np.ndarray, device,
                     rng: np.random.Generator):
        """Return (score_pos, score_neg): per-node scores for the real graph and
        a row-permuted (feature-corrupted) copy on the same propagation graph.
        """
        adj = torch.from_numpy(np.ascontiguousarray(adj_v)).float().to(device)
        adj_norm = normalize_adjacency(adj)

        z_pos = self._encode(feats, adj_norm, device)           # positive group
        perm = rng.permutation(feats.shape[0])                  # corruption
        z_neg = self._encode(feats[perm], adj_norm, device)     # negative group

        s_pos = self.proj(z_pos).sum(dim=-1)                    # (N,)
        s_neg = self.proj(z_neg).sum(dim=-1)
        return s_pos, s_neg
