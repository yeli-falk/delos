"""BGRL: Large-Scale Representation Learning on Graphs via Bootstrapping
(Thakoor et al., ICLR'22) -- STATIC negative-free self-supervised baseline.

BGRL is a single-graph, single-time method. It is the BYOL recipe on graphs: an online network
predicts the output of a slowly-moving target network, with no negatives and
no reconstruction. The difference that the comparison isolates is WHAT is
predicted -- BGRL predicts the same snapshot under another augmentation,
DELOS predicts a future snapshot.

The method, per Thakoor et al. (arXiv:2102.06514):

  1. two views of ONE static graph by the same two augmentations GRACE uses
     (edge removal with prob ``p_e``, feature-DIMENSION masking with prob
     ``p_f``), each view with its own rates,
  2. an ONLINE encoder ``f_theta`` and a TARGET encoder ``f_phi`` of the same
     architecture; a predictor MLP ``q_theta`` on the online side only. No
     projection head (the paper drops BYOL's projector),
  3. loss = negative cosine similarity between the online prediction of one
     view and the (stop-gradient) target embedding of the OTHER view,
     symmetrised::

        L = 2 - mean_i cos(q(f_theta(v1))_i, f_phi(v2)_i)
              - mean_i cos(q(f_theta(v2))_i, f_phi(v1)_i)

  4. after each optimiser step ``phi <- tau phi + (1 - tau) theta``, with
     ``tau`` cosine-annealed from ``tau_base`` (0.99) to 1.

Faithful to the reference (https://github.com/nerdslab/bgrl) in method:
column-wise feature masking, per-edge removal, ``Linear-PReLU-Linear``
predictor with hidden 512, symmetric cosine loss, cosine-annealed EMA, AdamW
with linear warm-up + cosine decay, and the ONLINE encoder is what is kept.
Reimplemented on this project's dense-adjacency snapshots; the encoder is the
shared GCN, which carries no BatchNorm -- the paper reports normalisation
helps stability but is not what prevents collapse, and the trainer logs the
embedding spread every epoch so a collapse is visible rather than silent.

The *dynamic-graph protocol* is the same as GRACE's and GraphMAE's: every
training snapshot is an independent static graph, a step draws one uniformly
and applies BGRL to it alone, so no temporal structure can reach the encoder
through the objective.
"""
from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baselines.grace.grace import drop_edges, drop_features
from src.world_models.graph_jepa_encoder import normalize_adjacency


def bgrl_loss(q1: torch.Tensor, y2: torch.Tensor,
              q2: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
    """Symmetric negative-cosine bootstrap loss over aligned (M, D) rows.

    ``q*`` are online predictions, ``y*`` target embeddings of the OTHER view
    (detached here, so a caller cannot leak gradient into the target).
    Range [0, 4]; 0 when every prediction is parallel to its target.
    """
    c12 = F.cosine_similarity(q1, y2.detach(), dim=-1).mean()
    c21 = F.cosine_similarity(q2, y1.detach(), dim=-1).mean()
    return 2.0 - c12 - c21


def ema_momentum(step: int, total_steps: int, base: float = 0.99) -> float:
    """Cosine anneal of the target momentum from ``base`` (step 0) to 1."""
    if total_steps <= 0:
        return base
    s = min(max(step, 0), total_steps)
    return 1.0 - (1.0 - base) * (math.cos(math.pi * s / total_steps) + 1.0) / 2.0


class BGRLPredictor(nn.Module):
    """The reference predictor: ``Linear -> PReLU -> Linear``."""

    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.PReLU(1),
                                 nn.Linear(hidden, dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BGRL(nn.Module):
    """Online encoder + predictor, EMA target encoder.

    ``self.encoder`` is the ONLINE encoder -- the one saved and probed.
    """

    def __init__(self, encoder: nn.Module, latent_dim: int,
                 pred_hidden: int = 512,
                 p_e1: float = 0.2, p_e2: float = 0.3,
                 p_f1: float = 0.2, p_f2: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = BGRLPredictor(latent_dim, pred_hidden)
        self.rates = ((p_e1, p_f1), (p_e2, p_f2))

    def trainable_parameters(self):
        return list(self.encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for p_o, p_t in zip(self.encoder.parameters(),
                            self.target_encoder.parameters()):
            p_t.mul_(momentum).add_(p_o.detach(), alpha=1.0 - momentum)
        for b_o, b_t in zip(self.encoder.buffers(),
                            self.target_encoder.buffers()):
            b_t.copy_(b_o)

    def _view(self, x, adj, p_e, p_f, generator):
        return (drop_features(x, p_f, generator)[None, None],
                normalize_adjacency(drop_edges(adj, p_e, generator)))

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                generator: Optional[torch.Generator] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One snapshot -> ``(q1, y2, q2, y1)``, each (N, D).

        ``x`` (N, F) float, ``adj`` (N, N) float. Both views pass through
        BOTH networks (the target never receives gradient).
        """
        (pe1, pf1), (pe2, pf2) = self.rates
        x1, a1 = self._view(x, adj, pe1, pf1, generator)
        x2, a2 = self._view(x, adj, pe2, pf2, generator)
        q1 = self.predictor(self.encoder(x1, a1)[0, 0])
        q2 = self.predictor(self.encoder(x2, a2)[0, 0])
        with torch.no_grad():
            y1 = self.target_encoder(x1, a1)[0, 0]
            y2 = self.target_encoder(x2, a2)[0, 0]
        return q1, y2, q2, y1

    @torch.no_grad()
    def embedding_spread(self, x: torch.Tensor, adj: torch.Tensor,
                         idx: torch.Tensor) -> float:
        """Mean per-dimension std of the L2-normalised online embeddings of
        the un-augmented snapshot. Its absolute level is set by the input
        features and the GCN's smoothing (a random-init GCN on degree-like
        features already sits far below 1/sqrt(D)), so read it RELATIVE to
        its value at initialisation: a collapse drives it towards 0."""
        z = self.encoder(x[None, None], normalize_adjacency(adj))[0, 0][idx]
        return float(F.normalize(z, dim=-1).std(dim=0).mean())
