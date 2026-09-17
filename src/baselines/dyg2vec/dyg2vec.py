"""DyG2Vec: Efficient Representation Learning for Dynamic Graphs
(Alomrani, Biparva, Zhang & Coates, TMLR'24; arXiv 2210.16906) — the
self-supervised baseline.

Implemented from the paper (v3, TMLR); the official repository
(huawei-noah/noah-research, ``graph_atlas/DyG2Vec``) contains only a README
and a figure, so there is no reference code to follow. Two parts:

  1. **Encoder** — the window-based temporal attention encoder of Sec. 4.1
     lives in the encoder registry as ``--encoder_kind dyg2vec``
     (:class:`~src.world_models.graph_jepa_encoder.DyG2VecEncoder`), because
     it IS the method's representation learner and has to be what the probe
     sees, exactly as for MaskDGNN. It encodes a history window once, with no
     node memory, and is what ``encoder.ckpt`` stores.

  2. **SSL objective** (Sec. 4.3, App. A.1/A.3) — a VICReg joint embedding of
     two *distorted views of the same history window*. NOT a predictive /
     target-encoder scheme: the two views go through the SAME encoder
     ``g_theta`` (shared weights, no EMA, no stop-gradient), then a shared
     2-layer MLP predictor ``p_phi`` gives ``Z' = p_phi(H')``, ``Z'' =
     p_phi(H'')`` and

         L_SSL = lambda * s(Z', Z'') + mu * [v(Z') + v(Z'')] + nu * [c(Z') + c(Z'')]

     with the invariance ``s`` = mean squared distance, the variance hinge
     ``v(Z) = mean_j max(0, gamma - sqrt(Var(Z_:j) + eps))`` and the
     covariance ``c(Z) = sum_{i != j} C(Z)_ij^2 / d`` (eq. 9-10, A.1);
     ``lambda = mu = 25, nu = 1, gamma = 1`` (Bardes et al., 2022). The
     distortions are **edge dropout** and **edge-feature masking** applied to
     the structural temporal edge encodings ``z_p, c_p``, both with
     ``p_d = 0.3`` (A.3; validated over {0.1, 0.15, 0.2, 0.3}). The loss is
     node-level, over the nodes of the window graph.

The paper's default results are end-to-end supervised; its SSL results
("SSL-init", Table 9) freeze the pre-trained encoder and train a downstream
decoder on top, which is precisely this pipeline's frozen-probe protocol. Only
the encoder is kept and probed with the SAME link-prediction /
node-classification probe as DyG-JEPA.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.world_models.graph_jepa_encoder import normalize_adjacency


def symmetric_edge_mask(adj: torch.Tensor, p: float,
                        generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Per-window Bernoulli(1 - p) keep-mask over UNDIRECTED edges.

    ``adj`` (T, N, N) bool, symmetric. Returns a (T, N, N) bool mask that is
    symmetric and True on a kept edge's both directions, so a dropped
    interaction disappears for both endpoints (and the same mask serves the
    structural-feature masking, where an interaction's ``z_p / c_p`` are
    hidden for both endpoints at once).
    """
    T, N, _ = adj.shape
    u = torch.rand(T, N, N, generator=generator, device=adj.device)
    u = torch.triu(u, diagonal=1)
    u = u + u.transpose(-1, -2)                       # symmetric uniforms
    return (u >= p) & adj


class Predictor(nn.Module):
    """``p_phi``: "a simple 2-layer MLP that maps node embeddings H to node
    representations Z" (A.3). No projector/expander (v1 Sec. 4.1)."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def _variance(z: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def _covariance(z: torch.Tensor) -> torch.Tensor:
    n, d = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(n - 1, 1)
    return (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / d


def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor, lam: float = 25.0,
                mu: float = 25.0, nu: float = 1.0, gamma: float = 1.0,
                eps: float = 1e-4):
    """Eq. 9-10 / App. A.1. ``z1``, ``z2`` (n, d) representations of the same
    nodes under the two views. Returns ``(total, inv, var, cov)``."""
    if z1.shape[0] < 2:
        zero = z1.sum() * 0.0
        return zero, zero, zero, zero
    inv = F.mse_loss(z1, z2, reduction="none").sum(dim=1).mean()   # mean ||z'-z''||^2
    var = _variance(z1, gamma, eps) + _variance(z2, gamma, eps)
    cov = _covariance(z1) + _covariance(z2)
    return lam * inv + mu * var + nu * cov, inv, var, cov


class DyG2VecSSL(nn.Module):
    """Shared window encoder + predictor trained with VICReg on two
    distorted views of each history window."""

    def __init__(self, encoder: nn.Module, latent_dim: int,
                 p_edge: float = 0.3, p_struct: float = 0.3,
                 lam: float = 25.0, mu: float = 25.0, nu: float = 1.0,
                 gamma: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.predictor = Predictor(latent_dim)
        self.p_edge, self.p_struct = p_edge, p_struct
        self.lam, self.mu, self.nu, self.gamma = lam, mu, nu, gamma

    def _view(self, x: torch.Tensor, adj: torch.Tensor,
              generator: Optional[torch.Generator]) -> torch.Tensor:
        """One distorted view -> (N, D) representations of the full window
        (the encoder's last output step)."""
        keep_e = symmetric_edge_mask(adj, self.p_edge, generator)
        keep_s = symmetric_edge_mask(adj, self.p_struct, generator)
        adj_v = (adj & keep_e).float()
        kw = {"struct_keep": keep_s[None]} if self.p_struct > 0 else {}
        h = self.encoder(x[None], normalize_adjacency(adj_v), **kw)[0]  # (T,N,D)
        return self.predictor(h[-1])

    def window_loss(self, feats_w: np.ndarray, adj_w: np.ndarray, device,
                    generator: Optional[torch.Generator] = None):
        """VICReg loss of one window: ``feats_w`` (T, N, F), ``adj_w``
        (T, N, N) bool. The loss is over the nodes with >= 1 interaction in
        the (undistorted) window — the nodes of the window graph."""
        x = torch.from_numpy(np.ascontiguousarray(feats_w)).float().to(device)
        adj = torch.from_numpy(np.ascontiguousarray(adj_w)).to(device)
        present = adj.any(dim=0).any(dim=1)                           # (N,)
        z1 = self._view(x, adj, generator)[present]
        z2 = self._view(x, adj, generator)[present]
        return vicreg_loss(z1, z2, self.lam, self.mu, self.nu, self.gamma)
