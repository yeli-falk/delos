"""GRACE: Deep Graph Contrastive Representation Learning (Zhu et al., 2020) —
STATIC self-supervised baseline.

GRACE is a single-graph, single-time contrastive method: it knows nothing
about the timeline. It is the *static* SSL control, the row that answers
"does the self-supervised objective need to be temporal at all?": same GCN
encoder, same node features, same frozen probe on the last history snapshot,
only the objective differs.

The method, per Zhu et al. (arXiv:2006.04131, GRML @ ICML'20):

  1. build two views of ONE static graph with two stochastic augmentations:
       * removing edges:  each edge dropped independently with prob ``p_e``
       * masking features: each feature DIMENSION zeroed (for every node)
         with prob ``p_f``
     each view has its own rates ``(p_e1, p_f1)`` / ``(p_e2, p_f2)``,
  2. encode both views with a SHARED GNN, project with a 2-layer MLP
     (Linear-ELU-Linear), then cosine similarity ``theta``,
  3. for each node ``i`` the positive is its own embedding in the other view;
     the negatives are BOTH the inter-view (``v_k``, ``k != i``) and the
     intra-view (``u_k``, ``k != i``) embeddings::

        l(u_i, v_i) = -log  e^{theta(u_i,v_i)/tau}
                            / ( e^{theta(u_i,v_i)/tau}
                                + sum_{k!=i} e^{theta(u_i,v_k)/tau}     inter
                                + sum_{k!=i} e^{theta(u_i,u_k)/tau} )   intra

     and the objective is the symmetric mean ``(l(u_i,v_i) + l(v_i,u_i)) / 2``
     over nodes. The intra-view negatives are what distinguish this from the
     plain NT-Xent that CLDG uses.

Faithful to the reference (https://github.com/CRIPAC-DIG/GRACE) in method:
``drop_feature`` masks columns, ``dropout_adj`` drops edges, the projection
head is ``fc1 -> ELU -> fc2``, ``semi_loss`` has the inter + intra
denominator, Cora defaults ``tau=0.4, p_e=(0.2, 0.4), p_f=(0.3, 0.4)``. It
is reimplemented on this project's dense-adjacency snapshots so it needs no
PyG and consumes the pipeline's node features, so the encoder input matches
DELOS exactly.

The *dynamic-graph protocol* is the standard one for a static method on a
snapshot sequence (the per-snapshot GCN rows of EvolveGCN / ROLAND): every
training snapshot is an independent static graph; a step draws one snapshot
uniformly from the training range and applies GRACE to it alone. No history
is ever aggregated, so the encoder cannot pick up temporal structure from the
objective -- which is exactly what the row is for.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.world_models.delos_encoder import normalize_adjacency


def drop_edges(adj: torch.Tensor, p: float,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Remove each edge of a dense (N, N) adjacency with probability ``p``.

    The keep-mask is symmetric, so an undirected edge is dropped or kept as
    one unit (its two directed entries never disagree); a directed adjacency
    just gets correlated drops for ``(i, j)`` and ``(j, i)``. Self-loops are
    untouched here -- ``normalize_adjacency`` adds them afterwards anyway.
    """
    if p <= 0.0:
        return adj
    n = adj.shape[-1]
    r = torch.rand(n, n, generator=generator, device=adj.device)
    r = torch.triu(r, diagonal=1)
    r = r + r.transpose(0, 1)                         # symmetric U(0,1) off-diagonal
    keep = r >= p
    keep.fill_diagonal_(True)
    return adj * keep.to(adj.dtype)


def drop_features(x: torch.Tensor, p: float,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Zero each feature DIMENSION (for every node) with probability ``p``.

    Mirrors the reference ``drop_feature``: the mask is over columns, not
    entries, so a masked attribute vanishes for the whole graph in that view.
    """
    if p <= 0.0:
        return x
    f = x.shape[-1]
    keep = torch.rand(f, generator=generator, device=x.device) >= p
    return x * keep.to(x.dtype)


def grace_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.4
               ) -> torch.Tensor:
    """Symmetric GRACE objective over aligned node embeddings.

    ``z1``/``z2`` are (M, D) projected embeddings of the SAME M nodes seen
    through the two views (cosine similarity is taken inside). Denominator =
    positive + inter-view negatives + intra-view negatives, as in the
    reference ``semi_loss``; returned as the mean over nodes of the two
    directions' average.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    def semi(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        refl = torch.exp(a @ a.t() / tau)             # intra-view (M, M)
        between = torch.exp(a @ b.t() / tau)          # inter-view (M, M)
        pos = between.diag()
        denom = refl.sum(dim=1) - refl.diag() + between.sum(dim=1)
        return -torch.log(pos / denom)

    return (0.5 * (semi(z1, z2) + semi(z2, z1))).mean()


class GRACEProjection(nn.Module):
    """The reference projection head: ``fc1 -> ELU -> fc2`` (discarded after
    pretraining; the probe sees the encoder output, not this)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.elu(self.fc1(z)))


class GRACE(nn.Module):
    """Shared GNN encoder + projection head for static graph contrast."""

    def __init__(self, encoder: nn.Module, latent_dim: int,
                 proj_hidden: Optional[int] = None, tau: float = 0.4,
                 p_e1: float = 0.2, p_e2: float = 0.4,
                 p_f1: float = 0.3, p_f2: float = 0.4):
        super().__init__()
        self.encoder = encoder
        self.proj = GRACEProjection(latent_dim, proj_hidden or latent_dim)
        self.tau = tau
        self.rates = ((p_e1, p_f1), (p_e2, p_f2))

    def embed_view(self, x: torch.Tensor, adj: torch.Tensor,
                   p_e: float, p_f: float,
                   generator: Optional[torch.Generator] = None
                   ) -> torch.Tensor:
        """Augment one static snapshot and encode it -> (N, D) projected.

        ``x`` is (N, F) float, ``adj`` is (N, N) float on the same device.
        """
        x_v = drop_features(x, p_f, generator)
        adj_v = drop_edges(adj, p_e, generator)
        z = self.encoder(x_v[None, None], normalize_adjacency(adj_v))   # (1,1,N,D)
        return self.proj(z[0, 0])

    def two_views(self, x: torch.Tensor, adj: torch.Tensor,
                  generator: Optional[torch.Generator] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        (pe1, pf1), (pe2, pf2) = self.rates
        return (self.embed_view(x, adj, pe1, pf1, generator),
                self.embed_view(x, adj, pe2, pf2, generator))


def snapshot_tensors(adj_t: np.ndarray, feats_t: np.ndarray, device
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """One snapshot's ``(x (N,F) float, adj (N,N) float)`` on ``device``."""
    x = torch.from_numpy(np.ascontiguousarray(feats_t)).float().to(device)
    adj = torch.from_numpy(np.ascontiguousarray(adj_t)).float().to(device)
    return x, adj
