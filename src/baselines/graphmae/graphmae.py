"""GraphMAE: Self-Supervised Masked Graph Autoencoders (Hou et al., KDD'22) —
STATIC generative self-supervised baseline.

GraphMAE is a single-graph, single-time method, like GRACE (``grace.py``); it
is the *generative* half of the static-SSL control block (GRACE is the
contrastive half). Both answer "does the self-supervised objective need to be
temporal at all?" on the same GCN encoder, the same node features and the
same frozen last-history-snapshot probe -- only the objective differs.

The method, per Hou et al. (arXiv:2205.10803):

  1. sample a fraction ``mask_rate`` of the nodes; replace their input
     features with a learnable ``[MASK]`` token (a ``replace_rate`` share of
     them instead get another random node's features, the paper's
     "random substitution" that keeps the encoder from over-fitting the
     token),
  2. encode the corrupted graph with a GNN, map through a linear
     encoder-to-decoder layer, then RE-MASK: the masked nodes' latents are
     zeroed before decoding, so the decoder must rebuild them from
     neighbours (the paper's key difference from a plain feature AE),
  3. a single-layer GNN decoder reconstructs the input features,
  4. scaled cosine error on the masked nodes only::

        L = mean_{i in masked} ( 1 - <x_i, x_hat_i> / (|x_i| |x_hat_i|) )^gamma

     with ``gamma >= 1`` down-weighting the easy (already well-reconstructed)
     nodes.

Faithful to the reference (https://github.com/THUDM/GraphMAE) in method:
learnable zero-initialised mask token, random substitution, linear
encoder-to-decoder without bias, re-masking to zero, one GNN decoder layer,
SCE loss, Adam with weight decay 2e-4 and mask_rate 0.5. Reimplemented on
this project's dense-adjacency snapshots so it needs no DGL; the encoder is
the shared GCN (the paper uses GAT, but the comparison here pins the encoder
so the row difference is the objective) and the decoder is a matching dense
GCN layer.

How this differs from ``pretrain_encoder.py --pretext feature_mask`` (the
repo's own masked-feature pretext): that one masks only the semantic channels
(past ``--graph_feat_dim``), uses no mask token, does not re-mask, decodes
with an MLP and minimises MSE. GraphMAE masks whole node vectors, re-masks,
decodes with a GNN and uses SCE -- the design choices the paper attributes
its gains to. Keep both: the pretext is the JEPA encoder's pretraining, this
is a published baseline.

The *dynamic-graph protocol* is the same as GRACE's: every training snapshot
is an independent static graph; a step draws one uniformly and applies
GraphMAE to it alone, so no temporal structure can reach the encoder through
the objective.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.world_models.delos_encoder import _spatial_aggregate, normalize_adjacency


def sce_loss(x: torch.Tensor, y: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Scaled cosine error, mean over rows: ``(1 - cos(x, y))^gamma``."""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return (1.0 - (x * y).sum(dim=-1)).pow(gamma).mean()


def sample_mask(active: torch.Tensor, mask_rate: float, replace_rate: float,
                generator: Optional[torch.Generator] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the masked nodes of one snapshot.

    Returns ``(masked, token_nodes, replaced)``: boolean (N,) masks. ``masked``
    is a uniform ``mask_rate`` fraction of the ACTIVE nodes (an inactive
    node's all-zero row is not a reconstruction target); ``replaced`` is the
    ``replace_rate`` share of them that receive a random active node's
    features instead of the token, ``token_nodes`` the rest.
    """
    device = active.device
    idx = torch.nonzero(active, as_tuple=False).flatten()
    n_mask = int(mask_rate * idx.numel())
    masked = torch.zeros_like(active)
    token_nodes = torch.zeros_like(active)
    replaced = torch.zeros_like(active)
    if n_mask == 0:
        return masked, token_nodes, replaced
    perm = torch.randperm(idx.numel(), generator=generator, device=device)
    chosen = idx[perm[:n_mask]]
    masked[chosen] = True
    n_rep = int(replace_rate * n_mask)
    replaced[chosen[:n_rep]] = True
    token_nodes[chosen[n_rep:]] = True
    return masked, token_nodes, replaced


class DenseGCNDecoder(nn.Module):
    """One dense GCN layer ``A_norm (H W)`` mapping latents back to features.

    The reference decodes with a single GNN layer of the encoder's type; the
    shared encoder here is a GCN, so this is its decoder counterpart.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # h: (N, D) -> (N, F); go through (1,1,N,*) for _spatial_aggregate.
        return _spatial_aggregate(adj_norm, self.lin(h)[None, None])[0, 0]


class GraphMAE(nn.Module):
    """Shared GNN encoder + mask token + linear bridge + GNN decoder."""

    def __init__(self, encoder: nn.Module, in_dim: int, latent_dim: int,
                 mask_rate: float = 0.5, replace_rate: float = 0.05,
                 gamma: float = 2.0):
        super().__init__()
        self.encoder = encoder
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.enc_to_dec = nn.Linear(latent_dim, latent_dim, bias=False)
        self.decoder = DenseGCNDecoder(latent_dim, in_dim)
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.gamma = gamma

    def corrupt(self, x: torch.Tensor, active: torch.Tensor,
                generator: Optional[torch.Generator] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_corrupted, masked)`` for one snapshot's (N, F) features."""
        masked, token_nodes, replaced = sample_mask(
            active, self.mask_rate, self.replace_rate, generator)
        x_c = x.clone()
        if replaced.any():
            src = torch.nonzero(active, as_tuple=False).flatten()
            pick = torch.randint(src.numel(), (int(replaced.sum()),),
                                 generator=generator, device=x.device)
            x_c[replaced] = x[src[pick]]
        x_c[token_nodes] = 0.0
        x_c[token_nodes] = x_c[token_nodes] + self.mask_token
        return x_c, masked

    def forward(self, x: torch.Tensor, adj: torch.Tensor, active: torch.Tensor,
                generator: Optional[torch.Generator] = None,
                target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One GraphMAE step on a static snapshot -> SCE loss.

        ``x`` (N, F) float, ``adj`` (N, N) float, ``active`` (N,) bool.
        ``target`` (N, F') optionally restricts the reconstruction target to
        a channel subset (the decoder output is sliced to match).
        """
        adj_norm = normalize_adjacency(adj)
        x_c, masked = self.corrupt(x, active, generator)
        if not masked.any():
            return x.new_zeros(())
        z = self.encoder(x_c[None, None], adj_norm)[0, 0]          # (N, D)
        rep = self.enc_to_dec(z)
        rep = rep.masked_fill(masked[:, None], 0.0)                 # re-mask
        recon = self.decoder(rep, adj_norm)                         # (N, F)
        tgt = x if target is None else target
        if recon.shape[-1] != tgt.shape[-1]:
            recon = recon[..., -tgt.shape[-1]:]
        return sce_loss(recon[masked], tgt[masked], self.gamma)
