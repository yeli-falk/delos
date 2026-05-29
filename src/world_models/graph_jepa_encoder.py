"""Frozen node encoders for graph C-JEPA.

Fixed encoder so its outputs can serve as
prediction targets without the predictor finding shortcuts in encoder space.

  - LinearNodeEncoder: per-node, per-time MLP. Ignores graph structure.
    Cheap baseline; lets you confirm the predictor is wiring history -> future.
  - GCNNodeEncoder:    2-hop normalised GCN over the fixed adjacency. Adds
    spatial mixing and is the closer analogue of "object-centric encoder".

Both modules expose the same forward signature:
    encoder(x, adj=None) -> z  with x: (B, T, N, F_in) and z: (B, T, N, D)
and are constructed with `requires_grad_(False)` so callers can freeze.

A small `LinearProbe` decoder is also provided for the standard "predict
embeddings, regress speeds via a linear map fit on train" eval protocol.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """Symmetric-normalised adjacency: D^{-1/2} (A + I) D^{-1/2}.

    Accepts a (..., N, N) tensor in float; returns the same shape. Works for
    a single static graph (N, N), per-timestep (T, N, N), per-batch (B, N, N),
    or per-(batch,time) (B, T, N, N).
    """
    if add_self_loops:
        N = adj.shape[-1]
        eye = torch.eye(N, device=adj.device, dtype=adj.dtype)
        adj = adj + eye
    deg = adj.sum(dim=-1).clamp(min=1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    return adj * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)


def _spatial_aggregate(adj_norm: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Aggregate node features over neighbours, using whatever adj shape.

    h is always (B, T, N, F). adj_norm is one of:
      (N, N)        - static graph, broadcast across batch and time.
      (B, N, N)     - per-sample static graph (rare).
      (T, N, N)     - dynamic graph shared across batch (DTDG).
      (B, T, N, N)  - per-(batch, timestep) graph.
    """
    if adj_norm.dim() == 2:
        return torch.einsum("ij,btjf->btif", adj_norm, h)
    if adj_norm.dim() == 3:
        if adj_norm.shape[0] == h.shape[1]:   # (T, N, N)
            return torch.einsum("tij,btjf->btif", adj_norm, h)
        return torch.einsum("bij,btjf->btif", adj_norm, h)   # (B, N, N)
    if adj_norm.dim() == 4:
        return torch.einsum("btij,btjf->btif", adj_norm, h)
    raise ValueError(f"adj_norm must have 2-4 dims, got shape {tuple(adj_norm.shape)}")


class LinearNodeEncoder(nn.Module):
    """Per-node MLP encoder. Ignores adjacency."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(x)


class GCNNodeEncoder(nn.Module):
    """2-layer GCN with a fixed (B-broadcast) adjacency."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F_in)  adj_norm: any of {(N,N), (B,N,N), (T,N,N), (B,T,N,N)}
        h = self.lin1(x)
        h = _spatial_aggregate(adj_norm, h)
        h = F.gelu(h)
        h = self.lin2(h)
        h = _spatial_aggregate(adj_norm, h)
        return h


class TGCNEncoder(nn.Module):
    """Temporal Graph Convolutional Network (Zhao et al., 2019).

    A clean implementation of T-GCN for use as a frozen encoder:

      spatial mix (GCN-style aggregation) -> temporal mix (bidirectional GRU)
                                          -> linear projection to out_dim

    Spatial mixing is applied per timestep, then a bidirectional GRU is run
    per-node along the time axis. The bidirectional GRU lets each (t, n)
    representation see both past and future context, which is the right
    inductive bias for masked-reconstruction pretraining (the model needs to
    infer hidden positions from surrounding context, not just history).

    Output shape matches the other encoders: (B, T, N, out_dim).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 bidirectional: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.spatial_lin = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        gru_out = hidden_dim * (2 if bidirectional else 1)
        self.out_lin = nn.Linear(gru_out, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F_in)  adj_norm: (N, N) or (B, N, N)
        B, T, N, _ = x.shape

        # 1) Per-timestep spatial mixing.
        h = self.spatial_lin(x)                                 # (B, T, N, hidden)
        h = _spatial_aggregate(adj_norm, h)
        h = F.gelu(h)

        # 2) Per-node temporal mixing via (bi-)GRU.
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        h, _ = self.gru(h)                                      # (B*N, T, hidden*num_dir)
        h = h.reshape(B, N, T, -1).permute(0, 2, 1, 3)          # (B, T, N, hidden*num_dir)

        # 3) Output projection.
        return self.out_lin(h)


class EncoderPretrainer(nn.Module):
    """Encoder + small reconstruction decoder for GraphMAE-style pretraining.

    Random-masks a fraction of `(t, n)` speed values, replaces them with a
    learned mask token, runs the (un-frozen) encoder, and asks a 2-layer MLP
    decoder to reconstruct the masked speeds from the encoder's latents.

    Once trained, you save `encoder.state_dict()` and load it into the frozen
    encoder slot of the C-JEPA training loop. The decoder + mask token are
    discarded -- they were only there to drive the encoder.
    """

    def __init__(self, encoder: nn.Module, latent_dim: int,
                 decoder_hidden: int = 64, speed_channel: int = 0):
        super().__init__()
        self.encoder = encoder
        self.speed_channel = speed_channel
        # Only the speed channel is masked; we replace it with a learned
        # scalar so the encoder has a stable "this is hidden" signal.
        # Time-of-day features stay visible -- in deployment you always know
        # the clock, so blanking it during pretraining just makes the task
        # artificially harder without any matching real-world payoff.
        self.mask_token = nn.Parameter(torch.zeros(1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, 1),
        )

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor | None,
                mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F)  mask: (B, T, N) bool — True = hidden, train target
        speed = x[..., self.speed_channel]
        masked_speed = torch.where(mask, self.mask_token.expand_as(speed), speed)
        x_in = x.clone()
        x_in[..., self.speed_channel] = masked_speed
        z = self.encoder(x_in, adj_norm)             # (B, T, N, D)
        return self.decoder(z).squeeze(-1)           # (B, T, N) predicted speed_norm


class LinkPredictionProbe(nn.Module):
    """Pairwise readout: maps two node latents -> edge logit.

    Standard formulation: concat([z_i, z_j, z_i*z_j]) -> MLP -> scalar logit.
    The Hadamard term is what makes BiLinear-style ranking metrics meaningful
    (Hits@k, MRR) even with a small probe.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(3 * in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        # z_i, z_j: (..., D)
        feat = torch.cat([z_i, z_j, z_i * z_j], dim=-1)
        return self.head(feat).squeeze(-1)


@torch.no_grad()
def link_ranking_metrics(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    k_values=(1, 3, 10),
) -> dict:
    """MRR + Hits@k for one batch of positive edges vs many negatives each.

    pos_scores: (P,)         logit for each positive edge
    neg_scores: (P, K_neg)   logits for K_neg negatives sharing the source
    """
    # Rank of the positive among (positive + negatives).
    all_scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
    ranks = (all_scores.argsort(dim=-1, descending=True) == 0).nonzero()[:, 1] + 1
    out = {"MRR": (1.0 / ranks.float()).mean().item()}
    for k in k_values:
        out[f"Hits@{k}"] = (ranks <= k).float().mean().item()
    return out


class LinearProbe(nn.Module):
    """Small decoder for the eval protocol: latent slot -> scalar speed."""

    def __init__(self, in_dim: int, hidden_dim: int = 0):
        super().__init__()
        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.head = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z).squeeze(-1)


def build_encoder(kind: str, in_dim: int, hidden_dim: int, out_dim: int, freeze: bool = True) -> nn.Module:
    if kind == "linear":
        enc = LinearNodeEncoder(in_dim, hidden_dim, out_dim)
    elif kind == "gcn":
        enc = GCNNodeEncoder(in_dim, hidden_dim, out_dim)
    elif kind == "tgcn":
        enc = TGCNEncoder(in_dim, hidden_dim, out_dim)
    else:
        raise ValueError(
            f"Unknown encoder kind '{kind}'. Expected one of: linear, gcn, tgcn."
        )
    if freeze:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
    return enc
