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

Encoders register themselves in ``ENCODER_REGISTRY`` via ``@register_encoder``,
so adding a new architecture (GAT, GraphSAGE, a temporal transformer) is a
one-file drop-in: subclass :class:`EncoderBase`, decorate it, done — no edit
to ``build_encoder`` needed.
"""
from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Pretext heads / samplers used to live here; re-export for backward compat
# so `from ...graph_jepa_encoder import EdgePretrainer` keeps working.
from src.world_models.pretrainers import (  # noqa: F401
    EdgePretrainer,
    EncoderPretrainer,
    FeatureMaskPretrainer,
    sample_feature_mask,
    sample_masked_edges,
)


def normalize_adjacency(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """Symmetric-normalised adjacency: D^{-1/2} (A + I) D^{-1/2}.

    Accepts a (..., N, N) tensor in float; returns the same shape. Works for
    a single static graph (N, N), per-timestep (T, N, N), per-batch (B, N, N),
    or per-(batch,time) (B, T, N, N).
    """
    # ONE full-size (..., N, N) float tensor, updated in place. The previous
    # form (``adj + eye`` then ``adj * d_i * d_j``) materialised three extra
    # N^2 temporaries; at full-N MOOC (N=7144, B=8, T=4) each is 6 GB, which
    # is what OOMed the JEPA arm on a 32 GB V100. Values are unchanged: the
    # diagonal add equals ``+ eye`` and the two in-place multiplies apply the
    # same factors in the same order.
    out = adj.float() if adj.dtype != torch.float32 else adj.clone()
    if add_self_loops:
        out.diagonal(dim1=-2, dim2=-1).add_(1.0)
    deg = out.sum(dim=-1).clamp(min=1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    out.mul_(d_inv_sqrt.unsqueeze(-1)).mul_(d_inv_sqrt.unsqueeze(-2))
    return out


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


# ---------------------------------------------------------------------------
# Encoder interface + registry
# ---------------------------------------------------------------------------
class EncoderBase(nn.Module):
    """Common interface for all DyG-JEPA node encoders.

    Contract: ``forward(x, adj_norm) -> z`` with ``x: (B, T, N, F_in)`` and
    ``z: (B, T, N, out_dim)``. ``adj_norm`` is a (pre-normalised) adjacency in
    any shape accepted by :func:`_spatial_aggregate`, or ``None`` for encoders
    that ignore graph structure.
    """

    def forward(self, x: torch.Tensor,
                adj_norm: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


# name -> builder(in_dim, hidden_dim, out_dim, bidirectional) -> EncoderBase
ENCODER_REGISTRY: Dict[str, Callable[..., EncoderBase]] = {}


def register_encoder(name: str):
    """Class decorator registering an encoder builder under ``name``.

    The decorated class must accept ``(in_dim, hidden_dim, out_dim)`` and may
    accept ``bidirectional``; the registry adapter forwards it only when the
    constructor takes it.
    """
    def deco(cls):
        import inspect
        takes_bidir = "bidirectional" in inspect.signature(cls.__init__).parameters

        def builder(in_dim, hidden_dim, out_dim, bidirectional=True):
            if takes_bidir:
                return cls(in_dim, hidden_dim, out_dim, bidirectional=bidirectional)
            return cls(in_dim, hidden_dim, out_dim)

        ENCODER_REGISTRY[name] = builder
        cls.registry_name = name
        return cls
    return deco


@register_encoder("linear")
class LinearNodeEncoder(EncoderBase):
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


@register_encoder("gcn")
class GCNNodeEncoder(EncoderBase):
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


@register_encoder("tgcn")
class TGCNEncoder(EncoderBase):
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


@register_encoder("ppr_gcn")
class PPRGCNEncoder(EncoderBase):
    """APPNP-style encoder over personalised-PageRank-propagated features.

    This is the dense, framework-consistent analogue of the dynamic PPR
    propagation in InstantGNN (Zheng et al.) used by the IDOL baseline
    (``src/baselines/idol``, "Topology-monitorable Contrastive Learning on
    Dynamic Graphs", KDD'24). InstantGNN incrementally maintains the
    personalised-PageRank matrix as the graph evolves; on this project's dense
    per-snapshot adjacency the *converged* result is exactly the APPNP/PPNP
    propagation

        H0 = MLP(X);   H^{k+1} = (1 - alpha) * A_hat @ H^k + alpha * H0,

    i.e. ``K`` power-iteration steps toward ``alpha (I - (1-alpha) A_hat)^{-1} H0``.
    The teleport ``alpha`` and step count ``K`` are fixed (not learnable) so the
    propagation is identical at pretrain and probe time — the only learnable
    parameters live in the feature MLP, matching GGD's GCN/MLP head.

    Output shape matches the other encoders: (B, T, N, out_dim).
    """

    PPR_ALPHA: float = 0.1
    PPR_ITERS: int = 10

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.alpha = self.PPR_ALPHA
        self.iters = self.PPR_ITERS
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F_in)  adj_norm: any shape accepted by _spatial_aggregate.
        h0 = self.lin2(F.gelu(self.lin1(x)))                    # (B, T, N, out)
        h = h0
        for _ in range(self.iters):
            h = (1.0 - self.alpha) * _spatial_aggregate(adj_norm, h) + self.alpha * h0
        return h


class AdaptiveFrequencyLayer(nn.Module):
    """Adaptive frequency enhancing layer (MaskDGNN, He et al., IJCAI'25;
    after FreeDyG, Tian et al. 2023).

    Along the TIME axis of a ``(B, T, N, D)`` latent sequence:

        H_f  = rFFT_T(H)                    (B, T//2+1, N, D) complex
        H_f' = W (.) H_f                    W in C^{(T//2+1) x D}, learnable
        H'   = irFFT_T(H_f')
        H   <- H + Dropout(H')              (paper eq. 11-14)

    The paper's filter is sized to one fixed window length. Here the filter is
    stored on a fixed grid of ``max_bins`` relative frequencies and linearly
    resampled to however many rFFT bins the input actually has, so the SAME
    parameters serve a pretraining window of ``w`` snapshots and a probe window
    of ``history_size`` snapshots (bin ``k`` of a ``T``-long window reads the
    filter at relative frequency ``k / (T//2)``). ``T = 1`` degenerates to a
    single (DC) bin -- a per-channel gain -- which is the right limit.

    ``W`` is initialised ~N(0, 0.02) (FreeDyG's init), so at step 0 the branch
    is near zero and the layer starts as (almost) the identity via the residual.
    """

    def __init__(self, dim: int, max_bins: int = 33, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.max_bins = max_bins
        # (bins, dim, 2) real/imag; view_as_complex needs the last dim == 2.
        self.weight = nn.Parameter(torch.randn(max_bins, dim, 2) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def _filter(self, n_bins: int) -> torch.Tensor:
        """Resample the (max_bins, D) filter to ``n_bins`` -> complex (n_bins, D)."""
        w = self.weight
        if n_bins == 1:
            w = w[:1]
        elif n_bins != self.max_bins:
            # Deterministic linear resampling on the relative-frequency axis
            # (F.interpolate's CUDA backward is non-deterministic; this is a
            # gather, so set_deterministic stays honoured).
            pos = torch.linspace(0.0, self.max_bins - 1, n_bins, device=w.device)
            lo = pos.floor().long().clamp(max=self.max_bins - 1)
            hi = (lo + 1).clamp(max=self.max_bins - 1)
            frac = (pos - lo.float()).view(-1, 1, 1)
            w = w[lo] * (1.0 - frac) + w[hi] * frac
        return torch.view_as_complex(w.contiguous())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, N, D)
        T = h.shape[1]
        h_f = torch.fft.rfft(h.float(), dim=1)                    # (B, T//2+1, N, D)
        w = self._filter(h_f.shape[1])                             # (T//2+1, D)
        h_f = h_f * w[None, :, None, :]
        h_hat = torch.fft.irfft(h_f, n=T, dim=1).to(h.dtype)       # (B, T, N, D)
        return h + self.dropout(h_hat)


@register_encoder("maskdgnn")
class MaskDGNNEncoder(EncoderBase):
    """MaskDGNN's graph representation learner (He et al., IJCAI'25):
    a per-snapshot GCN followed by the adaptive frequency enhancing layer.

    The paper's encoder is an ``l``-layer *decoupled* GCN (eq. 10, ``H <- A_hat
    H W``, no nonlinearity) with the frequency layer on its output. Here the
    spatial part is the pipeline's shared 2-layer :class:`GCNNodeEncoder`
    (identical parameters and input to every other baseline), so the ONLY
    architectural difference from the ``gcn`` arms is the frequency layer --
    which is what makes the comparison a comparison of objectives plus that
    one component, not of two unrelated networks.

    Output shape matches the other encoders: (B, T, N, out_dim). The frequency
    layer mixes along T, so unlike ``gcn`` this encoder is NOT per-snapshot:
    with a history window of length ``T`` every output step sees the whole
    window (both directions, like the bidirectional TGCN).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1, freq_bins: int = 33):
        super().__init__()
        self.gcn = GCNNodeEncoder(in_dim, hidden_dim, out_dim)
        self.freq = AdaptiveFrequencyLayer(out_dim, max_bins=freq_bins,
                                           dropout=dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return self.freq(self.gcn(x, adj_norm))


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
    # Per-source AUC: mean over (pos, K negatives) of P(pos > neg).
    # Ties contribute 0.5 (standard ROC convention).
    pos_exp = pos_scores.unsqueeze(-1)
    gt = (pos_exp > neg_scores).float() + 0.5 * (pos_exp == neg_scores).float()
    out["AUC"] = gt.mean().item()
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


def build_encoder(kind: str, in_dim: int, hidden_dim: int, out_dim: int,
                  freeze: bool = True, bidirectional: bool = True) -> nn.Module:
    """Build an encoder. `bidirectional` applies only to TGCN; ignored
    otherwise. Setting `bidirectional=False` gives a temporally causal
    encoder with no future-leak -- required for the formal counterfactual
    claim (assumption A4 in the theory) and the variant where the JEPA
    predictor's job is genuinely to forecast unseen future.

    Encoders are looked up in :data:`ENCODER_REGISTRY`; register a new one
    with ``@register_encoder("name")`` and it becomes selectable here and via
    ``--encoder_kind`` with no edit to this function."""
    if kind not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder kind '{kind}'. Registered: "
            f"{sorted(ENCODER_REGISTRY)}."
        )
    enc = ENCODER_REGISTRY[kind](in_dim, hidden_dim, out_dim,
                                 bidirectional=bidirectional)
    if freeze:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
    return enc
