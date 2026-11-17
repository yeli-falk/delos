"""Node encoders for DELOS.

Every encoder maps ``x: (B, T, N, F_in)`` (plus an optional adjacency) to
``z: (B, T, N, D)``. Encoders register themselves in ``ENCODER_REGISTRY`` via
``@register_encoder``; ``build_encoder`` looks the kind up there, so a new
architecture only needs to subclass :class:`EncoderBase` and be decorated.

Registered kinds: ``linear``, ``gcn``, ``gcn_root``, ``tgcn``, ``ppr_gcn``,
``maskdgnn``, ``maskdgnn_root``, ``dysat``, ``dygwm``, ``dygwm_root``,
``dyg2vec``.

The file also holds the ``LinearProbe`` and ``LinkPredictionProbe`` readout
heads and the link-ranking metrics.
"""
from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Pretext heads / samplers live in pretrainers.py; re-exported here so
# `from ...delos_encoder import EdgePretrainer` keeps working.
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
    # ONE full-size (..., N, N) float tensor, updated in place: adding ``eye``
    # and multiplying by ``d_i * d_j`` out of place would materialise three
    # extra N^2 temporaries, which does not fit in GPU memory on large graphs.
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
    """Common interface for all DELOS node encoders.

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


@register_encoder("gcn_root")
class GCNRootNodeEncoder(EncoderBase):
    """2-layer GCN with a ROOT WEIGHT (GraphSAGE-style): each layer is
    ``W_root x_i + A_hat (W_nbr x)_i`` instead of ``A_hat (W x)_i``.

    Why: with the shared normalised adjacency (self-loops included) a node's
    own features enter each layer with weight ~1/(deg+1), so two layers of
    plain GCN return a neighbourhood average. Per-node signals that are
    independent across neighbours (on the synthetic generator the hidden
    velocity, i.e. the part of the dynamics only the trajectory reveals) are
    averaged away. Keeping the node's own features on a separate path
    preserves them: the root path carries the per-node signal, the
    aggregation path the structural context, and the encoder learns the mix
    instead of the degree fixing it.

    Same interface, same hidden / output sizes as ``gcn``; two extra linear
    maps. Selectable everywhere via ``--encoder_kind gcn_root`` (ENC_KIND).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.root1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.root2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = self.root1(x) + _spatial_aggregate(adj_norm, self.lin1(x))
        h = F.gelu(h)
        return self.root2(h) + _spatial_aggregate(adj_norm, self.lin2(h))


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

    SPATIAL = GCNNodeEncoder          # the shared spatial encoder class

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1, freq_bins: int = 33):
        super().__init__()
        self.gcn = self.SPATIAL(in_dim, hidden_dim, out_dim)
        self.freq = AdaptiveFrequencyLayer(out_dim, max_bins=freq_bins,
                                           dropout=dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return self.freq(self.gcn(x, adj_norm))


@register_encoder("maskdgnn_root")
class MaskDGNNRootEncoder(MaskDGNNEncoder):
    """MaskDGNN's frequency layer on the ROOT-WEIGHT GCN (``gcn_root``), so a
    table whose shared spatial encoder is ``gcn_root`` keeps the "same GCN +
    the frequency layer" relation. slurm_all_methods.sh selects it when
    ENC_KIND=gcn_root."""
    SPATIAL = GCNRootNodeEncoder


# ---------------------------------------------------------------------------
# DySAT-style (Sankar et al., WSDM'20) structural + temporal self-attention
# ---------------------------------------------------------------------------
@register_encoder("dysat")
class DySATEncoder(EncoderBase):
    """DySAT-style window encoder: per-snapshot structural block, then masked
    self-attention over each node's sequence of snapshot embeddings.

    DySAT (Sankar, Wu, Gou, Zhang & Yang, WSDM'20, Sec. 3.3) runs structural
    attention on every snapshot and then a temporal self-attention layer per
    node over the T structural embeddings, with learned position embeddings
    added first and a CAUSAL mask (position t attends to t' <= t, eq. 5's
    ``M_ij = 0 if i <= j else -inf``), followed by a position-wise feed-forward
    and residual connections.

    Two deliberate departures, both for the encoder ladder this exists for:

    * The structural block is the pipeline's shared 2-layer
      :class:`GCNNodeEncoder`, not DySAT's GAT -- identical to the ``gcn``,
      ``tgcn`` and ``maskdgnn`` kinds, so the ONLY difference between those
      encoders is how a node's per-snapshot embeddings are mixed along time
      (nothing / GRU / spectral filter / causal self-attention). A dense GAT
      over full-N MOOC would also not fit in memory.
    * Position embeddings index the window position ``t`` (0 = oldest), so the
      same parameters serve any window length up to ``max_len``.

    Output at step ``t`` therefore encodes the prefix ``[0, t]`` of the window
    (like ``dyg2vec``, unlike the bidirectional ``tgcn`` / ``maskdgnn``): the
    last history step is the full history-window encoding, and no future
    step leaks into an earlier one. Shape contract as every encoder:
    ``(B, T, N, F_in) -> (B, T, N, out_dim)``.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 heads: int = 4, n_layers: int = 1, dropout: float = 0.1,
                 max_len: int = 64):
        super().__init__()
        self.gcn = GCNNodeEncoder(in_dim, hidden_dim, out_dim)
        self.pos = nn.Embedding(max_len, out_dim)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(out_dim, heads, dropout=dropout,
                                              batch_first=True),
                "ln1": nn.LayerNorm(out_dim),
                "ffn": nn.Sequential(nn.Linear(out_dim, 2 * out_dim), nn.GELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(2 * out_dim, out_dim)),
                "ln2": nn.LayerNorm(out_dim),
            }))
        self.dropout = nn.Dropout(dropout)
        self.attn_chunk = max(1, 32768 // heads)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = self.gcn(x, adj_norm)                                  # (B, T, N, D)
        B, T, N, D = h.shape
        if T > self.pos.num_embeddings:
            raise ValueError(f"dysat: window of {T} steps exceeds max_len "
                             f"{self.pos.num_embeddings}")
        seq = h.permute(0, 2, 1, 3).reshape(B * N, T, D)          # one sequence per node
        seq = seq + self.pos(torch.arange(T, device=h.device))[None]
        # causal: query t may attend to keys t' <= t (True = masked out)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device),
                            diagonal=1)
        for layer in self.layers:
            # chunk over nodes: the fused CUDA attention kernel fails with
            # "invalid configuration argument" once B*N*heads passes ~2^16
            a = torch.cat([
                layer["attn"](s, s, s, attn_mask=causal, need_weights=False)[0]
                for s in seq.split(self.attn_chunk)], dim=0)
            seq = layer["ln1"](seq + self.dropout(a))
            seq = layer["ln2"](seq + self.dropout(layer["ffn"](seq)))
        return seq.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()


# ---------------------------------------------------------------------------
# DyG-WM (Li et al., 2026, arXiv 2610.19364) context encoder
# ---------------------------------------------------------------------------
def neighbour_mean(adj_norm: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Mean of each node's NEIGHBOURS' features, self excluded; 0 for a node
    with no neighbour (DyG-WM's ``h_bar_u^s``, set to 0 for isolated nodes).

    ``adj_norm`` is any layout :func:`_spatial_aggregate` accepts; only its
    support (``> 0``) is used, so a normalised or a raw adjacency both work.
    """
    a = (adj_norm > 0).to(h.dtype)
    eye = torch.eye(a.shape[-1], dtype=torch.bool, device=a.device)
    a = a.masked_fill(eye, 0.0)
    deg = a.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return _spatial_aggregate(a / deg, h)


@register_encoder("dygwm")
class DyGWMEncoder(EncoderBase):
    """DyG-WM's context encoder (Li et al., 2026, Sec. 3.2) on the shared GCN.

    DyG-WM encodes every historical snapshot with a shared online GNN, adds
    each node's mean neighbourhood representation (0 for isolated nodes), runs
    SEPARATE recurrent encoders over the node trajectory and the neighbourhood
    trajectory, and fuses both with the node's own snapshot embedding through
    a gated residual into the context ``c_u^t``. The per-snapshot GNN alone is
    what the EMA target encoder applies to the target snapshot
    (:meth:`snapshot`); the fused context is what the predictors read and
    what the probe sees.

    As for ``maskdgnn`` / ``dysat`` the spatial block is the pipeline's shared
    2-layer :class:`GCNNodeEncoder` (same parameters and input as every other
    arm), so the ONLY architectural difference from the ``gcn`` arms is the
    two trajectory GRUs + gate -- the method's temporal mixer. Output at step
    ``t`` encodes the prefix ``[0, t]`` (GRUs are causal): the last history
    step is the full-history context and no future step leaks backwards.
    Shape contract as every encoder: ``(B, T, N, F_in) -> (B, T, N, out_dim)``.
    """

    SPATIAL = GCNNodeEncoder

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.gcn = self.SPATIAL(in_dim, hidden_dim, out_dim)
        self.gru_node = nn.GRU(out_dim, out_dim, batch_first=True)
        self.gru_nbr = nn.GRU(out_dim, out_dim, batch_first=True)
        self.gate = nn.Linear(3 * out_dim, out_dim)
        self.fuse = nn.Linear(2 * out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def snapshot(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """``GNN_theta`` on snapshots alone (no temporal mixing): the map the
        EMA target encoder applies to the target snapshot."""
        return self.gcn(x, adj_norm)

    def encode(self, x: torch.Tensor, adj_norm: torch.Tensor):
        """Return ``(h, c)``: per-snapshot GNN embeddings and fused contexts,
        both ``(B, T, N, D)``."""
        h = self.gcn(x, adj_norm)
        h_bar = neighbour_mean(adj_norm, h)
        B, T, N, D = h.shape
        seq = h.permute(0, 2, 1, 3).reshape(B * N, T, D)
        seq_bar = h_bar.permute(0, 2, 1, 3).reshape(B * N, T, D)
        r_node, _ = self.gru_node(seq)
        r_nbr, _ = self.gru_nbr(seq_bar)
        r = torch.cat([r_node, r_nbr], dim=-1)
        g = torch.sigmoid(self.gate(torch.cat([seq, r], dim=-1)))
        c = seq + g * self.dropout(self.fuse(r))
        return h, c.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return self.encode(x, adj_norm)[1]


@register_encoder("dygwm_root")
class DyGWMRootEncoder(DyGWMEncoder):
    """DyG-WM's trajectory GRUs on the ROOT-WEIGHT GCN (``gcn_root``), so a
    table whose shared spatial encoder is ``gcn_root`` keeps the "same GCN +
    the method's temporal mixer" relation (as ``maskdgnn_root``)."""
    SPATIAL = GCNRootNodeEncoder


# ---------------------------------------------------------------------------
# DyG2Vec (Alomrani, Biparva, Zhang & Coates, TMLR'24) window encoder
# ---------------------------------------------------------------------------
class Time2Vec(nn.Module):
    """Learnable cosine time encoding ``phi(dt) = [cos(w_k dt + b_k)]_k``
    (DyG2Vec eq. 6, "a learnable Time2Vec module"). Frequencies are
    initialised geometrically over ``10^0 .. 10^-1.5`` so the coordinates span
    the snapshot-unit gaps a history window can contain."""

    def __init__(self, dim: int):
        super().__init__()
        self.freq = nn.Parameter(torch.logspace(0.0, -1.5, dim))
        self.phase = nn.Parameter(torch.zeros(dim))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.cos(dt.unsqueeze(-1).float() * self.freq + self.phase)


class TemporalAttentionLayer(nn.Module):
    """One DyG2Vec message-passing layer (eq. 2-5) on an interaction list.

    ``h^l_i = sigma( W1 h^{l-1}_i + MHA(q = h^{l-1}_i, K = V = {[h^{l-1}_{u_p}
    || f_p]}_{p in N(i)}) )``. The key/value of interaction ``p`` is the
    concatenation of the neighbour's embedding and the interaction's temporal
    edge encoding ``f_p``; a linear map of a concatenation is the sum of two
    linear maps, so ``K = k_h(h_u) + k_f(f)`` (same for ``V``). Softmax is
    per query over ITS interactions (scatter-softmax on the edge list), so a
    node attends over one entry per interaction, not per neighbour.
    """

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads, self.dh = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.k_h = nn.Linear(dim, dim)
        self.k_f = nn.Linear(dim, dim, bias=False)
        self.v_h = nn.Linear(dim, dim)
        self.v_f = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim)
        self.skip = nn.Linear(dim, dim)                      # W1 (eq. 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, q_idx: torch.Tensor, n_idx: torch.Tensor,
                f: torch.Tensor) -> torch.Tensor:
        """``h`` (Q, D) node states; ``q_idx``/``n_idx`` (E,) query and
        neighbour rows of each interaction; ``f`` (E, D) its edge encoding."""
        Q, D = h.shape
        H, dh = self.heads, self.dh
        q = self.q(h).view(Q, H, dh)
        kh = self.k_h(h).view(Q, H, dh)
        vh = self.v_h(h).view(Q, H, dh)
        agg = torch.zeros(Q, H, dh, device=h.device, dtype=h.dtype)
        if q_idx.numel() > 0:
            kf = self.k_f(f).view(-1, H, dh)
            vf = self.v_f(f).view(-1, H, dh)
            score = (q[q_idx] * (kh[n_idx] + kf)).sum(-1) / (dh ** 0.5)  # (E,H)
            idx = q_idx[:, None].expand(-1, H)
            m = torch.full((Q, H), float("-inf"), device=h.device, dtype=score.dtype)
            m = m.scatter_reduce(0, idx, score, reduce="amax", include_self=True)
            m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
            w = torch.exp(score - m[q_idx])
            denom = torch.zeros(Q, H, device=h.device, dtype=w.dtype).index_add_(0, q_idx, w)
            w = w / (denom[q_idx] + 1e-12)
            agg = agg.index_add(0, q_idx, (vh[n_idx] + vf) * w[..., None])
        out = self.skip(h) + self.o(agg.reshape(Q, D))
        return self.drop(F.relu(out))


@register_encoder("dyg2vec")
class DyG2VecEncoder(EncoderBase):
    """DyG2Vec's window-based temporal attention encoder (Alomrani et al.,
    TMLR'24, Sec. 4.1) on this project's snapshot windows.

    The paper encodes a *history graph* -- the multigraph of the last ``W``
    interactions -- once, with ``L = 3`` layers of temporal attention whose
    keys are the individual interactions ``(u_p, t_p)`` of a node, each
    carrying a Time2Vec encoding of the gap to the node's LATEST interaction
    (``t_bar_i - t_p``, eq. 6-7; so the encoding is independent of any
    prediction target) plus a linear map ``W2`` of three structural counts
    (eq. 8): the two endpoints' degrees and their common-neighbour count at
    ``t_p``. Message passing is "flat" and non-causal within the window, with
    no node memory. Here:

    * the window is the ``T`` snapshots the encoder is handed; every edge
      ``(i, j)`` present in snapshot ``s`` is one interaction at time ``s``,
      and the output at step ``t`` encodes the prefix window ``[0, t]`` (the
      history graph ending at ``t``). The last step is therefore exactly the
      paper's encoding of the probe's history window; earlier steps are the
      same encoder on shorter histories.
    * structural counts are taken on snapshot ``s`` (degree in ``s``,
      common neighbours in ``s``) and ``log1p``-compressed before ``W2``.
      There are no per-edge features in the DTDG loader (they are aggregated
      into the node features), so the ``m_p`` term of eq. 5 is absent -- as
      on the paper's LastFM / UCI / Enron / SocialEvol.
    * ``h^0`` is a linear map of the pipeline's node features, so the input
      matches every other baseline; a final linear map gives ``out_dim``.

    ``struct_keep`` (B, T, N, N) bool, when given, zeroes the structural
    encoding of the interactions it marks False -- the paper's "edge feature
    masking" augmentation, applied to ``z_p`` / ``c_p`` during SSL. Edge
    dropout is applied by the caller to the adjacency itself.

    Defaults are the paper's: 3 layers, 2 heads, dropout 0.1, ReLU.
    """

    LAYERS: int = 3
    HEADS: int = 2

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 layers: int = LAYERS, heads: int = HEADS, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.time_enc = Time2Vec(hidden_dim)
        self.struct = nn.Linear(3, hidden_dim, bias=False)   # W2 (eq. 8)
        self.layers = nn.ModuleList(
            TemporalAttentionLayer(hidden_dim, heads, dropout) for _ in range(layers))
        self.out = nn.Linear(hidden_dim, out_dim)

    # -- interaction list of one window ---------------------------------------
    @staticmethod
    def _edges_of(A: torch.Tensor):
        """(i, j, deg_i, deg_j, cn_ij) of the edges of one (N, N) bool snapshot.
        Common neighbours are gathered in chunks so no N x N float is built."""
        deg = A.sum(dim=1)
        ij = A.nonzero(as_tuple=False)                         # (E, 2)
        i, j = ij[:, 0], ij[:, 1]
        cn = torch.empty(i.shape[0], device=A.device, dtype=deg.dtype)
        chunk = max(1, (1 << 26) // max(A.shape[0], 1))       # ~64M bools per chunk
        for a in range(0, i.shape[0], chunk):
            sl = slice(a, a + chunk)
            cn[sl] = (A[i[sl]] & A[j[sl]]).sum(dim=1)
        return i, j, deg, cn

    def _window_edges(self, adj_b, T: int, N: int, keep_b):
        """Interaction list of one batch item over all output steps.

        Returns ``(q_idx, n_idx, f)``: flat ``t*N + node`` indices of the
        query and the neighbour for every (output step ``t``, snapshot ``s <=
        t``, edge ``(i, j)`` of ``s``) triple, and the edge encoding
        ``f = phi(t_bar_i(t) - s) + W2[...]`` (E, D).
        """
        fixed = adj_b.dim() == 2
        per_s = []
        if fixed:
            per_s = [self._edges_of(adj_b)] * T
        else:
            per_s = [self._edges_of(adj_b[s]) for s in range(T)]
        deg_all = torch.stack([e[2] for e in per_s])           # (T, N)
        # t_bar_i(t): latest snapshot <= t in which i has an interaction.
        s_ar = torch.arange(T, device=adj_b.device)
        act = torch.where(deg_all > 0, s_ar[:, None].expand(T, N),
                          torch.full_like(deg_all, -1, dtype=torch.long))
        last_active = torch.cummax(act, dim=0).values           # (T, N)
        qs, ns, dts, feats, keeps = [], [], [], [], []
        for t in range(T):
            for s in range(t + 1):
                i, j, deg, cn = per_s[s]
                if i.numel() == 0:
                    continue
                qs.append(t * N + i); ns.append(t * N + j)
                dts.append(last_active[t, i] - s)
                feats.append(torch.stack([deg[i], deg[j], cn], dim=1))
                if keep_b is not None:
                    keeps.append(keep_b[s][i, j] if not fixed else keep_b[i, j])
        if not qs:
            D = self.hidden_dim
            e = torch.zeros(0, dtype=torch.long, device=adj_b.device)
            return e, e, torch.zeros(0, D, device=adj_b.device)
        q_idx, n_idx = torch.cat(qs), torch.cat(ns)
        dt = torch.cat(dts).clamp(min=0).float()
        st = torch.log1p(torch.cat(feats).float())
        if keeps:
            st = st * torch.cat(keeps).float()[:, None]
        f = self.time_enc(dt) + self.struct(st)
        return q_idx, n_idx, f

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor,
                struct_keep: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, N, F)  adj_norm: (N,N) | (T,N,N) | (B,N,N) | (B,T,N,N)
        B, T, N, _ = x.shape
        A = adj_norm > 0
        if A.dim() >= 2:
            A = A & ~torch.eye(N, dtype=torch.bool, device=A.device)  # drop self-loops
        outs = []
        for b in range(B):
            if A.dim() == 4:
                adj_b = A[b]
            elif A.dim() == 3:
                adj_b = A if A.shape[0] == T else A[b]        # (T,N,N) shared vs (B,N,N)
            else:
                adj_b = A
            keep_b = None
            if struct_keep is not None:
                keep_b = struct_keep[b] if struct_keep.dim() == 4 else struct_keep
            q_idx, n_idx, f = self._window_edges(adj_b, T, N, keep_b)
            h = self.inp(x[b]).reshape(T * N, self.hidden_dim)
            for layer in self.layers:
                h = layer(h, q_idx, n_idx, f)
            outs.append(self.out(h).view(T, N, -1))
        return torch.stack(outs)                                 # (B, T, N, out)


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
    # NaN guard: torch sorts NaN as the LARGEST value, so a NaN positive (from
    # a diverged encoder) would rank FIRST and score MRR = 1. Treat NaN as
    # -inf and give a NaN positive the worst rank.
    all_scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
    pos_nan = torch.isnan(pos_scores)
    if pos_nan.any():
        print(f"[warn] link_ranking_metrics: {int(pos_nan.sum())}/{pos_nan.numel()} "
              "positive scores are NaN -> ranked last", flush=True)
    all_scores = torch.nan_to_num(all_scores, nan=float("-inf"))
    ranks = (all_scores.argsort(dim=-1, descending=True) == 0).nonzero()[:, 1] + 1
    ranks = torch.where(pos_nan, torch.full_like(ranks, all_scores.shape[-1]), ranks)
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
