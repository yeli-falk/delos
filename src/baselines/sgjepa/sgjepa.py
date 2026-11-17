"""SG-JEPA: Scalable and Efficient Joint Spiking Embedding Predictive
Architecture for Large-Scale Dynamic Graphs (Zhang, Zhu, Zhong, Chen & Zheng,
2026; arXiv 2607.18412) -- the self-supervised baseline.

Implemented from the paper; no reference code was found. The timeline is
split into windows of ``w`` snapshots: the LAST snapshot of a window is the
target, the earlier ``w - 1`` are the context (Sec. 4.1).

  * **Target encoder** -- a full-precision GNN on the target snapshot under
    stop-gradient ("the choice of the graph neural network is not
    restricted"); here the pipeline's shared encoder (``--encoder_kind gcn``),
    which keeps the probe's input identical to every other arm. The paper
    feeds features || random-walk positional encoding || sinusoidal time
    encoding to the GNN; those extra input channels are NOT added, because the
    probe rebuilds the encoder on the pipeline's features (shared-backbone
    rule). The time encoding enters the predictor instead (Eq. 13 adds it
    there as well).
  * **Spiking context encoder** -- the same GNN's outputs on the context
    snapshots ``z_{v,1..w-1}`` are driven, in snapshot order, through
    ``--snn_layers`` spiking layers (synaptic ``Linear`` + Parametric LIF
    neuron, Fang et al. 2021: learnable leak ``sigmoid(beta)``, threshold
    ``v_th``, hard reset, arctan surrogate gradient). Spike COUNTS are prefix
    sums over snapshots (``c_{v,t} = sum_{t' <= t} s_{v,t'}``, the
    "coarse-to-fine" reading of Eq. 11; the printed summation range is
    ambiguous).
  * **Predictor** (Eq. 12-14) -- a projection ``W`` of the counts, plus the
    sinusoidal encoding of the gap to the target snapshot and a shared
    learnable token, pooled over the prefix with learnable softmax weights and
    passed through a shared MLP. "Inspired by Matryoshka representation
    learning": a prediction is formed from EVERY prefix length and the loss
    averages over them (``--no_matryoshka`` keeps only the full prefix), which
    is what lets inference use fewer than ``w - 1`` context snapshots
    (the paper's "adaptive computation").
  * **Loss** (Eq. 15-16) -- per-node InfoNCE between the prediction and the
    stop-gradient target, the other target nodes of the window as negatives;
    cosine similarity at temperature ``tau`` (the paper names neither; flags).
    Gradients reach the GNN through the context path only.

The paper reports node classification with "a lightweight MLP as the
task-specific classification head"; whether the encoder is frozen is not
stated. Here the encoder is frozen and probed exactly like every other arm.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.world_models.delos_encoder import normalize_adjacency


class _ATanSpike(torch.autograd.Function):
    """Heaviside forward, arctan surrogate backward (spikingjelly's default,
    ``alpha = 2``): ``d/dx ~ alpha/2 / (1 + (pi/2 * alpha * x)^2)``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        sg = (a / 2.0) / (1.0 + (math.pi / 2.0 * a * x) ** 2)
        return grad * sg, None


def spike(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return _ATanSpike.apply(x, alpha)


class PLIF(nn.Module):
    """Parametric leaky integrate-and-fire neuron (Fang et al., ICCV'21).

    ``v_t = v_{t-1} + sigmoid(beta) * (I_t - v_{t-1})``, spike when
    ``v_t >= v_th``, hard reset to 0. ``beta`` is learnable, initialised so
    that ``1 / tau = 1 / init_tau`` (``init_tau = 2`` -> ``beta = 0``).
    Operates along dim 1 of ``(n, T, D)`` currents and returns spikes in
    ``{0, 1}`` of the same shape.
    """

    def __init__(self, init_tau: float = 2.0, v_th: float = 1.0, alpha: float = 2.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(-math.log(init_tau - 1.0)))
        self.v_th = v_th
        self.alpha = alpha

    def forward(self, currents: torch.Tensor) -> torch.Tensor:
        leak = torch.sigmoid(self.beta)
        v = torch.zeros_like(currents[:, 0])
        out = []
        for t in range(currents.shape[1]):
            v = v + leak * (currents[:, t] - v)
            s = spike(v - self.v_th, self.alpha)
            v = v * (1.0 - s)
            out.append(s)
        return torch.stack(out, dim=1)


def sinusoidal_time_encoding(gaps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding of snapshot gaps (Eq. 8), ``(L,) -> (L, dim)``."""
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=gaps.device) / max(half, 1))
    ang = gaps.float()[:, None] * freq[None]
    enc = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if enc.shape[-1] < dim:                                   # odd dim
        enc = F.pad(enc, (0, dim - enc.shape[-1]))
    return enc


def info_nce(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """Eq. 15: ``-log exp(sim(p_v, z_v)/tau) / sum_u exp(sim(p_v, z_u)/tau)``
    over the window's target nodes, cosine similarity."""
    if pred.shape[0] < 2:
        return pred.sum() * 0.0
    logits = F.normalize(pred, dim=-1) @ F.normalize(target, dim=-1).T / tau
    labels = torch.arange(pred.shape[0], device=pred.device)
    return F.cross_entropy(logits, labels)


class SGJEPA(nn.Module):
    """Shared GNN + spiking context encoder + Matryoshka prefix predictor,
    trained with per-node InfoNCE against the stop-gradient target."""

    def __init__(self, encoder: nn.Module, latent_dim: int, snn_layers: int = 1,
                 tau: float = 0.1, init_tau: float = 2.0, v_th: float = 1.0,
                 matryoshka: bool = True, current_norm: bool = True,
                 max_len: int = 64):
        super().__init__()
        D = latent_dim
        self.encoder = encoder
        self.tau = tau
        self.matryoshka = matryoshka
        layers = []
        for _ in range(max(1, snn_layers)):
            layers.append(nn.Linear(D, D))
            if current_norm:
                # Threshold-scale normalisation of the synaptic current
                # (tdBN-style, Zheng et al. 2021). The shared GCN's latents sit
                # far below a unit threshold, so without it no neuron fires at
                # initialisation and the InfoNCE stays at log(n). The paper
                # does not state its input normalisation; --no_current_norm
                # removes this one.
                layers.append(nn.LayerNorm(D))
            layers.append(PLIF(init_tau, v_th))
        self.snn = nn.ModuleList(layers)
        self.proj = nn.Linear(D, D, bias=False)                  # W on spike counts
        self.token = nn.Parameter(torch.zeros(D))                # shared learnable token
        self.pool_logits = nn.Parameter(torch.zeros(max_len))    # weighted pooling
        self.mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))

    def spikes(self, z_ctx: torch.Tensor) -> torch.Tensor:
        """``(n, L, D)`` context latents -> ``(n, L, D)`` spikes of the last layer."""
        s = z_ctx
        for layer in self.snn:
            s = layer(s)
        return s

    def predict(self, z_ctx: torch.Tensor) -> torch.Tensor:
        """``(n, L, D)`` context latents -> ``(n, L, D)`` predictions, one per
        prefix length ``t = 1..L`` (the last is the full-context prediction)."""
        n, L, D = z_ctx.shape
        if L > self.pool_logits.numel():
            raise ValueError(f"sgjepa: context of {L} snapshots exceeds max_len "
                             f"{self.pool_logits.numel()}")
        counts = torch.cumsum(self.spikes(z_ctx), dim=1)         # c_{v,t}
        gaps = torch.arange(L, 0, -1, device=z_ctx.device)       # snapshots to target
        tok = self.proj(counts) + sinusoidal_time_encoding(gaps, D)[None] + self.token
        w = self.pool_logits[:L]
        preds = []
        for t in range(L):
            a = torch.softmax(w[:t + 1], dim=0)
            preds.append(self.mlp((tok[:, :t + 1] * a[None, :, None]).sum(dim=1)))
        return torch.stack(preds, dim=1)

    def window_loss(self, feats_w: np.ndarray, adj_w: np.ndarray,
                    active_w: np.ndarray, device,
                    generator: Optional[torch.Generator] = None,
                    node_batch: int = 256) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One window: ``feats_w`` (w, N, F), ``adj_w`` (w, N, N) bool,
        ``active_w`` (w, N). Targets are ``node_batch`` nodes active in the
        last snapshot; the earlier ``w - 1`` snapshots are the context."""
        x = torch.from_numpy(np.ascontiguousarray(feats_w)).float().to(device)
        adj = torch.from_numpy(np.ascontiguousarray(adj_w)).to(device).bool()
        act = torch.from_numpy(np.ascontiguousarray(active_w)).to(device).bool()
        z = self.encoder(x[None], normalize_adjacency(adj.float()))[0]   # (w, N, D)
        z_tgt = z[-1].detach()                                            # stop-gradient
        z_ctx = z[:-1]
        nodes = torch.nonzero(act[-1], as_tuple=False).squeeze(1)
        if nodes.numel() > node_batch:
            perm = torch.randperm(nodes.numel(), generator=generator, device=device)
            nodes = nodes[perm[:node_batch]]
        preds = self.predict(z_ctx[:, nodes].permute(1, 0, 2))           # (n, L, D)
        target = z_tgt[nodes]
        L = preds.shape[1]
        if self.matryoshka:
            loss = torch.stack([info_nce(preds[:, t], target, self.tau)
                                for t in range(L)]).mean()
        else:
            loss = info_nce(preds[:, -1], target, self.tau)
        with torch.no_grad():
            rate = float(self.spikes(z_ctx[:, nodes].permute(1, 0, 2)).mean()) \
                if nodes.numel() else 0.0
        return loss, {"nce": float(loss.detach()), "spike_rate": rate,
                      "n_nodes": float(nodes.numel())}
