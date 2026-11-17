"""DDGCL: Debiased Dynamic Graph Contrastive Learning
(Tian et al., "Self-supervised representation learning on dynamic graphs",
CIKM'21) — baseline.

Faithful reimplementation of the reference (https://github.com/ckldan520/DDGCL)
on this project's dense per-snapshot adjacency. The four defining ingredients
of the reference are all reproduced here; only the *encoder* is swapped for the
pipeline's GNN so the input matches DELOS exactly (the reference uses a
TGAT-style transformer over sampled temporal neighbours):

  1. **MoCo framework** (``moco_builder.Moco``). A query encoder + a
     momentum-updated key encoder (``m = 0.999``) feed a FIFO **negative queue**
     (size 512). Negatives are drawn cross-batch from the queue, not just from
     the current batch, and each queued key stores the snapshot time it was
     produced at so it is scored through ITS OWN time gap to the query (the
     reference's per-negative ``W_neg``). Set ``queue_size = 0`` (or
     ``momentum = 0``) to fall back to symmetric in-batch negatives.

  2. **Time-aware bilinear similarity** (the reference's ``W_pos`` / ``W_neg``).
     The scalar time gap ``Δt`` is encoded, then a *time-conditioned* matrix
     ``W(Δt) ∈ R^{D×D}`` is generated from it and the score of a pair is the
     bilinear form ``qᵀ · W(Δt) · k`` — NOT a concatenation + MLP head. ``W`` is
     initialised to the identity so training starts from plain cosine similarity.

  3. **Bochner / TGAT functional time encoding** (``Base_encoder.time_coding``):
     ``t -> [cos(ω·log(Δt+1)), sin(ω·log(Δt+1))]`` with learnable frequencies
     initialised to ``1/10^{linspace(0,1.5)}``.

  4. **Debiased + hard-negative-reweighted loss** (``loss_cpc_debais``). This is
     a Jensen-Shannon / BCE estimator (``softplus``), NOT softmax-InfoNCE:
       * positive term  ``softplus(-s_pos)``            (BCE against label 1);
       * hard-negative reweighting ``softmax(β·s_neg)`` (Robinson et al. 2021),
         ``β = 1.5``;
       * debias correction with class-prior ``τ = 1/100`` (Chuang et al. 2020).
     ``β = 0`` gives uniform negative weights; ``τ = 0`` drops the debias term.

The "dynamic" regime is realised by sampling, per step, a node's two temporal
views as two nearby TRAIN snapshots ``t1 < t2`` (gap ``Δt``); ``Δt`` conditions
the time-aware bilinear score. Only the (query) encoder is kept and probed with
the SAME link-prediction / node-classification probe as DELOS
(``src.train.analyze_jepa --skip_diagnostic --link_sources encoder``).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.world_models.delos_encoder import normalize_adjacency


class TimeEncoder(nn.Module):
    """Bochner / TGAT functional time encoding (``Base_encoder.time_coding``).

    ``Δt -> [cos(ω·log(Δt+1)), sin(ω·log(Δt+1))]``. The learnable frequencies
    ``ω`` are initialised geometrically (``1/10^{linspace(0,1.5)}``) so different
    coordinates respond to time scales spanning several orders of magnitude.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2:
            raise ValueError("time_dim must be even (a cos half and a sin half)")
        half = dim // 2
        basis = 1.0 / (10.0 ** np.linspace(0.0, 1.5, half))
        self.basis_freq = nn.Parameter(torch.from_numpy(basis).float())

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """``dt`` (...,) time gaps -> (..., dim) encoding (scalar dt -> (dim,))."""
        x = torch.log1p(dt.clamp(min=0.0))           # log(Δt + 1)
        m = x.unsqueeze(-1) * self.basis_freq        # (..., half)
        return torch.cat([torch.cos(m), torch.sin(m)], dim=-1)


class TimeBilinear(nn.Module):
    """Time-conditioned bilinear scorer (the reference's ``W_pos`` / ``W_neg``).

    Generates a ``(D, D)`` matrix ``W(Δt)`` from the time encoding so a pair is
    scored ``qᵀ · W(Δt) · k``. Initialised to the identity (zero weights, eye
    bias) so at step 0 the score is exactly the cosine similarity of ``q`` and
    ``k`` and time-conditioning is learned as a departure from it.
    """

    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.dim = dim
        self.gen = nn.Linear(time_dim, dim * dim)
        nn.init.zeros_(self.gen.weight)
        with torch.no_grad():
            self.gen.bias.copy_(torch.eye(dim).flatten())

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """``t_emb`` (..., time_dim) -> ``W`` (..., dim, dim)."""
        return self.gen(t_emb).view(*t_emb.shape[:-1], self.dim, self.dim)


def debiased_hardneg(s_pos: torch.Tensor, s_neg: torch.Tensor,
                     tau: float = 0.01, beta: float = 1.5) -> torch.Tensor:
    """Reference ``loss_cpc_debais`` — debiased + hard-negative-reweighted loss.

    ``s_pos`` (M,) are positive-pair scores, ``s_neg`` (M, Q) are the scores of
    each anchor against its Q negatives (RAW bilinear scores, not divided by any
    temperature). Reproduces the reference exactly (JS/BCE estimator):

        loss_pos = softplus(-s_pos)                          # -log σ(s_pos)
        w        = softmax(β · s_neg, dim=1)                 # hard-neg mining
        loss_neg = ( Σ_j w_ij·softplus(s_neg_ij)
                     − τ·softplus(s_pos_i) ) / (1 − τ)       # debias correction
        loss     = mean( loss_pos + loss_neg )

    ``β = 0`` -> uniform negative weights; ``τ = 0`` -> no debias term.
    Note ``log(1/(1+e^x)) = −softplus(x)`` — the reference's ``loss_neg_org``
    and ``bais`` reduce to the ``softplus`` forms above.
    """
    if s_neg.shape[1] < 1:
        return F.softplus(-s_pos).mean()
    loss_pos = F.softplus(-s_pos)                            # (M,)
    w = torch.softmax(beta * s_neg, dim=1)                   # (M, Q)
    weighted_neg = (w * F.softplus(s_neg)).sum(dim=1)        # (M,)
    loss_neg = (weighted_neg - tau * F.softplus(s_pos)) / (1.0 - tau)
    loss_neg = loss_neg.clamp(min=0.0)
    return (loss_pos + loss_neg).mean()


class DDGCL(nn.Module):
    """MoCo query/key encoders + time-aware bilinear head for debiased dynamic
    graph contrastive learning. The ``encoder`` is fully pluggable — any module
    with the pipeline's ``encoder(x[B,T,N,F], adj) -> (B,T,N,D)`` signature works
    (``LinearNodeEncoder`` / ``GCNNodeEncoder`` / ``TGCNEncoder`` / …).
    """

    def __init__(self, encoder: nn.Module, latent_dim: int, time_dim: int = 32,
                 tau: float = 0.01, beta: float = 1.5, momentum: float = 0.999,
                 queue_size: int = 512):
        super().__init__()
        self.encoder = encoder
        self.time_enc = TimeEncoder(time_dim)
        self.bilinear = TimeBilinear(latent_dim, time_dim)
        self.tau = tau
        self.beta = beta
        self.momentum = momentum
        self.use_queue = queue_size > 0 and momentum > 0.0
        if self.use_queue:
            self.queue_size = queue_size
            self.key_encoder = copy.deepcopy(encoder)
            for p in self.key_encoder.parameters():
                p.requires_grad_(False)
            self.register_buffer(
                "queue", F.normalize(torch.randn(latent_dim, queue_size), dim=0))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_time", torch.zeros(queue_size))

    # -- MoCo bookkeeping ---------------------------------------------------
    @torch.no_grad()
    def _momentum_update(self) -> None:
        m = self.momentum
        for pq, pk in zip(self.encoder.parameters(),
                          self.key_encoder.parameters()):
            pk.data.mul_(m).add_(pq.data, alpha=1.0 - m)

    @torch.no_grad()
    def _push_queue(self, keys: torch.Tensor, key_time: float) -> None:
        """Enqueue ``keys`` (M, D) L2-normalised (all produced at snapshot
        ``key_time``) into the FIFO negative queue."""
        m = keys.shape[0]
        t = float(key_time)
        if m >= self.queue_size:
            self.queue.copy_(keys[: self.queue_size].t())
            self.queue_time.fill_(t)
            self.queue_ptr[0] = 0
            return
        ptr = int(self.queue_ptr)
        end = ptr + m
        if end <= self.queue_size:
            self.queue[:, ptr:end] = keys.t()
            self.queue_time[ptr:end] = t
        else:
            first = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first].t()
            self.queue_time[ptr:] = t
            self.queue[:, : end - self.queue_size] = keys[first:].t()
            self.queue_time[: end - self.queue_size] = t
        self.queue_ptr[0] = end % self.queue_size

    # -- encoding -----------------------------------------------------------
    def _encode(self, feats: np.ndarray, adj: np.ndarray, device,
                encoder: nn.Module) -> torch.Tensor:
        """Encode one (N, F) feature matrix on snapshot ``adj`` -> (N, D)."""
        x = torch.from_numpy(np.ascontiguousarray(feats)).float().to(device)
        a = torch.from_numpy(np.ascontiguousarray(adj)).float().to(device)
        z = encoder(x[None, None], normalize_adjacency(a))    # (1,1,N,D)
        return z[0, 0]                                         # (N, D)

    def _inbatch(self, q: torch.Tensor, k: torch.Tensor,
                 W: torch.Tensor) -> torch.Tensor:
        """One direction of the symmetric in-batch loss: positives on the
        diagonal, the other M-1 rows as negatives."""
        M = q.shape[0]
        S = (q @ W) @ k.t()                                    # (M, M)
        s_pos = S.diagonal()
        mask = ~torch.eye(M, dtype=torch.bool, device=q.device)
        s_neg = S[mask].view(M, M - 1)
        return debiased_hardneg(s_pos, s_neg, self.tau, self.beta)

    def contrastive_step(self, feats1, adj1, feats2, adj2, idx: torch.Tensor,
                         t_query: float, t_key: float, device,
                         update: bool = True) -> torch.Tensor:
        """Loss for one contrastive step: the ``idx`` nodes seen at snapshot
        ``t_query`` (view 1 / query) and ``t_key`` (view 2 / positive key).

        With a queue: MoCo (query encoder + momentum key encoder + FIFO
        negatives), where each queued negative is scored through ITS OWN time
        gap ``|t_query − t_key_j|`` — the reference's per-negative ``W_neg``.
        Without a queue: symmetric in-batch negatives, all sharing the step's
        gap, through the shared encoder.

        ``update=False`` scores the step without advancing the momentum encoder
        or pushing to the negative queue — required for val-loss evaluation,
        which must not let held-out snapshots leak into the model's state.
        """
        dt = torch.as_tensor(float(t_key) - float(t_query), device=device)
        W = self.bilinear(self.time_enc(dt))                  # (D, D), time-cond.

        if self.use_queue:
            q = F.normalize(self._encode(feats1, adj1, device, self.encoder)[idx],
                            dim=-1)
            with torch.no_grad():
                if update:
                    self._momentum_update()
                k = F.normalize(
                    self._encode(feats2, adj2, device, self.key_encoder)[idx],
                    dim=-1)
            s_pos = ((q @ W) * k).sum(dim=-1)                 # (M,)
            # Per-negative time gap: all queries share t_query, so the gap
            # depends only on the queued key -> a (queue, D, D) transform, not a
            # (M, queue, D, D) one. Fold W_neg into each key: kk_j = W_neg_j·k_j.
            gaps = (float(t_query) - self.queue_time).abs()   # (queue,)
            W_neg = self.bilinear(self.time_enc(gaps))        # (queue, D, D)
            kk = torch.einsum("qde,qe->qd", W_neg,
                              self.queue.t().detach().clone())  # (queue, D)
            s_neg = q @ kk.t()                                # (M, queue)
            loss = debiased_hardneg(s_pos, s_neg, self.tau, self.beta)
            if update:
                self._push_queue(k, t_key)
            return loss

        z1 = F.normalize(self._encode(feats1, adj1, device, self.encoder)[idx],
                         dim=-1)
        z2 = F.normalize(self._encode(feats2, adj2, device, self.encoder)[idx],
                         dim=-1)
        return 0.5 * (self._inbatch(z1, z2, W) + self._inbatch(z2, z1, W))
