"""DyG-WM: Dynamic Graph World Modelling -- Learning Future Dynamics from
Temporal Context (Li et al., 2026; arXiv 2610.19364) -- the self-supervised
baseline.

Implemented from the paper; no reference code was found. The paper's link
prediction headline is a SUPERVISED model (binary link loss + ranking loss at
weight 1.0, its JEPA terms at weight 0.02). What is reproduced here is the
self-supervised core only -- the setting the paper itself uses for node
classification, where "all encoders are frozen and an MLP probe ... is
fitted" -- so the row is comparable with every other frozen-probe baseline:

  * **Context encoder** (Sec. 3.2) -- lives in the encoder registry as
    ``--encoder_kind dygwm`` (:class:`DyGWMEncoder`): shared per-snapshot GNN,
    mean neighbourhood representation, separate GRUs over the node and the
    neighbourhood trajectory, gated residual fusion into ``c_u^t``. It IS the
    representation learner, so it is what ``encoder.ckpt`` stores and what
    the probe sees (as for ``maskdgnn`` / ``dyg2vec``).
  * **Pair history** -- the endpoints plus the top-``B`` other nodes ranked by
    recency-decayed adjacency to either endpoint (decay ``lambda = 0.8``) plus
    a common-neighbour bonus (``beta = 1.0``); their per-snapshot means go
    through a GRU -> ``g_uv``.
  * **Path signatures** (Sec. 3.3) -- truncated second-order signature of the
    node descriptor path ``nu_u^s`` (normalised degree, degree change,
    arriving / departing / persistent neighbour proportions, activity) and
    of the pair descriptor path ``mu_uv^s`` (direct connectivity, endpoint
    degrees, common neighbours, Jaccard overlap, context coverage), each
    projected by an MLP (``phi_N`` / ``phi_R``).
  * **Targets** -- an EMA copy ``theta_bar`` of the online per-snapshot GNN
    applied to the TARGET snapshot under stop-gradient gives ``y_u^t``; the
    relation target is ``sg(g_theta_bar(chi(y_u, y_v)))`` over observed
    pairs, ``chi(a, b) = [a || b || a*b || a-b]``.
  * **Loss** (Eq. 9-11) -- ``D(y_hat, y) = || y_hat/|y_hat| - sg(y/|y|) ||^2``
    averaged over target nodes (``L^N``) and observed pairs (``L^R``), plus
    variance / covariance regularisers on the predicted latents. The paper
    does not print the var/cov formulas; VICReg's (hinge at ``gamma = 1``,
    off-diagonal covariance) are used. Weights: the paper's ``lambda_N =
    lambda_R = 0.02`` are relative to unit-weight link losses that are absent
    here, so the JEPA terms get weight 1 and var / cov keep the paper's
    RATIOS to them (``0.005/0.02 = 0.25``, ``0.0005/0.02 = 0.025``).

Deviations, all documented in the trainer's ``--help``: shared GCN spatial
block instead of the paper's unspecified GNN; one-bin horizon only (the
paper's); no link / rank loss (SSL-only row); window = the probe's history
length (paper W = 3 for node tasks, i.e. 2 context bins).
"""
from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.world_models.graph_jepa_encoder import normalize_adjacency

NODE_DESC = 6     # dims of nu_u^s
PAIR_DESC = 6     # dims of mu_uv^s


def chi(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``chi(a, b) = [a || b || a * b || a - b]`` (DyG-WM's pair operator)."""
    return torch.cat([a, b, a * b, a - b], dim=-1)


def signature2(path: torch.Tensor) -> torch.Tensor:
    """Truncated second-order path signature of ``path`` (P, L, d) -> (P, d + d*d).

    Level 1 is the total increment; level 2 is the iterated integral, which
    for a piecewise-linear path is (Chen)
    ``S2_ij = sum_{s<s'} dx_s,i dx_s',j + 1/2 sum_s dx_s,i dx_s,j``.
    A path with fewer than two points has zero signature.
    """
    P, L, d = path.shape
    if L < 2:
        return path.new_zeros(P, d + d * d)
    inc = path[:, 1:] - path[:, :-1]                         # (P, L-1, d)
    s1 = inc.sum(dim=1)
    before = torch.cumsum(inc, dim=1) - inc                 # sum_{s' < s} dx_s'
    s2 = torch.einsum("pli,plj->pij", before, inc) \
        + 0.5 * torch.einsum("pli,plj->pij", inc, inc)
    return torch.cat([s1, s2.reshape(P, d * d)], dim=-1)


def node_descriptors(adj: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """``nu_u^s`` for every snapshot: (T, N, NODE_DESC).

    [normalised degree, degree change, arriving, departing, persistent
    neighbour proportions, activity]. The first snapshot has no predecessor,
    so its change / arrival / departure terms are 0 and persistence is 1.
    """
    T, N, _ = adj.shape
    out = adj.new_zeros(T, N, NODE_DESC, dtype=torch.float32)
    prev = adj[0]
    for t in range(T):
        a = adj[t]
        deg = a.sum(dim=-1).float()
        deg_prev = prev.sum(dim=-1).float()
        inter = (a & prev).sum(dim=-1).float()
        dmax = deg.max().clamp(min=1.0)
        out[t, :, 0] = deg / dmax
        out[t, :, 1] = (deg - deg_prev) / torch.maximum(deg, deg_prev).clamp(min=1.0)
        out[t, :, 2] = (deg - inter) / deg.clamp(min=1.0)
        out[t, :, 3] = (deg_prev - inter) / deg_prev.clamp(min=1.0)
        out[t, :, 4] = inter / (deg + deg_prev - inter).clamp(min=1.0)
        out[t, :, 5] = active[t].float()
        prev = a
    return out


def context_nodes(adj: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                  n_ctx: int, decay: float, cn_bonus: float) -> torch.Tensor:
    """Pair-context membership mask (P, N): the endpoints plus the top-``n_ctx``
    other nodes by ``sum_s decay^(L-1-s) (A_s[u,j] + A_s[v,j]) + cn_bonus *
    [j is a common neighbour of u, v in some snapshot]`` (recency + bridging).
    Nodes with zero score are never selected."""
    L, N, _ = adj.shape
    P = u.shape[0]
    score = adj.new_zeros(P, N, dtype=torch.float32)
    cn_any = adj.new_zeros(P, N, dtype=torch.bool)
    for s in range(L):
        au, av = adj[s][u], adj[s][v]                         # (P, N) bool
        score += (decay ** (L - 1 - s)) * (au.float() + av.float())
        cn_any |= au & av
    score += cn_bonus * cn_any.float()
    rows = torch.arange(P, device=adj.device)
    score[rows, u] = 0.0
    score[rows, v] = 0.0
    k = min(n_ctx, N)
    vals, idx = score.topk(k, dim=1)
    mask = torch.zeros(P, N, dtype=torch.bool, device=adj.device)
    mask.scatter_(1, idx, vals > 0)
    mask[rows, u] = True
    mask[rows, v] = True
    return mask


def pair_descriptors(adj: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                     ctx_mask: torch.Tensor) -> torch.Tensor:
    """``mu_uv^s`` for every snapshot: (T, P, PAIR_DESC) = [direct connectivity,
    deg_u, deg_v, common neighbours (all / max degree), Jaccard overlap,
    context coverage = fraction of the pair's context nodes adjacent to an
    endpoint at ``s``]."""
    T = adj.shape[0]
    P = u.shape[0]
    out = adj.new_zeros(T, P, PAIR_DESC, dtype=torch.float32)
    ctx_f = ctx_mask.float()
    ctx_n = ctx_f.sum(dim=1).clamp(min=1.0)
    for t in range(T):
        a = adj[t]
        au, av = a[u].float(), a[v].float()                    # (P, N)
        dmax = a.sum(dim=-1).max().float().clamp(min=1.0)
        du, dv = au.sum(1), av.sum(1)
        cn = (au * av).sum(1)
        out[t, :, 0] = au.gather(1, v[:, None]).squeeze(1)
        out[t, :, 1] = du / dmax
        out[t, :, 2] = dv / dmax
        out[t, :, 3] = cn / dmax
        out[t, :, 4] = cn / (du + dv - cn).clamp(min=1.0)
        out[t, :, 5] = (((au + av) > 0).float() * ctx_f).sum(1) / ctx_n
    return out


def normalised_sq_dist(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Eq. 9: ``|| pred/|pred| - sg(target/|target|) ||^2``, averaged."""
    if pred.shape[0] == 0:
        return pred.sum() * 0.0
    d = F.normalize(pred, dim=-1) - F.normalize(target.detach(), dim=-1)
    return d.pow(2).sum(dim=-1).mean()


def _variance(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.sum() * 0.0
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def _covariance(z: torch.Tensor) -> torch.Tensor:
    n, d = z.shape
    if n < 2:
        return z.sum() * 0.0
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (n - 1)
    return (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / d


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class DyGWMSSL(nn.Module):
    """DyG-WM's self-supervised objective around a context encoder.

    ``encoder`` is ideally :class:`DyGWMEncoder` (``encode`` -> per-snapshot
    ``h`` and fused context ``c``; ``snapshot`` = the per-snapshot GNN the EMA
    target applies). Any registry encoder works as a fallback: its output is
    then both ``h`` and ``c`` and the EMA copy of the whole encoder is applied
    to the target snapshot alone -- that is "the DyG-WM objective on encoder
    X", not DyG-WM, and the trainer warns about it.
    """

    def __init__(self, encoder: nn.Module, latent_dim: int, n_ctx: int = 32,
                 decay: float = 0.8, cn_bonus: float = 1.0,
                 ema_momentum: float = 0.995, pred_hidden: int = 256,
                 lam_node: float = 1.0, lam_rel: float = 1.0,
                 lam_var: float = 0.25, lam_cov: float = 0.025,
                 gamma: float = 1.0, max_horizon: int = 8):
        super().__init__()
        D = latent_dim
        self.encoder = encoder
        self.n_ctx, self.decay, self.cn_bonus = n_ctx, decay, cn_bonus
        self.momentum = ema_momentum
        self.lam_node, self.lam_rel = lam_node, lam_rel
        self.lam_var, self.lam_cov, self.gamma = lam_var, lam_cov, gamma
        sig_dim = NODE_DESC + NODE_DESC ** 2
        self.rel_embed = _mlp(4 * D, D, D)                       # g_theta
        self.phi_node = _mlp(sig_dim, D, D)                      # phi_N
        self.phi_rel = _mlp(PAIR_DESC + PAIR_DESC ** 2, D, D)    # phi_R
        self.ctx_gru = nn.GRU(D, D, batch_first=True)            # -> g_uv
        self.phi_fuse = nn.Sequential(nn.Linear(3 * D, D), nn.GELU())
        self.horizon = nn.Embedding(max_horizon, D)              # e_{delta t}
        self.pred_node = _mlp(3 * D, pred_hidden, D)             # p_N
        self.pred_rel = _mlp(2 * D, pred_hidden, D)              # p_R
        # EMA targets: the per-snapshot GNN and the relation embedder.
        self._online_target_pairs = []
        self.target_gnn = copy.deepcopy(self._snapshot_module())
        self.target_rel = copy.deepcopy(self.rel_embed)
        for m in (self.target_gnn, self.target_rel):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    # -- encoder adapters ------------------------------------------------------
    def _snapshot_module(self) -> nn.Module:
        return getattr(self.encoder, "gcn", self.encoder)

    def _encode(self, x: torch.Tensor, adj_norm: torch.Tensor):
        if hasattr(self.encoder, "encode"):
            return self.encoder.encode(x, adj_norm)
        c = self.encoder(x, adj_norm)
        return c, c

    @torch.no_grad()
    def ema_update(self) -> None:
        m = self.momentum
        for online, target in ((self._snapshot_module(), self.target_gnn),
                               (self.rel_embed, self.target_rel)):
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.mul_(m).add_(po.detach(), alpha=1.0 - m)

    # -- loss ------------------------------------------------------------------
    def window_loss(self, feats_ctx: np.ndarray, adj_ctx: np.ndarray,
                    active_ctx: np.ndarray, feats_tgt: np.ndarray,
                    adj_tgt: np.ndarray, active_tgt: np.ndarray, device,
                    generator: Optional[torch.Generator] = None,
                    node_batch: int = 256, pair_batch: int = 256
                    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Loss of one (context window, target snapshot) pair.

        ``feats_ctx`` (L, N, F), ``adj_ctx`` (L, N, N) bool, ``active_ctx``
        (L, N); ``feats_tgt`` (N, F), ``adj_tgt`` (N, N) bool, ``active_tgt``
        (N,). Node targets are ``node_batch`` nodes active in the target
        snapshot; relation targets are ``pair_batch`` observed target edges.
        """
        x = torch.from_numpy(np.ascontiguousarray(feats_ctx)).float().to(device)
        adj = torch.from_numpy(np.ascontiguousarray(adj_ctx)).to(device).bool()
        act = torch.from_numpy(np.ascontiguousarray(active_ctx)).to(device).bool()
        x_t = torch.from_numpy(np.ascontiguousarray(feats_tgt)).float().to(device)
        a_t = torch.from_numpy(np.ascontiguousarray(adj_tgt)).to(device).bool()
        act_t = torch.from_numpy(np.ascontiguousarray(active_tgt)).to(device).bool()

        h, c = self._encode(x[None], normalize_adjacency(adj.float()))
        h, c_last = h[0], c[0, -1]                                 # (L,N,D), (N,D)
        with torch.no_grad():
            y = self.target_gnn(x_t[None, None],
                                normalize_adjacency(a_t.float()))[0, 0]   # (N, D)
        e = self.horizon.weight[0]                                 # one bin ahead

        # --- node branch -------------------------------------------------------
        nodes = torch.nonzero(act_t, as_tuple=False).squeeze(1)
        nodes = _subsample(nodes, node_batch, generator)
        nu = node_descriptors(adj, act)                            # (L, N, 6)
        s_u = self.phi_node(signature2(nu[:, nodes].permute(1, 0, 2)))
        y_hat_n = self.pred_node(torch.cat(
            [c_last[nodes], s_u, e.expand(nodes.shape[0], -1)], dim=-1))
        loss_n = normalised_sq_dist(y_hat_n, y[nodes])

        # --- relation branch ---------------------------------------------------
        pairs = torch.nonzero(torch.triu(a_t, diagonal=1), as_tuple=False)
        pairs = pairs[_subsample(torch.arange(pairs.shape[0], device=device),
                                 pair_batch, generator)]
        if pairs.shape[0] > 0:
            u, v = pairs[:, 0], pairs[:, 1]
            mask = context_nodes(adj, u, v, self.n_ctx, self.decay, self.cn_bonus)
            w = mask.float() / mask.float().sum(dim=1, keepdim=True)
            means = torch.einsum("pn,lnd->pld", w, h)              # (P, L, D)
            g_uv = self.ctx_gru(means)[0][:, -1]
            mu = pair_descriptors(adj, u, v, mask)                 # (L, P, 6)
            s_uv = self.phi_rel(signature2(mu.permute(1, 0, 2)))
            q = self.phi_fuse(torch.cat(
                [self.rel_embed(chi(c_last[u], c_last[v])), g_uv, s_uv], dim=-1))
            y_hat_r = self.pred_rel(torch.cat([q, e.expand(q.shape[0], -1)], dim=-1))
            with torch.no_grad():
                y_r = self.target_rel(chi(y[u], y[v]))
            loss_r = normalised_sq_dist(y_hat_r, y_r)
        else:
            y_hat_r = y_hat_n.new_zeros(0, y_hat_n.shape[-1])
            loss_r = y_hat_n.sum() * 0.0

        var = _variance(y_hat_n, self.gamma) + _variance(y_hat_r, self.gamma)
        cov = _covariance(y_hat_n) + _covariance(y_hat_r)
        total = (self.lam_node * loss_n + self.lam_rel * loss_r
                 + self.lam_var * var + self.lam_cov * cov)
        parts = {"node": float(loss_n.detach()), "rel": float(loss_r.detach()),
                 "var": float(var.detach()), "cov": float(cov.detach()),
                 "n_pairs": float(pairs.shape[0])}
        return total, parts


def _subsample(idx: torch.Tensor, k: int, generator: Optional[torch.Generator]) -> torch.Tensor:
    if idx.numel() <= k:
        return idx
    perm = torch.randperm(idx.numel(), generator=generator, device=idx.device)[:k]
    return idx[perm]
