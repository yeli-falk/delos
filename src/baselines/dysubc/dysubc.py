"""DySubC: Self-Supervised Dynamic Graph Representation Learning via Temporal
Subgraph Contrast (Chen, Liu, Jiang & Chen, ACM TKDD'23; arXiv 2112.08733) —
baseline.

Faithful reimplementation of the paper (cross-checked against the released
code, https://github.com/1221045504/DySubC) on this project's dense
per-snapshot adjacency. The method contrasts a node against its *temporal
subgraph*; every ingredient is reproduced here and only the node encoder is
swapped for the pipeline's shared GNN so the encoder input matches DELOS
exactly (the reference uses a 1-layer GCN + PReLU on one-hot features):

  1. **Temporal graph.** The reference trains on one static graph whose edge
     weight is the *normalised latest timestamp* of the interaction. On a
     discrete-time dynamic graph the analogue at anchor snapshot ``t`` is the
     union of the last ``lookback`` snapshots, each edge weighted by the
     normalised index of the latest snapshot in which it appears
     (:func:`build_temporal_graph`). Node degree counts interactions across
     the window — the reference's MultiGraph degree.

  2. **Temporal subgraph sampling** (Sec. 3.3, Algorithm 1). Layer-by-layer
     BFS from the central node until ``k`` nodes are collected; only the hop
     that would overflow ``k`` is ranked, by the importance score
     ``S_j = S_time(j) + alpha * S_structure(j)`` (Eq. 1-2) with
     ``S_structure = degree`` and ``S_time`` the normalised latest timestamp of
     the edge on the sampling path. The subgraph is ``A_i = A[idx, idx]``
     (Eq. 3), once with the time weights (``G_i``) and once unweighted
     (``G'_i``). :func:`sample_temporal_subgraphs` does this for a batch of
     centres, vectorised. The released code sorts the overflow hop
     *ascending* (a bug w.r.t. the paper); the paper's "largest S" is used.

  3. **Two encoders** (Eq. 4-5): ``H_i = E1(X_i, A_i)`` on the time-weighted
     subgraph and ``H'_i = E2(X'_i, A'_i)`` on the unweighted one, separate
     weights, GCN normalisation ``D^-1/2 (A + I) D^-1/2`` (with the weights).
     ``h_i = P(H_i)`` (Eq. 6) picks the central node's row out of ``E1``. Only
     ``E1`` is kept and probed.

  4. **Readouts** (Eq. 7-9). The time-weighted summary is the influence-
     weighted average ``s_i = sum_j Inf_j h_j / sum_j Inf_j`` with
     ``Inf_j = tau_j + beta / Dist(i, j)``; the reference implements ``tau_j``
     as node ``j``'s min-max-normalised latest activity time (not the pairwise
     ``i``-``j`` timestamp the prose suggests, which is undefined for 2-hop
     nodes) and gives the centre ``Dist = 1`` — both followed here. The
     unweighted summary ``s'_i`` is the plain mean of ``H'_i``.

  5. **Losses** (Eq. 10-13). Margin (triplet) losses on sigmoid dot-product
     scores, as in Sub-Con: the *structural* loss contrasts ``s_i`` against a
     batch-shuffled summary ``s~_i`` (another node's subgraph), the *temporal*
     loss contrasts ``s_i`` against the same subgraph's unweighted summary
     ``s'_i``. ``L = L1 + lambda * L2``. Paper defaults: margins 0.75,
     ``lambda = 0.5`` (the released code uses margin 0.5).

The "dynamic" regime is realised by sampling, per step, an anchor snapshot
``t`` from the TRAIN range and building the temporal graph from the window
ending at ``t``; the encoder therefore sees the time structure of exactly the
history the downstream probe hands it. Only ``E1`` is kept and probed with the
SAME link-prediction / node-classification probe as DELOS
(``src.train.analyze_jepa --skip_diagnostic --link_sources encoder``).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.world_models.delos_encoder import normalize_adjacency

# Reference normalises timestamps to [0, 10] before using them as edge weights
# and importance scores (``load_data_as_graph``: ``(t - t_min) / (t_max -
# t_min) * 10``).
TIME_SCALE: float = 10.0
# Safety cap on BFS depth.
MAX_HOPS: int = 8


class TemporalGraph:
    """Time-weighted union of a snapshot window (``build_temporal_graph``).

    ``W`` (N, N) float: 0 where no edge in the window, else the normalised
    latest snapshot ``TIME_SCALE * (s - s0 + 1) / L`` in ``(0, TIME_SCALE]``
    (the paper maps the earliest edge to weight 0, which silently drops it
    from the GCN; mapping to ``(0, 10]`` keeps every edge). ``deg`` (N,) is
    the interaction count over the window (MultiGraph degree). ``tau`` (N,)
    in [0, 1] is each node's normalised latest activity time — the readout's
    ``tau_j``. ``present`` (N,) bool marks nodes with >= 1 edge in the window
    (the only valid centres). ``levels`` are the distinct edge weights, in
    ascending order.
    """

    def __init__(self, W: torch.Tensor, deg: torch.Tensor, tau: torch.Tensor,
                 present: torch.Tensor, levels: torch.Tensor):
        self.W, self.deg, self.tau, self.present, self.levels = (
            W, deg, tau, present, levels)

    @property
    def num_nodes(self) -> int:
        return int(self.W.shape[0])


def build_temporal_graph(adj, t: int, lookback: int, device) -> TemporalGraph:
    """Union of snapshots ``[t - lookback + 1, t]`` with latest-time weights.

    ``adj`` is a (T, N, N) bool stack (or a ``ConstantAdjacency``). Later
    snapshots overwrite earlier ones, so each edge carries the weight of the
    LATEST snapshot it appears in.
    """
    s0 = max(0, t - lookback + 1)
    L = t - s0 + 1
    N = adj.shape[-1]
    W = torch.zeros(N, N, device=device)
    deg = torch.zeros(N, device=device)
    last = torch.full((N,), -1, device=device, dtype=torch.long)
    levels = []
    for s in range(s0, t + 1):
        A = torch.from_numpy(np.ascontiguousarray(adj[s])).to(device)
        lvl = TIME_SCALE * (s - s0 + 1) / L
        levels.append(lvl)
        W = torch.where(A, torch.full_like(W, lvl), W)
        deg += A.sum(dim=1).float()
        last = torch.where(A.any(dim=1), torch.full_like(last, s), last)
    present = last >= 0
    tau = ((last - s0).float() / max(L - 1, 1)).clamp(min=0.0)
    tau[~present] = 0.0
    return TemporalGraph(W, deg, tau, present,
                         torch.tensor(levels, device=device))


def sample_temporal_subgraphs(
    g: TemporalGraph, centres: torch.Tensor, k: int, alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Algorithm 1 for a batch of centres, vectorised over the batch.

    Returns ``(idx (M, k) long, mask (M, k) bool, dist (M, k) float)``: the
    sampled node ids per centre (centre at position 0, then by hop), a
    validity mask (a component smaller than ``k`` gives a partial subgraph,
    padded with the centre id and masked out), and the hop distance to the
    centre (``0`` for the centre itself).

    Hop by hop: every node adjacent to the current frontier and not yet
    selected is a candidate. If the candidates fit in the remaining room
    they are all taken; otherwise the ``room`` with the largest importance
    ``S_j = S_time + alpha * S_structure`` are (Eq. 1-2), where ``S_time`` is
    the latest-time weight of the edge that reaches ``j`` from the frontier
    (max over frontier nodes) and ``S_structure`` is ``j``'s degree. Both
    cases are one ``topk`` over the candidate scores.
    """
    device = g.W.device
    M, N = int(centres.shape[0]), g.num_nodes
    ar = torch.arange(M, device=device)
    sel = torch.zeros(M, N, dtype=torch.bool, device=device)
    sel[ar, centres] = True
    dist = torch.full((M, N), float("inf"), device=device)
    dist[ar, centres] = 0.0
    frontier = sel.clone()
    count = torch.ones(M, device=device, dtype=torch.long)
    W_bin = None
    for hop in range(1, MAX_HOPS + 1):
        if hop == 1:
            tscore = g.W[centres]                                  # (M, N)
            reach = tscore > 0
        else:
            if W_bin is None:
                W_bin = (g.W > 0).float()
            f = frontier.float()
            reach = (f @ W_bin) > 0
            # max over frontier nodes of the edge time: one matmul per
            # distinct level (levels are few: one per snapshot in the window).
            tscore = torch.zeros(M, N, device=device)
            for lvl in g.levels.tolist():
                hit = (f @ (g.W == lvl).float()) > 0
                tscore = torch.where(hit, torch.full_like(tscore, lvl), tscore)
        cand = reach & ~sel
        room = (k - count).clamp(min=0)                            # (M,)
        if not bool(cand.any()) or not bool((room > 0).any()):
            break
        score = tscore + alpha * g.deg[None, :]
        score = torch.where(cand, score, torch.full_like(score, float("-inf")))
        kk = min(k - 1, N)
        topv, topi = score.topk(kk, dim=1)                         # (M, kk)
        take = torch.isfinite(topv) & (torch.arange(kk, device=device)[None, :]
                                       < room[:, None])
        new = torch.zeros(M, N, dtype=torch.bool, device=device)
        rows = ar[:, None].expand(M, kk)[take]
        new[rows, topi[take]] = True
        if not bool(new.any()):
            break
        sel |= new
        dist = torch.where(new, torch.full_like(dist, float(hop)), dist)
        count = count + new.sum(dim=1)
        frontier = new
        if bool((count >= k).all()):
            break
    key = torch.where(sel, dist, torch.full_like(dist, float("inf")))
    vals, order = key.sort(dim=1)                                  # centre first
    idx = order[:, :k]
    dist_k = vals[:, :k]
    mask = torch.isfinite(dist_k)
    idx = torch.where(mask, idx, centres[:, None].expand(M, k))
    dist_k = torch.where(mask, dist_k, torch.ones_like(dist_k))
    return idx, mask, dist_k


def margin_loss(pos: torch.Tensor, neg: torch.Tensor, margin: float) -> torch.Tensor:
    """Sub-Con margin loss ``mean(max(0, neg - pos + margin))`` on sigmoid
    scores — the implemented form of Eq. 11 / 12 (the reference's
    ``MarginRankingLoss(margin)(pos, neg, ones)``)."""
    return F.relu(neg - pos + margin).mean()


class DySubC(nn.Module):
    """Two subgraph encoders + influence-weighted readout + margin losses.

    ``encoder`` (E1, time-weighted subgraph) is the module that is kept and
    probed; ``encoder_unweighted`` (E2) only produces the temporal negative
    ``s'_i``. Both follow the pipeline's ``encoder(x[B,T,N,F], adj) ->
    (B,T,N,D)`` signature, so any registry encoder works.
    """

    def __init__(self, encoder: nn.Module, encoder_unweighted: nn.Module,
                 subgraph_size: int = 20, alpha: float = 5.0, beta: float = 1.0,
                 margin_struct: float = 0.75, margin_time: float = 0.75,
                 lam: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.encoder_unweighted = encoder_unweighted
        self.k = subgraph_size
        self.alpha = alpha
        self.beta = beta
        self.margin_struct = margin_struct
        self.margin_time = margin_time
        self.lam = lam

    # -- pieces -------------------------------------------------------------
    def _encode_subgraphs(self, encoder: nn.Module, x: torch.Tensor,
                          A: torch.Tensor) -> torch.Tensor:
        """``x`` (M, k, F), ``A`` (M, k, k) weighted or binary -> (M, k, D)."""
        adj_norm = normalize_adjacency(A)[:, None]                 # (M,1,k,k)
        return encoder(x[:, None], adj_norm)[:, 0]                 # (M,k,D)

    @staticmethod
    def readout_weighted(H: torch.Tensor, inf: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        """Eq. 8: ``s_i = sum_j Inf_j h_j / sum_j Inf_j`` over valid nodes."""
        w = torch.where(mask, inf, torch.zeros_like(inf))          # (M,k)
        return (w[..., None] * H).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp(min=1e-8)

    @staticmethod
    def readout_mean(H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Eq. 9: plain mean over the subgraph's valid nodes."""
        m = mask.float()
        return (m[..., None] * H).sum(dim=1) / m.sum(dim=1, keepdim=True).clamp(min=1.0)

    def forward_subgraphs(self, g: TemporalGraph, feats_t: torch.Tensor,
                          centres: torch.Tensor):
        """Sample + encode + read out the subgraphs of ``centres``.

        Returns ``(h, s, s_unw)``: central-node embeddings (M, D) from E1,
        the influence-weighted summaries (M, D) from E1 and the mean
        summaries (M, D) from E2.
        """
        with torch.no_grad():
            idx, mask, dist = sample_temporal_subgraphs(g, centres, self.k,
                                                        self.alpha)
            A = g.W[idx[:, :, None], idx[:, None, :]]              # (M,k,k)
            valid = mask[:, :, None] & mask[:, None, :]
            A = torch.where(valid, A, torch.zeros_like(A))
            A_bin = (A > 0).float()
            # Eq. 7 influence: tau_j + beta / Dist(i, j); centre counts as
            # distance 1 (reference ``dis[centre] = 1``).
            inf = g.tau[idx] + self.beta / dist.clamp(min=1.0)
        x = feats_t[idx]                                           # (M,k,F)
        H1 = self._encode_subgraphs(self.encoder, x, A)            # time-weighted
        H2 = self._encode_subgraphs(self.encoder_unweighted, x, A_bin)
        h = H1[:, 0]                                               # Eq. 6
        s = self.readout_weighted(H1, inf, mask)                   # Eq. 8
        s_unw = self.readout_mean(H2, mask)                        # Eq. 9
        return h, s, s_unw

    def loss(self, h: torch.Tensor, s: torch.Tensor, s_unw: torch.Tensor,
             perm: torch.Tensor):
        """Eq. 11-13 on sigmoid dot-product scores. ``perm`` is the batch
        shuffle giving the structural negative ``s~_i = s_perm(i)``
        (Eq. 10). Returns ``(total, L_struct, L_time)``."""
        pos = torch.sigmoid((h * s).sum(-1))
        neg_struct = torch.sigmoid((h * s[perm]).sum(-1))
        neg_time = torch.sigmoid((h * s_unw).sum(-1))
        l1 = margin_loss(pos, neg_struct, self.margin_struct)
        l2 = margin_loss(pos, neg_time, self.margin_time)
        return l1 + self.lam * l2, l1, l2

    def contrastive_step(self, g: TemporalGraph, feats_t: np.ndarray,
                         centres: torch.Tensor, perm: torch.Tensor, device):
        """Full loss for one anchor snapshot: ``feats_t`` (N, F) node features
        at the anchor, ``centres`` (M,) the sampled central nodes."""
        x = torch.from_numpy(np.ascontiguousarray(feats_t)).float().to(device)
        h, s, s_unw = self.forward_subgraphs(g, x, centres)
        return self.loss(h, s, s_unw, perm)
