"""MaskDGNN: Self-Supervised Dynamic Graph Neural Networks with Activeness-aware
Temporal Masking (He et al., IJCAI'25) — baseline.

Reimplemented from the paper (https://www.ijcai.org/proceedings/2025/0322.pdf);
the footnoted reference repository is not publicly reachable, so where the
paper is ambiguous the choice is recorded below. The method has three parts:

  1. **Node activeness** (eq. 2-5). Per snapshot ``t`` every node gets
     ``A = alpha * D + (1 - alpha) * S``:
       * ``D`` (dynamics): the size of the symmetric difference between its
         neighbour sets at ``t-1`` and ``t``, ``|N_t (+) N_{t-1}|``, scaled by
         ``lam / sqrt(deg_{t-1} + eta)`` for nodes that existed at ``t-1`` and
         ``lam / sqrt(delta + gamma)`` for nodes new at ``t``;
       * ``S`` (significance): PageRank on the snapshot (damping 0.85).
     Both are max-normalised over the active nodes before fusing, so ``A`` is
     in ``[0, 1]`` and the mean over an edge's endpoints is a probability (the
     paper never states a normalisation but eq. 6 needs one: PageRank sums to
     one over N nodes while D is O(1)).

  2. **Activeness-aware edge masking** (eq. 6-8). Each edge's mask probability
     is ``p = 1 - mean(A_i, A_j)``, rescaled so the expected number of masked
     edges is ``mask_ratio * |E|`` (eq. 7, clamped at 1), then drawn Bernoulli.
     Low-activeness edges are masked, high-activeness ones retained — the
     behaviour the abstract and Section 4.1 describe. (Eq. 8 as printed keeps
     the edge when the Bernoulli fires, which contradicts eq. 6-7 and the prose;
     the prose is followed.) The retained graph is what the encoder sees.

  3. **Objective** (eq. 9 + 16). Three BCE terms on the pairwise edge head
     (an MLP on ``concat(h_u, h_v)``, eq. 15):
       * masked edges of the input snapshot as positives ("L_ssl");
       * the target snapshot's edges as positives;
       * sampled non-edges of the target snapshot as negatives.
     ``target='next'`` (default) scores the edges of snapshot ``t+1`` from the
     retained graph at ``t`` — the paper's problem definition (eq. 1) and the
     outer (meta-validation) objective of WinGNN, whose sliding-window training
     MaskDGNN adopts. ``target='same'`` reconstructs snapshot ``t``'s own
     edges, the inner WinGNN step / the masked-autoencoder reading of eq. 16.

The paper optimises with WinGNN's inner/outer gradient-aggregation windows. In
this repository every baseline is trained with AdamW, best-val-loss selection
and the shared early-stopping rule (``src.baselines.common``), so the
comparison is one of objectives, not optimisers: a window of ``w`` consecutive
TRAIN snapshots is encoded as one ``(1, w, N, F)`` sequence (the frequency
layer of the ``maskdgnn`` encoder filters along ``w``) and the summed window
loss takes one AdamW step. Only the encoder is kept for the frozen probes; the
encoder + edge head are also scored end-to-end through the shared 99-negative
link evaluator, which is the paper's own protocol.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.world_models.graph_jepa_encoder import normalize_adjacency


# ---------------------------------------------------------------------------
# Node activeness (eq. 2-5)
# ---------------------------------------------------------------------------
@torch.no_grad()
def pagerank(adj: torch.Tensor, active: torch.Tensor, damping: float = 0.85,
             n_iter: int = 50, tol: float = 1e-6) -> torch.Tensor:
    """Power-iteration PageRank on a (N, N) adjacency; inactive nodes get 0 and
    hold no teleport mass. Same recurrence as ``analyze_jepa._pagerank``."""
    N = adj.shape[0]
    A = adj.float()
    act = active.bool()
    n_active = int(act.sum())
    out = torch.zeros(N, device=adj.device)
    if n_active < 2:
        return out
    deg = A.sum(dim=-1)
    P = A / deg.clamp(min=1.0).unsqueeze(-1)              # row-stochastic
    teleport = act.float() / n_active
    r = teleport.clone()
    dangling = (deg == 0) & act
    for _ in range(n_iter):
        r_new = damping * (P.t() @ r) + (1.0 - damping) * teleport
        if bool(dangling.any()):
            r_new = r_new + damping * r[dangling].sum() * teleport
        if float((r_new - r).abs().sum()) < tol:
            r = r_new
            break
        r = r_new
    out[act] = r[act]
    return out


@torch.no_grad()
def dynamics_score(adj_t: torch.Tensor, adj_prev: Optional[torch.Tensor],
                   active_t: torch.Tensor, active_prev: Optional[torch.Tensor],
                   lam: float = 1.0, eta: float = 2.0, gamma: float = -0.5
                   ) -> torch.Tensor:
    """Eq. 2-3: ``lam * delta / sqrt(deg_prev + eta)`` for nodes present at
    ``t-1``, ``lam * delta / sqrt(delta + gamma)`` for nodes new at ``t``, with
    ``delta = |N_t (+) N_{t-1}|`` (symmetric difference of neighbour sets).

    With no previous snapshot (the first snapshot of a split) every node is
    treated as new, so ``delta = deg_t``. ``delta + gamma`` can be <= 0 for the
    paper's ``gamma = -0.5`` when ``delta = 0``; such nodes changed nothing and
    score 0. Inactive nodes score 0.
    """
    N = adj_t.shape[0]
    a_t = adj_t.bool()
    if adj_prev is None:
        a_prev = torch.zeros_like(a_t)
        was_present = torch.zeros(N, dtype=torch.bool, device=adj_t.device)
    else:
        a_prev = adj_prev.bool()
        was_present = active_prev.bool()
    delta = (a_t ^ a_prev).sum(dim=-1).float()                # (N,)
    deg_prev = a_prev.sum(dim=-1).float()
    d_old = lam * delta / torch.sqrt((deg_prev + eta).clamp(min=1e-6))
    d_new = torch.where(delta + gamma > 0,
                        lam * delta / torch.sqrt((delta + gamma).clamp(min=1e-6)),
                        torch.zeros_like(delta))
    D = torch.where(was_present, d_old, d_new)
    return D * active_t.bool().float()


@torch.no_grad()
def node_activeness(adj_t: torch.Tensor, adj_prev: Optional[torch.Tensor],
                    active_t: torch.Tensor, active_prev: Optional[torch.Tensor],
                    alpha: float = 0.7, lam: float = 1.0, eta: float = 2.0,
                    gamma: float = -0.5, damping: float = 0.85) -> torch.Tensor:
    """Eq. 5: ``A = alpha * D~ + (1 - alpha) * S~`` in [0, 1] (each term
    max-normalised over the active nodes; see the module docstring)."""
    D = dynamics_score(adj_t, adj_prev, active_t, active_prev, lam, eta, gamma)
    S = pagerank(adj_t, active_t, damping)

    def _norm(v):
        m = float(v.max())
        return v / m if m > 0 else torch.zeros_like(v)

    return alpha * _norm(D) + (1.0 - alpha) * _norm(S)


# ---------------------------------------------------------------------------
# Activeness-aware edge masking (eq. 6-8)
# ---------------------------------------------------------------------------
@torch.no_grad()
def activeness_edge_mask(adj_t: torch.Tensor, activeness: torch.Tensor,
                         mask_ratio: float,
                         generator: Optional[torch.Generator] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask ~``mask_ratio * |E|`` edges of a symmetric (N, N) bool adjacency,
    preferring edges between LOW-activeness endpoints.

    Returns ``(adj_retained (N, N) bool, masked (M, 2) long)`` with each masked
    edge listed once as ``i < j`` and zeroed in both directions.
    """
    a = adj_t.bool()
    triu = torch.triu(torch.ones_like(a), diagonal=1)
    i, j = (a & triu).nonzero(as_tuple=True)
    E = i.numel()
    empty = torch.empty((0, 2), dtype=torch.long, device=a.device)
    if E == 0 or mask_ratio <= 0.0:
        return a.clone(), empty
    n_mask = int(E * mask_ratio)
    if n_mask == 0:
        return a.clone(), empty
    p = 1.0 - 0.5 * (activeness[i] + activeness[j])            # eq. 6
    total = float(p.sum())
    if total > 0:
        p_hat = (p * n_mask / total).clamp(max=1.0)            # eq. 7
    else:                                                      # all nodes maximally active
        p_hat = torch.full_like(p, float(mask_ratio))
    u = torch.rand(E, generator=generator, device=a.device)
    masked = u < p_hat                                         # eq. 8 (prose reading)
    mi, mj = i[masked], j[masked]
    retained = a.clone()
    retained[mi, mj] = False
    retained[mj, mi] = False
    return retained, torch.stack([mi, mj], dim=-1)


@torch.no_grad()
def sample_negative_pairs(adj_target: torch.Tensor, active: torch.Tensor,
                          n: int, generator: Optional[torch.Generator] = None,
                          tries: int = 4) -> torch.Tensor:
    """``n`` random (i, j) pairs of ACTIVE nodes that are not edges of
    ``adj_target`` and satisfy ``i != j`` (rejection sampling, like
    ``pretrainers.sample_masked_edges``). Returns (n', 2) long, n' <= n."""
    idx = active.bool().nonzero(as_tuple=True)[0]
    if idx.numel() < 2 or n <= 0:
        return torch.empty((0, 2), dtype=torch.long, device=adj_target.device)
    a = adj_target.bool()

    def draw(k):
        return idx[torch.randint(0, idx.numel(), (k,), generator=generator,
                                 device=idx.device)]

    ri, rj = draw(n), draw(n)
    for _ in range(tries):
        bad = (ri == rj) | a[ri, rj]
        if not bool(bad.any()):
            break
        ri = torch.where(bad, draw(n), ri)
        rj = torch.where(bad, draw(n), rj)
    ok = (ri != rj) & ~a[ri, rj]
    return torch.stack([ri[ok], rj[ok]], dim=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class EdgeMLP(nn.Module):
    """Eq. 15: ``y_uv = MLP(concat(h_u, h_v))``. Broadcasts over leading dims,
    so it doubles as the ``probe(z_i, z_j)`` callable of the shared link
    evaluator (``analyze_jepa.eval_link_pred``)."""

    def __init__(self, dim: int, hidden: int = 0, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_i, z_j], dim=-1)).squeeze(-1)


class ActivenessCache:
    """Per-snapshot activeness vectors for one split's ``(adj, active, feats)``
    snapshots, computed on first use. The score of snapshot ``t`` depends on
    ``t-1`` within the same split (the first snapshot has no predecessor and is
    scored with every node treated as new)."""

    def __init__(self, adj, active, device, alpha: float, lam: float,
                 eta: float, gamma: float, damping: float = 0.85):
        self.adj, self.active, self.device = adj, active, device
        self.kw = dict(alpha=alpha, lam=lam, eta=eta, gamma=gamma, damping=damping)
        self._cache: Dict[int, torch.Tensor] = {}

    def _snap(self, t: int):
        a = torch.from_numpy(np.ascontiguousarray(np.asarray(self.adj[t]))).to(self.device)
        act = torch.from_numpy(np.ascontiguousarray(self.active[t])).to(self.device)
        return a, act

    def get(self, t: int) -> torch.Tensor:
        if t not in self._cache:
            a_t, act_t = self._snap(t)
            if t > 0:
                a_p, act_p = self._snap(t - 1)
            else:
                a_p, act_p = None, None
            self._cache[t] = node_activeness(a_t, a_p, act_t, act_p, **self.kw).cpu()
        return self._cache[t]


class MaskDGNN(nn.Module):
    """Encoder + edge head trained with the MaskDGNN objective over a window.

    ``encoder`` is pluggable (``encoder(x[B,T,N,F], adj_norm) -> (B,T,N,D)``);
    the paper's is the ``maskdgnn`` registry encoder (GCN + adaptive frequency
    layer along T).
    """

    def __init__(self, encoder: nn.Module, latent_dim: int, mask_ratio: float = 0.2,
                 target: str = "next", neg_per_pos: int = 1,
                 edge_hidden: int = 0, dropout: float = 0.1,
                 max_pos_per_snapshot: int = 0):
        super().__init__()
        if target not in ("next", "same"):
            raise ValueError("target must be 'next' or 'same'")
        self.encoder = encoder
        self.edge_head = EdgeMLP(latent_dim, edge_hidden, dropout)
        self.mask_ratio = mask_ratio
        self.target = target
        self.neg_per_pos = neg_per_pos
        self.max_pos = max_pos_per_snapshot

    @property
    def extra_target_snapshots(self) -> int:
        """Snapshots a window needs beyond its ``w`` inputs (1 for 'next')."""
        return 1 if self.target == "next" else 0

    def window_loss(self, feats_w: np.ndarray, adj_w: np.ndarray,
                    active_w: np.ndarray, act_scores: torch.Tensor, device,
                    generator: Optional[torch.Generator] = None,
                    return_parts: bool = False):
        """Loss of one window.

        ``feats_w`` (L, N, F), ``adj_w`` (L + extra, N, N) bool, ``active_w``
        (L + extra, N) bool, ``act_scores`` (L, N) activeness of the L INPUT
        snapshots. Input snapshot ``k`` is masked, all L retained graphs are
        encoded as one sequence, and snapshot ``k``'s latents score (i) its own
        masked edges, (ii) the target snapshot's edges (``k+1`` for 'next',
        ``k`` for 'same') and (iii) sampled negatives of the target snapshot.
        """
        L = feats_w.shape[0]
        adj_t = torch.from_numpy(np.ascontiguousarray(adj_w)).to(device)
        act_t = torch.from_numpy(np.ascontiguousarray(active_w)).to(device)
        x = torch.from_numpy(np.ascontiguousarray(feats_w)).float().to(device)

        retained, masked = [], []
        for k in range(L):
            r, m = activeness_edge_mask(adj_t[k], act_scores[k].to(device),
                                        self.mask_ratio, generator)
            retained.append(r); masked.append(m)
        adj_r = torch.stack(retained).float()                  # (L, N, N)
        z = self.encoder(x[None], normalize_adjacency(adj_r))[0]  # (L, N, D)

        ssl, pos, neg, n_terms = 0.0, 0.0, 0.0, 0
        for k in range(L):
            tgt = k + self.extra_target_snapshots
            a_tgt, act_tgt = adj_t[tgt], act_t[tgt]
            triu = torch.triu(torch.ones_like(a_tgt), diagonal=1)
            pi, pj = (a_tgt & triu).nonzero(as_tuple=True)
            if pi.numel() == 0:
                continue
            if self.max_pos and pi.numel() > self.max_pos:
                sel = torch.randperm(pi.numel(), generator=generator,
                                     device=pi.device)[: self.max_pos]
                pi, pj = pi[sel], pj[sel]
            zk = z[k]
            terms = []
            if masked[k].numel() > 0:
                s_m = self.edge_head(zk[masked[k][:, 0]], zk[masked[k][:, 1]])
                l_m = F.binary_cross_entropy_with_logits(s_m, torch.ones_like(s_m))
                ssl = ssl + l_m; terms.append(l_m)
            s_p = self.edge_head(zk[pi], zk[pj])
            l_p = F.binary_cross_entropy_with_logits(s_p, torch.ones_like(s_p))
            pos = pos + l_p; terms.append(l_p)
            negs = sample_negative_pairs(a_tgt, act_tgt, pi.numel() * self.neg_per_pos,
                                         generator)
            if negs.numel() > 0:
                s_n = self.edge_head(zk[negs[:, 0]], zk[negs[:, 1]])
                l_n = F.binary_cross_entropy_with_logits(s_n, torch.zeros_like(s_n))
                neg = neg + l_n; terms.append(l_n)
            n_terms += 1
        if n_terms == 0:
            loss = z.sum() * 0.0
        else:
            loss = (ssl + pos + neg) / n_terms
        if return_parts:
            def _f(v):
                v = float(v.detach()) if torch.is_tensor(v) else float(v)
                return v / max(n_terms, 1)
            return loss, {"ssl": _f(ssl), "pos": _f(pos), "neg": _f(neg)}
        return loss
