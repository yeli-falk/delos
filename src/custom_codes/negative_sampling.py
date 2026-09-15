"""Unified negative sampling for dynamic link prediction.

One definition of "negative edge", shared by every stage that needs one:
  - the link-prediction eval (MRR / Hits@k ranking),
  - the link-prediction probe fitting,
  - the self-supervised edge objectives (encoder pretraining + JEPA edge head).

Three strategies, following Poursafaei et al. (NeurIPS 2022), "Towards Better
Evaluation for Dynamic Link Prediction":

  random      uniform over destinations that are active and NOT a current edge.
  historical  destinations that WERE connected to the source at some earlier
              snapshot in the available history but are NOT connected now.
              Tests whether the model tracks edge *disappearance*, not just
              "this pair was seen at some point".
  inductive   historical destinations restricted to edges NOT present anywhere
              in the training period -- structure that is novel to the eval
              phase. Requires a train-period edge-union mask.

For historical / inductive, if a source has fewer qualifying candidates than
k_per_pos, the shortfall is filled with random negatives (the TGB convention)
so every positive always receives exactly k_per_pos negatives.

All strategies use *destination corruption*: the source node of each positive
edge is kept fixed and only the target is replaced. This matches the ranking
protocol (each positive is ranked against negatives sharing its source) and
makes the SSL contrastive negatives consistent with the eval negatives.

Two entry points:
  sample_negative_dst(...)  -> (P, k) negative destinations per positive edge,
                               the ranking / probe interface.
  sample_negative_bt(...)   -> (Q, 4) [b, t, i, j] negative tuples, the
                               BCE / contrastive interface used by SSL objectives.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

STRATEGIES = ("random", "historical", "inductive")


def _draw(cand: torch.Tensor, k: int, random_pool: torch.Tensor, src_i: int,
          generator: Optional[torch.Generator], device) -> torch.Tensor:
    """Draw k destinations from `cand`, filling any shortfall with random
    samples from `random_pool` (bool mask of valid random destinations).

    Sampling is without replacement when `cand` is large enough, with
    replacement (random fill) otherwise. Degenerate fallback to `src_i` only
    when nothing valid exists at all -- those rows are excluded from metrics
    upstream anyway."""
    if cand.numel() >= k:
        perm = torch.randperm(cand.numel(), generator=generator, device=device)[:k]
        return cand[perm]

    out = torch.full((k,), src_i, dtype=torch.long, device=device)
    out[: cand.numel()] = cand
    need = k - cand.numel()
    pool = random_pool.nonzero(as_tuple=True)[0]
    if pool.numel() > 0:
        idx = torch.randint(0, pool.numel(), (need,), generator=generator, device=device)
        out[cand.numel():] = pool[idx]
    return out


def sample_negative_dst(
    src: torch.Tensor,
    dst: torch.Tensor,
    adj_now: torch.Tensor,
    active_now: torch.Tensor,
    *,
    strategy: str = "random",
    k_per_pos: int = 99,
    history_adj: Optional[torch.Tensor] = None,
    train_edge_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-positive negative destinations for ranking / probe training.

    Args:
      src, dst:        (P,) long  positive edges at the target snapshot.
      adj_now:         (N, N) bool adjacency at the target snapshot.
      active_now:      (N,) bool   node existence at the target snapshot.
      strategy:        one of STRATEGIES.
      k_per_pos:       negatives per positive.
      history_adj:     (T_hist, N, N) or (N, N) bool union of prior snapshots.
                       Required for historical / inductive.
      train_edge_mask: (N, N) bool union of edges seen in the training period.
                       Required for inductive.
    Returns:
      (P, k_per_pos) long of negative destination indices (source-shared).
    """
    device = adj_now.device
    N = adj_now.shape[0]
    P = src.shape[0]
    if P == 0:
        return torch.empty((0, k_per_pos), dtype=torch.long, device=device)

    if strategy == "random":
        src_exp = src.view(P, 1).expand(P, k_per_pos)
        rand = torch.randint(0, N, (P, k_per_pos), generator=generator, device=device)
        # A few rejection rounds: resample current edges, self-loops, inactive.
        for _ in range(6):
            bad = (rand == src_exp) | adj_now[src_exp, rand] | ~active_now[rand]
            if not bad.any():
                break
            rand = torch.where(
                bad,
                torch.randint(0, N, (P, k_per_pos), generator=generator, device=device),
                rand,
            )
        return rand

    if strategy not in ("historical", "inductive"):
        raise ValueError(f"unknown negative strategy {strategy!r}; expected {STRATEGIES}")
    if history_adj is None:
        raise ValueError(f"strategy={strategy!r} requires history_adj")
    hist_union = history_adj if history_adj.dim() == 2 else history_adj.any(dim=0)
    if strategy == "inductive" and train_edge_mask is None:
        raise ValueError("strategy='inductive' requires train_edge_mask")

    neg = torch.empty((P, k_per_pos), dtype=torch.long, device=device)
    for p in range(P):
        i = int(src[p])
        forbidden = adj_now[i].clone()
        forbidden[i] = True
        forbidden |= ~active_now
        seen = hist_union[i].clone()
        if strategy == "inductive":
            seen &= ~train_edge_mask[i]
        cand = (seen & ~forbidden).nonzero(as_tuple=True)[0]
        neg[p] = _draw(cand, k_per_pos, ~forbidden, i, generator, device)
    return neg


def sample_negative_bt(
    adj: torch.Tensor,
    active: torch.Tensor,
    pos_idx: torch.Tensor,
    *,
    neg_per_pos: int = 1,
    strategy: str = "random",
    train_edge_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Negative (b, t, i, j) tuples for the SSL edge objectives (BCE).

    Destination-corrupted: each positive (b, t, i, j) yields neg_per_pos
    negatives (b, t, i, j') with j' a non-edge under the same strategy. The
    `historical` / `inductive` strategies use the in-window prefix adj[b, :t]
    as the source's edge history.

    Args:
      adj:         (B, T, N, N) bool  the *true* graph (negatives are checked
                   against this, so a hidden positive can never become a
                   negative).
      active:      (B, T, N) bool
      pos_idx:     (P, 4) long  [b, t, i, j] positive edges.
    Returns:
      (P * neg_per_pos, 4) long  [b, t, i, j'] negatives.
    """
    device = adj.device
    N = adj.shape[-1]
    if pos_idx.numel() == 0 or neg_per_pos == 0:
        return torch.empty((0, 4), dtype=torch.long, device=device)

    pb, pt, pi, pj = pos_idx.unbind(-1)
    P = pb.numel()
    rb = pb.repeat_interleave(neg_per_pos)
    rt = pt.repeat_interleave(neg_per_pos)
    ri = pi.repeat_interleave(neg_per_pos)

    if strategy == "random":
        Q = P * neg_per_pos
        rj = torch.randint(0, N, (Q,), generator=generator, device=device)
        for _ in range(6):
            bad = (
                (rj == ri)
                | adj[rb, rt, ri, rj]
                | ~active[rb, rt, ri]
                | ~active[rb, rt, rj]
            )
            if not bad.any():
                break
            rj = torch.where(
                bad, torch.randint(0, N, (Q,), generator=generator, device=device), rj,
            )
        return torch.stack([rb, rt, ri, rj], dim=-1)

    # historical / inductive: group by (b, t), reuse the per-source sampler.
    rj = torch.empty(P * neg_per_pos, dtype=torch.long, device=device)
    T = adj.shape[1]
    key = pb * T + pt
    for k in torch.unique(key):
        sel = (key == k).nonzero(as_tuple=True)[0]
        b = int(pb[sel[0]])
        t = int(pt[sel[0]])
        hist = adj[b, :t] if t > 0 else adj[b, t:t + 1] & False
        negs = sample_negative_dst(
            pi[sel], pj[sel], adj[b, t], active[b, t], strategy=strategy,
            k_per_pos=neg_per_pos, history_adj=hist,
            train_edge_mask=train_edge_mask, generator=generator,
        )  # (len(sel), neg_per_pos)
        for local, p_global in enumerate(sel.tolist()):
            rj[p_global * neg_per_pos:(p_global + 1) * neg_per_pos] = negs[local]
    return torch.stack([rb, rt, ri, rj], dim=-1)


def build_train_edge_union(dataset) -> torch.Tensor:
    """Union of all edges observed in the training period of a DTDG dataset.

    Used as `train_edge_mask` for inductive negatives. Reads the dataset's
    full adjacency (`dataset.adj`, numpy) and its `SPLITS` to find the train
    boundary -- matching the chronological split the dataset itself uses."""
    adj = dataset.adj
    T_total = adj.shape[0]
    train_end = int(T_total * dataset.SPLITS[0])
    union = adj[:train_end].any(axis=0)
    return torch.from_numpy(np.ascontiguousarray(union))


if __name__ == "__main__":
    # Minimal self-test: shapes, no current-edge / self leakage, fill behaviour.
    torch.manual_seed(0)
    N, T = 12, 4
    adj = (torch.rand(1, T, N, N) > 0.7)
    adj = adj | adj.transpose(-1, -2)
    eye = torch.eye(N, dtype=torch.bool)
    adj[:, :, eye] = False
    active = torch.ones(1, T, N, dtype=torch.bool)

    adj_now = adj[0, -1]
    active_now = active[0, -1]
    src, dst = adj_now.nonzero(as_tuple=True)
    hist = adj[0, :-1]

    for strat in STRATEGIES:
        tem = adj[0, :2].any(0) if strat == "inductive" else None
        neg = sample_negative_dst(
            src, dst, adj_now, active_now, strategy=strat, k_per_pos=5,
            history_adj=hist, train_edge_mask=tem,
        )
        assert neg.shape == (src.numel(), 5), (strat, neg.shape)
        # No negative is a current edge or a self-loop.
        leak = adj_now[src.view(-1, 1).expand_as(neg), neg]
        assert not leak.any(), f"{strat}: sampled a current edge as negative"
        assert not (neg == src.view(-1, 1)).any(), f"{strat}: self-loop negative"
        print(f"[ok] dst {strat:<11} shape={tuple(neg.shape)}")

    pos_idx = torch.stack([torch.zeros_like(src), torch.full_like(src, T - 1),
                           src, dst], dim=-1)
    for strat in STRATEGIES:
        tem = adj[0, :2].any(0) if strat == "inductive" else None
        bt = sample_negative_bt(adj, active, pos_idx, neg_per_pos=3,
                                strategy=strat, train_edge_mask=tem)
        assert bt.shape == (src.numel() * 3, 4), (strat, bt.shape)
        print(f"[ok] bt  {strat:<11} shape={tuple(bt.shape)}")
    print("all negative-sampling self-tests passed")
