"""Sparse (edge-list) structural attention must equal the dense SDPA path.

Run from the repo root:
    PYTHONPATH=. python tests/test_sparse_attention.py        # plain
    PYTHONPATH=. python -m pytest tests/test_sparse_attention.py

Checks, in float64 and eval mode (no dropout):
  1. build_spacetime_edge_index enumerates exactly the True entries of
     build_spacetime_mask (same pair set, no duplicates) across the
     hidden-edge / temporal-neighbour / inactive-node combinations.
  2. DyGJEPAPredictor forward outputs (and edge-head logits) match between
     attn_impl='dense' and 'sparse' with shared weights.
  3. Gradients w.r.t. the input latents match.
"""
import itertools

import torch

from src.cjepa_graph_predictor import (
    DyGJEPAPredictor, build_spacetime_edge_index, build_spacetime_mask,
)


def _random_graph(B, T, N, density, seed, with_inactive=True, with_hidden=True,
                  self_loops=True):
    g = torch.Generator().manual_seed(seed)
    adj = torch.rand(B, T, N, N, generator=g) < density
    if self_loops:      # some datasets carry explicit self-loops
        adj |= torch.eye(N, dtype=torch.bool).view(1, 1, N, N) & (torch.rand(B, T, N, 1, generator=g) < 0.5)
    active = torch.ones(B, T, N, dtype=torch.bool)
    if with_inactive:
        # node appears at a random time and stays; a couple never appear at t=0
        first = torch.randint(0, T, (B, N), generator=g)
        first[:, 0] = 0
        active = torch.arange(T).view(1, T, 1) >= first.unsqueeze(1)
    hidden = (torch.rand(B, T, N, N, generator=g) < 0.3) & adj if with_hidden else None
    return adj, active, hidden


def test_edge_index_matches_dense_mask():
    B, T, N = 2, 4, 7
    for seed, tn, inactive, hidden in itertools.product(
            range(3), (False, True), (False, True), (False, True)):
        adj, active, hid = _random_graph(B, T, N, 0.25, seed, inactive, hidden)
        dense = build_spacetime_mask(adj, active, hid, True, tn)[:, 0]     # (B, L, L)
        b, q, k = build_spacetime_edge_index(adj, active, hid, True, tn)
        rebuilt = torch.zeros_like(dense)
        rebuilt[b, q, k] = True
        assert torch.equal(rebuilt, dense), (seed, tn, inactive, hidden)
        assert b.numel() == dense.sum().item(), "duplicate pairs"      # no dups


def _pair(attn_impl, **kw):
    torch.manual_seed(0)
    return DyGJEPAPredictor(
        num_nodes=kw["N"], node_dim=16, history_frames=kw["T"] - 1, pred_frames=1,
        num_masked_nodes=2, depth=2, heads=2, dim_head=8, mlp_dim=32, dropout=0.1,
        allow_temporal_neighbors=kw["tn"], use_edge_head=True,
        residual_head=kw["res"], attn_impl=attn_impl,
    ).double().eval()


def test_predictor_dense_vs_sparse():
    B, T, N = 2, 4, 9
    for seed, tn, res in itertools.product(range(2), (False, True), (False, True)):
        adj, active, hid = _random_graph(B, T, N, 0.3, 10 + seed)
        pd_ = _pair("dense", N=N, T=T, tn=tn, res=res)
        ps_ = _pair("sparse", N=N, T=T, tn=tn, res=res)
        ps_.load_state_dict(pd_.state_dict())
        x = torch.randn(B, T - 1, N, 16, dtype=torch.double)
        xd = x.clone().requires_grad_(True)
        xs = x.clone().requires_grad_(True)
        forced = torch.tensor([1, 4])
        pred_d, _, (idx_d, log_d) = pd_(xd, adj, active, hidden_edges=hid, forced_masked_idx=forced)
        pred_s, _, (idx_s, log_s) = ps_(xs, adj, active, hidden_edges=hid, forced_masked_idx=forced)
        assert torch.allclose(pred_d, pred_s, atol=1e-10), (seed, tn, res, (pred_d - pred_s).abs().max())
        assert torch.allclose(log_d, log_s, atol=1e-10)
        pred_d.pow(2).sum().backward()
        pred_s.pow(2).sum().backward()
        assert torch.allclose(xd.grad, xs.grad, atol=1e-9), (xd.grad - xs.grad).abs().max()
        # inference() path (future adj/active carried forward, no masking)
        assert torch.allclose(pd_.inference(xd, adj, active), ps_.inference(xs, adj, active), atol=1e-10)


if __name__ == "__main__":
    test_edge_index_matches_dense_mask()
    test_predictor_dense_vs_sparse()
    print("ok: sparse == dense")
