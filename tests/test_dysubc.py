"""Unit tests for the DySubC baseline's temporal graph + subgraph sampler."""
import numpy as np
import torch

from src.baselines.dysubc.dysubc import (
    TIME_SCALE,
    DySubC,
    build_temporal_graph,
    sample_temporal_subgraphs,
)
from src.world_models.graph_jepa_encoder import build_encoder


def _star_plus_chain():
    """3 snapshots over 8 nodes: node 0 is a hub whose edges appear at
    different times; node 7 is only reachable through node 6 (2 hops)."""
    N, T = 8, 3
    adj = np.zeros((T, N, N), dtype=bool)

    def add(t, u, v):
        adj[t, u, v] = adj[t, v, u] = True

    add(0, 0, 1); add(0, 0, 2); add(1, 0, 3); add(2, 0, 4); add(2, 0, 5)
    add(0, 0, 6); add(2, 0, 6)                # 0-6 seen twice, latest at t=2
    add(1, 6, 7)                              # 7 is 2 hops from 0
    add(2, 1, 2)                              # 1-2 clique edge
    return adj


def test_temporal_graph_latest_time_and_degree():
    adj = _star_plus_chain()
    g = build_temporal_graph(adj, t=2, lookback=3, device="cpu")
    # latest snapshot wins; weights on a (0, 10] scale.
    assert torch.isclose(g.W[0, 1], torch.tensor(TIME_SCALE * 1 / 3))
    assert torch.isclose(g.W[0, 3], torch.tensor(TIME_SCALE * 2 / 3))
    assert torch.isclose(g.W[0, 6], torch.tensor(TIME_SCALE))
    assert g.W[3, 4] == 0
    # MultiGraph degree counts interactions over the window: node 0 has 7.
    assert g.deg[0] == 7 and g.deg[6] == 3 and g.deg[7] == 1
    # node-level latest activity in [0, 1]
    assert g.tau[0] == 1.0 and g.tau[3] == 0.5 and g.tau[7] == 0.5
    assert bool(g.present.all())
    assert g.levels.tolist() == sorted(g.levels.tolist())


def test_sampler_takes_whole_hops_then_ranks_overflow():
    adj = _star_plus_chain()
    g = build_temporal_graph(adj, t=2, lookback=3, device="cpu")
    centres = torch.tensor([0, 7])
    # k=4 from node 0: hop 1 has 6 candidates > room 3 -> the 3 largest
    # S = time + alpha*deg. With alpha=0 the score is time: nodes 4,5,6 (t=2).
    idx, mask, dist = sample_temporal_subgraphs(g, centres, k=4, alpha=0.0)
    assert idx.shape == (2, 4) and bool(mask.all())
    assert idx[0, 0] == 0 and dist[0, 0] == 0
    assert set(idx[0, 1:].tolist()) == {4, 5, 6} and bool((dist[0, 1:] == 1).all())
    # From node 7: hop 1 = {6} fits, hop 2 = {0} fits; then hop 3 overflows
    # (5 candidates for 1 slot) -> highest time among 0's other neighbours.
    assert idx[1, 0] == 7 and idx[1, 1] == 6 and idx[1, 2] == 0
    assert dist[1].tolist() == [0, 1, 2, 3]
    assert idx[1, 3].item() in {4, 5}
    # alpha large: degree dominates -> node 6 (deg 3) and 1, 2 (deg 2) win.
    idx, _, _ = sample_temporal_subgraphs(g, centres[:1], k=4, alpha=100.0)
    assert set(idx[0, 1:].tolist()) == {6, 1, 2}


def test_sampler_pads_small_components():
    N = 6
    adj = np.zeros((1, N, N), dtype=bool)
    adj[0, 0, 1] = adj[0, 1, 0] = True          # component {0, 1}
    g = build_temporal_graph(adj, t=0, lookback=1, device="cpu")
    idx, mask, dist = sample_temporal_subgraphs(g, torch.tensor([0]), k=5, alpha=1.0)
    assert mask[0].tolist() == [True, True, False, False, False]
    assert idx[0, :2].tolist() == [0, 1]
    assert bool((idx[0, 2:] == 0).all())         # padded with the centre id
    assert not g.present[2]


def test_forward_and_loss_shapes():
    torch.manual_seed(0)
    adj = _star_plus_chain()
    g = build_temporal_graph(adj, t=2, lookback=3, device="cpu")
    F_in, D = 5, 16
    enc = lambda: build_encoder("gcn", in_dim=F_in, hidden_dim=8, out_dim=D, freeze=False)
    model = DySubC(enc(), enc(), subgraph_size=4, alpha=1.0, beta=1.0)
    feats = torch.randn(8, F_in)
    centres = torch.tensor([0, 6, 7])
    h, s, s_unw = model.forward_subgraphs(g, feats, centres)
    assert h.shape == s.shape == s_unw.shape == (3, D)
    total, l1, l2 = model.loss(h, s, s_unw, torch.tensor([1, 2, 0]))
    assert total.item() >= 0 and torch.isfinite(total)
    assert torch.isclose(total, l1 + 0.5 * l2)
    total.backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.encoder_unweighted.parameters())
