"""Unit tests for the DyG2Vec baseline: window encoder + VICReg SSL."""
import numpy as np
import torch

from src.baselines.dyg2vec.dyg2vec import DyG2VecSSL, symmetric_edge_mask, vicreg_loss
from src.world_models.graph_jepa_encoder import (
    DyG2VecEncoder,
    build_encoder,
    normalize_adjacency,
)


def _random_window(B, T, N, p=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    adj = torch.rand(B, T, N, N, generator=g) < p
    adj = adj | adj.transpose(-1, -2)
    adj &= ~torch.eye(N, dtype=torch.bool)
    return adj


def test_encoder_shapes_and_registry():
    enc = build_encoder("dyg2vec", in_dim=5, hidden_dim=16, out_dim=8, freeze=False)
    assert isinstance(enc, DyG2VecEncoder) and len(enc.layers) == 3
    B, T, N = 2, 3, 12
    x = torch.randn(B, T, N, 5)
    adj = _random_window(B, T, N)
    assert enc(x, normalize_adjacency(adj.float())).shape == (B, T, N, 8)
    assert enc(x, normalize_adjacency(adj[0, 0].float())).shape == (B, T, N, 8)   # fixed
    assert enc(x, normalize_adjacency(adj[0].float())).shape == (B, T, N, 8)      # (T,N,N)
    z = enc(x, normalize_adjacency(torch.zeros(B, T, N, N)))                       # no edges
    assert torch.isfinite(z).all()


def test_encoder_output_t_uses_prefix_window_only():
    enc = build_encoder("dyg2vec", in_dim=4, hidden_dim=8, out_dim=8, freeze=False).eval()
    B, T, N = 1, 3, 10
    x = torch.randn(B, T, N, 4)
    a = _random_window(B, T, N, seed=1)
    b = a.clone(); b[:, 2] = _random_window(1, 1, N, seed=2)[:, 0]
    za, zb = enc(x, normalize_adjacency(a.float())), enc(x, normalize_adjacency(b.float()))
    assert torch.allclose(za[:, :2], zb[:, :2], atol=1e-6)     # steps 0,1 unchanged
    assert not torch.allclose(za[:, 2], zb[:, 2])              # step 2 sees snapshot 2


def test_window_edges_time_and_structure():
    enc = DyG2VecEncoder(3, 8, 8)
    N = 4
    adj = torch.zeros(2, N, N, dtype=torch.bool)
    adj[0, 0, 1] = adj[0, 1, 0] = True
    adj[1, 0, 2] = adj[1, 2, 0] = True
    adj[1, 1, 2] = adj[1, 2, 1] = True
    q, n, f = enc._window_edges(adj, T=2, N=N, keep_b=None)
    # t=0: 2 directed edges from s=0; t=1: 2 (s=0) + 4 (s=1)
    assert q.numel() == 8 and f.shape == (8, 8)
    # at t=1, node 0's latest interaction is s=1, so its s=0 edge has dt=1
    # and node 1's latest is s=1 too -> dt=1 for (1,0)@s=0.
    i_e = q % N; t_e = q // N
    mask = (t_e == 1) & (i_e == 0)
    assert mask.sum() == 2
    # structural masking: keep=False zeroes only the W2 part
    keep = torch.zeros_like(adj)
    _, _, f0 = enc._window_edges(adj, T=2, N=N, keep_b=keep)
    dt = torch.zeros(8)
    assert not torch.allclose(f, f0)


def test_symmetric_edge_mask():
    adj = _random_window(1, 3, 20, p=0.5)[0]
    keep = symmetric_edge_mask(adj, 0.3, torch.Generator().manual_seed(0))
    assert torch.equal(keep, keep.transpose(-1, -2))
    assert bool((keep <= adj).all())
    frac = keep.sum().item() / adj.sum().item()
    assert 0.5 < frac < 0.9
    assert torch.equal(symmetric_edge_mask(adj, 0.0), adj)


def test_vicreg_and_ssl_step():
    torch.manual_seed(0)
    z = torch.randn(64, 8)
    total, inv, var, cov = vicreg_loss(z, z)
    assert inv.item() == 0.0 and total.item() >= 0
    enc = build_encoder("dyg2vec", in_dim=5, hidden_dim=16, out_dim=16, freeze=False)
    model = DyG2VecSSL(enc, latent_dim=16)
    T, N = 3, 15
    adj = _random_window(1, T, N)[0].numpy()
    feats = np.random.randn(T, N, 5).astype(np.float32)
    total, inv, var, cov = model.window_loss(feats, adj, "cpu",
                                             torch.Generator().manual_seed(0))
    assert torch.isfinite(total) and total.item() > 0
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.encoder.parameters())
    assert all(p.grad is not None for p in model.predictor.parameters())
