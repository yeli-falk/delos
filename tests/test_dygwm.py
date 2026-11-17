"""Unit tests for the DyG-WM baseline: context encoder kind, descriptors,
path signature, EMA-target objective."""
import numpy as np
import torch

from src.baselines.dygwm.dygwm import (
    DyGWMSSL,
    chi,
    context_nodes,
    node_descriptors,
    pair_descriptors,
    signature2,
)
from src.world_models.delos_encoder import (
    ENCODER_REGISTRY,
    DyGWMEncoder,
    build_encoder,
    neighbour_mean,
    normalize_adjacency,
)


def _inputs(B=2, T=4, N=6, F=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, N, F, generator=g)
    a = (torch.rand(T, N, N, generator=g) < 0.4)
    a = a | a.transpose(1, 2)
    a &= ~torch.eye(N, dtype=torch.bool)
    return x, a


def test_registered_and_shape_contract():
    assert "dygwm" in ENCODER_REGISTRY and "dygwm_root" in ENCODER_REGISTRY
    enc = build_encoder("dygwm", 5, 8, 8, freeze=False).eval()
    assert isinstance(enc, DyGWMEncoder)
    x, a = _inputs()
    adj = normalize_adjacency(a.float())
    assert enc(x, adj).shape == (2, 4, 6, 8)                             # (T, N, N)
    assert enc(x, adj[0]).shape == (2, 4, 6, 8)                          # (N, N)
    assert enc(x, adj[None].expand(2, -1, -1, -1)).shape == (2, 4, 6, 8)  # (B, T, N, N)
    h, c = enc.encode(x, adj)
    assert torch.allclose(h, enc.snapshot(x, adj))                      # h is the bare GNN


def test_causal_and_uses_history():
    torch.manual_seed(0)
    enc = build_encoder("dygwm", 5, 8, 8, freeze=False).eval()
    x, a = _inputs()
    adj = normalize_adjacency(a.float())
    z = enc(x, adj)
    x2 = x.clone(); x2[:, 3] += 10.0
    z2 = enc(x2, adj)
    assert torch.allclose(z[:, :3], z2[:, :3], atol=1e-6)     # no future leak
    assert not torch.allclose(z[:, 3], z2[:, 3])
    x3 = x.clone(); x3[:, 0] += 10.0
    assert not torch.allclose(z[:, 3], enc(x3, adj)[:, 3])    # last step sees step 0


def test_neighbour_mean_excludes_self_and_zeroes_isolated():
    N = 4
    a = torch.zeros(N, N)
    a[0, 1] = a[1, 0] = 1.0                      # node 2, 3 isolated
    a[0, 2] = a[2, 0] = 1.0
    h = torch.arange(N, dtype=torch.float32).view(1, 1, N, 1) + 1.0
    m = neighbour_mean(normalize_adjacency(a), h)[0, 0, :, 0]
    assert torch.allclose(m, torch.tensor([2.5, 1.0, 1.0, 0.0]))


def test_signature2_two_point_path():
    d = 3
    path = torch.zeros(1, 2, d)
    path[0, 1] = torch.tensor([1.0, 2.0, -1.0])
    s = signature2(path)[0]
    assert torch.allclose(s[:d], path[0, 1])
    s2 = s[d:].view(d, d)
    assert torch.allclose(s2, 0.5 * torch.outer(path[0, 1], path[0, 1]))
    assert torch.equal(signature2(torch.zeros(2, 1, d)), torch.zeros(2, d + d * d))
    # three-point path: level-2 of (dx1, dx2) has the Chen cross term
    path3 = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
    s3 = signature2(path3)[0, 2:].view(2, 2)
    assert torch.allclose(s3, torch.tensor([[0.5, 1.0], [0.0, 0.5]]))


def test_node_and_pair_descriptors():
    N = 4
    adj = torch.zeros(2, N, N, dtype=torch.bool)
    adj[0, 0, 1] = adj[0, 1, 0] = True
    adj[1, 0, 1] = adj[1, 1, 0] = True
    adj[1, 0, 2] = adj[1, 2, 0] = True
    active = torch.ones(2, N, dtype=torch.bool)
    nu = node_descriptors(adj, active)
    assert nu.shape == (2, N, 6)
    assert torch.allclose(nu[0, :, 1:4], torch.zeros(N, 3))        # first bin: no change
    # node 0 at t=1: degree 2 (max), +1 change, 1 arriving of 2, 0 departing, 1 persistent of 2
    assert torch.allclose(nu[1, 0], torch.tensor([1.0, 0.5, 0.5, 0.0, 0.5, 1.0]))
    u = torch.tensor([0]); v = torch.tensor([1])
    mask = context_nodes(adj, u, v, n_ctx=2, decay=0.8, cn_bonus=1.0)
    assert mask[0, 0] and mask[0, 1] and mask[0, 2] and not mask[0, 3]  # 3 has zero score
    mu = pair_descriptors(adj, u, v, mask)
    assert mu.shape == (2, 1, 6)
    assert mu[1, 0, 0] == 1.0                                        # direct edge
    assert torch.isfinite(mu).all()


def test_ssl_step_backward_and_ema():
    torch.manual_seed(0)
    enc = build_encoder("dygwm", 5, 16, 16, freeze=False)
    model = DyGWMSSL(enc, latent_dim=16, n_ctx=4)
    L, N = 3, 15
    x, a = _inputs(B=1, T=L + 1, N=N, F=5, seed=3)
    adj = a.numpy(); feats = x[0].numpy(); act = np.ones((L + 1, N), dtype=bool)
    total, parts = model.window_loss(feats[:L], adj[:L], act[:L], feats[L], adj[L],
                                     act[L], "cpu", torch.Generator().manual_seed(0))
    assert torch.isfinite(total) and parts["n_pairs"] > 0
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.encoder.gcn.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.encoder.gru_node.parameters())
    assert all(p.grad is None for p in model.target_gnn.parameters())   # stop-gradient
    before = [p.clone() for p in model.target_gnn.parameters()]
    with torch.no_grad():
        for p in model.encoder.gcn.parameters():
            p.add_(1.0)
    model.ema_update()
    assert all(not torch.equal(b, p) for b, p in zip(before, model.target_gnn.parameters()))


def test_chi_shape():
    a = torch.randn(3, 4); b = torch.randn(3, 4)
    assert chi(a, b).shape == (3, 16)
