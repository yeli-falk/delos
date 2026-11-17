"""Unit tests for the GRACE (static SSL) baseline's augmentations and loss."""
import math

import torch

from src.baselines.grace.grace import (
    GRACE,
    drop_edges,
    drop_features,
    grace_loss,
)
from src.world_models.delos_encoder import build_encoder


def _ring(n=8):
    adj = torch.zeros(n, n)
    for i in range(n):
        adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = 1.0
    return adj


def test_drop_edges_is_symmetric_and_rate_matches():
    torch.manual_seed(0)
    n = 200
    adj = torch.ones(n, n) - torch.eye(n)
    g = torch.Generator().manual_seed(1)
    out = drop_edges(adj, 0.3, g)
    assert torch.equal(out, out.t())                # undirected edges drop as a unit
    assert out.diag().sum() == 0                    # diagonal untouched (was 0)
    kept = out.sum() / adj.sum()
    assert abs(kept - 0.7) < 0.03
    assert torch.equal(drop_edges(adj, 0.0, g), adj)


def test_drop_features_masks_whole_columns():
    torch.manual_seed(0)
    x = torch.ones(50, 400)
    g = torch.Generator().manual_seed(2)
    out = drop_features(x, 0.4, g)
    col = out[0]
    assert torch.equal(out, col.expand_as(out))     # same mask for every node
    assert abs(1 - col.mean().item() - 0.4) < 0.08
    assert torch.equal(drop_features(x, 0.0, g), x)


def test_grace_loss_matches_reference_formula():
    torch.manual_seed(0)
    m, d, tau = 6, 5, 0.4
    z1, z2 = torch.randn(m, d), torch.randn(m, d)
    loss = grace_loss(z1, z2, tau)
    # hand-rolled semi_loss from the reference implementation
    a = torch.nn.functional.normalize(z1, dim=-1)
    b = torch.nn.functional.normalize(z2, dim=-1)
    ref = 0.0
    for u, v in ((a, b), (b, a)):
        for i in range(m):
            pos = math.exp(float(u[i] @ v[i]) / tau)
            inter = sum(math.exp(float(u[i] @ v[k]) / tau) for k in range(m))
            intra = sum(math.exp(float(u[i] @ u[k]) / tau) for k in range(m) if k != i)
            ref += -math.log(pos / (inter + intra))
    ref /= 2 * m
    assert abs(loss.item() - ref) < 1e-5
    # identical views: intra-view negatives keep the loss strictly positive
    assert grace_loss(z1, z1, tau).item() > 0


def test_two_views_shapes_and_gradients():
    torch.manual_seed(0)
    n, f_in, d = 8, 5, 16
    enc = build_encoder("gcn", in_dim=f_in, hidden_dim=8, out_dim=d, freeze=False)
    model = GRACE(enc, latent_dim=d, tau=0.4)
    x, adj = torch.randn(n, f_in), _ring(n)
    g = torch.Generator().manual_seed(3)
    z1, z2 = model.two_views(x, adj, g)
    assert z1.shape == z2.shape == (n, d)
    assert not torch.allclose(z1, z2)               # the two views differ
    loss = grace_loss(z1, z2, model.tau)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.proj.parameters())
