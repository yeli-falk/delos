"""DySAT-style encoder: shape contract, causality, and that time mixing happens.

Mirrors the checks the other registry encoders get: (B, T, N, F) -> (B, T, N, D)
for every adjacency layout the pipeline hands out; output at step t must not
depend on steps > t (DySAT's causal mask); output at the last step must depend
on earlier steps (otherwise it is the plain GCN with extra parameters).
"""
import torch

from src.world_models.delos_encoder import ENCODER_REGISTRY, build_encoder


def _inputs(B=2, T=4, N=6, F=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, N, F, generator=g)
    a = (torch.rand(T, N, N, generator=g) < 0.4).float()
    a = ((a + a.transpose(1, 2)) > 0).float()
    a = a + torch.eye(N)[None]
    d = a.sum(-1, keepdim=True)
    return x, a / d


def test_registered_and_shape_contract():
    assert "dysat" in ENCODER_REGISTRY
    enc = build_encoder("dysat", 5, 8, 8, freeze=False).eval()
    x, adj = _inputs()
    assert enc(x, adj).shape == (2, 4, 6, 8)                 # (T, N, N)
    assert enc(x, adj[0]).shape == (2, 4, 6, 8)              # (N, N)
    assert enc(x, adj[None].expand(2, -1, -1, -1)).shape == (2, 4, 6, 8)  # (B, T, N, N)


def test_causal_within_window():
    torch.manual_seed(0)
    enc = build_encoder("dysat", 5, 8, 8, freeze=False).eval()
    x, adj = _inputs()
    z = enc(x, adj)
    x2 = x.clone(); x2[:, 3] += 10.0                          # perturb the LAST step only
    z2 = enc(x2, adj)
    assert torch.allclose(z[:, :3], z2[:, :3], atol=1e-6)     # steps 0..2 unchanged
    assert not torch.allclose(z[:, 3], z2[:, 3])              # step 3 changed


def test_last_step_sees_history():
    torch.manual_seed(0)
    enc = build_encoder("dysat", 5, 8, 8, freeze=False).eval()
    x, adj = _inputs()
    z = enc(x, adj)
    x2 = x.clone(); x2[:, 0] += 10.0                          # perturb the FIRST step
    z2 = enc(x2, adj)
    assert not torch.allclose(z[:, 3], z2[:, 3])              # last step depends on it


def test_single_step_window_runs():
    enc = build_encoder("dysat", 5, 8, 8, freeze=False).eval()
    x, adj = _inputs(T=1)
    assert enc(x, adj).shape == (2, 1, 6, 8)
