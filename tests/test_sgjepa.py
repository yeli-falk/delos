"""Unit tests for the SG-JEPA baseline: PLIF spiking, surrogate gradient,
prefix predictor, InfoNCE objective through the shared GCN."""
import numpy as np
import torch

from src.baselines.sgjepa.sgjepa import (
    PLIF,
    SGJEPA,
    info_nce,
    sinusoidal_time_encoding,
    spike,
)
from src.world_models.delos_encoder import build_encoder


def test_spike_forward_binary_and_surrogate_grad():
    x = torch.tensor([-1.0, 0.0, 0.5], requires_grad=True)
    s = spike(x)
    assert torch.equal(s, torch.tensor([0.0, 1.0, 1.0]))
    s.sum().backward()
    assert (x.grad > 0).all()                       # arctan surrogate is nonzero everywhere
    assert x.grad[1] > x.grad[0]                    # and peaks at the threshold


def test_plif_integrates_and_resets():
    neuron = PLIF(init_tau=2.0, v_th=1.0)
    assert torch.isclose(torch.sigmoid(neuron.beta), torch.tensor(0.5))
    cur = torch.full((1, 4, 1), 1.0)                # constant current 1 -> v: .5, .75, .875 ... never 1
    s = neuron(cur)
    assert s.shape == (1, 4, 1) and s.sum() == 0
    cur = torch.full((1, 3, 1), 3.0)                # v: 1.5 -> spike, reset, 1.5 -> spike ...
    s = neuron(cur)
    assert torch.equal(s[0, :, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert set(torch.unique(s).tolist()) <= {0.0, 1.0}


def test_time_encoding_and_info_nce():
    te = sinusoidal_time_encoding(torch.arange(3, 0, -1), 8)
    assert te.shape == (3, 8) and torch.isfinite(te).all()
    z = torch.eye(5)
    assert info_nce(z, z, tau=0.1) < info_nce(z, z.roll(1, 0), tau=0.1)
    assert info_nce(z[:1], z[:1], tau=0.1) == 0.0


def test_predictor_prefixes_and_loss_backward():
    torch.manual_seed(0)
    enc = build_encoder("gcn", 5, 16, 16, freeze=False)
    model = SGJEPA(enc, latent_dim=16, snn_layers=1, tau=0.1)
    n, L, D = 7, 3, 16
    preds = model.predict(torch.randn(n, L, D) * 3)
    assert preds.shape == (n, L, D)
    # prefix t depends only on context steps <= t
    z = torch.randn(n, L, D) * 3
    z2 = z.clone(); z2[:, -1] += 5.0
    p1, p2 = model.predict(z), model.predict(z2)
    assert torch.allclose(p1[:, :-1], p2[:, :-1], atol=1e-6)

    T, N = L + 1, 20
    g = torch.Generator().manual_seed(1)
    adj = torch.rand(T, N, N, generator=g) < 0.3
    adj = (adj | adj.transpose(1, 2)) & ~torch.eye(N, dtype=torch.bool)
    feats = (torch.randn(T, N, 5, generator=g) * 2).numpy()
    act = np.ones((T, N), dtype=bool)
    loss, parts = model.window_loss(feats, adj.numpy(), act, "cpu",
                                    torch.Generator().manual_seed(0), node_batch=16)
    assert torch.isfinite(loss) and parts["n_nodes"] == 16
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.encoder.parameters())          # gradient via the context path
    assert all(p.grad is not None for p in model.snn.parameters())
    model.matryoshka = False
    loss2, _ = model.window_loss(feats, adj.numpy(), act, "cpu",
                                 torch.Generator().manual_seed(0), node_batch=16)
    assert torch.isfinite(loss2)
