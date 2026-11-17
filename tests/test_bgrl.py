"""Unit tests for the BGRL (static negative-free SSL) baseline."""
import torch

from src.baselines.bgrl.bgrl import BGRL, bgrl_loss, ema_momentum
from src.baselines.bgrl.train_bgrl import lr_at
from src.world_models.delos_encoder import build_encoder


def _ring(n=8):
    adj = torch.zeros(n, n)
    for i in range(n):
        adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = 1.0
    return adj


def _model(n=8, f_in=5, d=16):
    torch.manual_seed(0)
    enc = build_encoder("gcn", in_dim=f_in, hidden_dim=8, out_dim=d, freeze=False)
    return BGRL(enc, latent_dim=d, pred_hidden=32), torch.randn(n, f_in), _ring(n)


def test_loss_range_and_stop_gradient():
    q = torch.randn(6, 4, requires_grad=True)
    y = torch.randn(6, 4, requires_grad=True)
    assert bgrl_loss(q, q.detach(), q, q.detach()).item() < 1e-5      # parallel -> 0
    assert abs(bgrl_loss(q, -q.detach(), q, -q.detach()).item() - 4.0) < 1e-5
    bgrl_loss(q, y, q, y).backward()
    assert q.grad is not None and y.grad is None                      # targets detached


def test_momentum_and_lr_schedules():
    assert abs(ema_momentum(0, 100, 0.99) - 0.99) < 1e-9
    assert abs(ema_momentum(100, 100, 0.99) - 1.0) < 1e-9
    assert 0.99 < ema_momentum(50, 100, 0.99) < 1.0
    assert abs(lr_at(0, 100, 1e-3, 10) - 1e-4) < 1e-12                 # warm-up start
    assert abs(lr_at(9, 100, 1e-3, 10) - 1e-3) < 1e-12                 # warm-up end
    assert lr_at(100, 100, 1e-3, 10) < 1e-9                            # decayed to 0


def test_target_gets_no_gradient_and_tracks_online_by_ema():
    model, x, adj = _model()
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
    q1, y2, q2, y1 = model(x, adj, torch.Generator().manual_seed(1))
    assert q1.shape == y2.shape == (8, 16) and not y1.requires_grad
    loss = bgrl_loss(q1, y2, q2, y1)
    loss.backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
    # one SGD-like nudge to the online net, then an EMA step
    before = [p.detach().clone() for p in model.target_encoder.parameters()]
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    model.update_target(0.9)
    for b, p_t, p_o in zip(before, model.target_encoder.parameters(),
                           model.encoder.parameters()):
        assert torch.allclose(p_t, 0.9 * b + 0.1 * p_o, atol=1e-6)
    model.update_target(1.0)                                           # tau=1: frozen
    after = [p.detach().clone() for p in model.target_encoder.parameters()]
    model.update_target(1.0)
    assert all(torch.equal(a, p) for a, p in zip(after, model.target_encoder.parameters()))


def test_trainable_parameters_exclude_target():
    model, _, _ = _model()
    ids = {id(p) for p in model.trainable_parameters()}
    assert all(id(p) not in ids for p in model.target_encoder.parameters())
    assert all(id(p) in ids for p in model.predictor.parameters())
    # only the ONLINE encoder is what gets saved: same keys as a bare encoder
    assert set(model.encoder.state_dict()) == set(model.target_encoder.state_dict())
