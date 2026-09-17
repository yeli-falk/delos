"""Unit tests for the GraphMAE (static generative SSL) baseline."""
import torch

from src.baselines.graphmae.graphmae import GraphMAE, sample_mask, sce_loss
from src.world_models.graph_jepa_encoder import build_encoder


def _ring(n=8):
    adj = torch.zeros(n, n)
    for i in range(n):
        adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = 1.0
    return adj


def test_sce_loss_zero_on_identical_and_scaled_by_gamma():
    x = torch.randn(5, 7)
    assert sce_loss(x, x, 2.0).item() < 1e-6
    assert sce_loss(x, 3.0 * x, 2.0).item() < 1e-6          # cosine: scale-free
    y = -x                                                  # cos = -1 -> (2)^gamma
    assert abs(sce_loss(x, y, 1.0).item() - 2.0) < 1e-5
    assert abs(sce_loss(x, y, 2.0).item() - 4.0) < 1e-5


def test_sample_mask_respects_active_and_rates():
    active = torch.zeros(100, dtype=torch.bool)
    active[:60] = True
    g = torch.Generator().manual_seed(0)
    masked, token, replaced = sample_mask(active, 0.5, 0.1, g)
    assert int(masked.sum()) == 30                            # 50% of the 60 active
    assert not masked[60:].any()                              # never an inactive node
    assert torch.equal(masked, token | replaced)
    assert not (token & replaced).any()
    assert int(replaced.sum()) == 3                           # 10% of the masked


def test_forward_remasks_and_backprops():
    torch.manual_seed(0)
    n, f_in, d = 8, 5, 16
    enc = build_encoder("gcn", in_dim=f_in, hidden_dim=8, out_dim=d, freeze=False)
    model = GraphMAE(enc, in_dim=f_in, latent_dim=d, mask_rate=0.5,
                     replace_rate=0.0, gamma=2.0)
    x, adj = torch.randn(n, f_in), _ring(n)
    active = torch.ones(n, dtype=torch.bool)
    g = torch.Generator().manual_seed(1)
    # corrupt(): masked nodes carry exactly the (zero-init) token
    x_c, masked = model.corrupt(x, active, g)
    assert int(masked.sum()) == 4
    assert torch.equal(x_c[masked], torch.zeros(4, f_in))
    assert torch.equal(x_c[~masked], x[~masked])
    loss = model(x, adj, active, torch.Generator().manual_seed(1))
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert model.mask_token.grad is not None
    assert model.decoder.lin.weight.grad is not None


def test_semantic_target_slices_decoder_output():
    torch.manual_seed(0)
    n, f_in, d = 8, 10, 16
    enc = build_encoder("gcn", in_dim=f_in, hidden_dim=8, out_dim=d, freeze=False)
    model = GraphMAE(enc, in_dim=f_in, latent_dim=d)
    x, adj = torch.randn(n, f_in), _ring(n)
    active = torch.ones(n, dtype=torch.bool)
    loss = model(x, adj, active, torch.Generator().manual_seed(2), target=x[:, 8:])
    assert torch.isfinite(loss)
