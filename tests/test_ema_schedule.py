"""EMA target: anneal horizon and the encoder-freeze pause.

Guards two properties behind DELOS's ema_cos arm and the encoder-freeze
knob:
  * the cosine anneal runs over ``total_steps`` updates and reaches
    ``momentum_end`` there, regardless of how long the run then continues
    (train_delos maps --ema_anneal_epochs onto total_steps);
  * a paused target neither moves nor advances its anneal clock, so the
    schedule starts when the encoder is unfrozen.
"""
import copy
import math

import torch

from src.train.jepa import EMAEncoder
from src.baselines.bgrl.bgrl import ema_momentum


def _enc():
    torch.manual_seed(0)
    return torch.nn.Linear(4, 3)


def test_cosine_reaches_end_at_horizon():
    online = _enc()
    ema = EMAEncoder(online, momentum=0.99, momentum_end=1.0, total_steps=100)
    assert math.isclose(ema.current_momentum(), 0.99)
    for _ in range(50):
        ema.update(online)
    assert math.isclose(ema.current_momentum(), 0.995, abs_tol=1e-9)
    for _ in range(50):
        ema.update(online)
    assert math.isclose(ema.current_momentum(), 1.0)
    for _ in range(25):                      # run continues past the horizon
        ema.update(online)
    assert math.isclose(ema.current_momentum(), 1.0)


def test_paused_target_neither_moves_nor_ticks():
    online = _enc()
    ema = EMAEncoder(online, momentum=0.9, momentum_end=1.0, total_steps=10)
    before = copy.deepcopy(ema.target.state_dict())
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
    ema.paused = True
    for _ in range(5):
        ema.update(online)
    assert ema.step == 0
    for k, v in ema.target.state_dict().items():
        assert torch.equal(v, before[k])
    ema.paused = False
    ema.update(online)
    assert ema.step == 1
    for k, v in ema.target.state_dict().items():
        assert not torch.equal(v, before[k])


def test_bgrl_anneal_uses_its_own_horizon():
    steps = 20
    assert math.isclose(ema_momentum(50 * steps, 50 * steps, 0.99), 1.0)
    # whole-run horizon at the same step: nowhere near 1 (the bug)
    assert ema_momentum(50 * steps, 100 * steps, 0.99) < 0.996
