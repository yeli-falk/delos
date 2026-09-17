"""STD-MAE baseline: spatial-temporal-decoupled masked pre-training.

Reference: Gao, Hu, Chen, Zhang, Xu, Zhang, "Spatial-Temporal-Decoupled Masked
Pre-training for Spatiotemporal Forecasting", IJCAI 2024.
Original implementation: https://github.com/Jimmy-7664/STD-MAE (BasicTS /
EasyTorch).

The port lives in three modules:

  * :mod:`src.baselines.stdmae.stdmae`  -- the architecture (S-MAE / T-MAE
    masked autoencoders + the Graph WaveNet forecasting backend), lifted from
    the reference with its BasicTS/EasyTorch, ``timm`` and
    ``positional_encodings`` dependencies removed.
  * :mod:`src.baselines.long_history`   -- the long-history plumbing, shared
    with STEP. STD-MAE needs a much longer input window than the forecast
    history (3 days on METR-LA), which this repo's per-split window datasets do
    not carry.
  * ``pretrain_stdmae.py`` / ``train_stdmae.py`` -- the two training stages.

See ``scripts/regression/README.md`` for how the arm is run and scored.
"""
