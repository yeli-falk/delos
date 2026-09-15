"""STEP baseline: pre-training-enhanced spatial-temporal graph neural network.

Reference: Shao, Zhang, Wang, Xu, "Pre-training Enhanced Spatial-temporal Graph
Neural Network for Multivariate Time Series Forecasting", SIGKDD 2022.
Original implementation: https://github.com/GestaltCogTeam/STEP (BasicTS /
EasyTorch), branch ``github``.

Ported the same way as the STD-MAE arm, and sharing its plumbing on purpose:

  * :mod:`src.baselines.step.step`      -- TSFormer (the masked pre-training
    model), DiscreteGraphLearning (the learned dependency graph) and the STEP
    wrapper, lifted from the reference with its BasicTS/EasyTorch, ``timm`` and
    hard-coded per-dataset constants removed.
  * :mod:`src.baselines.masked_pretrain` -- the patchifier / transformer stack /
    mask generator / ``masked_mae``, byte-identical in both papers.
  * :mod:`src.baselines.graphwavenet`   -- the Graph WaveNet backend, likewise.
  * :mod:`src.baselines.long_history`   -- the long-window plumbing, so STEP and
    STD-MAE read byte-identical inputs.
  * ``pretrain_tsformer.py`` / ``train_step.py`` -- the two training stages.

See ``scripts/regression/README.md`` for how the arm is run and scored.
"""
