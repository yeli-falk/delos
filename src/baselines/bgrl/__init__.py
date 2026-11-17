"""BGRL baseline (Large-Scale Representation Learning on Graphs via
Bootstrapping, Thakoor et al., ICLR'22) -- the STATIC negative-free control."""
from src.baselines.bgrl.bgrl import BGRL, BGRLPredictor, bgrl_loss, ema_momentum

__all__ = ["BGRL", "BGRLPredictor", "bgrl_loss", "ema_momentum"]
