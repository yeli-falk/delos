"""DyG2Vec baseline (Efficient Representation Learning for Dynamic Graphs,
Alomrani et al., TMLR'24): window-based temporal attention encoder
(``--encoder_kind dyg2vec``) + VICReg on two distorted views of each window."""
from src.baselines.dyg2vec.dyg2vec import (
    DyG2VecSSL,
    Predictor,
    symmetric_edge_mask,
    vicreg_loss,
)

__all__ = ["DyG2VecSSL", "Predictor", "symmetric_edge_mask", "vicreg_loss"]
