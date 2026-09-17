"""GRACE baseline (Deep Graph Contrastive Representation Learning, Zhu et
al., 2020) -- the STATIC self-supervised control."""
from src.baselines.grace.grace import (
    GRACE,
    GRACEProjection,
    drop_edges,
    drop_features,
    grace_loss,
    snapshot_tensors,
)

__all__ = [
    "GRACE",
    "GRACEProjection",
    "drop_edges",
    "drop_features",
    "grace_loss",
    "snapshot_tensors",
]
