"""GraphMAE baseline (Self-Supervised Masked Graph Autoencoders, Hou et al.,
KDD'22) -- the STATIC generative self-supervised control."""
from src.baselines.graphmae.graphmae import (
    DenseGCNDecoder,
    GraphMAE,
    sample_mask,
    sce_loss,
)

__all__ = ["DenseGCNDecoder", "GraphMAE", "sample_mask", "sce_loss"]
