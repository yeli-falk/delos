"""MaskDGNN baseline (Self-Supervised Dynamic GNNs with Activeness-aware
Temporal Masking, He et al., IJCAI'25): activeness-aware edge masking +
masked/next-edge BCE on a GCN with an adaptive frequency layer."""
from src.baselines.maskdgnn.maskdgnn import (
    ActivenessCache,
    EdgeMLP,
    MaskDGNN,
    activeness_edge_mask,
    dynamics_score,
    node_activeness,
    pagerank,
    sample_negative_pairs,
)

__all__ = [
    "ActivenessCache",
    "EdgeMLP",
    "MaskDGNN",
    "activeness_edge_mask",
    "dynamics_score",
    "node_activeness",
    "pagerank",
    "sample_negative_pairs",
]
