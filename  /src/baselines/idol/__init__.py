"""IDOL baseline (Topology-monitorable Contrastive Learning on Dynamic Graphs,
Zhu et al., KDD'24): InstantGNN dynamic PPR propagation + GGD group
discrimination."""
from src.baselines.idol.idol import (
    GGD,
    ProjectionMLP,
    ggd_group_loss,
)

__all__ = [
    "GGD",
    "ProjectionMLP",
    "ggd_group_loss",
]
