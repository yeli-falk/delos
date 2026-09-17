"""CLDG baseline (Contrastive Learning on Dynamic Graphs, Xu et al., ICDE'23)."""
from src.baselines.cldg.cldg import (
    CLDG,
    ProjectionHead,
    nt_xent_loss,
    sampling_layer,
    view_arrays,
)

__all__ = [
    "CLDG",
    "ProjectionHead",
    "nt_xent_loss",
    "sampling_layer",
    "view_arrays",
]
