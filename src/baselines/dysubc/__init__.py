"""DySubC baseline (Self-Supervised Dynamic Graph Representation Learning via
Temporal Subgraph Contrast, Chen et al., ACM TKDD'23): temporal subgraph
sampling + time-weighted / unweighted subgraph encoders + margin losses."""
from src.baselines.dysubc.dysubc import (
    DySubC,
    TemporalGraph,
    build_temporal_graph,
    margin_loss,
    sample_temporal_subgraphs,
)

__all__ = [
    "DySubC",
    "TemporalGraph",
    "build_temporal_graph",
    "margin_loss",
    "sample_temporal_subgraphs",
]
