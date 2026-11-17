"""Universal discrete-time dynamic-graph (DTDG) data pipeline for DELOS.

This subpackage holds the shared machinery that every dynamic-graph dataset in
the project reuses, so the per-dataset files in ``src/custom_codes`` shrink to
thin parsers (raw file -> canonical representation -> :class:`DTDGDataset`):

  - :mod:`.canonical`  -- intermediate dataclasses (``SnapshotSequence``,
                          ``EdgeStream``, ``GraphPayload``).
  - :mod:`.features`   -- per-(t, node) feature engineering (graph stats,
                          edge-feature aggregation, static descriptors).
  - :mod:`.binning`    -- CTDG event stream -> DTDG snapshots (top-k filtering,
                          equal-width binning, symmetrisation).
  - :mod:`.dataset`    -- :class:`DTDGDataset`, the universal sliding-window
                          container emitting the canonical batch dict.
"""
from __future__ import annotations

from src.custom_codes.dtdg.canonical import (
    EdgeStream,
    GraphPayload,
    SnapshotSequence,
)
from src.custom_codes.dtdg.dataset import DTDGDataset

__all__ = [
    "EdgeStream",
    "GraphPayload",
    "SnapshotSequence",
    "DTDGDataset",
]
