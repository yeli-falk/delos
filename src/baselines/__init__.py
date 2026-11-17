"""Baseline methods for comparison against DELOS.

Each baseline is self-contained and reuses the project's universal DTDG loader
(Block 1), encoder registry, and probe/eval harness (Block 4) so comparisons are
controlled (same data, same encoder architecture, same downstream probe). Only
the self-supervised objective differs.
"""
