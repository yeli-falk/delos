"""Per-(t, node) feature engineering for DELOS datasets.

Single home for the feature builders that used to live in (and be imported
across) the per-dataset files:

  - :func:`build_graph_features`        -- 8-dim local graph statistics, the
        base feature every dynamic-graph dataset uses. Formerly
        ``dblp_dataset._features_from_history``.
  - :func:`aggregate_edge_features_to_nodes` -- mean-pool per-edge features
        (JODIE LIWC, SNAP PROPERTIES) onto (bin, node) positions. Formerly in
        ``dblp_dataset``.
  - :func:`concept_static_features`     -- EHRshot vocab one-hot + log-freq.
        Formerly ``ehrshot_dataset._concept_static_features``.
  - :func:`time_of_day`                 -- fractional time-of-day, used by the
        METR-LA traffic features. Formerly ``metr_la_dataset._time_of_day``.

The numerics here are byte-for-byte identical to the originals; this module
only relocates them so a single edit changes the feature stack everywhere.
"""
from __future__ import annotations

import numpy as np

# EHRshot concept-vocabulary type cardinality; see
# scripts/dblp/preprocess_ehrshot.py VOCAB_TYPE_ID.
VOCAB_TYPE_DIM = 9


def build_graph_features(adj: np.ndarray, active: np.ndarray,
                         feat_dim: int = 8, seed: int = 0) -> np.ndarray:
    """Cheap per-(t, n) feature vector built from local graph stats.

      [degree, log(1+degree), is_new_this_year, mean_neighbour_degree,
       clustering_coef_proxy, year_index/T, sin(2pi*t/T), cos(2pi*t/T)]
    """
    T, N, _ = adj.shape
    feats = np.zeros((T, N, feat_dim), dtype=np.float32)
    deg = adj.sum(-1).astype(np.float32)
    for t in range(T):
        deg_t = deg[t]
        feats[t, :, 0] = deg_t / max(N, 1)
        feats[t, :, 1] = np.log1p(deg_t)
        feats[t, :, 2] = (active[t].astype(np.float32)
                          - (active[t - 1].astype(np.float32) if t > 0 else 0.0))
        # Mean neighbour degree (z-scored).
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_nb_deg = np.where(
                deg_t > 0, (adj[t] @ deg_t) / np.maximum(deg_t, 1), 0.0,
            )
        feats[t, :, 3] = mean_nb_deg / max(N, 1)
        feats[t, :, 4] = (deg_t > 0).astype(np.float32)
        feats[t, :, 5] = t / max(T - 1, 1)
        feats[t, :, 6] = np.sin(2 * np.pi * t / max(T, 1))
        feats[t, :, 7] = np.cos(2 * np.pi * t / max(T, 1))
    feats *= active[..., None].astype(np.float32)
    return feats


def aggregate_edge_features_to_nodes(
    edge_features: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    bin_idx: np.ndarray,
    num_bins: int,
    num_nodes: int,
    normalize: bool = True,
) -> np.ndarray:
    """Mean-pool per-edge features over the edges incident to each (bin, node).

    For each bin t and node i, the result is the average of `edge_features`
    over all edges that touched i during bin t (counting both as source and
    target). Useful for folding edge-level semantic features (e.g. JODIE's
    172-dim LIWC vectors) into the (T, N, F) node-feature tensor that the
    encoder expects.

    Args:
      edge_features: (E, F) float array of features per event/edge.
      sources, targets: (E,) int arrays of dense node indices.
      bin_idx:           (E,) int array, the temporal bin of each event.
      num_bins:          T
      num_nodes:         N
      normalize:         if True (default), z-score each output feature dim
                         using statistics computed over ACTIVE (b, n)
                         positions only. Required for stable MSE training:
                         JODIE LIWC and SNAP PROPERTIES have native per-dim
                         stds spanning O(1)..O(10^2), and the multi-seed
                         pretrain MSE loss becomes bimodal without this.
                         Inactive positions stay at zero post-normalisation,
                         preserving the "no edge -> zero feature" semantics.

    Returns:
      (T, N, F) float32 array of per-(bin, node) mean edge features.
    """
    F = edge_features.shape[1]
    feat_sum = np.zeros((num_bins, num_nodes, F), dtype=np.float64)
    count = np.zeros((num_bins, num_nodes), dtype=np.int64)
    # np.add.at is the scatter-add we need; in a loop over F because
    # add.at doesn't broadcast the last dim cleanly across all flavours.
    for k in range(F):
        np.add.at(feat_sum[..., k], (bin_idx, sources), edge_features[:, k])
        np.add.at(feat_sum[..., k], (bin_idx, targets), edge_features[:, k])
    np.add.at(count, (bin_idx, sources), 1)
    np.add.at(count, (bin_idx, targets), 1)
    safe = np.maximum(count, 1)[..., None].astype(np.float64)
    result = (feat_sum / safe).astype(np.float32)

    if normalize:
        active = (count > 0)                                  # (T, N) bool
        if active.any():
            # Z-score per-dim using only active positions, then zero out
            # inactive entries so "no edge present" stays distinguishable
            # from "edge present with feature = mean".
            for k in range(F):
                vals = result[active, k]
                if vals.size < 2:
                    continue
                m = float(vals.mean())
                s = float(vals.std()) + 1e-6
                result[..., k] = (result[..., k] - m) / s
            result *= active[..., None].astype(np.float32)
    return result


def concept_static_features(code_vocab_type: np.ndarray,
                            code_freq: np.ndarray,
                            num_nodes: int) -> np.ndarray:
    """Per-node static features: vocabulary type one-hot + log frequency.

    Shape: (N, VOCAB_TYPE_DIM + 1). Same for every patient, every t. The
    encoder treats these as time-invariant node descriptors; the dynamic
    structure comes from `active` and the predictor's masked-codes pretext.
    """
    feats = np.zeros((num_nodes, VOCAB_TYPE_DIM + 1), dtype=np.float32)
    feats[np.arange(num_nodes), code_vocab_type] = 1.0
    # log1p of frequency, normalised to [0, 1] for stability across vocabs
    f = np.log1p(code_freq.astype(np.float64))
    f = f / max(f.max(), 1.0)
    feats[:, VOCAB_TYPE_DIM] = f.astype(np.float32)
    return feats


def time_of_day(times: np.ndarray) -> np.ndarray:
    """Fractional time-of-day in [0, 1)."""
    nanos = times.astype("datetime64[ns]").astype("int64")
    seconds = (nanos // 1_000_000_000) % (24 * 3600)
    return (seconds / (24 * 3600)).astype(np.float32)
