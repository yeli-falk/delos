"""CTDG event stream -> DTDG snapshot binning.

Consolidates the near-identical binning code that was copy-pasted between
``jodie_dataset`` (bipartite user x item) and ``reddit_dataset`` (homogeneous
subreddit x subreddit). Both:

  1. select a top-K active node subset,
  2. remap surviving events to dense node ids,
  3. bin event timestamps into ``num_bins`` equal-width snapshots,
  4. OR-reduce events per bin into a symmetric bool adjacency + active mask,
  5. (optionally) mean-pool per-edge features onto (bin, node) positions.

Steps 3-4 are shared (:func:`equal_width_bins`, :func:`scatter_snapshots`);
the only real difference is the top-K / remap step, captured by the two
public entry points :func:`bin_bipartite` and :func:`bin_homogeneous`.
Numerics are identical to the originals.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.custom_codes.dtdg.features import aggregate_edge_features_to_nodes


# ---------------------------------------------------------------------------
# Node vocabulary
# ---------------------------------------------------------------------------
def bipartite_node_ids(users: np.ndarray,
                       items: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sorted unique (user_ids, item_ids) of a bipartite stream.

    Every node is kept: bipartite datasets (JODIE Wikipedia / Reddit / MOOC)
    are always used at their full size. Dense node indices are the position
    in these sorted arrays (users first, then items).
    """
    return np.unique(users), np.unique(items)


def top_k_combined(sources: np.ndarray, targets: np.ndarray,
                   top_k: int) -> np.ndarray:
    """Top-K nodes by combined source+target appearance count (shared vocab).

    Combined because some nodes are mostly sources and some mostly targets;
    ranking by either alone misses half the picture.
    """
    all_ids = np.concatenate([sources, targets])
    unique, counts = np.unique(all_ids, return_counts=True)
    return unique[np.argsort(counts)[::-1][:top_k]]


# ---------------------------------------------------------------------------
# Shared binning core
# ---------------------------------------------------------------------------
def equal_width_bins(times_k: np.ndarray, num_bins: int) -> np.ndarray:
    """Discretise timestamps into ``num_bins`` equal-width bins over [min, max]."""
    t_lo, t_hi = float(times_k.min()), float(times_k.max())
    span = max(t_hi - t_lo, 1.0)
    return np.clip(
        ((times_k - t_lo) / span * num_bins).astype(np.int64),
        0, num_bins - 1,
    )


def scatter_snapshots(src_idx: np.ndarray, dst_idx: np.ndarray,
                      bin_idx: np.ndarray, num_nodes: int,
                      num_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Scatter dense-indexed events into symmetric bool snapshots.

    Returns (adj (T,N,N) bool symmetric no-self-loops, active (T,N) bool).
    Boolean fancy indexing is idempotent so duplicate events collapse to a
    single bool edge per (bin, i, j).
    """
    adj = np.zeros((num_bins, num_nodes, num_nodes), dtype=bool)
    active = np.zeros((num_bins, num_nodes), dtype=bool)
    sl = src_idx != dst_idx                      # drop accidental self-loops
    b = bin_idx[sl]; si = src_idx[sl]; ti = dst_idx[sl]
    adj[b, si, ti] = True
    adj[b, ti, si] = True
    active[b, si] = True
    active[b, ti] = True
    return adj, active


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def bin_bipartite(
    users: np.ndarray, items: np.ndarray, times: np.ndarray, num_bins: int,
    edge_features: Optional[np.ndarray] = None,
    normalize_edge_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Bin a bipartite (user, item) CTDG stream into DTDG snapshots.

    The whole stream is used (every user and item). Items occupy node indices
    [n_users, n_users + n_items). Returns (adj, active, node_edge_feats) where
    the third entry is the per-(bin, node) mean of ``edge_features`` if
    provided, else ``None``.
    """
    keep_u_arr, keep_i_arr = bipartite_node_ids(users, items)
    n_users = len(keep_u_arr)
    n_items = len(keep_i_arr)
    N = n_users + n_items

    u_in = np.isin(users, keep_u_arr)
    i_in = np.isin(items, keep_i_arr)
    keep_mask = u_in & i_in
    users_k = users[keep_mask]; items_k = items[keep_mask]; times_k = times[keep_mask]
    if times_k.size == 0:
        raise ValueError(
            "JODIE stream has no events after parsing "
            f"(unique users={len(np.unique(users))}, "
            f"unique items={len(np.unique(items))}); inspect "
            "dataset/jodie/<dataset>.npz with np.load."
        )
    print(f"[jodie] kept {len(users_k)}/{len(users)} events "
          f"({100*len(users_k)/len(users):.1f}%)  "
          f"unique_u={len(keep_u_arr)} unique_i={len(keep_i_arr)}",
          flush=True)

    u_idx = np.searchsorted(keep_u_arr, users_k)
    i_idx = np.searchsorted(keep_i_arr, items_k) + n_users
    bin_idx = equal_width_bins(times_k, num_bins)
    adj, active = scatter_snapshots(u_idx, i_idx, bin_idx, N, num_bins)

    node_feats = None
    if edge_features is not None:
        feats_k = edge_features[keep_mask]
        sl = u_idx != i_idx
        node_feats = aggregate_edge_features_to_nodes(
            feats_k[sl], u_idx[sl], i_idx[sl], bin_idx[sl],
            num_bins=num_bins, num_nodes=N,
            normalize=normalize_edge_features,
        )
    return adj, active, node_feats


def bin_homogeneous(
    sources: np.ndarray, targets: np.ndarray, times: np.ndarray,
    top_k: int, num_bins: int,
    edge_features: Optional[np.ndarray] = None,
    normalize_edge_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Bin a homogeneous (src, dst) CTDG stream into DTDG snapshots.

    Source and target share one node vocabulary. Returns (adj, active,
    node_edge_feats) as :func:`bin_bipartite`.
    """
    keep_sorted = np.sort(top_k_combined(sources, targets, top_k))
    N = len(keep_sorted)

    s_in = np.isin(sources, keep_sorted)
    t_in = np.isin(targets, keep_sorted)
    keep_mask = s_in & t_in
    src_k = sources[keep_mask]; tgt_k = targets[keep_mask]; times_k = times[keep_mask]
    if times_k.size == 0:
        raise ValueError(
            "After top-K filtering no events remain.\n"
            f"  total events:         {len(sources)}\n"
            f"  events with kept src: {int(s_in.sum())}\n"
            f"  events with kept tgt: {int(t_in.sum())}\n"
            f"  unique nodes in data: {len(np.unique(np.concatenate([sources, targets])))}\n"
            f"  kept top:             {top_k}\n"
            "Raise --reddit_top_k."
        )

    src_idx = np.searchsorted(keep_sorted, src_k)
    tgt_idx = np.searchsorted(keep_sorted, tgt_k)
    bin_idx = equal_width_bins(times_k, num_bins)
    adj, active = scatter_snapshots(src_idx, tgt_idx, bin_idx, N, num_bins)

    print(f"[reddit] kept {len(src_k)}/{len(sources)} events "
          f"({100*len(src_k)/len(sources):.1f}%)  N={N}  bins={num_bins}",
          flush=True)

    node_feats = None
    if edge_features is not None:
        feats_k = edge_features[keep_mask]
        sl = src_idx != tgt_idx
        node_feats = aggregate_edge_features_to_nodes(
            feats_k[sl], src_idx[sl], tgt_idx[sl], bin_idx[sl],
            num_bins=num_bins, num_nodes=N,
            normalize=normalize_edge_features,
        )
    return adj, active, node_feats
