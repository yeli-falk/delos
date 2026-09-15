"""JODIE Reddit / Wikipedia binned into discrete-time dynamic graphs.

JODIE (Kumar et al., 2019) ships two of the most-benchmarked CTDG datasets:

  - Reddit:   user x subreddit interactions, 10k users, 1k subreddits,
              ~672k events over one month, 172-dim LIWC edge features.
  - Wikipedia: editor x page edits, ~8.2k editors, 1k pages, ~157k events,
              same edge feature dim.
  - MOOC:     student x course-item interactions, ~7k students, 97 items,
              ~412k events, 4-dim edge features, 4,066 dropout state-change
              events — by far the densest state-label signal of the three
              (reddit has ~366 bans, wikipedia ~217).

Both are CTDG (continuous-time event streams). DyG-JEPA expects DTDG
(per-snapshot adjacency + active mask + node features). We bridge the two by
binning timestamps into `num_bins` equal-width windows (see
:mod:`src.custom_codes.dtdg.binning`). Top-K most-active entities are kept to
bound the dense (T, N, N) representation, and optional LIWC edge features are
mean-pooled onto (bin, node) positions.

CSV format expected (matches Stanford's JODIE release exactly):

    user_id,item_id,timestamp,state_label,comma_separated_172_features

Download once via:
    wget http://snap.stanford.edu/jodie/reddit.csv    -O dataset/jodie/reddit.csv
    wget http://snap.stanford.edu/jodie/wikipedia.csv -O dataset/jodie/wikipedia.csv

This file holds only JODIE-specific parsing (CSV -> event arrays + optional
edge features / state labels); binning, feature engineering, windowing and
splitting all live in the universal DTDG pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from src.custom_codes.dtdg.binning import bin_bipartite, top_k_bipartite
from src.custom_codes.dtdg.canonical import SnapshotSequence
from src.custom_codes.dtdg.dataset import DTDGDataset
from src.custom_codes.dtdg.features import build_graph_features


JODIE_URLS = {
    "reddit": "http://snap.stanford.edu/jodie/reddit.csv",
    "wikipedia": "http://snap.stanford.edu/jodie/wikipedia.csv",
    "mooc": "http://snap.stanford.edu/jodie/mooc.csv",
}


def _jodie_num_feats(path: Path) -> int:
    """Count the per-event feature columns by sniffing the first data row.

    Reddit/Wikipedia carry 172 LIWC features, MOOC only 4 — the fixed
    4 leading columns (user, item, timestamp, state_label) are shared, so
    n_feats = n_fields - 4.
    """
    with open(path) as f:
        f.readline()                              # discard header
        line = f.readline()
    n_fields = line.count(",") + 1
    return max(n_fields - 4, 0)


def _jodie_names(path: Path) -> list:
    return (["user_id", "item_id", "timestamp", "state_label"]
            + [f"f{k}" for k in range(_jodie_num_feats(path))])


def _parse_jodie_csv_with_state_labels(path: Path) -> tuple:
    """Parse JODIE CSV and additionally return per-event state_label (col 3).

    The JODIE state_label is 1 at the event where a user undergoes a state
    change (e.g. ban) and 0 otherwise — see Kumar et al. 2019. Useful for
    dynamic node classification, the standard JODIE benchmark task.

    Returns (users, items, times, state_labels). Cached as .state.npz.
    """
    cache = path.with_suffix(".state.npz")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        d = np.load(cache)
        return d["users"], d["items"], d["times"], d["states"]

    import pandas as pd
    df = pd.read_csv(
        path, header=0, names=_jodie_names(path),
        usecols=["user_id", "item_id", "timestamp", "state_label"],
        dtype={"user_id": str, "item_id": str,
               "timestamp": np.float64, "state_label": np.float64},
        engine="c", low_memory=False,
    )
    df = df.dropna(subset=["user_id", "item_id", "timestamp", "state_label"])
    users_codes, _ = pd.factorize(df["user_id"])
    items_codes, _ = pd.factorize(df["item_id"])
    users = users_codes.astype(np.int64)
    items = items_codes.astype(np.int64)
    times = df["timestamp"].to_numpy(dtype=np.float64)
    states = df["state_label"].to_numpy(dtype=np.int64)

    n_state_change = int((states > 0).sum())
    print(f"[jodie] parsed {len(users)} events + state_label from {path.name}: "
          f"{n_state_change} state-change events", flush=True)
    np.savez(cache, users=users, items=items, times=times, states=states)
    return users, items, times, states


def _parse_jodie_csv_with_features(path: Path) -> tuple:
    """Same as `_parse_jodie_csv` but also returns the 172-dim LIWC features.

    Returns (users, items, times, features) where features is (E, 172). Has
    its own cache file (suffix `.feat.npz`) so the no-features call still
    hits the smaller cache.
    """
    cache = path.with_suffix(".feat.npz")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        d = np.load(cache)
        return d["users"], d["items"], d["times"], d["features"]

    import pandas as pd
    names = _jodie_names(path)
    df = pd.read_csv(
        path, header=0, names=names, engine="c", low_memory=False,
        dtype={"user_id": str, "item_id": str, "timestamp": np.float64},
    )
    df = df.dropna(subset=["user_id", "item_id", "timestamp"])
    users_codes, _ = pd.factorize(df["user_id"])
    items_codes, _ = pd.factorize(df["item_id"])
    users = users_codes.astype(np.int64)
    items = items_codes.astype(np.int64)
    times = df["timestamp"].to_numpy(dtype=np.float64)
    feat_cols = names[4:]
    features = df[feat_cols].to_numpy(dtype=np.float32)

    print(f"[jodie] parsed {len(users)} events + {features.shape[1]}-dim "
          f"LIWC features from {path.name}", flush=True)
    np.savez(cache, users=users, items=items, times=times, features=features)
    return users, items, times, features


def _parse_jodie_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a JODIE CSV. Returns (user_ids, item_ids, timestamps) as 1-D arrays.

    JODIE files have a 5-field header line ("user_id,item_id,timestamp,
    state_label,comma_separated_list_of_features") followed by data rows
    with 1 user + 1 item + 1 ts + 1 state + 172 features = 176 fields each.
    Pandas mis-handles this header/body mismatch under several flag
    combinations (silently NaNs every row, or raises). We sidestep the
    inference by passing an explicit `names=` schema covering all 176
    columns and using `header=0` to discard the original header line.

    IDs are treated as opaque strings then factorized to dense ints — that's
    what the upstream JODIE loader does, and it's robust to both numeric
    and non-numeric IDs.

    Cached as .npz alongside the CSV; later runs skip the parse entirely.
    """
    cache = path.with_suffix(".npz")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        d = np.load(cache)
        return d["users"], d["items"], d["times"]

    users = items = times = None
    try:
        import pandas as pd
        df = pd.read_csv(
            path, header=0, names=_jodie_names(path),
            usecols=["user_id", "item_id", "timestamp"],
            dtype={"user_id": str, "item_id": str, "timestamp": np.float64},
            engine="c", low_memory=False,
        )
        df = df.dropna(subset=["user_id", "item_id", "timestamp"])
        if len(df) > 0:
            users_codes, _ = pd.factorize(df["user_id"])
            items_codes, _ = pd.factorize(df["item_id"])
            users = users_codes.astype(np.int64)
            items = items_codes.astype(np.int64)
            times = df["timestamp"].to_numpy(dtype=np.float64)
    except Exception as e:
        print(f"[jodie] pandas parse failed ({e!r}); falling back to csv module",
              flush=True)

    if users is None or len(users) == 0:
        # Failsafe: stdlib csv. ~30-60s on the 2.3 GB Reddit file, one-time
        # cost since the npz cache absorbs it for every subsequent run.
        import csv
        u_strs, i_strs, t_floats = [], [], []
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)                                # discard header
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    t = float(row[2])
                except (ValueError, TypeError):
                    continue
                u_strs.append(row[0]); i_strs.append(row[1]); t_floats.append(t)
        if not u_strs:
            raise ValueError(f"Parsed zero rows from {path}; CSV malformed?")
        u_arr = np.array(u_strs); i_arr = np.array(i_strs)
        _, users = np.unique(u_arr, return_inverse=True)
        _, items = np.unique(i_arr, return_inverse=True)
        users = users.astype(np.int64); items = items.astype(np.int64)
        times = np.asarray(t_floats, dtype=np.float64)

    print(f"[jodie] parsed {len(users)} events from {path.name}: "
          f"{len(np.unique(users))} unique users, "
          f"{len(np.unique(items))} unique items", flush=True)
    np.savez(cache, users=users, items=items, times=times)
    return users, items, times


def _state_label_vector(path: Path, num_nodes: int, top_k_users: Optional[int],
                        top_k_items: Optional[int]) -> np.ndarray:
    """Per-node binary 'ever underwent a state change' label, JODIE's canonical
    dynamic-node-classification target. Users in the top-K subset (every user
    when top_k_users is 0/None) get 0/1; everyone else (and all items) gets -1
    (excluded from the probe)."""
    users_full, items_full, _, states_full = _parse_jodie_csv_with_state_labels(path)
    keep_u, _ = top_k_bipartite(users_full, items_full, top_k_users, top_k_items)
    keep_u_arr = np.array(sorted(keep_u), dtype=np.int64)
    in_subset = np.isin(users_full, keep_u_arr)
    users_k = users_full[in_subset]
    states_k = states_full[in_subset]
    u_idx = np.searchsorted(keep_u_arr, users_k)
    n_users = len(keep_u_arr)
    labels = np.full(num_nodes, -1, dtype=np.int64)
    labels[:n_users] = 0                # users seen, none flagged yet
    np.maximum.at(labels[:n_users], u_idx, states_k)   # OR-reduce ban flags
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"[jodie] state_labels: pos={n_pos}  neg={n_neg}  "
          f"excluded={(labels < 0).sum()}", flush=True)
    return labels


def _state_label_matrix(path: Path, num_nodes: int, num_bins: int,
                        top_k_users: Optional[int], top_k_items: Optional[int],
                        mode: str = "event") -> np.ndarray:
    """Per-(bin, node) binary state-change label — the *dynamic* JODIE target.

    Unlike :func:`_state_label_vector`, which OR-reduces every user's ban flags
    into one static bit, this keeps the time axis: it returns a
    ``(num_bins, num_nodes)`` matrix so a probe can ask a different question at
    each snapshot. Kept users get 0/1; excluded users and all items get -1 at
    every bin (filtered out downstream).

      mode='event':  Y[t, u] = 1 iff a state_label==1 event for user u falls in
                     snapshot t. The faithful dynamic benchmark target (a ban
                     is localised to the bin where it happens); very sparse.
      mode='state':  Y[t, u] = 1 for every bin at or after u's first flagged
                     bin (cumulative 'currently banned' state); denser, still
                     time-varying (0 before the ban, 1 after).

    The bins reproduce the graph's exact edges: equal-width over the timestamps
    of events where BOTH endpoints are kept (matching ``bin_bipartite``), so
    ``Y[t]`` aligns with snapshot ``t`` of ``ds.adj``.
    """
    assert mode in ("event", "state"), f"unknown state_label_mode {mode!r}"
    users_full, items_full, times_full, states_full = \
        _parse_jodie_csv_with_state_labels(path)
    keep_u, keep_i = top_k_bipartite(users_full, items_full,
                                     top_k_users, top_k_items)
    keep_u_arr = np.array(sorted(keep_u), dtype=np.int64)
    keep_i_arr = np.array(sorted(keep_i), dtype=np.int64)
    n_users = len(keep_u_arr)

    # Reproduce the graph's bin edges from the both-endpoints-kept subset.
    keep_mask = np.isin(users_full, keep_u_arr) & np.isin(items_full, keep_i_arr)
    times_k = times_full[keep_mask]
    if times_k.size == 0:
        raise ValueError("no kept events; cannot bin dynamic state labels")
    t_lo = float(times_k.min())
    span = max(float(times_k.max()) - t_lo, 1.0)

    labels = np.full((num_bins, num_nodes), -1, dtype=np.int64)
    labels[:, :n_users] = 0                              # kept users default 0

    # Positive state-change events for kept users, binned by their timestamp
    # with the *same* edges (independent of whether that interaction's item was
    # kept, so a ban on a non-kept item still lands in the right snapshot).
    pos = (states_full == 1) & np.isin(users_full, keep_u_arr)
    pos_u = np.searchsorted(keep_u_arr, users_full[pos])           # 0..n_users-1
    pos_bin = np.clip(((times_full[pos] - t_lo) / span * num_bins).astype(np.int64),
                      0, num_bins - 1)

    if mode == "event":
        labels[pos_bin, pos_u] = 1
    else:  # 'state' — cumulative from the first flagged bin onward
        first_bin = np.full(n_users, num_bins, dtype=np.int64)
        np.minimum.at(first_bin, pos_u, pos_bin)
        banned = np.arange(num_bins)[:, None] >= first_bin[None, :]  # (T, n_users)
        labels[:, :n_users] = np.where(banned, 1, 0)

    n_pos = int((labels == 1).sum())
    per_bin = (labels == 1).sum(axis=1)
    print(f"[jodie] dynamic state_labels (mode={mode}): "
          f"pos_cells={n_pos}  bins_with_pos={int((per_bin > 0).sum())}/{num_bins}  "
          f"max_per_bin={int(per_bin.max())}  kept_users={n_users}", flush=True)
    return labels


class JODIEDynamicGraphDataset(Dataset):
    """JODIE Reddit or Wikipedia binned into DTDG snapshots.

    Same return signature as DBLPDynamicGraphDataset so it drops into the
    existing train / pretrain / diagnose / ablate scripts unchanged. Thin
    wrapper over the universal :class:`DTDGDataset`.

    Args:
      data_path: path to the JODIE CSV. If None, falls back to
                 dataset/jodie/<dataset>.csv.
      dataset:   "reddit", "wikipedia" or "mooc" (controls default path +
                 cache key).
      num_bins:  number of DTDG snapshots to bin events into (default 30).
      top_k_users / top_k_items: keep only the most-active K of each side.
                 0 / None (the default) keeps every node, i.e. the dataset
                 is used at its full size.
      history_size / num_preds: window lengths.
      split:     "train" | "val" | "test", chronological 70/10/20.
    """

    SPLITS: Tuple[float, float, float] = (0.7, 0.1, 0.2)

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: str = "train",
        history_size: int = 3,
        num_preds: int = 1,
        dataset: str = "reddit",
        num_bins: int = 30,
        top_k_users: Optional[int] = 0,
        top_k_items: Optional[int] = 0,
        feat_dim: int = 8,
        stride: int = 1,
        seed: int = 42,
        with_edge_features: bool = False,
        normalize_edge_features: bool = True,
        with_state_labels: bool = False,
        state_label_mode: str = "ever",
        split_mode: str = "chronological",
    ):
        assert split in ("train", "val", "test")
        assert dataset in JODIE_URLS, f"unknown jodie dataset {dataset!r}"
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.window = history_size + num_preds
        self.stride = stride
        self.dataset = dataset
        self.with_edge_features = with_edge_features

        if data_path is None:
            data_path = f"dataset/jodie/{dataset}.csv"
        path = Path(data_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"JODIE {dataset} CSV not found at {path}. Download via:\n"
                f"  wget {JODIE_URLS[dataset]} -O {path}"
            )

        if with_edge_features:
            users, items, times, edge_feats = _parse_jodie_csv_with_features(path)
        else:
            users, items, times = _parse_jodie_csv(path)
            edge_feats = None
        adj, active, node_edge_feats = bin_bipartite(
            users, items, times, top_k_users=top_k_users,
            top_k_items=top_k_items, num_bins=num_bins,
            edge_features=edge_feats,
            normalize_edge_features=normalize_edge_features,
        )

        self.num_nodes = adj.shape[1]
        graph_feats = build_graph_features(adj, active, feat_dim=feat_dim, seed=seed)
        if node_edge_feats is not None:
            feats = np.concatenate([graph_feats, node_edge_feats], axis=-1)
            self.feat_dim = graph_feats.shape[-1] + node_edge_feats.shape[-1]
        else:
            feats = graph_feats
            self.feat_dim = feat_dim

        # State-change labels (optional). 'ever' keeps the legacy static
        # per-node "ever banned" bit; 'event'/'state' additionally build a
        # per-(bin, node) dynamic matrix on ds.node_labels_dyn while still
        # exposing an OR-reduced static vector on ds.node_labels for backward
        # compatibility with compute_block_labels(source="dataset").
        self.node_labels = None
        self.node_labels_dyn = None
        self.state_label_mode = state_label_mode
        if with_state_labels:
            if state_label_mode == "ever":
                self.node_labels = _state_label_vector(
                    path, self.num_nodes, top_k_users, top_k_items)
            else:
                self.node_labels_dyn = _state_label_matrix(
                    path, self.num_nodes, num_bins, top_k_users, top_k_items,
                    mode=state_label_mode)
                # Static compat vector = ever-flagged (max over bins keeps -1
                # for excluded, OR-reduces 0/1 for kept users).
                self.node_labels = self.node_labels_dyn.max(axis=0)

        seq = SnapshotSequence(adj=adj, active=active, feats=feats,
                               node_labels=self.node_labels)
        self._ds = DTDGDataset.from_global(
            seq, split=split, history_size=history_size, num_preds=num_preds,
            stride=stride, splits=self.SPLITS,
            split_mode=split_mode, split_seed=seed,
        )
        self.starts = self._ds.starts

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]

    @property
    def feature_dim(self) -> int:
        return self.feat_dim
