"""Reddit subreddit-hyperlink graph (Kumar et al., 2018) binned into DTDG.

This is the OTHER "Reddit" dataset in the dynamic-graph SSL literature,
distinct from the bipartite user x subreddit JODIE Reddit:

  - JODIE Reddit:  user x subreddit interactions, bipartite, ~672k events
  - Reddit-Body:   subreddit x subreddit hyperlinks (from post body text),
                   homogeneous, signed edges, ~286k edges, ~36k subreddits,
                   2014-01-01 to 2017-04-30 (~3.3 years of data).

Format (SNAP soc-redditHyperlinks-body.tsv, tab-separated):

    SOURCE_SUBREDDIT \\t TARGET_SUBREDDIT \\t POST_ID \\t TIMESTAMP \\t LINK_SENTIMENT \\t PROPERTIES

  - SOURCE/TARGET are subreddit *names* (strings like 'leagueoflegends')
  - TIMESTAMP is 'YYYY-MM-DD HH:MM:SS'
  - LINK_SENTIMENT is +1 (positive) or -1 (negative)
  - PROPERTIES is a comma-separated vector of 86 LIWC + sentiment features

We ignore POST_ID and LINK_SENTIMENT; only (source, target, time) and the
optional 86-dim PROPERTIES vector are used. Source and target share one node
vocabulary (homogeneous graph), so factorisation is shared.

Download once via:
    wget https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv  -O dataset/reddit/soc-redditHyperlinks-body.tsv
    wget https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv -O dataset/reddit/soc-redditHyperlinks-title.tsv

This file holds only the SNAP-TSV parsing; binning, feature engineering,
windowing and splitting all live in the universal DTDG pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from src.custom_codes.dtdg.binning import bin_homogeneous
from src.custom_codes.dtdg.canonical import SnapshotSequence
from src.custom_codes.dtdg.dataset import DTDGDataset
from src.custom_codes.dtdg.features import build_graph_features


REDDIT_URLS = {
    "body":  "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv",
    "title": "https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv",
}


def _parse_reddit_tsv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a SNAP Reddit hyperlinks TSV.

    Returns (sources, targets, times) as int64/int64/float64 arrays.
    Subreddit names are factorised into a SHARED dense node vocabulary —
    source and target are both subreddits, so they live in one ID space.

    Cached as .npz alongside the TSV; subsequent loads skip the parse.
    """
    cache = path.with_suffix(".npz")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        d = np.load(cache)
        return d["sources"], d["targets"], d["times"]

    import pandas as pd
    df = pd.read_csv(
        path, sep="\t", header=0,
        usecols=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "TIMESTAMP"],
        dtype={"SOURCE_SUBREDDIT": str, "TARGET_SUBREDDIT": str,
               "TIMESTAMP": str},
    )
    df = df.dropna(subset=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "TIMESTAMP"])

    # Parse 'YYYY-MM-DD HH:MM:SS' -> int64 nanoseconds -> float64 seconds.
    ts = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    valid = ~ts.isna()
    df = df[valid]
    ts = ts[valid]
    times = ts.astype("int64").to_numpy() / 1e9

    # Shared vocabulary: factorise the union of source and target names.
    combined = pd.concat([df["SOURCE_SUBREDDIT"], df["TARGET_SUBREDDIT"]],
                         ignore_index=True)
    codes, uniques = pd.factorize(combined)
    n = len(df)
    sources = codes[:n].astype(np.int64)
    targets = codes[n:].astype(np.int64)

    print(f"[reddit] parsed {len(sources)} hyperlinks from {path.name}: "
          f"{len(uniques)} unique subreddits", flush=True)
    np.savez(cache, sources=sources, targets=targets, times=times)
    return sources, targets, times


def _parse_reddit_tsv_with_features(path: Path) -> tuple:
    """Same as `_parse_reddit_tsv` but also returns the 86-dim PROPERTIES
    vector per hyperlink. Cached separately (.feat.npz)."""
    cache = path.with_suffix(".feat.npz")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        d = np.load(cache)
        return d["sources"], d["targets"], d["times"], d["features"]

    import pandas as pd
    df = pd.read_csv(
        path, sep="\t", header=0,
        usecols=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "TIMESTAMP",
                 "PROPERTIES"],
        dtype={"SOURCE_SUBREDDIT": str, "TARGET_SUBREDDIT": str,
               "TIMESTAMP": str, "PROPERTIES": str},
    )
    df = df.dropna(subset=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT",
                           "TIMESTAMP", "PROPERTIES"])
    ts = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    valid = ~ts.isna()
    df = df[valid]; ts = ts[valid]
    times = ts.astype("int64").to_numpy() / 1e9

    combined = pd.concat([df["SOURCE_SUBREDDIT"], df["TARGET_SUBREDDIT"]],
                         ignore_index=True)
    codes, uniques = pd.factorize(combined)
    n = len(df)
    sources = codes[:n].astype(np.int64)
    targets = codes[n:].astype(np.int64)

    # PROPERTIES is a comma-separated vector inside a TAB-separated row.
    feat_strs = df["PROPERTIES"].str.split(",", expand=True)
    features = feat_strs.to_numpy(dtype=np.float32)

    print(f"[reddit] parsed {len(sources)} hyperlinks + "
          f"{features.shape[1]}-dim PROPERTIES from {path.name}: "
          f"{len(uniques)} unique subreddits", flush=True)
    np.savez(cache, sources=sources, targets=targets, times=times,
             features=features)
    return sources, targets, times, features


class RedditHyperlinkDynamicGraphDataset(Dataset):
    """Reddit subreddit-hyperlink (body or title) graph as DTDG snapshots.

    Same return signature as DBLPDynamicGraphDataset. Thin wrapper over the
    universal :class:`DTDGDataset`.

    Args:
      data_path: path to the SNAP TSV. If None, defaults to
                 dataset/reddit/soc-redditHyperlinks-<variant>.tsv.
      variant:   "body" or "title" — which of the two SNAP files to use.
      num_bins:  number of DTDG snapshots (default 36 ~ monthly over 3 years).
      top_k:     keep only the K most-active subreddits (combined src+tgt).
    """

    SPLITS: Tuple[float, float, float] = (0.7, 0.1, 0.2)

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: str = "train",
        history_size: int = 3,
        num_preds: int = 1,
        variant: str = "body",
        num_bins: int = 36,
        top_k: int = 1000,
        feat_dim: int = 8,
        stride: int = 1,
        seed: int = 42,
        with_edge_features: bool = False,
        normalize_edge_features: bool = True,
    ):
        assert split in ("train", "val", "test")
        assert variant in REDDIT_URLS, f"unknown reddit variant {variant!r}"
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.window = history_size + num_preds
        self.stride = stride
        self.variant = variant
        self.with_edge_features = with_edge_features

        if data_path is None:
            data_path = f"dataset/reddit/soc-redditHyperlinks-{variant}.tsv"
        path = Path(data_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Reddit-{variant} TSV not found at {path}. Download via:\n"
                f"  wget {REDDIT_URLS[variant]} -O {path}"
            )

        if with_edge_features:
            sources, targets, times, edge_feats = (
                _parse_reddit_tsv_with_features(path))
        else:
            sources, targets, times = _parse_reddit_tsv(path)
            edge_feats = None
        adj, active, node_edge_feats = bin_homogeneous(
            sources, targets, times, top_k=top_k, num_bins=num_bins,
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

        seq = SnapshotSequence(adj=adj, active=active, feats=feats)
        self._ds = DTDGDataset.from_global(
            seq, split=split, history_size=history_size, num_preds=num_preds,
            stride=stride, splits=self.SPLITS,
        )
        self.starts = self._ds.starts

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]

    @property
    def feature_dim(self) -> int:
        return self.feat_dim
