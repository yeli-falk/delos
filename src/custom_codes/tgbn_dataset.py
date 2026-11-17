"""TGB node property prediction: tgbn-trade (Huang et al., NeurIPS'23 D&B).

The node label is time-varying: for nation ``u`` in year ``t`` it is the
vector of ``u``'s export shares to every other nation in year ``t`` (sums to
one). Evaluation is NDCG@10 of the predicted shares against the true ones
(:func:`src.train.analyze_jepa.ndcg_at_k` reimplements TGB's evaluator). The
label at ``t`` is predicted from everything observed before ``t``, which is
the "target step after the history window" geometry of the DTDG pipeline.
TGB's persistence baseline maps onto the ``label_persistence`` column.

Raw files (fetch with ``scripts/dblp/fetch_tgbn_trade.sh`` ->
``dataset/tgb/tgbn-trade/``):

    tgbn-trade_edgelist.csv     year,nation,trading nation,weight   1986..2016
    tgbn-trade_node_labels.csv  year,nation,trading nation,weight   1987..2016

255 nations, 31 years. The label file is the edge file for the same year
(identical weights), so ``label(t)`` is the row-normalised weighted
out-adjacency at ``t``. The snapshot adjacency the encoders see is the usual
symmetric bool graph; the weights survive in the label tensor and the
optional per-node weight features.

Snapshots are the native years (1986 has edges but no label, so its rows are
marked invalid). Node ids are nations in sorted name order.

Exposed for the probe (``run_block_probe.py --task ranking``):

    node_label_dist   (T, N, N) float32   label vector per (year, nation)
    label_valid       (T, N)    bool      row has a label (sums to one)
    state_label_mode  "ranking"
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

from src.custom_codes.dtdg.binning import scatter_snapshots
from src.custom_codes.dtdg.canonical import SnapshotSequence
from src.custom_codes.dtdg.dataset import DTDGDataset
from src.custom_codes.dtdg.features import build_graph_features


TGBN_TRADE_URL = ("https://object-arbutus.alliancecan.ca/swift/v1/"
                  "14c95234f6cd4a21a47deafe20cce2a7/tgb/tgbn-trade.zip")
EDGE_FILE = "tgbn-trade_edgelist.csv"
LABEL_FILE = "tgbn-trade_node_labels.csv"
WEIGHT_FEAT_DIM = 5


def _read(path: Path):
    import pandas as pd
    df = pd.read_csv(path, header=0, names=["year", "src", "dst", "w"],
                     dtype={"year": np.int64, "src": str, "dst": str, "w": np.float64})
    return df.dropna()


def parse_tgbn_trade(data_dir: Path) -> dict:
    """Both CSVs -> dense arrays, cached as ``tgbn-trade.npz`` next to them.

    Returns dict(years (T,), nations (N,), edges (E, 4: t_idx, src, dst, w),
    labels (T, N, N) float32).
    """
    edge_path, label_path = data_dir / EDGE_FILE, data_dir / LABEL_FILE
    for p in (edge_path, label_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Fetch tgbn-trade with\n"
                f"  bash scripts/dblp/fetch_tgbn_trade.sh\n"
                f"(downloads {TGBN_TRADE_URL} and unzips into {data_dir}).")
    cache = data_dir / "tgbn-trade.npz"
    newest = max(edge_path.stat().st_mtime, label_path.stat().st_mtime)
    if cache.exists() and cache.stat().st_mtime >= newest:
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in ("years", "nations", "edges", "labels")}

    e, l = _read(edge_path), _read(label_path)
    nations = np.array(sorted(set(e.src) | set(e.dst) | set(l.src) | set(l.dst)))
    nid = {n: i for i, n in enumerate(nations)}
    years = np.arange(min(e.year.min(), l.year.min()),
                      max(e.year.max(), l.year.max()) + 1)
    y0 = int(years[0]); T, N = len(years), len(nations)

    edges = np.stack([(e.year.to_numpy() - y0).astype(np.float64),
                      e.src.map(nid).to_numpy().astype(np.float64),
                      e.dst.map(nid).to_numpy().astype(np.float64),
                      e.w.to_numpy()], axis=1)
    labels = np.zeros((T, N, N), dtype=np.float32)
    labels[(l.year.to_numpy() - y0), l.src.map(nid).to_numpy(),
           l.dst.map(nid).to_numpy()] = l.w.to_numpy().astype(np.float32)

    print(f"[tgbn-trade] parsed {len(e)} edges / {len(l)} label rows: "
          f"N={N} nations, T={T} years ({y0}..{int(years[-1])}), "
          f"labelled (year, nation) rows={int((labels.sum(-1) > 0).sum())}",
          flush=True)
    out = dict(years=years, nations=nations, edges=edges, labels=labels)
    np.savez(cache, **out)
    return out


def weight_features(edges: np.ndarray, T: int, N: int) -> np.ndarray:
    """Per-(year, nation) descriptors of the WEIGHTED trade the bool snapshot
    drops: [max export share, export entropy / log N, export partners / N,
    total import share received / N, import partners / N]. Computed from the
    edges of that year, i.e. what is observed at that step."""
    W = np.zeros((T, N, N), dtype=np.float64)
    t, s, d = edges[:, 0].astype(int), edges[:, 1].astype(int), edges[:, 2].astype(int)
    np.add.at(W, (t, s, d), edges[:, 3])
    out = np.zeros((T, N, WEIGHT_FEAT_DIM), dtype=np.float32)
    row_sum = W.sum(-1, keepdims=True)
    P = np.divide(W, row_sum, out=np.zeros_like(W), where=row_sum > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(P * np.log(np.where(P > 0, P, 1.0))).sum(-1) / np.log(max(N, 2))
    out[..., 0] = P.max(-1)
    out[..., 1] = ent
    out[..., 2] = (W > 0).sum(-1) / N
    out[..., 3] = W.sum(1) / N                      # incoming share mass
    out[..., 4] = (W > 0).sum(1) / N
    return out


class TGBNTradeDataset(Dataset):
    """tgbn-trade as yearly DTDG snapshots with a per-(year, nation) label
    distribution. Same batch signature as every other loader (thin wrapper
    over :class:`DTDGDataset`); the ranking labels ride along as attributes.

    Args:
      data_dir:  folder with the two TGB CSVs.
      with_weight_features: append :func:`weight_features` (5 dims) to the 8
                 graph statistics.
      split:     "train" | "val" | "test", chronological 70/10/20 over the
                 31 years (train 1986-2006, val 2007-2009, test 2010-2016).
    """

    SPLITS: Tuple[float, float, float] = (0.7, 0.1, 0.2)

    def __init__(
        self,
        data_dir: str = "dataset/tgb/tgbn-trade",
        split: str = "train",
        history_size: int = 3,
        num_preds: int = 1,
        with_weight_features: bool = True,
        feat_dim: int = 8,
        stride: int = 1,
        seed: int = 42,
        split_mode: str = "chronological",
    ):
        assert split in ("train", "val", "test")
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.window = history_size + num_preds
        self.stride = stride
        self.with_weight_features = with_weight_features

        d = parse_tgbn_trade(Path(data_dir).expanduser())
        edges, labels = d["edges"], d["labels"]
        T, N = labels.shape[0], labels.shape[1]
        t_idx = edges[:, 0].astype(np.int64)
        src = edges[:, 1].astype(np.int64)
        dst = edges[:, 2].astype(np.int64)
        adj, active = scatter_snapshots(src, dst, t_idx, N, T)

        feats = build_graph_features(adj, active, feat_dim=feat_dim, seed=seed)
        if with_weight_features:
            wf = weight_features(edges, T, N) * active[..., None].astype(np.float32)
            feats = np.concatenate([feats, wf], axis=-1)

        self.num_nodes = N
        self.feat_dim = feats.shape[-1]
        self.years = d["years"]
        self.nations = d["nations"]
        self.node_label_dist = labels                        # (T, N, N)
        self.label_valid = labels.sum(-1) > 0                # (T, N)
        self.num_classes = N
        self.state_label_mode = "ranking"
        self.node_labels = None
        self.node_labels_dyn = None

        seq = SnapshotSequence(adj=adj, active=active, feats=feats)
        self._ds = DTDGDataset.from_global(
            seq, split=split, history_size=history_size, num_preds=num_preds,
            stride=stride, splits=self.SPLITS,
            split_mode=split_mode, split_seed=seed,
        )
        self.starts = self._ds.starts
        print(f"[tgbn-trade] split={split} windows={len(self.starts)} "
              f"N={N} T={T} feat_dim={self.feat_dim} "
              f"(weight feats={'on' if with_weight_features else 'off'})",
              flush=True)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]

    @property
    def feature_dim(self) -> int:
        return self.feat_dim


# ---------------------------------------------------------------------------
# tgbn-genre / tgbn-reddit / tgbn-token: weekly user -> item interaction labels
# ---------------------------------------------------------------------------
# Each is a bipartite interaction stream user -> class (music genre /
# subreddit / token) with a weight. The label of user ``u`` at label time
# ``L`` is the vector of ``u``'s interaction weight over the classes during
# the following week [L, L + 7d). Edges have unix-second timestamps.
#
# The stream is binned into 7-day snapshots whose starts are label days, so
# label(b, u) describes user ``u``'s interactions in snapshot ``b`` and is
# predicted from the snapshots before ``b``. On tgbn-genre the labels are
# daily, each covering the next 7 days; the loader keeps the label at each
# bin's first day. The loader prints how well the label agrees with the
# snapshot's own weighted class distribution (``label/snapshot NDCG@10
# agreement``). The agreement is well below 1 on genre because TGB builds the
# label from information the edgelist does not carry; this is not a binning
# error, and the label file is the ground truth.
#
# The probe reports TGB's two floors (``label_persistence`` and
# ``label_moving_average``); the moving average is the stronger one.
#
# The dense (T, N, N) snapshot pipeline fits genre (1,505 nodes) outright.
# Reddit (11,766 nodes, ~730 weeks) needs a time cut (``max_bins``: keep the
# last K labelled weeks and only the nodes seen in them). Token (61,756
# nodes) does not fit dense at any cut and is wired for completeness only.
#
# Raw CSVs (``bash scripts/dblp/fetch_tgbn.sh <genre|reddit|token>``), all
# with a header row, columns by POSITION as py-tgb reads them
# (tgb/utils/pre_process.py):
#   tgbn-genre_edgelist.csv    time, user_id, genre, weight
#   tgbn-reddit_edgelist.csv   ts, src, subreddit, num_words, score   (w = score)
#   tgbn-token_edgelist.csv    timestamp, user_address, token_address, value, IsSender
#   tgbn-*_node_labels.csv     ts, src, dst, w                        (all three)
# Node ids: users first (sorted), then the C classes (sorted); the label
# vector is over the C classes; class nodes carry no label.

TGBN_SPECS = {
    "trade":  dict(edge_w_col=3, n_edge_cols=4),
    "genre":  dict(edge_w_col=3, n_edge_cols=4),
    "reddit": dict(edge_w_col=4, n_edge_cols=5),
    "token":  dict(edge_w_col=3, n_edge_cols=5),
}
TGBN_URL = ("https://object-arbutus.alliancecan.ca/swift/v1/"
            "14c95234f6cd4a21a47deafe20cce2a7/tgb/tgbn-{name}.zip")
DAY = 86400


def tgbn_files(name: str) -> Tuple[str, str]:
    return f"tgbn-{name}_edgelist.csv", f"tgbn-{name}_node_labels.csv"


def _read_interactions(path: Path, w_col: int, n_cols: int):
    """Positional read of a TGB edge/label CSV -> DataFrame(ts, src, dst, w)."""
    import pandas as pd
    df = pd.read_csv(path, header=0, usecols=[0, 1, 2, w_col],
                     names=[f"c{i}" for i in range(n_cols)],
                     dtype={"c0": np.int64, "c1": str, "c2": str, f"c{w_col}": np.float64})
    df = df.rename(columns={"c0": "ts", "c1": "src", "c2": "dst", f"c{w_col}": "w"})
    return df.dropna()


def weight_features_sparse(edges: np.ndarray, T: int, N: int) -> np.ndarray:
    """Same five per-(step, node) descriptors as :func:`weight_features`
    ([max out share, out entropy / log N, out partners / N, in mass / N,
    in partners / N]) computed from the edge list without the (T, N, N)
    weight tensor, which the larger TGB graphs cannot afford."""
    import pandas as pd
    df = pd.DataFrame({"t": edges[:, 0].astype(np.int64),
                       "s": edges[:, 1].astype(np.int64),
                       "d": edges[:, 2].astype(np.int64), "w": edges[:, 3]})
    g = df.groupby(["t", "s", "d"], sort=False)["w"].sum().reset_index()
    out = np.zeros((T, N, WEIGHT_FEAT_DIM), dtype=np.float32)
    if len(g) == 0:
        return out
    tot = g.groupby(["t", "s"])["w"].transform("sum")
    g["p"] = np.divide(g["w"], tot, out=np.zeros(len(g)), where=tot > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g["plogp"] = np.where(g["p"] > 0, g["p"] * np.log(g["p"]), 0.0)
    src = g.groupby(["t", "s"]).agg(pmax=("p", "max"), ent=("plogp", "sum"),
                                    deg=("d", "size")).reset_index()
    ti, si = src["t"].to_numpy(), src["s"].to_numpy()
    out[ti, si, 0] = src["pmax"].to_numpy()
    out[ti, si, 1] = -src["ent"].to_numpy() / np.log(max(N, 2))
    out[ti, si, 2] = src["deg"].to_numpy() / N
    dst = g.groupby(["t", "d"]).agg(mass=("p", "sum"), deg=("s", "size")).reset_index()
    ti, di = dst["t"].to_numpy(), dst["d"].to_numpy()
    out[ti, di, 3] = dst["mass"].to_numpy() / N
    out[ti, di, 4] = dst["deg"].to_numpy() / N
    return out


def parse_tgbn_interaction(data_dir: Path, name: str, bin_days: int = 7,
                           max_bins: int = 0) -> dict:
    """Both CSVs of tgbn-<name> -> snapshot-aligned arrays, cached as
    ``tgbn-<name>_b<bin_days>_k<max_bins>.npz`` next to them.

    Returns dict(bin_starts (T,) unix seconds, nodes (N,) str, n_users,
    classes (C,) str, edges (E, 4: t_idx, src, dst, w), labels (T, N, C)
    float32 row-normalised, agreement float).

    Binning: ``delta = bin_days`` days; the origin is one bin before the first
    label time, so bin 0 is the unlabelled lead-in (edges older than that are
    folded into it). A bin's label is the label row with the earliest label
    time inside the bin; with a bin coarser than 7 days that label still spans
    one week, so the alignment diagnostic drops. ``max_bins > 0`` keeps only
    the last ``max_bins`` labelled bins plus the lead-in and drops nodes never
    seen in them.
    """
    if name not in TGBN_SPECS or name == "trade":
        raise ValueError(f"parse_tgbn_interaction: name must be genre, reddit or "
                         f"token, got {name!r} (trade has its own parser)")
    edge_file, label_file = tgbn_files(name)
    edge_path, label_path = data_dir / edge_file, data_dir / label_file
    for p in (edge_path, label_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Fetch tgbn-{name} with\n"
                f"  bash scripts/dblp/fetch_tgbn.sh {name}\n"
                f"(downloads {TGBN_URL.format(name=name)} and unzips into {data_dir}).")
    cache = data_dir / f"tgbn-{name}_b{bin_days}_k{max_bins}.npz"
    newest = max(edge_path.stat().st_mtime, label_path.stat().st_mtime)
    keys = ("bin_starts", "nodes", "n_users", "classes", "edges", "labels", "agreement")
    if cache.exists() and cache.stat().st_mtime >= newest:
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in keys}

    spec = TGBN_SPECS[name]
    e = _read_interactions(edge_path, spec["edge_w_col"], spec["n_edge_cols"])
    l = _read_interactions(label_path, 3, 4)
    delta = int(bin_days) * DAY
    lab_ts = np.sort(l.ts.unique())
    origin = int(lab_ts[0]) - delta
    l["bin"] = (l.ts.to_numpy() - origin) // delta
    # one label time per bin: the earliest queried inside it
    first_in_bin = l.groupby("bin")["ts"].transform("min")
    l = l[l.ts.to_numpy() == first_in_bin.to_numpy()]
    e["bin"] = np.maximum((e.ts.to_numpy() - origin) // delta, 0)
    n_pre = int((e.ts.to_numpy() < origin).sum())
    T = int(max(l.bin.max(), e.bin.max())) + 1
    if max_bins and max_bins > 0:
        lo = max(int(l.bin.max()) - int(max_bins) + 1, 1)   # keep bins lo..last
        e = e[e.bin >= lo - 1]; l = l[l.bin >= lo]          # bin lo-1 = lead-in
        e["bin"] -= (lo - 1); l["bin"] -= (lo - 1)
        origin += (lo - 1) * delta
        T = int(max(l.bin.max(), e.bin.max())) + 1
    users = np.array(sorted(set(e.src) | set(l.src)))
    classes = np.array(sorted(set(e.dst) | set(l.dst)))
    U, C = len(users), len(classes)
    N = U + C
    uid = {u: i for i, u in enumerate(users)}
    cid = {c: i for i, c in enumerate(classes)}
    edges = np.stack([e.bin.to_numpy().astype(np.float64),
                      e.src.map(uid).to_numpy().astype(np.float64),
                      U + e.dst.map(cid).to_numpy().astype(np.float64),
                      e.w.to_numpy()], axis=1)
    labels = np.zeros((T, N, C), dtype=np.float32)
    lb, lu, lc = l.bin.to_numpy(), l.src.map(uid).to_numpy(), l.dst.map(cid).to_numpy()
    np.add.at(labels, (lb, lu, lc), l.w.to_numpy().astype(np.float32))
    row = labels.sum(-1, keepdims=True)
    labels = np.divide(labels, row, out=np.zeros_like(labels), where=row > 0)

    # alignment diagnostic: label(b, u) vs u's weighted class distribution in
    # snapshot b (should rank identically when labels = next-week interactions)
    agreement = float("nan")
    lab_rows = np.argwhere(row[..., 0] > 0)
    if len(lab_rows):
        rng = np.random.RandomState(0)
        pick = lab_rows[rng.choice(len(lab_rows), size=min(2000, len(lab_rows)), replace=False)]
        snap = np.zeros((len(pick), C), dtype=np.float64)
        key = {(int(b), int(u)): i for i, (b, u) in enumerate(pick)}
        tb, ts_, td = edges[:, 0].astype(int), edges[:, 1].astype(int), edges[:, 2].astype(int) - U
        for b_, s_, d_, w_ in zip(tb, ts_, td, edges[:, 3]):
            i = key.get((b_, s_))
            if i is not None and d_ >= 0:
                snap[i, d_] += w_
        from src.train.analyze_jepa import ndcg_at_k
        ok = snap.sum(1) > 0
        if ok.any():
            agreement = float(ndcg_at_k(labels[pick[ok, 0], pick[ok, 1]], snap[ok], 10).mean())
    bin_starts = origin + delta * np.arange(T)
    print(f"[tgbn-{name}] parsed {len(e)} edges / {len(l)} label rows: users={U} "
          f"classes={C} (N={N}), T={T} bins of {bin_days}d "
          f"({n_pre} edges older than the lead-in folded into bin 0), labelled "
          f"(bin, user) rows={int((row[..., 0] > 0).sum())}, label/snapshot "
          f"NDCG@10 agreement={agreement:.3f}", flush=True)
    out = dict(bin_starts=bin_starts, nodes=np.concatenate([users, classes]),
               n_users=np.int64(U), classes=classes, edges=edges, labels=labels,
               agreement=np.float64(agreement))
    np.savez(cache, **out)
    return out


class TGBNInteractionDataset(Dataset):
    """tgbn-genre / tgbn-reddit / tgbn-token as weekly bipartite DTDG snapshots
    with a per-(week, user) label distribution over the C classes. Same batch
    signature and ranking attributes as :class:`TGBNTradeDataset`
    (``node_label_dist`` (T, N, C), ``label_valid`` (T, N),
    ``state_label_mode = "ranking"``), so ``run_block_probe.py --task ranking``
    and every slurm pipeline read it unchanged. Split 70/10/20 chronological
    over bins, as for trade.
    """

    SPLITS: Tuple[float, float, float] = (0.7, 0.1, 0.2)

    def __init__(
        self,
        name: str = "genre",
        data_dir: str | None = None,
        split: str = "train",
        history_size: int = 3,
        num_preds: int = 1,
        bin_days: int = 7,
        max_bins: int = 0,
        with_weight_features: bool = True,
        feat_dim: int = 8,
        stride: int = 1,
        seed: int = 42,
        split_mode: str = "chronological",
    ):
        assert split in ("train", "val", "test")
        self.name = name
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.window = history_size + num_preds
        self.stride = stride
        self.with_weight_features = with_weight_features
        data_dir = data_dir or f"dataset/tgb/tgbn-{name}"

        d = parse_tgbn_interaction(Path(data_dir).expanduser(), name,
                                   bin_days=bin_days, max_bins=max_bins)
        edges, labels = d["edges"], d["labels"]
        T, N, C = labels.shape
        t_idx = edges[:, 0].astype(np.int64)
        src = edges[:, 1].astype(np.int64)
        dst = edges[:, 2].astype(np.int64)
        adj, active = scatter_snapshots(src, dst, t_idx, N, T)

        feats = build_graph_features(adj, active, feat_dim=feat_dim, seed=seed)
        if with_weight_features:
            wf = weight_features_sparse(edges, T, N) * active[..., None].astype(np.float32)
            feats = np.concatenate([feats, wf], axis=-1)

        self.num_nodes = N
        self.num_users = int(d["n_users"])
        self.feat_dim = feats.shape[-1]
        self.bin_starts = d["bin_starts"]
        self.nodes = d["nodes"]
        self.classes = d["classes"]
        self.node_label_dist = labels                        # (T, N, C)
        self.label_valid = labels.sum(-1) > 0                # (T, N)
        self.num_classes = C
        self.label_agreement = float(d["agreement"])
        self.state_label_mode = "ranking"
        self.node_labels = None
        self.node_labels_dyn = None

        seq = SnapshotSequence(adj=adj, active=active, feats=feats)
        self._ds = DTDGDataset.from_global(
            seq, split=split, history_size=history_size, num_preds=num_preds,
            stride=stride, splits=self.SPLITS,
            split_mode=split_mode, split_seed=seed,
        )
        self.starts = self._ds.starts
        print(f"[tgbn-{name}] split={split} windows={len(self.starts)} "
              f"N={N} (users={self.num_users}, classes={C}) T={T} "
              f"feat_dim={self.feat_dim} "
              f"(weight feats={'on' if with_weight_features else 'off'})",
              flush=True)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]

    @property
    def feature_dim(self) -> int:
        return self.feat_dim
