"""BTS airport delay-propagation dataset: an evolving flight network with a
recurring per-airport delay state.

The graph is the flights flown in each bin (it changes with the schedule, the
season and cancellations), and the label is whether an airport is in a delayed
state in the bin: a state of the node that switches on and off many times per
season and propagates along the current graph (aircraft rotations, downstream
congestion). It is derived from the flights' delay fields, not from the
node's degree.

Raw data: the Bureau of Transportation Statistics "Reporting Carrier On-Time
Performance (1987-present)" monthly files,
    https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_<YYYY>_<M>.zip
(``bash scripts/dblp/fetch_bts_flights.sh 2018 2019``), each a zip with one
CSV (header row, ~110 columns, a trailing comma per row). Columns used:
FlightDate, CRSDepTime (local hhmm), Origin, Dest, DepDelayMinutes, DepDel15,
ArrDelayMinutes, ArrDel15, Cancelled, Diverted.

Binning: ``bin_hours`` (default 24 = one snapshot per calendar day, from
FlightDate; sub-daily bins use the scheduled local departure hour, i.e. each
airport's own clock). Nodes: every airport code seen (Origin or Dest),
sorted. Edges: flown (not cancelled) flights origin -> dest in the bin,
symmetrised to the pipeline's bool snapshot. Active: any flown flight.

Per-(bin, airport) operational counts (``ops``, cached):
    n_dep, n_arr, n_dep_del15, sum_dep_delay_min, n_arr_del15,
    sum_arr_delay_min, n_cancelled_dep, n_scheduled_dep
Label (``node_labels_dyn``, -1 = invalid): with ``label_kind="share_del15"``
    1 if n_dep_del15 / n_dep > threshold (default 0.30; BTS' own "on-time"
    definition is a 15-minute departure delay), valid when
    n_dep >= min_departures (default 5); ``"mean_delay"`` thresholds the mean
    departure delay in minutes instead (default threshold 15).
Features: the 8 graph statistics + (unless ``with_ops_features=False``)
    [log1p n_dep, log1p n_arr, share dep del15, mean dep delay / 60,
     share arr del15, mean arr delay / 60, cancellation rate,
     sin / cos day-of-week, sin / cos hour-of-day (sub-daily bins only)].
    The delay state of the observed bins is therefore visible to every
    method (the label at t+1 is forecast from the state at t).
``node_signal`` (T, N) float32 keeps the continuous share-delayed for a
regression variant. The loader prints the label audit on construction:
positive rate, share of airports whose state reverts, persistence accuracy,
switch rate.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

from src.custom_codes.dtdg.binning import scatter_snapshots
from src.custom_codes.dtdg.canonical import SnapshotSequence
from src.custom_codes.dtdg.dataset import DTDGDataset
from src.custom_codes.dtdg.features import build_graph_features

BTS_URL = ("https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_"
           "On_Time_Performance_1987_present_{year}_{month}.zip")
BTS_STEM = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present"
COLS = ["FlightDate", "CRSDepTime", "Origin", "Dest", "DepDelayMinutes",
        "DepDel15", "ArrDelayMinutes", "ArrDel15", "Cancelled", "Diverted"]
OPS_COLS = ["n_dep", "n_arr", "n_dep_del15", "sum_dep_delay", "n_arr_del15",
            "sum_arr_delay", "n_cancelled_dep", "n_scheduled_dep"]


def bts_files(data_dir: Path, years: Sequence[int]) -> List[Path]:
    """The monthly files for ``years`` (zip or extracted csv), in order.
    Months with no file are skipped with a warning, a missing year fails."""
    out = []
    for y in years:
        found = 0
        for m in range(1, 13):
            for cand in (data_dir / f"{BTS_STEM}_{y}_{m}.zip",
                         data_dir / f"{BTS_STEM}_{y}_{m}.csv",
                         data_dir / f"On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{y}_{m}.csv"):
                if cand.exists():
                    out.append(cand); found += 1; break
        if found == 0:
            raise FileNotFoundError(
                f"no BTS on-time file for {y} under {data_dir}. Fetch with\n"
                f"  bash scripts/dblp/fetch_bts_flights.sh {y}\n(login node).")
        if found < 12:
            print(f"[bts] warning: only {found}/12 monthly files for {y}", flush=True)
    return out


def _read_month(path: Path):
    import pandas as pd
    kw = dict(usecols=COLS, index_col=False, low_memory=False,
              dtype={"Origin": str, "Dest": str, "FlightDate": str})
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            member = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            with z.open(member) as f:
                df = pd.read_csv(f, **kw)
    else:
        df = pd.read_csv(path, **kw)
    return df


def parse_bts_flights(data_dir: Path, years: Sequence[int], bin_hours: int = 24) -> dict:
    """Monthly files -> per-bin flight edges and per-(bin, airport) counts,
    cached as ``bts_flights_<y0>-<y1>_h<bin_hours>.npz`` in ``data_dir``.

    Returns dict(airports (N,) str, bin_start_day (T,) int days since the
    first day, bin_hour (T,) int, edges (E, 4: bin, src, dst, n_flights),
    ops (T, N, 8) float32, dow (T,) int day-of-week of the bin).
    """
    import pandas as pd
    years = sorted(int(y) for y in years)
    files = bts_files(Path(data_dir), years)
    per_day = 24 // int(bin_hours)
    if per_day * int(bin_hours) != 24:
        raise ValueError("bin_hours must divide 24")
    cache = Path(data_dir) / f"bts_flights_{years[0]}-{years[-1]}_h{bin_hours}.npz"
    newest = max(p.stat().st_mtime for p in files)
    keys = ("airports", "bin_start_day", "bin_hour", "edges", "ops", "dow")
    if cache.exists() and cache.stat().st_mtime >= newest:
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in keys}

    frames = []
    for p in files:
        df = _read_month(p)
        df = df.dropna(subset=["FlightDate", "Origin", "Dest"])
        frames.append(df)
        print(f"[bts] {p.name}: {len(df)} flights", flush=True)
    df = pd.concat(frames, ignore_index=True); del frames
    date = pd.to_datetime(df.FlightDate, format="%Y-%m-%d", errors="coerce")
    df = df[~date.isna()]; date = date[~date.isna()]
    day0 = date.min().normalize()
    day_idx = ((date - day0).dt.days).to_numpy().astype(np.int64)
    crs = pd.to_numeric(df.CRSDepTime, errors="coerce").fillna(0).to_numpy().astype(np.int64)
    hour = np.clip(crs // 100, 0, 23)
    bins = day_idx * per_day + hour // int(bin_hours)
    T = int(bins.max()) + 1

    airports = np.array(sorted(set(df.Origin) | set(df.Dest)))
    aid = {a: i for i, a in enumerate(airports)}
    N = len(airports)
    o = df.Origin.map(aid).to_numpy().astype(np.int64)
    d_ = df.Dest.map(aid).to_numpy().astype(np.int64)
    cancelled = pd.to_numeric(df.Cancelled, errors="coerce").fillna(0).to_numpy() > 0
    diverted = pd.to_numeric(df.Diverted, errors="coerce").fillna(0).to_numpy() > 0
    flown = ~cancelled
    dep_del15 = pd.to_numeric(df.DepDel15, errors="coerce").fillna(0).to_numpy()
    dep_min = pd.to_numeric(df.DepDelayMinutes, errors="coerce").fillna(0).to_numpy()
    arr_del15 = pd.to_numeric(df.ArrDel15, errors="coerce").fillna(0).to_numpy()
    arr_min = pd.to_numeric(df.ArrDelayMinutes, errors="coerce").fillna(0).to_numpy()
    arr_ok = flown & ~diverted

    ops = np.zeros((T, N, len(OPS_COLS)), dtype=np.float64)
    np.add.at(ops[..., 0], (bins[flown], o[flown]), 1.0)
    np.add.at(ops[..., 1], (bins[arr_ok], d_[arr_ok]), 1.0)
    np.add.at(ops[..., 2], (bins[flown], o[flown]), dep_del15[flown])
    np.add.at(ops[..., 3], (bins[flown], o[flown]), dep_min[flown])
    np.add.at(ops[..., 4], (bins[arr_ok], d_[arr_ok]), arr_del15[arr_ok])
    np.add.at(ops[..., 5], (bins[arr_ok], d_[arr_ok]), arr_min[arr_ok])
    np.add.at(ops[..., 6], (bins[cancelled], o[cancelled]), 1.0)
    np.add.at(ops[..., 7], (bins, o), 1.0)

    key = (bins[flown] * N + o[flown]) * N + d_[flown]
    uniq, cnt = np.unique(key, return_counts=True)
    edges = np.stack([(uniq // (N * N)).astype(np.float64),
                      ((uniq // N) % N).astype(np.float64),
                      (uniq % N).astype(np.float64), cnt.astype(np.float64)], axis=1)
    bin_start_day = np.arange(T) // per_day
    bin_hour = (np.arange(T) % per_day) * int(bin_hours)
    dow = ((day0.dayofweek + bin_start_day) % 7).astype(np.int64)
    print(f"[bts] {len(df)} flights {years[0]}-{years[-1]} ({int(cancelled.sum())} cancelled): "
          f"N={N} airports, T={T} bins of {bin_hours}h, {len(edges)} (bin, route) edges",
          flush=True)
    out = dict(airports=airports, bin_start_day=bin_start_day, bin_hour=bin_hour,
               edges=edges, ops=ops.astype(np.float32), dow=dow)
    np.savez(cache, **out)
    return out


def delay_labels(ops: np.ndarray, label_kind: str = "share_del15",
                 threshold: float = 0.30, min_departures: int = 5
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """(labels (T, N) int64 with -1 invalid, signal (T, N) float32)."""
    n_dep = ops[..., 0]
    valid = n_dep >= max(int(min_departures), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        if label_kind == "share_del15":
            sig = np.where(valid, ops[..., 2] / np.maximum(n_dep, 1), 0.0)
        elif label_kind == "mean_delay":
            sig = np.where(valid, ops[..., 3] / np.maximum(n_dep, 1), 0.0)
        else:
            raise ValueError(f"label_kind must be share_del15 or mean_delay, got {label_kind!r}")
    labels = np.where(valid, (sig > threshold).astype(np.int64), -1)
    return labels, sig.astype(np.float32)


def ops_features(ops: np.ndarray, dow: np.ndarray, bin_hour: np.ndarray,
                 sub_daily: bool) -> np.ndarray:
    n_dep, n_arr = ops[..., 0], ops[..., 1]
    T, N = n_dep.shape
    cols = [np.log1p(n_dep), np.log1p(n_arr),
            np.divide(ops[..., 2], n_dep, out=np.zeros_like(n_dep), where=n_dep > 0),
            np.divide(ops[..., 3], n_dep, out=np.zeros_like(n_dep), where=n_dep > 0) / 60.0,
            np.divide(ops[..., 4], n_arr, out=np.zeros_like(n_arr), where=n_arr > 0),
            np.divide(ops[..., 5], n_arr, out=np.zeros_like(n_arr), where=n_arr > 0) / 60.0,
            np.divide(ops[..., 6], ops[..., 7], out=np.zeros_like(n_dep), where=ops[..., 7] > 0)]
    ang = 2 * np.pi * dow / 7.0
    cols += [np.broadcast_to(np.sin(ang)[:, None], (T, N)),
             np.broadcast_to(np.cos(ang)[:, None], (T, N))]
    if sub_daily:
        ang = 2 * np.pi * bin_hour / 24.0
        cols += [np.broadcast_to(np.sin(ang)[:, None], (T, N)),
                 np.broadcast_to(np.cos(ang)[:, None], (T, N))]
    return np.stack(cols, axis=-1).astype(np.float32)


def label_audit(labels: np.ndarray, active: np.ndarray) -> dict:
    """The dataset criteria, from labels alone: positive rate over valid
    rows, fraction of nodes whose state changes >= 2 times and whose state
    ever REVERTS (1 -> 0), persistence accuracy label(t+1) == label(t) on
    rows valid at both steps, and the switch rate."""
    v = labels >= 0
    y = labels
    both = v[1:] & v[:-1]
    same = (y[1:] == y[:-1]) & both
    ch = ((y[1:] != y[:-1]) & both)
    n_ch = ch.sum(0)
    rev = ((y[1:] == 0) & (y[:-1] == 1) & both).any(0)
    return {
        "valid_rows": int(v.sum()), "rows": int(active.sum()),
        "pos_rate": float(y[v].mean()) if v.any() else float("nan"),
        "nodes_2plus_changes": float((n_ch >= 2).mean()),
        "nodes_reverting": float(rev.mean()),
        "persistence_acc": float(same.sum() / max(both.sum(), 1)),
        "switch_rate": float(ch.sum() / max(both.sum(), 1)),
    }


def structure_audit(adj: np.ndarray) -> dict:
    """How dynamic the structure is, from the bool snapshots alone: mean
    Jaccard overlap of consecutive snapshots' edge sets, mean share of a
    snapshot's edges absent from the previous snapshot (edges that "come"),
    mean share of the previous snapshot's edges absent from the current one
    (edges that "go"), and the share of the all-time edge set present in an
    average snapshot."""
    T = adj.shape[0]
    iu = np.triu_indices(adj.shape[1], k=1)
    E = adj[:, iu[0], iu[1]]                      # (T, n_pairs) bool
    if T < 2:
        return {"jaccard": float("nan"), "new_share": float("nan"),
                "gone_share": float("nan"), "ever_share": float("nan")}
    inter = (E[1:] & E[:-1]).sum(1); union = (E[1:] | E[:-1]).sum(1)
    cur, prev = E[1:].sum(1), E[:-1].sum(1)
    ever = E.any(0).sum()
    return {
        "jaccard": float(np.mean(inter / np.maximum(union, 1))),
        "new_share": float(np.mean((cur - inter) / np.maximum(cur, 1))),
        "gone_share": float(np.mean((prev - inter) / np.maximum(prev, 1))),
        "ever_share": float(np.mean(E.sum(1) / max(ever, 1))),
    }


class BTSFlightsDataset(Dataset):
    """Daily (or sub-daily) flight-network snapshots with a per-airport
    delay state. Exposes the dynamic-label contract of the JODIE / synthetic
    loaders (``node_labels_dyn`` (T, N) with -1 invalid,
    ``state_label_mode``), so ``run_block_probe.py --task binary`` reads it
    unchanged, and ``node_signal`` for a regression variant. Split 70/10/20
    chronological over bins.
    """

    SPLITS: Tuple[float, float, float] = (0.7, 0.1, 0.2)

    def __init__(
        self,
        data_dir: str = "dataset/bts",
        years: Sequence[int] = (2019,),
        split: str = "train",
        history_size: int = 3,
        num_preds: int = 1,
        bin_hours: int = 24,
        label_kind: str = "share_del15",
        threshold: float = 0.30,
        min_departures: int = 5,
        with_ops_features: bool = True,
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
        self.years = [int(y) for y in years]
        self.bin_hours = int(bin_hours)

        d = parse_bts_flights(Path(data_dir).expanduser(), self.years, bin_hours=self.bin_hours)
        ops, edges = d["ops"].astype(np.float64), d["edges"]
        T, N = ops.shape[:2]
        t_idx = edges[:, 0].astype(np.int64)
        src = edges[:, 1].astype(np.int64)
        dst = edges[:, 2].astype(np.int64)
        adj, active = scatter_snapshots(src, dst, t_idx, N, T)

        feats = build_graph_features(adj, active, feat_dim=feat_dim, seed=seed)
        if with_ops_features:
            of = ops_features(ops, d["dow"], d["bin_hour"], sub_daily=self.bin_hours < 24)
            feats = np.concatenate([feats, of * active[..., None].astype(np.float32)], axis=-1)

        labels, signal = delay_labels(ops, label_kind, threshold, min_departures)
        labels = np.where(active, labels, -1)
        self.num_nodes = N
        self.feat_dim = feats.shape[-1]
        self.airports = d["airports"]
        self.bin_start_day = d["bin_start_day"]
        self.bin_hour = d["bin_hour"]
        self.ops = d["ops"]
        self.node_labels_dyn = labels
        self.node_signal = signal
        self.n_classes = 2
        self.state_label_mode = f"delay:{label_kind}>{threshold:g}"
        # static fallback (majority state per airport) for the SnapshotSequence
        maj = np.zeros(N, dtype=np.int64)
        for i in range(N):
            col = labels[:, i]; col = col[col >= 0]
            maj[i] = int(col.mean() > 0.5) if col.size else 0
        self.node_labels = maj
        self.audit = label_audit(labels, active)
        self.structure = structure_audit(adj)

        seq = SnapshotSequence(adj=adj, active=active, feats=feats, node_labels=maj)
        self._ds = DTDGDataset.from_global(
            seq, split=split, history_size=history_size, num_preds=num_preds,
            stride=stride, splits=self.SPLITS,
            split_mode=split_mode, split_seed=seed,
        )
        self.starts = self._ds.starts
        a = self.audit
        print(f"[bts] split={split} windows={len(self.starts)} N={N} T={T} "
              f"feat_dim={self.feat_dim} label={self.state_label_mode} "
              f"(min_dep={min_departures})", flush=True)
        if split == "train":
            print(f"[bts] label audit (all bins): valid rows={a['valid_rows']}/{a['rows']} "
                  f"pos_rate={a['pos_rate']:.3f} nodes_with_2+_changes={a['nodes_2plus_changes']:.2f} "
                  f"nodes_reverting={a['nodes_reverting']:.2f} persistence_acc={a['persistence_acc']:.3f} "
                  f"switch_rate={a['switch_rate']:.3f}", flush=True)
            st = self.structure
            print(f"[bts] structure audit: consecutive-snapshot Jaccard={st['jaccard']:.3f} "
                  f"edges new vs previous={st['new_share']:.3f} gone={st['gone_share']:.3f} "
                  f"share of all-time edges per snapshot={st['ever_share']:.3f}", flush=True)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]

    @property
    def feature_dim(self) -> int:
        return self.feat_dim
