"""NYC taxi mobility dataset: an evolving zone-to-zone trip network with a
recurring, periodic per-zone congestion state.

The mobility counterpart of ``bts_flights_dataset.py``. The graph is the trips
driven in each bin (pickup zone -> dropoff zone), and the label is whether a
zone is congested in the bin: the mean speed of the trips picked up there is
low for that zone. It is derived from the trips' distance / duration fields,
not from the node's degree. Congestion follows peak / off-peak and weekday /
weekend cycles, which makes "copy the same bin one period ago" a strong
trivial predictor, hence the seasonal floors in the audit.

Raw data: the NYC TLC trip records, one parquet per month,
    https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_<YYYY>-<MM>.parquet
(``bash scripts/dblp/fetch_nyc_taxi.sh 2019-01 2019-02 2019-03``; reading
parquet needs ``pyarrow``; a same-named ``.csv`` is read too). Columns used:
tpep_pickup_datetime, tpep_dropoff_datetime (lpep_* for green), PULocationID,
DOLocationID, trip_distance (miles). Timestamps are NYC local time.

Binning: ``bin_hours`` (default 1, must divide 24) on the PICKUP time, from
00:00 of the first requested month. Nodes: the 263 TLC taxi zones
(LocationID 1..263; 264 / 265 = unknown are dropped), restricted to zones
with at least one trip. Edges: trips pickup -> dropoff in the bin with at
least ``min_edge_trips`` trips, symmetrised to the pipeline's bool snapshot.

Per-(bin, zone) counts (``ops``, cached):
    n_pu, n_do, n_ok, sum_miles, sum_hours
where ``n_ok`` / sums cover the pickups that pass the sanity filter (duration
1 min - 3 h, distance 0.1 - 100 mi, speed 1 - 80 mph). Zone speed in a bin =
sum_miles / sum_hours (trip-length weighted), valid when n_ok >= min_trips.
Label (``node_labels_dyn``, -1 = invalid):
    ``label_kind="rel_speed"``  1 if the speed is below the zone's own
        ``threshold`` quantile (default 0.25) of its speeds over the train
        bins, so the label is a state of the zone and not of its location;
    ``label_kind="abs_speed"``  1 if the speed is below ``threshold`` mph.
``node_signal`` (T, N) float32 = speed / the zone's cut-off (< 1 = congested).
Features: the 8 graph statistics + (unless ``with_ops_features=False``)
    [log1p pickups, log1p dropoffs, speed / 20, speed / cut-off,
     mean trip miles / 10, mean trip hours, sin / cos day-of-week,
     sin / cos hour-of-day (sub-daily bins only)].
The loader prints the label and structure audits of the BTS loader plus the
trivial floors on the test bins (AUROC of copy-last, same bin yesterday, same
bin last week).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

from src.custom_codes.bts_flights_dataset import label_audit, structure_audit
from src.custom_codes.dtdg.binning import scatter_snapshots
from src.custom_codes.dtdg.canonical import SnapshotSequence
from src.custom_codes.dtdg.dataset import DTDGDataset
from src.custom_codes.dtdg.features import build_graph_features

NYC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/{kind}_tripdata_{month}.parquet"
N_ZONES = 263
OPS_COLS = ["n_pu", "n_do", "n_ok", "sum_miles", "sum_hours"]
TIME_PREFIX = {"yellow": "tpep", "green": "lpep"}


def nyc_files(data_dir: Path, months: Sequence[str], kind: str = "yellow") -> List[Path]:
    """The monthly files for ``months`` ("YYYY-MM"), parquet or csv, in order."""
    out = []
    for m in months:
        for cand in (data_dir / f"{kind}_tripdata_{m}.parquet",
                     data_dir / f"{kind}_tripdata_{m}.csv"):
            if cand.exists():
                out.append(cand); break
        else:
            raise FileNotFoundError(
                f"no {kind}_tripdata_{m}.parquet under {data_dir}. Fetch with\n"
                f"  bash scripts/dblp/fetch_nyc_taxi.sh {m}\n(login node).")
    return out


def _read_month(path: Path, kind: str):
    import pandas as pd
    pre = TIME_PREFIX[kind]
    cols = [f"{pre}_pickup_datetime", f"{pre}_dropoff_datetime",
            "PULocationID", "DOLocationID", "trip_distance"]
    if path.suffix == ".parquet":
        try:
            df = pd.read_parquet(path, columns=cols)
        except ImportError as e:
            raise ImportError(f"reading {path.name} needs pyarrow: pip install pyarrow") from e
    else:
        df = pd.read_csv(path, usecols=cols, low_memory=False)
    df.columns = ["pu_time", "do_time", "pu", "do", "miles"]
    df["pu_time"] = pd.to_datetime(df.pu_time, errors="coerce")
    df["do_time"] = pd.to_datetime(df.do_time, errors="coerce")
    return df


def parse_nyc_taxi(data_dir: Path, months: Sequence[str], bin_hours: int = 1,
                   kind: str = "yellow") -> dict:
    """Monthly files -> per-bin trip edges and per-(bin, zone) counts, cached
    as ``nyc_<kind>_<m0>_<m1>_h<bin_hours>.npz`` in ``data_dir``. Files are
    aggregated one at a time, so memory stays at one month of trips.

    Returns dict(zones (N,) int LocationID, bin_start_day (T,) int days since
    the first day, bin_hour (T,) int, edges (E, 4: bin, src, dst, n_trips)
    int32, ops (T, N, 5) float32, dow (T,) int day-of-week of the bin).
    N is always 263 here; the dataset drops the zones with no trip.
    """
    import pandas as pd
    if kind not in TIME_PREFIX:
        raise ValueError(f"kind must be yellow or green, got {kind!r}")
    months = sorted(str(m) for m in months)
    files = nyc_files(Path(data_dir), months, kind)
    bin_hours = int(bin_hours)
    if bin_hours < 1 or 24 % bin_hours:
        raise ValueError("bin_hours must divide 24")
    cache = Path(data_dir) / f"nyc_{kind}_{months[0]}_{months[-1]}_h{bin_hours}.npz"
    newest = max(p.stat().st_mtime for p in files)
    keys = ("zones", "bin_start_day", "bin_hour", "edges", "ops", "dow")
    if cache.exists() and cache.stat().st_mtime >= newest:
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in keys}

    start = pd.Timestamp(months[0] + "-01")
    end = pd.Timestamp(months[-1] + "-01") + pd.offsets.MonthBegin(1)
    T = int((end - start) / pd.Timedelta(hours=bin_hours))
    N = N_ZONES
    ops = np.zeros((T, N, len(OPS_COLS)), dtype=np.float64)
    edge_keys, edge_cnts, n_trips = [], [], 0
    for p in files:
        df = _read_month(p, kind)
        n_raw = len(df)
        # the monthly files carry stray timestamps from other years
        df = df[(df.pu_time >= start) & (df.pu_time < end)
                & df.pu.between(1, N) & df.do.between(1, N)]
        bins = ((df.pu_time - start) / pd.Timedelta(hours=bin_hours)).to_numpy().astype(np.int64)
        pu = df.pu.to_numpy().astype(np.int64) - 1
        do = df.do.to_numpy().astype(np.int64) - 1
        miles = pd.to_numeric(df.miles, errors="coerce").fillna(0).to_numpy().astype(np.float64)
        hours = ((df.do_time - df.pu_time) / pd.Timedelta(hours=1)).fillna(0).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            mph = miles / hours
        ok = ((hours >= 1 / 60) & (hours <= 3) & (miles >= 0.1) & (miles <= 100)
              & (mph >= 1) & (mph <= 80))
        np.add.at(ops[..., 0], (bins, pu), 1.0)
        np.add.at(ops[..., 1], (bins, do), 1.0)
        np.add.at(ops[..., 2], (bins[ok], pu[ok]), 1.0)
        np.add.at(ops[..., 3], (bins[ok], pu[ok]), miles[ok])
        np.add.at(ops[..., 4], (bins[ok], pu[ok]), hours[ok])
        uniq, cnt = np.unique((bins * N + pu) * N + do, return_counts=True)
        edge_keys.append(uniq); edge_cnts.append(cnt)
        n_trips += len(df)
        print(f"[nyc] {p.name}: {len(df)}/{n_raw} trips kept, {int(ok.sum())} with a valid speed",
              flush=True)
    key = np.concatenate(edge_keys); cnt = np.concatenate(edge_cnts)
    # a (bin, pu, do) key can only repeat across files at a month boundary
    uniq, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv, weights=cnt, minlength=len(uniq))
    edges = np.stack([uniq // (N * N), (uniq // N) % N, uniq % N, cnt], axis=1).astype(np.int32)
    hour_idx = np.arange(T) * bin_hours
    bin_start_day = hour_idx // 24
    bin_hour = hour_idx % 24
    dow = ((start.dayofweek + bin_start_day) % 7).astype(np.int64)
    print(f"[nyc] {n_trips} {kind} trips {months[0]}..{months[-1]}: N={N} zones, "
          f"T={T} bins of {bin_hours}h, {len(edges)} (bin, pickup, dropoff) edges", flush=True)
    out = dict(zones=np.arange(1, N + 1), bin_start_day=bin_start_day, bin_hour=bin_hour,
               edges=edges, ops=ops.astype(np.float32), dow=dow)
    np.savez(cache, **out)
    return out


def zone_speed(ops: np.ndarray, min_trips: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """(speed (T, N) mph, 0 where invalid; valid (T, N) bool)."""
    valid = ops[..., 2] >= max(int(min_trips), 1)
    speed = np.divide(ops[..., 3], ops[..., 4], out=np.zeros_like(ops[..., 3], dtype=np.float64),
                      where=valid & (ops[..., 4] > 0))
    return speed, valid


def congestion_labels(ops: np.ndarray, label_kind: str = "rel_speed",
                      threshold: float = 0.25, min_trips: int = 10,
                      n_train_bins: int | None = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(labels (T, N) int64 with -1 invalid, signal (T, N) float32 = speed /
    cut-off, cutoff (N,) mph). ``rel_speed``: the cut-off is the zone's
    ``threshold`` quantile of its valid speeds over the first
    ``n_train_bins`` bins (all bins when None); a zone with no valid train
    bin gets no label. ``abs_speed``: the cut-off is ``threshold`` mph."""
    speed, valid = zone_speed(ops, min_trips)
    T, N = speed.shape
    if label_kind == "abs_speed":
        cutoff = np.full(N, float(threshold))
    elif label_kind == "rel_speed":
        if not 0.0 < threshold < 1.0:
            raise ValueError("rel_speed threshold is a quantile in (0, 1)")
        ref = np.where(valid, speed, np.nan)[:n_train_bins]
        cutoff = np.full(N, np.nan)
        has = ~np.isnan(ref).all(0)
        cutoff[has] = np.nanquantile(ref[:, has], threshold, axis=0)
    else:
        raise ValueError(f"label_kind must be rel_speed or abs_speed, got {label_kind!r}")
    valid = valid & ~np.isnan(cutoff)[None, :]
    safe = np.where(np.isnan(cutoff) | (cutoff <= 0), 1.0, cutoff)
    sig = np.where(valid, speed / safe[None, :], 0.0)
    labels = np.where(valid, (sig < 1.0).astype(np.int64), -1)
    return labels, sig.astype(np.float32), cutoff


def ops_features(ops: np.ndarray, signal: np.ndarray, dow: np.ndarray,
                 bin_hour: np.ndarray, sub_daily: bool, min_trips: int = 10) -> np.ndarray:
    T, N = signal.shape
    speed, _ = zone_speed(ops, min_trips)
    n_ok = ops[..., 2]
    cols = [np.log1p(ops[..., 0]), np.log1p(ops[..., 1]), speed / 20.0, signal,
            np.divide(ops[..., 3], n_ok, out=np.zeros_like(n_ok), where=n_ok > 0) / 10.0,
            np.divide(ops[..., 4], n_ok, out=np.zeros_like(n_ok), where=n_ok > 0)]
    ang = 2 * np.pi * dow / 7.0
    cols += [np.broadcast_to(np.sin(ang)[:, None], (T, N)),
             np.broadcast_to(np.cos(ang)[:, None], (T, N))]
    if sub_daily:
        ang = 2 * np.pi * bin_hour / 24.0
        cols += [np.broadcast_to(np.sin(ang)[:, None], (T, N)),
                 np.broadcast_to(np.cos(ang)[:, None], (T, N))]
    return np.stack(cols, axis=-1).astype(np.float32)


def _auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank AUROC with tied scores sharing their mean rank."""
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    rank = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return float((rank[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def trivial_floors(labels: np.ndarray, signal: np.ndarray, first_bin: int,
                   lags: dict) -> dict:
    """AUROC of "the zone's speed ``lag`` bins ago" as the congestion score,
    over target bins >= ``first_bin`` and rows valid at both ends. ``lags``
    maps a name to a lag in bins. nan when a lag has no usable rows."""
    out = {}
    T = labels.shape[0]
    for name, lag in lags.items():
        t0 = max(int(first_bin), int(lag))
        if lag < 1 or t0 >= T:
            out[name] = float("nan"); continue
        y, prev_y, prev_s = labels[t0:], labels[t0 - lag:T - lag], signal[t0 - lag:T - lag]
        m = (y >= 0) & (prev_y >= 0)
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            out[name] = float("nan"); continue
        out[name] = _auroc(y[m], -prev_s[m])
    return out


class NYCTaxiDataset(Dataset):
    """Hourly (or coarser) taxi-trip snapshots with a per-zone congestion
    state. Same dynamic-label contract as ``BTSFlightsDataset``
    (``node_labels_dyn`` (T, N) with -1 invalid, ``state_label_mode``,
    ``node_signal``), so ``run_block_probe.py --task binary`` reads it
    unchanged. Split 70/10/20 chronological over bins.
    """

    SPLITS: Tuple[float, float, float] = (0.7, 0.1, 0.2)

    def __init__(
        self,
        data_dir: str = "dataset/nyc_taxi",
        months: Sequence[str] = ("2019-01", "2019-02", "2019-03"),
        split: str = "train",
        history_size: int = 3,
        num_preds: int = 1,
        bin_hours: int = 1,
        kind: str = "yellow",
        label_kind: str = "rel_speed",
        threshold: float = 0.25,
        min_trips: int = 10,
        min_edge_trips: int = 1,
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
        self.months = sorted(str(m) for m in months)
        self.bin_hours = int(bin_hours)

        d = parse_nyc_taxi(Path(data_dir).expanduser(), self.months,
                           bin_hours=self.bin_hours, kind=kind)
        keep = d["ops"][..., :2].sum((0, 2)) > 0            # zones with any trip
        remap = np.cumsum(keep) - 1
        ops = d["ops"][:, keep].astype(np.float64)
        T, N = ops.shape[:2]
        e = d["edges"][d["edges"][:, 3] >= int(min_edge_trips)]
        adj, active = scatter_snapshots(remap[e[:, 1]], remap[e[:, 2]],
                                        e[:, 0].astype(np.int64), N, T)

        n_train = int(self.SPLITS[0] * T)
        labels, signal, cutoff = congestion_labels(ops, label_kind, threshold, min_trips,
                                                   n_train_bins=n_train)
        labels = np.where(active, labels, -1)
        feats = build_graph_features(adj, active, feat_dim=feat_dim, seed=seed)
        if with_ops_features:
            of = ops_features(ops, signal, d["dow"], d["bin_hour"],
                              sub_daily=self.bin_hours < 24, min_trips=min_trips)
            feats = np.concatenate([feats, of * active[..., None].astype(np.float32)], axis=-1)

        self.num_nodes = N
        self.feat_dim = feats.shape[-1]
        self.zones = d["zones"][keep]
        self.bin_start_day = d["bin_start_day"]
        self.bin_hour = d["bin_hour"]
        self.ops = ops.astype(np.float32)
        self.speed_cutoff = cutoff
        self.node_labels_dyn = labels
        self.node_signal = signal
        self.n_classes = 2
        self.state_label_mode = f"congestion:{label_kind}<{threshold:g}"
        # static fallback (majority state per zone) for the SnapshotSequence
        maj = np.zeros(N, dtype=np.int64)
        for i in range(N):
            col = labels[:, i]; col = col[col >= 0]
            maj[i] = int(col.mean() > 0.5) if col.size else 0
        self.node_labels = maj
        self.audit = label_audit(labels, active)
        self.structure = structure_audit(adj)
        per_day = 24 // self.bin_hours
        test0 = int((self.SPLITS[0] + self.SPLITS[1]) * T)
        self.floors = trivial_floors(labels, signal, test0,
                                     {"copy_last": 1, "same_bin_yesterday": per_day,
                                      "same_bin_last_week": 7 * per_day})

        seq = SnapshotSequence(adj=adj, active=active, feats=feats, node_labels=maj)
        self._ds = DTDGDataset.from_global(
            seq, split=split, history_size=history_size, num_preds=num_preds,
            stride=stride, splits=self.SPLITS,
            split_mode=split_mode, split_seed=seed,
        )
        self.starts = self._ds.starts
        a = self.audit
        print(f"[nyc] split={split} windows={len(self.starts)} N={N} T={T} "
              f"feat_dim={self.feat_dim} label={self.state_label_mode} "
              f"(min_trips={min_trips}, min_edge_trips={min_edge_trips})", flush=True)
        if split == "train":
            print(f"[nyc] label audit (all bins): valid rows={a['valid_rows']}/{a['rows']} "
                  f"zones ever labelled={int((labels >= 0).any(0).sum())}/{N} "
                  f"pos_rate={a['pos_rate']:.3f} nodes_with_2+_changes={a['nodes_2plus_changes']:.2f} "
                  f"nodes_reverting={a['nodes_reverting']:.2f} persistence_acc={a['persistence_acc']:.3f} "
                  f"switch_rate={a['switch_rate']:.3f}", flush=True)
            st = self.structure
            print(f"[nyc] structure audit: consecutive-snapshot Jaccard={st['jaccard']:.3f} "
                  f"edges new vs previous={st['new_share']:.3f} gone={st['gone_share']:.3f} "
                  f"share of all-time edges per snapshot={st['ever_share']:.3f}", flush=True)
            f = self.floors
            print(f"[nyc] trivial floors (test bins, AUROC of the zone's past speed): "
                  f"copy-last={f['copy_last']:.3f} same-bin-yesterday={f['same_bin_yesterday']:.3f} "
                  f"same-bin-last-week={f['same_bin_last_week']:.3f}", flush=True)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]

    @property
    def feature_dim(self) -> int:
        return self.feat_dim
