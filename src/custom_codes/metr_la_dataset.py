"""METR-LA dataset for C-JEPA on dynamic graphs.

METR-LA is a fixed-topology spatiotemporal graph: 207 traffic-loop sensors on
LA highways, sampled every 5 minutes. The graph adjacency is constant; only
the per-node speed signal evolves.

Two on-disk layouts are supported, picked automatically by what is present
in `data_dir`:

  (1) DCRNN convention -- the original release:
        data_dir/metr-la.h5     pandas DataFrame, shape [T_total, 207]
        data_dir/adj_mx.pkl     tuple (ids, id_to_ind, adj) with adj 207x207

  (2) Hugging Face mirror `witgaw/METR-LA` -- pre-windowed parquet:
        data_dir/{train,val,test}.parquet     one row per (sensor, t0) window
        data_dir/sensor_graph/adj_mx.npy      (207, 207) numpy array

The HF mirror fixes history=12 / num_preds=12 so user-supplied window sizes
must match. Pass `synthetic=True` to bypass both and use a 50-node toy graph.
"""
from __future__ import annotations

import pickle as pkl
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.custom_codes.dtdg.features import time_of_day


def _load_speed_dataframe(h5_path: Path) -> np.ndarray:
    import pandas as pd

    df = pd.read_hdf(h5_path)
    # Index is a DatetimeIndex; values are float speeds (mph). Missing readings
    # are encoded as 0.0 in the DCRNN release; we treat them as such.
    speeds = df.values.astype(np.float32)               # (T_total, N)
    times = df.index.values.astype("datetime64[ns]")    # (T_total,)
    return speeds, times


def _load_adjacency(adj_path: Path) -> np.ndarray:
    with open(adj_path, "rb") as f:
        # DCRNN serialised the tuple with python2; allow latin1 for safety.
        try:
            payload = pkl.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            payload = pkl.load(f, encoding="latin1")
    if isinstance(payload, tuple):
        adj = payload[-1]
    else:
        adj = payload
    return np.asarray(adj, dtype=np.float32)


# ---------------------------------------------------------------------------
# Hugging Face parquet loader (witgaw/METR-LA)
# ---------------------------------------------------------------------------
def _load_hf_parquet(parquet_path: Path, history_size: int, num_preds: int):
    """Pivot the HF pre-windowed parquet into (n_samples, T, N, 2) windows.

    The HF schema has columns x_t-11_d0 ... x_t+0_d0 (12 history values) and
    y_t+1_d0 ... y_t+12_d0 (12 future values), with `_d1` mirrors holding the
    time-of-day signal. Each row is one (t0_timestamp, node_id) pair.
    """
    import pandas as pd

    if history_size != 12 or num_preds != 12:
        raise ValueError(
            "HF mirror is fixed at history=12 / num_preds=12. "
            "Either match those, or download the raw metr-la.h5."
        )

    df = pd.read_parquet(parquet_path)
    # Column names use x_t-11_d0 ... x_t-1_d0, x_t+0_d0 for history and
    # y_t+1_d0 ... y_t+12_d0 for the future. _d0 = traffic, _d1 = time-of-day.
    hist_offsets = [f"-{i}" for i in range(11, 0, -1)] + ["+0"]
    fut_offsets = [f"+{i}" for i in range(1, 13)]
    expected_hist = [f"x_t{o}_d0" for o in hist_offsets]
    expected_fut = [f"y_t{o}_d0" for o in fut_offsets]
    expected_hist_tod = [f"x_t{o}_d1" for o in hist_offsets]
    expected_fut_tod = [f"y_t{o}_d1" for o in fut_offsets]
    needed = ["t0_timestamp", "node_id", *expected_hist, *expected_fut,
              *expected_hist_tod, *expected_fut_tod]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet at {parquet_path} missing columns: {missing[:5]}...")

    df = df.sort_values(["t0_timestamp", "node_id"], kind="mergesort")
    n_nodes = int(df["node_id"].max()) + 1
    grouped = df.groupby("t0_timestamp", sort=True)
    n_samples = grouped.ngroups

    # Sanity check: every t0 should have all nodes present.
    counts = grouped.size().values
    if not (counts == n_nodes).all():
        raise ValueError(
            f"Inconsistent node count per timestamp in {parquet_path}: "
            f"min={counts.min()} max={counts.max()} expected={n_nodes}"
        )

    speed_cols = expected_hist + expected_fut                  # 24 columns
    tod_cols = expected_hist_tod + expected_fut_tod            # 24 columns
    speeds = df[speed_cols].values.astype(np.float32)          # (n_samples * N, 24)
    tods = df[tod_cols].values.astype(np.float32)              # (n_samples * N, 24)

    # Reshape: rows are sorted by (t0, node_id) so reshape gives (n_samples, N, 24).
    speeds = speeds.reshape(n_samples, n_nodes, history_size + num_preds)
    tods = tods.reshape(n_samples, n_nodes, history_size + num_preds)

    # Move time axis to position 1 -> (n_samples, T, N).
    speeds = speeds.transpose(0, 2, 1)
    tods = tods.transpose(0, 2, 1)
    timestamps = np.asarray(grouped.size().index.values)
    return speeds, tods, timestamps, n_nodes


def _make_synthetic(num_nodes: int, num_steps: int, seed: int = 0):
    """Random-walk adjacency + AR(1) speed signal. Useful for shape tests."""
    rng = np.random.RandomState(seed)
    coords = rng.randn(num_nodes, 2)
    d2 = ((coords[:, None] - coords[None]) ** 2).sum(-1)
    sigma2 = np.median(d2)
    adj = np.exp(-d2 / sigma2).astype(np.float32)
    np.fill_diagonal(adj, 0.0)

    speeds = np.zeros((num_steps, num_nodes), dtype=np.float32)
    speeds[0] = rng.randn(num_nodes) * 5 + 60
    rho = 0.85
    for t in range(1, num_steps):
        speeds[t] = rho * speeds[t - 1] + (1 - rho) * (60 + rng.randn(num_nodes) * 5)
    # 5-minute steps starting at the epoch.
    base = np.datetime64("2012-03-01T00:00:00", "m")
    times = base + (np.arange(num_steps) * 5).astype("timedelta64[m]")
    return speeds, times, adj


class MetrLADataset(Dataset):
    """Sliding-window dataset over METR-LA.

    Each item yields:
      x_hist  : (T_hist, N, F) raw node features (speed + tod sin/cos)
      x_fut   : (T_pred, N, F) ground-truth future features (for target encoding)
      signal  : (T_total, N) un-normalised speeds -- the regression target
      speeds  : alias of `signal`, kept for callers predating the rename
      active  : (T_total, N) bool, False where the sensor reading is missing
      mean    : (1,) speed mean used for de-normalisation at eval
      std     : (1,) speed std used for de-normalisation at eval

    Note there is deliberately NO per-item `adj`: the graph is constant, so
    materialising a (T_total, N, N) copy per window would dominate both the
    collate cost and GPU memory. Consumers read the dataset-level `self.adj`
    (N, N) instead -- see ``src/world_models/probes.py::graph_inputs``, which
    falls back to it whenever a batch carries no `adj`. That fallback is what
    lets the England-COVID probe/eval code run unchanged on METR-LA, so the
    two datasets report through one metric implementation.

    Args:
      data_dir:    folder containing metr-la.h5 + adj_mx.pkl
      split:       "train" | "val" | "test"  (chronological 70/10/20)
      history_size, num_preds: window lengths
      synthetic:   if True, ignore data_dir and use a synthetic 50-node graph
      stride:      step between consecutive windows (default 1, like DCRNN)
    """

    SPLITS: Tuple[float, float, float] = (0.7, 0.1, 0.2)

    def __init__(
        self,
        data_dir: Optional[str],
        split: str = "train",
        history_size: int = 12,
        num_preds: int = 12,
        synthetic: bool = False,
        stride: int = 1,
        synthetic_nodes: int = 50,
        synthetic_steps: int = 4000,
        normalize: bool = True,
    ):
        assert split in ("train", "val", "test")
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.window = history_size + num_preds
        self.stride = stride

        self.layout = "synthetic"
        if not synthetic and data_dir is not None:
            data_dir = Path(data_dir).expanduser()
            if (data_dir / f"{split}.parquet").exists():
                self.layout = "hf_parquet"
            elif (data_dir / "metr-la.h5").exists():
                self.layout = "h5"

        if self.layout == "synthetic":
            speeds, times, adj = _make_synthetic(synthetic_nodes, synthetic_steps)
            self._build_h5_style(
                speeds, times, adj, split, normalize, stride,
            )
        elif self.layout == "h5":
            speeds, times = _load_speed_dataframe(data_dir / "metr-la.h5")
            adj = _load_adjacency(data_dir / "adj_mx.pkl")
            self._build_h5_style(
                speeds, times, adj, split, normalize, stride,
            )
        else:
            self._build_hf_parquet(data_dir, split, history_size, num_preds,
                                   normalize, stride)

    # ------------------------------------------------------------------
    # H5 / synthetic path: contiguous time series, sliding windows
    # ------------------------------------------------------------------
    def _build_h5_style(self, speeds, times, adj, split, normalize, stride):
        self.num_nodes = speeds.shape[1]
        T_total = speeds.shape[0]
        train_end = int(T_total * self.SPLITS[0])
        val_end = train_end + int(T_total * self.SPLITS[1])

        if normalize:
            train_speeds = speeds[:train_end]
            mask = train_speeds > 0
            self.mean = float(train_speeds[mask].mean())
            self.std = float(train_speeds[mask].std() + 1e-6)
        else:
            self.mean, self.std = 0.0, 1.0

        speeds_norm = (speeds - self.mean) / self.std
        tod = time_of_day(times)
        tod_sin = np.sin(2 * np.pi * tod)
        tod_cos = np.cos(2 * np.pi * tod)

        feats = np.stack(
            [
                speeds_norm,
                np.broadcast_to(tod_sin[:, None], speeds.shape).copy(),
                np.broadcast_to(tod_cos[:, None], speeds.shape).copy(),
            ],
            axis=-1,
        ).astype(np.float32)

        if split == "train":
            seg, seg_speeds = feats[:train_end], speeds[:train_end]
        elif split == "val":
            seg, seg_speeds = feats[train_end:val_end], speeds[train_end:val_end]
        else:
            seg, seg_speeds = feats[val_end:], speeds[val_end:]

        self.feats = seg
        self.raw_speeds = seg_speeds
        self.adj = adj.astype(np.float32)

        max_start = len(seg) - self.window
        if max_start < 0:
            raise ValueError(
                f"Split '{split}' has only {len(seg)} steps but window = {self.window}."
            )
        self.starts = np.arange(0, max_start + 1, stride, dtype=np.int64)

    # ------------------------------------------------------------------
    # HF parquet path: rows are pre-windowed (sensor, t0) tuples
    # ------------------------------------------------------------------
    def _build_hf_parquet(self, data_dir, split, history_size, num_preds,
                          normalize, stride=1):
        speeds, tods, _, n_nodes = _load_hf_parquet(
            data_dir / f"{split}.parquet", history_size, num_preds,
        )
        # speeds: (n_samples, T, N)  tods: (n_samples, T, N)
        self.num_nodes = n_nodes

        if normalize:
            if split == "train":
                train_speeds = speeds
            else:
                train_path = data_dir / "train.parquet"
                if train_path.exists():
                    tr_speeds, *_ = _load_hf_parquet(
                        train_path, history_size, num_preds,
                    )
                    train_speeds = tr_speeds
                else:
                    train_speeds = speeds
            mask = train_speeds > 0
            self.mean = float(train_speeds[mask].mean()) if mask.any() else 0.0
            self.std = float(train_speeds[mask].std() + 1e-6) if mask.any() else 1.0
        else:
            self.mean, self.std = 0.0, 1.0

        speeds_norm = (speeds - self.mean) / self.std
        tod_sin = np.sin(2 * np.pi * tods)
        tod_cos = np.cos(2 * np.pi * tods)

        # (n_samples, T, N, 3): [speed_norm, tod_sin, tod_cos]
        windows = np.stack([speeds_norm, tod_sin, tod_cos], axis=-1).astype(np.float32)
        self.windows = windows
        self.raw_window_speeds = speeds.astype(np.float32)

        adj_path = data_dir / "sensor_graph" / "adj_mx.npy"
        adj = np.load(adj_path).astype(np.float32)
        self.adj = adj
        self.starts = np.arange(0, len(self.windows), stride, dtype=np.int64)

        # The parquet release is PRE-WINDOWED, so it has no contiguous (T, N, F)
        # series -- but the windows are stride-1, so window i begins at timestep
        # i and `windows[:, 0]` reconstructs that series exactly. Exposing it
        # under the same names the h5 path uses (`feats` / `raw_speeds`) is what
        # lets the snapshot consumers -- CLDG / DDGCL / IDOL via
        # baselines.common._fixed_topology_snapshots -- run on the REAL dataset.
        # Without it they see no `.feats` and abort; that gap was invisible when
        # testing against the synthetic generator, which takes the h5 path.
        # NB: derived from the FULL window list, not the strided subset, so the
        # snapshot timeline stays contiguous regardless of --metr_la_stride.
        self.feats = windows[:, 0]                          # (T, N, F)
        self.raw_speeds = self.raw_window_speeds[:, 0]      # (T, N)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        if self.layout == "hf_parquet":
            # Index through `starts`, not `idx`: otherwise --metr_la_stride
            # would shorten the epoch while still handing back the first N
            # consecutive windows, i.e. silently training on a contiguous
            # PREFIX of the split rather than a uniform subsample of it.
            start = int(self.starts[idx])
            clip = self.windows[start]                              # (T, N, F)
            speeds = self.raw_window_speeds[start]                  # (T, N)
        else:
            start = int(self.starts[idx])
            clip = self.feats[start : start + self.window]
            speeds = self.raw_speeds[start : start + self.window]
        x_hist = torch.from_numpy(clip[: self.history_size])
        x_fut = torch.from_numpy(clip[self.history_size :])
        speeds_t = torch.from_numpy(speeds)
        # A missing loop-detector reading is stored as 0.0 in the DCRNN release,
        # which is also what every published METR-LA number masks out. Exposing
        # it as `active` lets the shared probe apply the same rule it applies to
        # England COVID instead of each call site re-deriving `speeds > 0`.
        active_t = speeds_t > 0
        return {
            "x_hist": x_hist,
            "x_fut": x_fut,
            "signal": speeds_t,         # un-normalised, for MAE eval
            "speeds": speeds_t,         # legacy alias (fixed-topology probe)
            "active": active_t,
            "start_idx": start,
        }

    @property
    def feature_dim(self) -> int:
        if self.layout == "hf_parquet":
            return self.windows.shape[-1]
        return self.feats.shape[-1]
