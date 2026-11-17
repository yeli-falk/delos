"""NYC taxi loader: raw monthly format (csv, parquet when pyarrow is there),
hourly and coarser binning, the trip sanity filter, congestion labels and
validity, the train-only cut-off, features, the trivial floors, and the
probe's dynamic-label contract."""
from pathlib import Path

import numpy as np
import pytest

from src.custom_codes.nyc_taxi_dataset import (
    NYCTaxiDataset,
    congestion_labels,
    parse_nyc_taxi,
    trivial_floors,
)

ZONES = [10, 20, 30, 40, 50]          # LocationIDs; 50 stays below min_trips


def _toy_frame(month="2019-01", days=28, seed=0, speed_fn=None):
    """Hourly trips for ``days`` days. ``speed_fn(day, hour, zone_idx) -> mph``
    sets each zone-hour's speed exactly (default: slow in the 8h and 17h
    peaks on weekdays, i.e. a daily + weekly cycle)."""
    import pandas as pd
    rng = np.random.RandomState(seed)
    start = pd.Timestamp(month + "-01")
    rows = []
    for day in range(days):
        weekday = (start.dayofweek + day) % 7 < 5
        for hour in range(24):
            for i, z in enumerate(ZONES):
                n = 12 if z != 50 else 2
                if speed_fn is not None:
                    mph = speed_fn(day, hour, i)
                else:
                    mph = 8.0 if (weekday and hour in (8, 17)) else 20.0 + rng.rand()
                for k in range(n):
                    pu = start + pd.Timedelta(days=day, hours=hour, minutes=2 * k)
                    miles = 2.0
                    do = pu + pd.Timedelta(hours=miles / mph)
                    rows.append((pu, do, z, ZONES[(i + 1 + k) % len(ZONES)], miles))
    # junk the loader must drop: unknown zone, stray year, zero-duration trip
    rows.append((start, start + pd.Timedelta(minutes=5), 264, 10, 1.0))
    rows.append((pd.Timestamp("2009-01-01"), pd.Timestamp("2009-01-01 00:10"), 10, 20, 1.0))
    rows.append((start, start, 10, 20, 3.0))
    return pd.DataFrame(rows, columns=["tpep_pickup_datetime", "tpep_dropoff_datetime",
                                       "PULocationID", "DOLocationID", "trip_distance"])


def _write_toy(root: Path, months=("2019-01",), fmt="csv", **kw):
    root.mkdir(parents=True, exist_ok=True)
    for j, m in enumerate(months):
        df = _toy_frame(m, seed=j, **kw)
        df["fare_amount"] = 10.0                               # an unused column
        if fmt == "csv":
            df.to_csv(root / f"yellow_tripdata_{m}.csv", index=False)
        else:
            df.to_parquet(root / f"yellow_tripdata_{m}.parquet")


def test_parse_hourly(tmp_path):
    _write_toy(tmp_path)
    d = parse_nyc_taxi(tmp_path, ["2019-01"], bin_hours=1)
    T, N = d["ops"].shape[:2]
    assert (T, N) == (31 * 24, 263)
    ops = d["ops"]
    assert ops[0, 9, 0] == 12 + 1 and ops[0, 9, 2] == 12       # zero-duration trip: counted, no speed
    assert ops[:, 263 - 1].sum() == 0 and ops[28 * 24:].sum() == 0
    assert ops[0, 9, 3] / ops[0, 9, 4] == pytest.approx(20.5, abs=0.6)
    e = d["edges"]
    assert e.dtype == np.int32 and e[:, 0].max() == 28 * 24 - 1 and (e[:, 3] >= 1).all()
    assert d["dow"][0] == 1 and d["bin_hour"][25] == 1           # 2019-01-01 was a Tuesday
    assert (tmp_path / "nyc_yellow_2019-01_2019-01_h1.npz").exists()
    d2 = parse_nyc_taxi(tmp_path, ["2019-01"], bin_hours=1)      # from the cache
    assert (d2["ops"] == ops).all()
    with pytest.raises(ValueError):
        parse_nyc_taxi(tmp_path, ["2019-01"], bin_hours=5)
    with pytest.raises(FileNotFoundError):
        parse_nyc_taxi(tmp_path, ["2019-02"])


def test_parquet_and_coarse_bins(tmp_path):
    pytest.importorskip("pyarrow")
    _write_toy(tmp_path, months=("2019-01", "2019-02"), fmt="parquet")
    d = parse_nyc_taxi(tmp_path, ["2019-01", "2019-02"], bin_hours=6)
    assert d["ops"].shape[0] == (31 + 28) * 4
    assert d["ops"][31 * 4, 9, 0] == 6 * 12 + 1                  # February starts at bin 124


def test_labels_cutoff_and_validity(tmp_path):
    _write_toy(tmp_path)
    d = parse_nyc_taxi(tmp_path, ["2019-01"], bin_hours=1)
    ops = d["ops"][:28 * 24, [9, 19, 29, 39, 49]]
    lab, sig, cut = congestion_labels(ops, "rel_speed", 0.25, min_trips=10)
    assert (lab[:, 4] == -1).all() and np.isnan(cut[4])          # zone 50: 2 trips < 10
    # the zone's own lowest quarter of hours, which contains the weekday peaks
    assert lab[8, 0] == 1 and lab[17, 0] == 1
    assert lab[:, :4].mean() == pytest.approx(0.25, abs=0.01)
    assert (sig[lab == 1] < 1).all() and (sig[lab == 0] >= 1).all()
    # weekday peaks (8h, 17h) are the only hours under 10 mph: 2019-01-01 is a Tuesday
    lab_abs, _, cut_abs = congestion_labels(ops, "abs_speed", 10.0, min_trips=10)
    sat = 4 * 24                                                 # 2019-01-05
    assert (cut_abs == 10.0).all() and lab_abs[8, 0] == 1 and lab_abs[9, 0] == 0
    assert lab_abs[sat + 8, 0] == 0 and lab_abs[:, :4].mean() == pytest.approx(40 / 672)
    # the cut-off only reads the train bins
    _, _, cut_tr = congestion_labels(ops, "rel_speed", 0.25, 10, n_train_bins=8)
    assert cut_tr[0] > 19.0                                      # hours 0-7: no peak seen yet
    with pytest.raises(ValueError):
        congestion_labels(ops, "rel_speed", 10.0)


def test_trivial_floors():
    # a pure 4-bin cycle: copying the previous bin is wrong, copying one period back is exact
    T = 40
    y = np.tile(np.array([1, 0, 0, 0]), T // 4)[:, None].repeat(3, 1)
    sig = np.where(y == 1, 0.5, 1.5).astype(np.float32)
    f = trivial_floors(y, sig, first_bin=20, lags={"last": 1, "period": 4, "too_long": 99})
    assert f["period"] == pytest.approx(1.0) and f["last"] < 0.5 and np.isnan(f["too_long"])


def test_loader_and_probe_contract(tmp_path):
    _write_toy(tmp_path)
    ds = NYCTaxiDataset(data_dir=str(tmp_path), months=["2019-01"], split="train",
                        history_size=4, num_preds=1)
    assert ds.num_nodes == 5 and list(ds.zones) == ZONES and ds.feature_dim == 8 + 10
    assert ds.node_labels_dyn.shape == (31 * 24, 5) and ds.node_signal.shape == (31 * 24, 5)
    assert ds.state_label_mode.startswith("congestion:") and ds.n_classes == 2
    assert (ds.node_labels_dyn[:, 4] == -1).all()
    assert ds.floors["same_bin_last_week"] > ds.floors["copy_last"]
    item = ds[0]
    assert item["x_hist"].shape == (4, 5, 18) and item["adj"].shape == (5, 5, 5)
    assert (item["adj"] == item["adj"].transpose(1, 2)).all()
    ds2 = NYCTaxiDataset(data_dir=str(tmp_path), months=["2019-01"], split="train",
                         with_ops_features=False, min_edge_trips=4)
    assert ds2.feature_dim == 8
    assert ds2.structure["ever_share"] <= ds.structure["ever_share"] + 1e-9
    ds24 = NYCTaxiDataset(data_dir=str(tmp_path), months=["2019-01"], split="train", bin_hours=24)
    assert ds24.feature_dim == 8 + 8
    # the probe resolves per-row labels from node_labels_dyn
    import sys; sys.path.insert(0, "scripts/dblp")
    from run_block_probe import _per_row_labels
    node_id = np.array([0, 1, 4]); abs_bin = np.array([10, 10, 10])
    per_row, blk, src, prev = _per_row_labels(ds, node_id, abs_bin, "dataset", "binary")
    assert src.startswith("dataset_dynamic:congestion") and blk is None
    assert per_row[2] == -1 and set(per_row[:2]) <= {0, 1}
    assert (prev == ds.node_labels_dyn[9, node_id]).all()
