"""BTS flights loader: raw monthly format (zip and csv, header + trailing
comma), daily and sub-daily binning, delay labels and validity, features,
the label audit, and the probe's dynamic-label contract."""
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.custom_codes.bts_flights_dataset import (
    BTS_STEM,
    BTSFlightsDataset,
    delay_labels,
    label_audit,
    parse_bts_flights,
    structure_audit,
)

HEADER = ("Year,Quarter,Month,DayofMonth,DayOfWeek,FlightDate,Reporting_Airline,Tail_Number,"
          "Origin,Dest,CRSDepTime,DepTime,DepDelay,DepDelayMinutes,DepDel15,CRSArrTime,ArrTime,"
          "ArrDelay,ArrDelayMinutes,ArrDel15,Cancelled,Diverted,Flights,Distance,")
AIRPORTS = ["ATL", "BOS", "DEN", "JFK", "LAX", "SFO"]


def _write_toy(root: Path, year=2019, months=(1, 2), seed=0, as_zip=True,
               delay_pattern=None):
    """A few months of flights. ``delay_pattern(day, airport_idx) -> share
    delayed`` sets each airport-day's delay share exactly (default: a random
    share so the state recurs)."""
    import calendar
    rng = np.random.RandomState(seed)
    root.mkdir(parents=True, exist_ok=True)
    for m in months:
        rows = []
        for day in range(1, calendar.monthrange(year, m)[1] + 1):
            date = f"{year}-{m:02d}-{day:02d}"
            for i, o in enumerate(AIRPORTS):
                n = 8 if o != "SFO" else 2                       # SFO below min_departures
                share = (delay_pattern(day, i) if delay_pattern else rng.rand())
                n_del = int(round(share * n))
                for k in range(n):
                    d = AIRPORTS[(i + 1 + k) % len(AIRPORTS)]
                    dep = (6 + 2 * k) * 100 + 15                   # 06:15, 08:15, ...
                    delayed = k < n_del
                    mins = 40 if delayed else 0
                    cancelled = 1 if (k == n - 1 and rng.rand() < 0.1) else 0
                    rows.append(f"{year},1,{m},{day},1,{date},AA,N1,{o},{d},{dep},{dep},"
                                f"{mins},{mins},{int(delayed)},{dep+200},{dep+200},{mins},{mins},"
                                f"{int(delayed)},{cancelled},0,1,500,")
        body = HEADER + "\n" + "\n".join(rows) + "\n"
        if as_zip:
            with zipfile.ZipFile(root / f"{BTS_STEM}_{year}_{m}.zip", "w") as z:
                z.writestr(f"On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{m}.csv", body)
                z.writestr("readme.html", "<html/>")
        else:
            (root / f"{BTS_STEM}_{year}_{m}.csv").write_text(body)


@pytest.mark.parametrize("as_zip", [True, False])
def test_parse_daily(tmp_path, as_zip):
    _write_toy(tmp_path, as_zip=as_zip)
    d = parse_bts_flights(tmp_path, [2019], bin_hours=24)
    assert list(d["airports"]) == AIRPORTS
    T, N = d["ops"].shape[:2]
    assert (T, N) == (31 + 28, 6)
    assert d["ops"][..., 0].sum() > 0 and d["ops"][..., 7].sum() >= d["ops"][..., 0].sum()
    e = d["edges"]
    assert e[:, 0].max() == T - 1 and (e[:, 3] >= 1).all()
    assert (tmp_path / "bts_flights_2019-2019_h24.npz").exists()
    assert d["dow"][0] == 1                                       # 2019-01-01 was a Tuesday


def test_sub_daily_bins(tmp_path):
    _write_toy(tmp_path, months=(1,))
    d = parse_bts_flights(tmp_path, [2019], bin_hours=6)
    T = d["ops"].shape[0]
    assert T == 31 * 4
    # departures are scheduled 06:15..20:15 -> never in the 00-06 bin
    assert d["ops"][0::4, :, 0].sum() == 0 and d["ops"][1::4, :, 0].sum() > 0
    with pytest.raises(ValueError):
        parse_bts_flights(tmp_path, [2019], bin_hours=5)


def test_labels_threshold_and_validity(tmp_path):
    _write_toy(tmp_path, months=(1,), delay_pattern=lambda day, i: 0.5 if (day + i) % 2 else 0.0)
    d = parse_bts_flights(tmp_path, [2019], bin_hours=24)
    lab, sig = delay_labels(d["ops"], "share_del15", 0.30, min_departures=5)
    assert (lab[:, 5] == -1).all()                                # SFO: 2 departures < 5
    assert set(np.unique(lab[:, :5])) == {0, 1}
    # pattern: share 0.5 when (day + i) is odd; day 1 = bin 0
    assert lab[0, 0] == 1 and lab[1, 0] == 0 and lab[0, 1] == 0
    lab2, _ = delay_labels(d["ops"], "share_del15", 0.60, 5)
    assert lab2[:, :5].max() == 0                                 # 0.5 share below 0.6
    lab3, sig3 = delay_labels(d["ops"], "mean_delay", 15.0, 5)
    assert (lab3[:, :5] == lab[:, :5]).all()                      # 40-min delays: same state
    a = label_audit(lab, np.ones_like(lab, dtype=bool))
    assert a["nodes_reverting"] == pytest.approx(5 / 6) and a["persistence_acc"] == pytest.approx(0.0)
    assert a["switch_rate"] == pytest.approx(1.0)


def test_loader_and_probe_contract(tmp_path):
    _write_toy(tmp_path, months=(1, 2, 3))
    ds = BTSFlightsDataset(data_dir=str(tmp_path), years=[2019], split="train",
                           history_size=2, num_preds=1)
    assert ds.num_nodes == 6 and ds.feature_dim == 8 + 9
    assert ds.node_labels_dyn.shape == (90, 6) and ds.node_signal.shape == (90, 6)
    assert ds.state_label_mode.startswith("delay:") and ds.n_classes == 2
    assert (ds.node_labels_dyn[:, 5] == -1).all()
    item = ds[0]
    assert item["x_hist"].shape == (2, 6, 17) and item["adj"].shape == (3, 6, 6)
    assert (item["adj"] == item["adj"].transpose(1, 2)).all() and not np.diag(item["adj"][0].numpy()).any()
    ds2 = BTSFlightsDataset(data_dir=str(tmp_path), years=[2019], split="train",
                            with_ops_features=False)
    assert ds2.feature_dim == 8
    ds6 = BTSFlightsDataset(data_dir=str(tmp_path), years=[2019], split="train", bin_hours=6)
    assert ds6.feature_dim == 8 + 11
    # the probe resolves per-row labels from node_labels_dyn
    import sys; sys.path.insert(0, "scripts/dblp")
    from run_block_probe import _per_row_labels
    node_id = np.array([0, 1, 5]); abs_bin = np.array([10, 10, 10])
    per_row, blk, src, prev = _per_row_labels(ds, node_id, abs_bin, "dataset", "binary")
    assert src.startswith("dataset_dynamic:delay") and blk is None
    assert per_row[2] == -1 and set(per_row[:2]) <= {0, 1}
    assert (prev == ds.node_labels_dyn[9, node_id]).all()


def test_structure_audit():
    adj = np.zeros((3, 4, 4), dtype=bool)
    adj[0, 0, 1] = adj[0, 1, 0] = True                        # t0: {01}
    adj[1, 0, 1] = adj[1, 1, 0] = adj[1, 2, 3] = adj[1, 3, 2] = True   # t1: {01, 23}
    adj[2, 2, 3] = adj[2, 3, 2] = True                        # t2: {23}
    st = structure_audit(adj)
    assert st["jaccard"] == pytest.approx((0.5 + 0.5) / 2)
    assert st["new_share"] == pytest.approx((0.5 + 0.0) / 2)
    assert st["gone_share"] == pytest.approx((0.0 + 0.5) / 2)
    assert st["ever_share"] == pytest.approx((1 + 2 + 1) / 3 / 2)
