"""tgbn-genre / reddit / token loader: raw formats, weekly alignment, time
cut, sparse weight features, and the moving-average floor in the probe."""
from pathlib import Path

import numpy as np
import pytest

from src.custom_codes.tgbn_dataset import (
    DAY,
    TGBNInteractionDataset,
    parse_tgbn_interaction,
    parse_tgbn_trade,
    tgbn_files,
    weight_features,
    weight_features_sparse,
)
from src.train.analyze_jepa import ndcg_at_k

T0 = 1_600_000_000                     # some Monday-ish unix second


def _write_toy(root: Path, name: str, n_users=5, n_classes=4, weeks=8, seed=0,
               daily_labels=False):
    """A weekly user->class stream in the dataset's raw column layout. Labels
    for week k (k >= 1) are EXACTLY the normalised interactions of week k, as
    in TGB ('next week'), written at the week's start; week 0 is the lead-in.
    ``daily_labels`` also writes a (wrong-looking) copy of each week's label at
    every day of the week, to exercise the earliest-label-per-bin rule."""
    rng = np.random.RandomState(seed)
    users = [f"u{i}" for i in range(n_users)]
    classes = [f"c{i}" for i in range(n_classes)]
    e_rows, l_rows = [], []
    for k in range(weeks):
        wk = {}
        for u in users:
            for _ in range(rng.randint(2, 6)):
                c = classes[rng.randint(n_classes)]
                w = float(rng.randint(1, 5))
                ts = T0 + k * 7 * DAY + rng.randint(0, 7 * DAY)
                e_rows.append((ts, u, c, w))
                wk[(u, c)] = wk.get((u, c), 0.0) + w
        if k >= 1:
            for (u, c), w in wk.items():
                tot = sum(v for (uu, _), v in wk.items() if uu == u)
                days = range(7) if daily_labels else [0]
                for d in days:
                    l_rows.append((T0 + k * 7 * DAY + d * DAY, u, c, w / tot))
    root.mkdir(parents=True, exist_ok=True)
    ef, lf = tgbn_files(name)
    with open(root / ef, "w") as f:
        if name == "genre":
            f.write("time,user_id,genre,weight\n")
            for ts, u, c, w in e_rows: f.write(f"{ts},{u},{c},{w}\n")
        elif name == "reddit":
            f.write("ts,src,subreddit,num_words,score\n")
            for ts, u, c, w in e_rows: f.write(f"{ts},{u},{c},{int(w * 10)},{int(w)}\n")
        else:
            f.write("timestamp,user_address,token_address,value,IsSender\n")
            for ts, u, c, w in e_rows: f.write(f"{ts},{u},{c},{w},{1}\n")
    with open(root / lf, "w") as f:
        f.write("ts,src,dst,w\n")
        for ts, u, c, w in l_rows: f.write(f"{ts},{u},{c},{w:.8f}\n")
    return users, classes, weeks


@pytest.mark.parametrize("name", ["genre", "reddit", "token"])
def test_parse_formats_and_alignment(tmp_path, name):
    users, classes, weeks = _write_toy(tmp_path, name)
    d = parse_tgbn_interaction(tmp_path, name)
    U, C = len(users), len(classes)
    assert int(d["n_users"]) == U and list(d["classes"]) == classes
    assert list(d["nodes"][:U]) == users
    T = d["labels"].shape[0]
    assert T == weeks and d["labels"].shape == (T, U + C, C)
    assert not (d["labels"][0] > 0).any()                       # lead-in unlabelled
    assert np.allclose(d["labels"][1:, :U].sum(-1), 1.0, atol=1e-5)
    assert not (d["labels"][:, U:] > 0).any()                   # class nodes unlabelled
    assert d["agreement"] > 0.999                               # label == snapshot ranking
    assert (tmp_path / f"tgbn-{name}_b7_k0.npz").exists()
    # edges: user -> U + class, bins 0..T-1
    e = d["edges"]
    assert e[:, 1].max() < U and e[:, 2].min() >= U and e[:, 0].max() == T - 1


def test_daily_labels_take_earliest_per_bin(tmp_path):
    _write_toy(tmp_path, "genre", daily_labels=True)
    d = parse_tgbn_interaction(tmp_path, "genre")
    assert d["agreement"] > 0.999 and np.allclose(d["labels"][1:, :5].sum(-1), 1.0, atol=1e-5)


def test_max_bins_time_cut(tmp_path):
    users, classes, weeks = _write_toy(tmp_path, "genre", weeks=10)
    full = parse_tgbn_interaction(tmp_path, "genre")
    cut = parse_tgbn_interaction(tmp_path, "genre", max_bins=4)
    assert cut["labels"].shape[0] == 5                          # 4 labelled + lead-in
    assert not (cut["labels"][0] > 0).any() and (cut["labels"][1:].sum(-1) > 0).any()
    # the kept weeks' labels are the same vectors as in the full parse
    U = int(full["n_users"])
    assert np.allclose(cut["labels"][1:, :U], full["labels"][-4:, :U], atol=1e-6)


def test_sparse_weight_features_match_dense(tmp_path):
    from tests.test_tgbn_dataset import _write_toy as _write_trade
    _write_trade(tmp_path)
    d = parse_tgbn_trade(tmp_path)
    dense = weight_features(d["edges"], 8, 6)
    sparse = weight_features_sparse(d["edges"], 8, 6)
    assert np.allclose(dense, sparse, atol=1e-6)


def test_loader_batches_and_probe_attributes(tmp_path):
    users, classes, weeks = _write_toy(tmp_path, "reddit")
    ds = TGBNInteractionDataset(name="reddit", data_dir=str(tmp_path), split="train",
                                history_size=2, num_preds=1)
    U, C = len(users), len(classes)
    assert ds.num_nodes == U + C and ds.num_users == U and ds.num_classes == C
    assert ds.state_label_mode == "ranking" and ds.feature_dim == 13
    assert ds.node_label_dist.shape == (weeks, U + C, C) and ds.label_valid.shape == (weeks, U + C)
    assert not ds.label_valid[:, U:].any()
    item = ds[0]
    assert item["x_hist"].shape == (2, U + C, 13) and item["adj"].shape == (3, U + C, U + C)
    assert (item["adj"] == item["adj"].transpose(1, 2)).all()
    assert not item["adj"][:, :U, :U].any()                      # bipartite: no user-user edges


def test_moving_average_floor_in_probe(tmp_path):
    """The probe's two floors on a stream whose labels drift: the moving
    average over a window is at least as good as persistence in expectation,
    and both are reported with row counts."""
    import sys
    sys.path.insert(0, "scripts/dblp")
    from run_block_probe import _per_row_label_vectors
    users, classes, weeks = _write_toy(tmp_path, "genre", weeks=12)
    ds = TGBNInteractionDataset(name="genre", data_dir=str(tmp_path), split="train",
                                history_size=3, num_preds=1)
    U = len(users)
    node_id = np.repeat(np.arange(U), weeks - 4)
    abs_bin = np.tile(np.arange(4, weeks), U)
    Y, valid, Yprev, pv, Yavg, av = _per_row_label_vectors(ds, node_id, abs_bin, history_size=3)
    assert valid.all() and pv.all() and av.all()
    assert np.allclose(Yavg.sum(-1), 1.0, atol=1e-5)            # mean of 3 normalised rows
    # window of 1 == persistence
    _, _, _, _, Y1, _ = _per_row_label_vectors(ds, node_id, abs_bin, history_size=1)
    assert np.allclose(Y1, Yprev, atol=1e-6)
    assert np.isfinite(ndcg_at_k(Y, Yavg, 10).mean())
