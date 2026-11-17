"""CHASE loader: safe unpickling, top-K subset, label masking, paper splits,
feature normalisation, and the probe's dynamic-label contract."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.custom_codes import chase_dataset as cd  # noqa: E402


def _toy(T=24, N=12, F=5, C=5, seed=0):
    """A Music-shaped toy release: nodes appear over time, labels drift."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((T, N))
    first = rng.integers(0, T // 2, size=N); first[0] = 0
    for i in range(N):
        mask[first[i]:, i] = 1
    labels = np.zeros((T, N))
    cur = rng.integers(0, C, size=N)
    for t in range(T):
        flip = rng.random(N) < 0.2
        cur = np.where(flip, rng.integers(0, C, size=N), cur)
        labels[t] = np.where(mask[t] == 1, cur, 0)
    feats = rng.normal(size=(T, N, F)).astype(np.float32) * 10 + 3
    feats *= mask[..., None]
    edge_indices, edge_weights = [], []
    for t in range(T):
        act = np.where(mask[t] == 1)[0]
        pairs = [(a, b) for a in act for b in act if a < b and rng.random() < 0.5]
        if not pairs:
            pairs = [(act[0], act[-1])] if len(act) > 1 else [(0, 1)]
        e = np.array(pairs, dtype=np.int64).T
        e = np.concatenate([e, e[::-1]], axis=1)              # both directions
        edge_indices.append(e)
        edge_weights.append(np.ones(e.shape[1], dtype=np.float32))
    return dict(edge_indices=edge_indices, edge_weights=edge_weights,
                feats=feats, labels=labels, mask=mask)


@pytest.fixture
def toy_dir(tmp_path):
    with open(tmp_path / "Music.pkl", "wb") as f:
        pickle.dump(_toy(), f, protocol=4)
    return tmp_path


def test_unpickler_accepts_numpy_and_refuses_other_globals(tmp_path):
    p = tmp_path / "ok.pkl"
    with open(p, "wb") as f:
        pickle.dump({"a": np.arange(3), "b": np.float32(1.5)}, f)
    with open(p, "rb") as f:
        d = cd._NumpyOnlyUnpickler(f).load()
    assert d["a"].tolist() == [0, 1, 2]
    bad = tmp_path / "bad.pkl"
    with open(bad, "wb") as f:
        pickle.dump(len, f)                                     # builtins.len
    with open(bad, "rb") as f, pytest.raises(pickle.UnpicklingError):
        cd._NumpyOnlyUnpickler(f).load()


def test_parse_masks_inactive_labels_and_symmetrises(toy_dir):
    d = cd.parse_chase(toy_dir / "Music.pkl", top_k=0)
    adj, active, labels = d["adj"], d["active"], d["labels"]
    assert adj.dtype == bool and adj.shape[0] == 24 and adj.shape[1] == 12
    assert (adj == adj.transpose(0, 2, 1)).all()
    assert not adj[:, np.arange(12), np.arange(12)].any()
    assert (labels[~active] == -1).all() and (labels[active] >= 0).all()
    assert (d["node_feats"].shape == (24, 12, 5))
    cache = cd._cache_path(toy_dir / "Music.pkl", 0)
    assert cache.exists()
    d2 = cd.parse_chase(toy_dir / "Music.pkl", top_k=0)        # from cache
    assert (d2["labels"] == labels).all()


def test_top_k_keeps_most_active_nodes(toy_dir):
    full = cd.parse_chase(toy_dir / "Music.pkl", top_k=0)
    deg = full["adj"].sum((0, 2))
    sub = cd.parse_chase(toy_dir / "Music.pkl", top_k=5)
    assert sub["adj"].shape[1:] == (5, 5) and len(sub["keep"]) == 5
    assert set(sub["keep"].tolist()) == set(np.argsort(-deg, kind="stable")[:5].tolist())
    keep = sub["keep"]
    assert (sub["labels"] == full["labels"][:, keep]).all()
    assert (sub["adj"] == full["adj"][:, keep][:, :, keep]).all()


def test_normalise_fits_on_train_rows_only():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 4, 3)) * 5 + 2
    fit = np.zeros((6, 4), dtype=bool); fit[:3] = True
    z = cd.normalise_node_features(x, fit, zscore=True)
    assert np.allclose(z[fit].mean(0), 0, atol=1e-5) and np.allclose(z[fit].std(0), 1, atol=1e-5)
    zp = cd.normalise_node_features(x, fit, zscore=True, pca=2)
    assert zp.shape == (6, 4, 2)
    raw = cd.normalise_node_features(x, fit, zscore=False, pca=0)
    assert np.allclose(raw, x.astype(np.float32))


def test_dataset_paper_splits_and_contract(toy_dir):
    ds = {s: cd.CHASEDataset("music", data_dir=str(toy_dir), split=s,
                             history_size=3, num_preds=1)
          for s in ("train", "val", "test")}
    tr, va, te = ds["train"], ds["val"], ds["test"]
    # Paper counts 15 / 4 / 5 snapshots: targets of the test windows are the
    # last 5 snapshots, val the 4 before.
    W = 4
    assert (te.starts + W - 1).tolist() == [19, 20, 21, 22, 23]
    assert (va.starts + W - 1).tolist() == [15, 16, 17, 18]
    assert (tr.starts + W - 1).min() == 3 and (tr.starts + W - 1).max() == 14
    assert tr.n_classes == 5 and tr.dyn_label_task == "multiclass"
    assert tr.node_labels_dyn.shape == (24, 12) and tr.feature_dim == 8 + 5
    b = tr[0]
    assert b["x_hist"].shape == (3, 12, 13) and b["adj"].shape == (4, 12, 12)
    assert 0 < tr.audit["persistence_acc"] < 1
    assert "jaccard" in tr.structure


def test_binary_dataset_sets_binary_task(tmp_path):
    toy = _toy(T=37, C=2)                                       # GossipCop's 26/5/6
    with open(tmp_path / "GossipCop.pkl", "wb") as f:
        pickle.dump(toy, f, protocol=4)
    ds = cd.CHASEDataset("gossipcop", data_dir=str(tmp_path), split="train",
                         history_size=3, num_preds=1, with_node_features=False)
    assert ds.n_classes == 2 and ds.dyn_label_task == "binary"
    assert ds.feature_dim == 8


def test_probe_label_path_uses_dynamic_multiclass(toy_dir):
    """run_block_probe._per_row_labels must take the dynamic path for a
    multiclass CHASE dataset (no mode 'community' prefix)."""
    from scripts.dblp.run_block_probe import _per_row_labels
    ds = cd.CHASEDataset("music", data_dir=str(toy_dir), split="train",
                         history_size=3, num_preds=1, with_node_features=False)
    node_id = np.array([0, 1, 2]); abs_bin = np.array([5, 6, 7])
    per_row, block, src, prev = _per_row_labels(ds, node_id, abs_bin, "dataset", "multiclass")
    assert block is None and src.startswith("dataset_dynamic:chase")
    assert (per_row == ds.node_labels_dyn[abs_bin, node_id]).all()
    assert (prev == ds.node_labels_dyn[abs_bin - 1, node_id]).all()
