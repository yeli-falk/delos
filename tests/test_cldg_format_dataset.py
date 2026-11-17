"""Unit tests for the CLDG-format loader (DBLP-10 / Patent-6) and its binning."""
from pathlib import Path

import numpy as np
import pytest

from src.custom_codes.cldg_format_dataset import (
    CLDGFormatDynamicGraphDataset,
    bin_cldg_stream,
    build_cldg_snapshots,
)
from src.custom_codes.dtdg.binning import (
    equal_frequency_bins,
    equal_width_bins,
    native_bins,
)


def _write_toy(root: Path, name: str = "patent", n_nodes: int = 12,
               feat_dim: int = 4, seed: int = 0):
    """A tiny CLDG-format folder: a ring + a few extra edges over 4 distinct
    timestamps, 3 classes, node ids in shuffled order in node2label (as the
    real patent file has), sinusoid-like features."""
    rng = np.random.RandomState(seed)
    folder = root / name
    folder.mkdir(parents=True)
    ts = [0.0, 0.25, 0.25, 0.5, 1.0]
    rows = []
    for i in range(n_nodes):
        rows.append((i, (i + 1) % n_nodes, ts[i % len(ts)]))
    rows.append((0, 5, 1.0)); rows.append((0, 5, 1.0))          # duplicate edge
    rows.append((3, 3, 0.5))                                    # self loop
    with open(folder / f"{name}.txt", "w") as f:
        for s, d, t in rows:
            f.write(f"{s} {d} {t:.6f}\n")
    labels = np.array([1 + (i % 3) for i in range(n_nodes)])   # raw ids 1..3
    order = rng.permutation(n_nodes)
    with open(folder / "node2label.txt", "w") as f:
        for i in order:
            f.write(f"{i} {labels[i]}\n")
    feats = rng.randn(n_nodes, feat_dim).astype(np.float32)
    np.savetxt(folder / "feature.txt", feats, fmt="%.6f")
    return folder, np.array([[s, d, t] for s, d, t in rows]), labels, feats


# --------------------------------------------------------------------------
# binning primitives
# --------------------------------------------------------------------------
def test_native_bins_one_per_distinct_timestamp():
    t = np.array([0.0, 0.5, 0.5, 1.0, 0.0])
    b, T = native_bins(t)
    assert T == 3
    assert b.tolist() == [0, 1, 1, 2, 0]
    # reference stream decides the grid, the subset only gets mapped
    b2, T2 = native_bins(np.array([0.5, 1.0]), reference=t)
    assert (T2, b2.tolist()) == (3, [1, 2])


def test_equal_frequency_bins_balance_and_ties():
    rng = np.random.RandomState(0)
    t = np.sort(rng.beta(8, 1, size=10_000))          # heavily right-skewed
    b = equal_frequency_bins(t, 10)
    counts = np.bincount(b, minlength=10)
    assert counts.min() > 800 and counts.max() < 1200   # ~1000 each
    # equal-width on the same stream leaves early bins nearly empty
    w = np.bincount(equal_width_bins(t, 10), minlength=10)
    assert w[0] < 50
    # tied timestamps never split across bins
    tied = np.array([0.1] * 50 + [0.2] * 50 + [0.3] * 50)
    bt = equal_frequency_bins(tied, 3)
    for v in (0.1, 0.2, 0.3):
        assert len(np.unique(bt[tied == v])) == 1
    assert equal_frequency_bins(tied, 3).max() <= 2


def test_bin_cldg_stream_top_k_and_symmetry():
    rng = np.random.RandomState(1)
    n = 40
    src = rng.randint(0, n, 500); dst = rng.randint(0, n, 500); t = rng.rand(500)
    # make node 0 a hub so it is certainly in the top-K
    src[:100] = 0
    adj, active, keep, bins = bin_cldg_stream(
        src, dst, t, num_bins=5, binning="equal_freq", top_k=10)
    assert adj.shape == (5, 10, 10) and active.shape == (5, 10)
    assert keep.shape == (10,) and 0 in keep and np.all(np.diff(keep) > 0)
    assert (adj == adj.transpose(0, 2, 1)).all()
    assert not adj[:, np.arange(10), np.arange(10)].any()      # no self loops
    assert (active == adj.any(-1)).all()
    # full graph when top_k == 0
    adj_full, _, keep_full, _ = bin_cldg_stream(
        src, dst, t, num_bins=5, binning="equal_width", top_k=0)
    assert adj_full.shape[1] == n and keep_full.tolist() == list(range(n))


# --------------------------------------------------------------------------
# the loader on a toy folder
# --------------------------------------------------------------------------
def test_build_snapshots_native_labels_features(tmp_path):
    folder, rows, labels, feats = _write_toy(tmp_path)
    seq, meta = build_cldg_snapshots("patent", data_dir=str(tmp_path),
                                     num_bins=0, top_k=0, with_file_features=True)
    T, N = seq.num_steps, seq.num_nodes
    assert (T, N) == (4, 12)                          # 4 distinct timestamps
    # labels densified 1..3 -> 0..2 and aligned to node id despite shuffled file
    assert seq.node_labels.tolist() == [(i % 3) for i in range(12)]
    assert meta["classes"].tolist() == [1, 2, 3]
    # duplicate edge collapsed, self loop dropped, symmetric
    assert seq.adj[3, 0, 5] and seq.adj[3, 5, 0]
    assert not seq.adj[2, 3, 3]
    assert (seq.adj == seq.adj.transpose(0, 2, 1)).all()
    # features: 8 graph stats + 4 file dims, file part equals the row when
    # active and is zero when inactive
    assert seq.feat_dim == 12
    for t in range(T):
        for n in range(N):
            if seq.active[t, n]:
                assert np.allclose(seq.feats[t, n, 8:], feats[n], atol=1e-5)
            else:
                assert not seq.feats[t, n].any()
    # cache files were written and are reused
    assert (folder / "patent.npz").exists() and (folder / "feature.npy").exists()
    seq2, _ = build_cldg_snapshots("patent", data_dir=str(tmp_path),
                                   num_bins=0, top_k=0, with_file_features=False)
    assert seq2.feat_dim == 8 and (seq2.adj == seq.adj).all()


def test_top_k_reindexes_labels_and_features(tmp_path):
    _, _, labels, feats = _write_toy(tmp_path, n_nodes=20)
    seq, meta = build_cldg_snapshots("patent", data_dir=str(tmp_path),
                                     num_bins=0, top_k=8, with_file_features=True)
    keep = meta["node_index"]
    assert seq.num_nodes == 8 and len(keep) == 8
    assert seq.node_labels.tolist() == [(i % 3) for i in keep]
    t, n = np.argwhere(seq.active)[0]
    assert np.allclose(seq.feats[t, n, 8:], feats[keep[n]], atol=1e-5)


def test_dataset_windows_and_splits(tmp_path):
    _write_toy(tmp_path, n_nodes=16)
    kw = dict(data_dir=str(tmp_path), num_bins=0, top_k=0,
              history_size=1, num_preds=1)
    train = CLDGFormatDynamicGraphDataset("patent", split="train", **kw)
    assert train.num_classes == 3 and train.node_labels_dyn is None
    assert train.state_label_mode == "static"
    assert train.feature_dim == train.feat_dim == 12
    item = train[0]
    assert item["x_hist"].shape == (1, 16, 12) and item["x_fut"].shape == (1, 16, 12)
    assert item["adj"].shape == (2, 16, 16) and item["active"].shape == (2, 16)
    # 4 steps -> train_end = 2, val_end = 2 (int(4*0.1)=0): val has no window
    with pytest.raises(ValueError):
        CLDGFormatDynamicGraphDataset("patent", split="val", **kw)
    test = CLDGFormatDynamicGraphDataset("patent", split="test", **kw)
    assert len(test) >= 1


def test_bad_inputs(tmp_path):
    _write_toy(tmp_path)
    with pytest.raises(ValueError):
        build_cldg_snapshots("patent", data_dir=str(tmp_path), num_bins=3,
                             binning="quantile", top_k=0)
    with pytest.raises(ValueError):
        build_cldg_snapshots("bitcoin", data_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        build_cldg_snapshots("dblp", data_dir=str(tmp_path))


# --------------------------------------------------------------------------
# the real files, when present (skipped otherwise)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not Path("dataset/cldg/patent/patent.txt").exists(),
                    reason="dataset/cldg/patent not extracted")
def test_real_patent_defaults():
    seq, meta = build_cldg_snapshots("patent", num_bins=30, binning="equal_freq",
                                     top_k=2000, with_file_features=False)
    assert seq.num_steps == 30 and seq.num_nodes == 2000
    assert len(meta["classes"]) == 6 and (seq.node_labels >= 0).all()
    assert meta["edges_per_bin"].min() > 0            # every snapshot populated


@pytest.mark.skipif(not Path("dataset/cldg/dblp/dblp.txt").exists(),
                    reason="dataset/cldg/dblp not extracted")
def test_real_dblp_native_years():
    seq, meta = build_cldg_snapshots("dblp", num_bins=0, top_k=2000,
                                     with_file_features=False)
    assert seq.num_steps == 27 and seq.num_nodes == 2000
    assert len(meta["classes"]) == 10 and (seq.node_labels >= 0).all()
