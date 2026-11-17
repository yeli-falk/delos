"""Unit tests for the tgbn-trade loader, the NDCG@10 metric and the ranking head."""
from pathlib import Path

import numpy as np
import pytest

from src.custom_codes.tgbn_dataset import (
    EDGE_FILE,
    LABEL_FILE,
    TGBNTradeDataset,
    parse_tgbn_trade,
    weight_features,
)
from src.train.analyze_jepa import ndcg_at_k, train_node_probe


def _write_toy(root: Path, n=6, years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007), seed=0):
    """Tiny trade graph: every year each nation exports to 2-3 others with
    shares summing to one; labels exist for every year but the first."""
    rng = np.random.RandomState(seed)
    names = [f"N{i}" for i in range(n)]
    e_rows, l_rows = [], []
    for y in years:
        for u in range(n):
            partners = rng.choice([v for v in range(n) if v != u], size=rng.randint(2, 4), replace=False)
            w = rng.dirichlet(np.ones(len(partners)))
            for v, wv in zip(partners, w):
                e_rows.append((y, names[u], names[v], wv))
                if y != years[0]:
                    l_rows.append((y, names[u], names[v], wv))
    root.mkdir(parents=True, exist_ok=True)
    for fname, rows in ((EDGE_FILE, e_rows), (LABEL_FILE, l_rows)):
        with open(root / fname, "w") as f:
            f.write("year,nation,trading nation,weight\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.8f}\n")
    return names, e_rows, l_rows


def test_ndcg_matches_sklearn_when_available():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.RandomState(0)
    yt = rng.rand(40, 30) * (rng.rand(40, 30) < 0.2)
    ys = rng.randn(40, 30)
    assert abs(ndcg_at_k(yt, ys, 10).mean() - sk.ndcg_score(yt, ys, k=10)) < 1e-9


def test_ndcg_basic_properties():
    yt = np.array([[0.5, 0.3, 0.2, 0.0]])
    assert ndcg_at_k(yt, yt, 10)[0] == pytest.approx(1.0)          # perfect order
    worst = ndcg_at_k(yt, -yt, 10)[0]
    assert 0 < worst < 1
    assert ndcg_at_k(np.zeros((1, 4)), yt, 10)[0] == 0.0             # no relevance -> 0
    assert ndcg_at_k(yt, yt, 2)[0] == pytest.approx(1.0)             # k < C


def test_parse_and_loader(tmp_path):
    names, e_rows, l_rows = _write_toy(tmp_path)
    d = parse_tgbn_trade(tmp_path)
    assert d["nations"].tolist() == names and d["years"].tolist() == list(range(2000, 2008))
    T, N = d["labels"].shape[:2]
    assert (T, N) == (8, 6)
    assert not (d["labels"][0] > 0).any()                            # first year unlabelled
    assert np.allclose(d["labels"][1:].sum(-1), 1.0, atol=1e-6)
    assert (tmp_path / "tgbn-trade.npz").exists()

    ds = TGBNTradeDataset(data_dir=str(tmp_path), split="train", history_size=2, num_preds=1)
    assert ds.num_nodes == 6 and ds.feature_dim == 13 and ds.state_label_mode == "ranking"
    assert ds.node_label_dist.shape == (8, 6, 6) and ds.label_valid.shape == (8, 6)
    assert not ds.label_valid[0].any() and ds.label_valid[1:].all()
    item = ds[0]
    assert item["x_hist"].shape == (2, 6, 13) and item["adj"].shape == (3, 6, 6)
    assert (item["adj"] == item["adj"].transpose(1, 2)).all()
    ds2 = TGBNTradeDataset(data_dir=str(tmp_path), split="train", history_size=2,
                           num_preds=1, with_weight_features=False)
    assert ds2.feature_dim == 8


def test_weight_features_ranges(tmp_path):
    _write_toy(tmp_path)
    d = parse_tgbn_trade(tmp_path)
    wf = weight_features(d["edges"], 8, 6)
    assert wf.shape == (8, 6, 5)
    assert (wf >= 0).all() and (wf[..., :3] <= 1).all()
    # a nation with a single partner has max share 1 and entropy 0
    single = np.array([[0.0, 0, 1, 1.0]])
    w1 = weight_features(single, 1, 2)
    assert w1[0, 0, 0] == pytest.approx(1.0) and w1[0, 0, 1] == pytest.approx(0.0)


def test_ranking_probe_learns_identity_features():
    rng = np.random.RandomState(0)
    n, C = 300, 12
    Y = rng.dirichlet(np.ones(C) * 0.3, size=n).astype(np.float32)
    tr = np.zeros(n, bool); tr[:200] = True
    good = train_node_probe(Y, Y, tr, ~tr, "ranking", device="cpu", epochs=100)
    rand = train_node_probe(rng.randn(n, C), Y, tr, ~tr, "ranking", device="cpu", epochs=20)
    assert set(good) >= {"ndcg10_test", "n_classes", "n"} and good["n_classes"] == C
    assert good["ndcg10_test"] > rand["ndcg10_test"] + 0.2
    sub = train_node_probe(Y, Y, tr, ~tr, "ranking", device="cpu", epochs=10,
                           test_subsets={"first": np.arange(n) < 250})
    assert sub["subset_first"]["n"] == 50


@pytest.mark.skipif(not Path("dataset/tgb/tgbn-trade/" + LABEL_FILE).exists(),
                    reason="tgbn-trade not fetched")
def test_real_tgbn_trade():
    ds = TGBNTradeDataset(split="test")
    assert ds.num_nodes == 255 and ds.node_label_dist.shape[0] == 31
    assert not ds.label_valid[0].any() and ds.label_valid[1:].sum(1).min() >= 200
    # persistence floor is high on this benchmark (TGB reports ~0.85)
    L, V = ds.node_label_dist, ds.label_valid
    rows = V[30] & V[29]
    assert ndcg_at_k(L[30][rows], L[29][rows], 10).mean() > 0.7
