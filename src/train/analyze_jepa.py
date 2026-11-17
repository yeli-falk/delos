"""Unified analysis of a DELOS encoder + predictor on any dynamic-graph dataset.

A single entry point that runs two analyses and reports them side by side.
Every probe here is *trained by gradient descent* (no closed-form ridge): the
question is always "what can a learned readout pull out of this frozen
representation".

Two complementary views, both comparing the encoder-only baseline against the
JEPA predictor's forecast:

  (1) Representation diagnostics
      On a held-out split we collect two frozen per-node representations and
      compare them:
        - z_enc  = encoder(x_hist, adj_hist)[-1]      (embedding persistence:
                   the encoder's only honest forecast, history-only)
        - z_pred = predictor.inference(z_hist, ...)   (the JEPA forecast)
      Training-free metrics: effective rank, top-SV energy spectrum, linear
      CKA(z_enc, z_pred). Then *trained* linear/MLP probes for structural
      targets at the forecast step (degree, PageRank, k-core, is-hub, and an
      optional community/block id). If z_pred carries more structure than the
      encoder, its probe scores beat z_enc; if CKA ~ 1 and scores match, the
      predictor merely reproduced the encoder (collapse).

  (2) Downstream link-prediction ablation
      A trained LinkPredictionProbe scores candidate edges from a per-node
      representation. We compare four representation *sources* on MRR / Hits@k:
        - random      U(0,1) scores                        (chance baseline)
        - encoder     frozen encoder persistence + probe   (what a pretrained
                      encoder alone buys)
        - jepa        frozen encoder + frozen predictor     (does the predictor's
                      forecast beat encoder persistence downstream?)
        - supervised  fresh encoder + probe, end-to-end     (what direct
                      supervision buys)

Both analyses are dataset-agnostic: the dataset is built by the shared factory
(`build_dynamic_dataset`), so any `--data_source` (dblp / jodie / reddit / sbm /
tgb / ... ) works. The structural diagnostics apply to every dynamic-graph
dataset; the link-prediction ablation applies to link-prediction-style datasets
and is skipped with a warning on datasets that have no edges to rank.

Run (synthetic smoke, both analyses, cpu):
    python src/train/analyze_jepa.py --synthetic --device cpu \
        --link_epochs 2 --probe_epochs 30

Run (real checkpoints, DBLP):
    python src/train/analyze_jepa.py \
        --encoder_ckpt outputs/dblp_encoder/encoder.ckpt \
        --jepa_ckpt    outputs/dblp_cjepa/best.ckpt \
        --data_path    dataset/dblp/edges.csv \
        --out_dir      outputs/analyze_dblp

Run (only one half):
    python src/train/analyze_jepa.py ... --skip_linkpred      # diagnostics only
    python src/train/analyze_jepa.py ... --skip_diagnostic    # link-pred only
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.delos_predictor import MaskedNodePredictor, predictor_flags_from_args
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.world_models.delos_encoder import (
    ENCODER_REGISTRY,
    LinearProbe,
    LinkPredictionProbe,
    build_encoder,
    link_ranking_metrics,
    normalize_adjacency,
)
from src.world_models.probes import NodeClassProbe, sample_neg_edges, score_negatives_chunked


# ===========================================================================
# PART 1 — Representation diagnostics
# ===========================================================================
@torch.no_grad()
def collect_representations(encoder, predictor, loader, history_size, device):
    """Walk the loader once and stack per-active-future-node row vectors.

    Returns a dict with:
      Z_enc, Z_pred         (M, D)   stacked latents at future timesteps
      degree, pagerank, kcore (M,)   structural attribute targets
      window, t_offset, node_id (M,) bookkeeping (so we can split fairly)
    """
    enc_rows, pred_rows = [], []
    deg_rows, pr_rows, kc_rows = [], [], []
    win_rows, t_rows, node_rows = [], [], []
    encoder.eval(); predictor.eval()
    for wid, batch in enumerate(loader):
        x_hist = batch["x_hist"].to(device)
        adj = batch["adj"].to(device)             # (B, T, N, N) bool
        active = batch["active"].to(device)        # (B, T, N) bool

        adj_norm = normalize_adjacency(adj)
        # Encoder-only baseline MUST be history-only: feed it the history
        # adjacency and use the last-history-step embedding as its forecast of
        # the future node representation (embedding persistence) -- identical
        # information to what the JEPA predictor receives.
        adj_norm_hist = adj_norm[:, :history_size]
        assert adj_norm_hist.shape[1] == history_size, \
            "encoder-only path received non-history timesteps (leakage)"
        z_hist = encoder(x_hist, adj_norm_hist)                    # (B, T_hist, N, D)
        assert z_hist.shape[1] == history_size, \
            "z_hist must span exactly the history window (no future leak)"
        z_enc_hist_last = z_hist[:, -1]                            # (B, N, D)
        z_pred_fut = predictor.inference(z_hist, adj, active)      # (B, T_pred, N, D)

        B, T_total, N = active.shape
        T_pred = z_pred_fut.shape[1]
        for b in range(B):
            for tp in range(T_pred):
                t_global = history_size + tp
                act = active[b, t_global]
                if act.sum() < 2:
                    continue
                idx = act.nonzero(as_tuple=True)[0]
                A = adj[b, t_global].cpu().numpy().astype(np.float64)
                act_np = act.cpu().numpy()

                deg = A.sum(axis=-1)
                pr = _pagerank(A, act_np)
                kc = _kcore(A, act_np)

                idx_np = idx.cpu().numpy()
                enc_rows.append(z_enc_hist_last[b, idx].cpu())
                pred_rows.append(z_pred_fut[b, tp, idx].cpu())
                deg_rows.append(torch.from_numpy(deg[idx_np]))
                pr_rows.append(torch.from_numpy(pr[idx_np]))
                kc_rows.append(torch.from_numpy(kc[idx_np]))
                win_rows.append(torch.full((idx.numel(),), wid * 1000 + b))
                t_rows.append(torch.full((idx.numel(),), tp))
                node_rows.append(torch.from_numpy(idx_np.astype(np.int64)))

    if not enc_rows:
        raise RuntimeError(
            "collect_representations gathered no rows -- the eval split has no "
            "snapshot with >=2 active nodes. Check the dataset / split."
        )
    Z_enc = torch.cat(enc_rows, dim=0).numpy().astype(np.float64)
    Z_pred = torch.cat(pred_rows, dim=0).numpy().astype(np.float64)
    degree = torch.cat(deg_rows, dim=0).numpy().astype(np.float64)
    pagerank = torch.cat(pr_rows, dim=0).numpy().astype(np.float64)
    kcore = torch.cat(kc_rows, dim=0).numpy().astype(np.float64)
    window = torch.cat(win_rows, dim=0).numpy()
    t_off = torch.cat(t_rows, dim=0).numpy()
    node_id = torch.cat(node_rows, dim=0).numpy()
    return dict(Z_enc=Z_enc, Z_pred=Z_pred, degree=degree,
                pagerank=pagerank, kcore=kcore,
                window=window, t_offset=t_off, node_id=node_id)


def _pagerank(A: np.ndarray, active: np.ndarray, damping: float = 0.85,
              n_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """Power-iteration PageRank on an undirected adjacency (N, N). Inactive
    nodes get score 0 and are excluded from teleport mass."""
    N = A.shape[0]
    out = np.zeros(N, dtype=np.float64)
    active_mask = active.astype(bool)
    n_active = int(active_mask.sum())
    if n_active < 2:
        return out
    deg = A.sum(axis=-1)
    safe_deg = np.where(deg > 0, deg, 1.0)
    P = A / safe_deg[:, None]            # row-stochastic on out-edges
    teleport = active_mask.astype(np.float64) / n_active
    r = teleport.copy()
    for _ in range(n_iter):
        r_new = damping * (P.T @ r) + (1 - damping) * teleport
        dangling = (deg == 0) & active_mask
        if dangling.any():
            r_new += damping * r[dangling].sum() * teleport
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    out[active_mask] = r[active_mask]
    return out


def _kcore(A: np.ndarray, active: np.ndarray) -> np.ndarray:
    """k-core number per node (max k s.t. the node is in the k-core).
    Batagelj-Zaversnik peeling in pure numpy. Inactive nodes get 0."""
    N = A.shape[0]
    core = np.zeros(N, dtype=np.float64)
    active_mask = active.astype(bool)
    deg = A.sum(axis=-1).astype(np.int64) * active_mask
    alive = active_mask.copy()
    while alive.any():
        alive_idx = np.where(alive)[0]
        if alive_idx.size == 0:
            break
        d_alive = deg[alive_idx]
        d_min = d_alive.min()
        to_remove = alive_idx[d_alive == d_min]
        core[to_remove] = d_min
        for v in to_remove:
            nbrs = np.where((A[v] > 0) & alive)[0]
            deg[nbrs] = np.maximum(deg[nbrs] - 1, 0)
        alive[to_remove] = False
    return core


def effective_rank(Z: np.ndarray, eps: float = 1e-12) -> dict:
    """Roy-Vetterli effective rank from singular values of centered Z."""
    Z = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Z, full_matrices=False, compute_uv=False)
    p = (s ** 2)
    p = p / max(p.sum(), eps)
    H = -np.sum(p * np.log(np.clip(p, eps, None)))
    return {
        "eff_rank": float(np.exp(H)),
        "top10_energy": [float(x) for x in p[:10].tolist()],
        "nominal_dim": int(Z.shape[1]),
    }


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two row-aligned representations. Range [0, 1]."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    den = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(num / max(den, 1e-12))


def _train_test_split_by_window(window: np.ndarray, seed: int = 0,
                                test_frac: float = 0.3):
    """Hold out whole windows so train/test rows don't share a snapshot."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(window)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_windows = set(uniq[:n_test].tolist())
    test_mask = np.array([w in test_windows for w in window])
    return ~test_mask, test_mask


def _auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney U / (n_pos * n_neg). O(n log n)."""
    label = label.astype(bool)
    n_pos = label.sum(); n_neg = (~label).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(1, len(order) + 1)
    auc = (rank[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _auprc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision).

    The metric that matters on severely imbalanced state labels: AUROC stays
    deceptively high when positives are rare, AP does not. Computed as
    sum_k (R_k - R_{k-1}) * P_k over descending-score thresholds.
    """
    label = label.astype(bool)
    n_pos = int(label.sum())
    if n_pos == 0 or n_pos == len(label):
        return float("nan")
    order = np.argsort(score)[::-1]
    tp = np.cumsum(label[order])
    precision = tp / np.arange(1, len(label) + 1)
    recall = tp / n_pos
    d_recall = np.diff(recall, prepend=0.0)
    return float((precision * d_recall).sum())


def _best_f1_threshold(score: np.ndarray, label: np.ndarray) -> float:
    """Score cut that maximises F1 on (score, label); predict positive above it.

    Meant to be fitted on the TRAIN rows and applied to test: the BCE probe's
    natural cut (logit 0) predicts almost all-negative when positives are
    rare, so F1 there is near 0 for every method, and tuning the cut on test
    is leakage. Falls back to 0.0 (the logit cut) when a class is absent.
    """
    label = label.astype(bool)
    n_pos = int(label.sum())
    if n_pos == 0 or n_pos == len(label):
        return 0.0
    order = np.argsort(score)[::-1]
    s = score[order]
    tp = np.cumsum(label[order])
    f1 = 2.0 * tp / (np.arange(1, len(s) + 1) + n_pos)   # 2TP / (n_pred + n_pos)
    k = int(np.argmax(f1))
    return float((s[k] + s[k + 1]) / 2.0) if k + 1 < len(s) else float(s[k] - 1e-6)


def _prf(score: np.ndarray, label: np.ndarray, thr: float) -> dict:
    """Positive-class precision / recall / F1 at a fixed score cut."""
    label = label.astype(bool)
    if len(label) == 0 or not label.any():
        return {"f1_test": float("nan"), "precision_test": float("nan"),
                "recall_test": float("nan")}
    pred = score > thr
    tp = int((pred & label).sum())
    prec = tp / max(int(pred.sum()), 1)
    rec = tp / int(label.sum())
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"f1_test": float(f1), "precision_test": float(prec),
            "recall_test": float(rec)}


# ---------------------------------------------------------------------------
# Trained probes for the structural diagnostic (gradient descent, not ridge)
# ---------------------------------------------------------------------------
def _iter_minibatches(n, batch_size, generator):
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        yield perm[i:i + batch_size]


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-row NDCG@k, the TGB node-property metric (matches
    ``sklearn.metrics.ndcg_score(y_true, y_pred, k)`` row by row for rows
    without score ties). ``y_true`` (n, C) non-negative relevances (the label
    distribution), ``y_score`` (n, C) predicted scores. Rows whose ideal DCG
    is zero score 0, as in sklearn."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    k = min(k, y_true.shape[1])
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    top = np.argpartition(-y_score, k - 1, axis=1)[:, :k]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(y_score, top, 1), axis=1), 1)
    dcg = (np.take_along_axis(y_true, top, 1) * disc).sum(1)
    ideal = np.sort(y_true, axis=1)[:, ::-1][:, :k]
    idcg = (ideal * disc).sum(1)
    return np.where(idcg > 0, dcg / np.maximum(idcg, 1e-12), 0.0)


def train_node_probe(Z: np.ndarray, y: np.ndarray, train: np.ndarray,
                     test: np.ndarray, task: str, *, device, epochs: int = 200,
                     lr: float = 1e-2, weight_decay: float = 1e-4,
                     hidden_dim: int = 0, batch_size: int = 4096,
                     seed: int = 0,
                     test_subsets: "dict[str, np.ndarray] | None" = None) -> dict:
    """Train a readout head on a frozen representation by gradient descent.

    Features are standardised by train-set statistics; for regression the
    target is standardised too (R^2 is affine-invariant, so it is reported in
    standardised space). Returns a dict keyed by task:
      task='regression' -> {"r2_train", "r2_test"}
      task='binary'     -> {"acc_test", "auroc_test", "auprc_test", ...}
      task='multiclass' -> {"acc_test", "macro_f1_test", "n_classes"}
      task='ranking'    -> {"ndcg10_test", "n_classes"}: y is (rows, C) of
                           non-negative relevances (tgbn-trade share vectors);
                           a linear/MLP head to C logits trained with the
                           soft-target cross-entropy, scored with NDCG@10.

    test_subsets (binary / multiclass): optional {name: bool mask over ALL rows};
    the trained probe is additionally scored on `test & mask` and reported
    under "subset_<name>". Used e.g. for the onset subset (nodes not yet in
    the positive state at the previous snapshot), where persistence carries
    no signal.
    """
    gen = torch.Generator().manual_seed(seed)
    mu = Z[train].mean(axis=0, keepdims=True)
    sd = Z[train].std(axis=0, keepdims=True) + 1e-6
    Zs = ((Z - mu) / sd).astype(np.float32)
    n_in = Z.shape[1]

    if task == "multiclass":
        classes = np.unique(y)
        classes = classes[classes >= 0]        # drop -1 no-label sentinel
        K = len(classes)
        if K < 2:
            return {"acc_test": float("nan"), "macro_f1_test": float("nan"),
                    "n_classes": int(K)}
        cls_to_idx = {int(c): i for i, c in enumerate(classes)}
        tr_m = train & (y >= 0)
        te_m = test & (y >= 0)
        Xtr = torch.tensor(Zs[tr_m], device=device)
        Xte = torch.tensor(Zs[te_m], device=device)
        ytr = torch.tensor([cls_to_idx[int(c)] for c in y[tr_m]],
                           dtype=torch.long, device=device)
        yte_idx = np.array([cls_to_idx[int(c)] for c in y[te_m]])
        # Seed the head init from a FORKED RNG so it is reproducible across
        # calls without perturbing the caller's global torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            head = NodeClassProbe(n_in, K, hidden_dim=hidden_dim).to(device)
        loss_fn = F.cross_entropy
    elif task == "ranking":
        y = np.asarray(y, dtype=np.float32)
        K = y.shape[1]
        Xtr = torch.tensor(Zs[train], device=device)
        Xte = torch.tensor(Zs[test], device=device)
        ytr = torch.tensor(y[train], device=device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            head = NodeClassProbe(n_in, K, hidden_dim=hidden_dim).to(device)

        def loss_fn(logits, target):            # soft-target cross-entropy
            return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    else:                                       # regression or binary
        Xtr = torch.tensor(Zs[train], device=device)
        Xte = torch.tensor(Zs[test], device=device)
        with torch.random.fork_rng(devices=[]):    # reproducible init, no global side-effect
            torch.manual_seed(seed)
            head = LinearProbe(n_in, hidden_dim=hidden_dim).to(device)
        if task == "regression":
            ymu, ysd = y[train].mean(), y[train].std() + 1e-6
            ytr = torch.tensor(((y[train] - ymu) / ysd).astype(np.float32),
                               device=device)
            yte = torch.tensor(((y[test] - ymu) / ysd).astype(np.float32),
                               device=device)
            loss_fn = F.mse_loss
        elif task == "binary":
            ytr = torch.tensor(y[train].astype(np.float32), device=device)
            yte = torch.tensor(y[test].astype(np.float32), device=device)
            loss_fn = F.binary_cross_entropy_with_logits
        else:
            raise ValueError(f"unknown probe task {task!r}")

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    n = Xtr.shape[0]
    head.train()
    for _ in range(epochs):
        for sl in _iter_minibatches(n, batch_size, gen):
            out = head(Xtr[sl])
            loss = loss_fn(out, ytr[sl])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        if task == "regression":
            yhat_tr = head(Xtr).cpu().numpy()
            yhat_te = head(Xte).cpu().numpy()
            ytr_np, yte_np = ytr.cpu().numpy(), yte.cpu().numpy()

            def _r2(yt, yh):
                ss_res = ((yt - yh) ** 2).sum()
                ss_tot = ((yt - yt.mean()) ** 2).sum()
                return float(1.0 - ss_res / max(ss_tot, 1e-12))
            return {"r2_train": _r2(ytr_np, yhat_tr),
                    "r2_test": _r2(yte_np, yhat_te)}
        if task == "binary":
            logit_te = head(Xte).cpu().numpy()
            yte_np = yte.cpu().numpy()
            acc = float(((logit_te > 0.0) == (yte_np > 0.5)).mean())
            # F1 needs a cut; fit it on TRAIN (never test), see _best_f1_threshold.
            thr = _best_f1_threshold(head(Xtr).cpu().numpy(),
                                     ytr.cpu().numpy() > 0.5)
            out = {"acc_test": acc,
                   "auroc_test": _auroc(logit_te, yte_np),
                   "auprc_test": _auprc(logit_te, yte_np),
                   **_prf(logit_te, yte_np > 0.5, thr),
                   "f1_threshold": thr,
                   "n_pos_test": int((yte_np > 0.5).sum()),
                   "pos_rate_test": float((yte_np > 0.5).mean())}
            for name, sub in (test_subsets or {}).items():
                m = sub[test]                 # align full-row mask to test rows
                out[f"subset_{name}"] = {
                    "auroc_test": _auroc(logit_te[m], yte_np[m]),
                    "auprc_test": _auprc(logit_te[m], yte_np[m]),
                    **_prf(logit_te[m], yte_np[m] > 0.5, thr),
                    "n": int(m.sum()),
                    "n_pos": int((yte_np[m] > 0.5).sum()),
                }
            return out
        if task == "ranking":
            score_te = head(Xte).cpu().numpy()
            per_row = ndcg_at_k(y[test], score_te, k=10)
            out = {"ndcg10_test": float(per_row.mean()) if len(per_row) else float("nan"),
                   "n_classes": int(K), "n": int(len(per_row))}
            for name, sub in (test_subsets or {}).items():
                m = sub[test]
                out[f"subset_{name}"] = {
                    "ndcg10_test": float(per_row[m].mean()) if m.any() else float("nan"),
                    "n": int(m.sum())}
            return out
        # multiclass
        pred_idx = head(Xte).argmax(dim=-1).cpu().numpy()

        def _score(pi, yi):
            if len(yi) == 0:
                return {"acc_test": float("nan"), "macro_f1_test": float("nan")}
            f1s = []
            for k in range(K):
                tp = int(((pi == k) & (yi == k)).sum())
                fp = int(((pi == k) & (yi != k)).sum())
                fn = int(((pi != k) & (yi == k)).sum())
                if tp + fn == 0:              # class absent from this subset
                    continue
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
            return {"acc_test": float((pi == yi).mean()),
                    "macro_f1_test": float(np.mean(f1s)) if f1s else float("nan")}

        out = {**_score(pred_idx, yte_idx), "n_classes": int(K)}
        for name, sub in (test_subsets or {}).items():
            m = sub[te_m]                     # align full-row mask to test rows
            out[f"subset_{name}"] = {**_score(pred_idx[m], yte_idx[m]),
                                     "n": int(m.sum())}
        # Per-row predictions (original class ids) for derived metrics such as
        # change-detection precision/recall. Private keys: callers must pop
        # them before serialising the dict.
        out["_test_rows"] = np.flatnonzero(te_m)
        out["_pred_test"] = classes[pred_idx]
        return out


def compute_block_labels(ds, source: str = "louvain") -> np.ndarray:
    """Community label per node for the block-id probe.

    source='dataset' -> ds.node_labels; source='louvain' -> Louvain on the
    time-aggregated adjacency. Returns labels[i] (>=0) or -1 for never-active
    nodes; the probe ignores -1 rows.
    """
    if source == "dataset":
        if not hasattr(ds, "node_labels") or ds.node_labels is None:
            raise ValueError(
                "Block source='dataset' requested but ds has no .node_labels. "
                "Use --block_probe louvain or a dataset with ground-truth labels."
            )
        return np.asarray(ds.node_labels).astype(np.int64)

    if source == "louvain":
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "block_probe=louvain requires networkx; pip install networkx"
            ) from e
        N = ds.num_nodes
        agg = ds.adj.any(axis=0)               # (N, N) bool
        active_any = ds.active.any(axis=0)     # (N,) bool
        triu = np.triu(agg, k=1)
        src, dst = triu.nonzero()
        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from(zip(src.tolist(), dst.tolist()))
        communities = nx.algorithms.community.louvain_communities(G, seed=0)
        labels = np.full(N, -1, dtype=np.int64)
        for c_id, members in enumerate(communities):
            for node in members:
                labels[node] = c_id
        labels[~active_any] = -1
        n_clusters = len([1 for c in communities if len(c) >= 2])
        print(f"[diagnose] louvain: {n_clusters} non-trivial communities, "
              f"label coverage {(labels >= 0).mean():.2f}", flush=True)
        return labels

    raise ValueError(f"unknown block_probe source {source!r}")


def run_diagnostic(data: dict, out_dir: Path,
                   block_labels: "np.ndarray | None" = None, *,
                   device="cpu", probe_epochs: int = 200, probe_lr: float = 1e-2,
                   probe_hidden: int = 0) -> dict:
    """Effective rank / CKA + *trained* structural probes for encoder vs jepa.

    The probe training hyper-parameters are keyword-only; the first three
    positional args are also used by diagnose_trajectory_jepa.
    """
    Z_enc, Z_pred = data["Z_enc"], data["Z_pred"]
    deg, pr, kc = data["degree"], data["pagerank"], data["kcore"]
    win = data["window"]
    print(f"[diagnose] rows={len(deg):,}  D_enc={Z_enc.shape[1]}  "
          f"D_pred={Z_pred.shape[1]}", flush=True)

    results = {
        "n_rows": int(len(deg)),
        "encoder": {"effective_rank": effective_rank(Z_enc)},
        "jepa":    {"effective_rank": effective_rank(Z_pred)},
        "cka_enc_pred": linear_cka(Z_enc, Z_pred),
    }

    train_mask, test_mask = _train_test_split_by_window(win, seed=0)
    print(f"[diagnose] split: {train_mask.sum()} train rows / "
          f"{test_mask.sum()} test rows  (probes: {probe_epochs} epochs, "
          f"lr={probe_lr}, hidden={probe_hidden})", flush=True)

    hub_thresh = np.quantile(deg[train_mask], 0.9)
    is_hub = (deg >= hub_thresh).astype(np.int64)

    if block_labels is not None:
        per_row_block = block_labels[data["node_id"]]
    else:
        per_row_block = None

    def _probe(Z, y, task):
        return train_node_probe(Z, y, train_mask, test_mask, task,
                                device=device, epochs=probe_epochs,
                                lr=probe_lr, hidden_dim=probe_hidden)

    for name, Z in [("encoder", Z_enc), ("jepa", Z_pred)]:
        results[name]["probe_degree"] = _probe(Z, deg, "regression")
        results[name]["probe_pagerank"] = _probe(Z, pr, "regression")
        results[name]["probe_kcore"] = _probe(Z, kc, "regression")
        results[name]["probe_is_hub"] = _probe(Z, is_hub, "binary")
        if per_row_block is not None:
            results[name]["probe_block"] = _probe(Z, per_row_block, "multiclass")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)

    def _fmt(x): return f"{x:.4f}" if isinstance(x, float) else str(x)
    rows = [
        ("effective rank",
         results['encoder']['effective_rank']['eff_rank'],
         results['jepa']['effective_rank']['eff_rank']),
        ("top-1 sv energy",
         results['encoder']['effective_rank']['top10_energy'][0],
         results['jepa']['effective_rank']['top10_energy'][0]),
        ("probe degree R^2 (autocorr)",
         results['encoder']['probe_degree']['r2_test'],
         results['jepa']['probe_degree']['r2_test']),
        ("probe pagerank R^2",
         results['encoder']['probe_pagerank']['r2_test'],
         results['jepa']['probe_pagerank']['r2_test']),
        ("probe k-core R^2",
         results['encoder']['probe_kcore']['r2_test'],
         results['jepa']['probe_kcore']['r2_test']),
        ("probe is_hub AUROC",
         results['encoder']['probe_is_hub']['auroc_test'],
         results['jepa']['probe_is_hub']['auroc_test']),
    ]
    if per_row_block is not None:
        rows.extend([
            ("probe block-id acc",
             results['encoder']['probe_block']['acc_test'],
             results['jepa']['probe_block']['acc_test']),
            ("probe block-id macro-F1",
             results['encoder']['probe_block']['macro_f1_test'],
             results['jepa']['probe_block']['macro_f1_test']),
        ])
    print("\n" + "=" * 72)
    print("DIAGNOSTIC: encoder-persistence vs JEPA-forecast representation")
    print(f"{'metric':<28} {'encoder':>16} {'jepa':>16}  delta")
    print("-" * 72)
    for name, e, j in rows:
        delta = j - e
        flag = "  +" if delta > 0 else ("  -" if delta < 0 else "  =")
        print(f"{name:<28} {_fmt(e):>16} {_fmt(j):>16} {flag}{_fmt(abs(delta))}")
    print(f"{'CKA(encoder, jepa)':<28} {_fmt(results['cka_enc_pred']):>16}")
    print("=" * 72, flush=True)
    return results


# ===========================================================================
# PART 2 — Downstream link-prediction ablation
# ===========================================================================
LINK_SOURCES = ("random", "encoder", "jepa", "supervised")


def encode_history_last(encoder, x_hist, adj_b, history_size):
    """Encoder on the history-only window; return the last-history-step latent
    as the static per-node embedding (B, N, D)."""
    adj_norm = normalize_adjacency(adj_b[:, :history_size])
    z = encoder(x_hist, adj_norm)
    return z[:, -1]


def _batch_reprs(source, encoder, predictor, batch, history_size, device,
                 grad: bool):
    """Per-batch node representations for a link-pred source.

    Returns (z_static, z_pred):
      encoder / supervised -> (z_static (B,N,D), None)   encoder persistence
      jepa                 -> (None, z_pred (B,T_pred,N,D)) predictor forecast
      random               -> (None, None)
    """
    if source == "random":
        return None, None
    x_hist = batch["x_hist"].to(device, non_blocking=True)
    adj_b = batch["adj"].to(device, non_blocking=True)
    if source in ("encoder", "supervised"):
        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx:
            z_static = encode_history_last(encoder, x_hist, adj_b, history_size)
        return z_static, None
    if source == "jepa":
        active_b = batch["active"].to(device, non_blocking=True)
        with torch.no_grad():
            adj_norm = normalize_adjacency(adj_b)
            z_hist = encoder(x_hist, adj_norm[:, :history_size])
            z_pred = predictor.inference(z_hist, adj_b, active_b)
        return None, z_pred
    raise ValueError(f"unknown link source {source!r}")


def _step_z(source, z_static, z_pred, b, tp):
    """Node-embedding matrix (N, D) used to score edges at future step tp."""
    if source == "jepa":
        return z_pred[b, tp]
    return z_static[b]


def train_link_epoch(source, encoder, predictor, probe, loader, opt,
                     history_size, device, neg_per_pos=4):
    trainable_encoder = (source == "supervised")
    if trainable_encoder:
        encoder.train()
    elif encoder is not None:
        encoder.eval()
    if predictor is not None:
        predictor.eval()
    probe.train()

    total_loss, n_batches = 0.0, 0
    for batch in loader:
        z_static, z_pred = _batch_reprs(
            source, encoder, predictor, batch, history_size, device,
            grad=trainable_encoder,
        )
        adj_b = batch["adj"].to(device, non_blocking=True)
        active_b = batch["active"].to(device, non_blocking=True)
        B, T_total, N = active_b.shape

        all_pos, all_neg = [], []
        for b in range(B):
            for tp in range(T_total - history_size):
                t_global = history_size + tp
                adj_t = adj_b[b, t_global]
                active_t = active_b[b, t_global]
                if adj_t.sum() == 0 or active_t.sum() < 2:
                    continue
                src, dst = adj_t.nonzero(as_tuple=True)
                if src.numel() == 0:
                    continue
                z = _step_z(source, z_static, z_pred, b, tp)
                pos_logits = probe(z[src], z[dst])
                neg_dst = torch.randint(0, N, (src.numel(), neg_per_pos),
                                        device=device)
                src_rep = z[src].unsqueeze(1).expand(-1, neg_per_pos, -1)
                neg_logits = probe(src_rep, z[neg_dst]).flatten()
                all_pos.append(pos_logits)
                all_neg.append(neg_logits)

        if not all_pos:
            continue
        pos = torch.cat(all_pos)
        neg = torch.cat(all_neg)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos, neg]),
            torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        trainable = [p for p in opt.param_groups[0]["params"] if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_link_pred(source, encoder, predictor, probe, loader, history_size,
                   device, k_per_pos=99):
    """MRR / Hits@k for two positive subsets, mirroring
    ``probes.evaluate_link_prediction``:
      * ``all`` — every positive edge in the target snapshot
      * ``new`` — only edges not seen in any earlier snapshot of the window

    Returns ``{"all": {tp: {...}}, "new": {tp: {...}}}``.
    """
    if encoder is not None:
        encoder.eval()
    if predictor is not None:
        predictor.eval()
    if probe is not None:
        probe.eval()
    per_horizon_all: dict = {}
    per_horizon_new: dict = {}

    for batch_idx, batch in enumerate(loader):
        z_static, z_pred = _batch_reprs(
            source, encoder, predictor, batch, history_size, device, grad=False,
        )
        adj_b = batch["adj"].to(device)
        active_b = batch["active"].to(device)
        B, T_total, N = active_b.shape

        for b in range(B):
            for tp in range(T_total - history_size):
                t_global = history_size + tp
                adj_t = adj_b[b, t_global]
                active_t = active_b[b, t_global]
                if adj_t.sum() == 0 or active_t.sum() < 2:
                    continue
                src, dst = adj_t.nonzero(as_tuple=True)
                if src.numel() == 0:
                    continue
                neg_dst = sample_neg_edges(
                    adj_t, active_t, k_per_pos=k_per_pos,
                    seed=batch_idx * 1000 + b * 100 + tp,
                )
                if neg_dst.numel() == 0:
                    continue

                if source == "random":
                    g = torch.Generator(device=device).manual_seed(
                        batch_idx * 1000 + b * 100 + tp,
                    )
                    pos_scores = torch.rand(src.shape[0], generator=g, device=device)
                    neg_scores = torch.rand(
                        (src.shape[0], k_per_pos), generator=g, device=device,
                    )
                else:
                    z = _step_z(source, z_static, z_pred, b, tp)
                    pos_scores = probe(z[src], z[dst])
                    neg_scores = score_negatives_chunked(probe, z, src, neg_dst)

                m_all = link_ranking_metrics(pos_scores, neg_scores)
                per_horizon_all.setdefault(tp, []).append(m_all)

                # New-edge subset: positives absent from every earlier snapshot.
                prior_seen = adj_b[b, :t_global].any(dim=0)     # (N, N) bool
                is_new = ~prior_seen[src, dst]
                if int(is_new.sum()) == 0:
                    continue
                m_new = link_ranking_metrics(
                    pos_scores[is_new], neg_scores[is_new],
                )
                per_horizon_new.setdefault(tp, []).append(m_new)

    def _aggregate(per_horizon):
        out = {}
        for tp, ms in per_horizon.items():
            keys = ms[0].keys()
            out[tp] = {k: float(np.mean([m[k] for m in ms])) for k in keys}
        return out

    return {"all": _aggregate(per_horizon_all),
            "new": _aggregate(per_horizon_new)}


def _mean_mrr(metrics: dict) -> float:
    """Mean MRR over horizons on the ``all`` subset (used for model selection)."""
    per_horizon = metrics.get("all", metrics) if metrics else {}
    return (float(np.mean([m["MRR"] for m in per_horizon.values()]))
            if per_horizon else 0.0)


def run_link_source(source, *, base_encoder, predictor, fresh_encoder_fn,
                    node_dim, probe_hidden, train_loader, val_loader,
                    test_loader, history_size, device, epochs, lr, weight_decay,
                    neg_per_pos):
    """Train (if needed) and test one link-prediction representation source."""
    if source == "random":
        print("[link] random U(0,1) scoring (no training)", flush=True)
        return eval_link_pred("random", None, None, None, test_loader,
                              history_size, device)

    if source == "supervised":
        encoder = fresh_encoder_fn()
        freeze = False
    else:                                   # encoder | jepa
        encoder = base_encoder
        freeze = True

    probe = LinkPredictionProbe(in_dim=node_dim, hidden_dim=probe_hidden).to(device)
    if freeze:
        params = list(probe.parameters())
    else:
        params = list(encoder.parameters()) + list(probe.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    n_train = sum(p.numel() for p in params if p.requires_grad)
    print(f"[link:{source}] trainable params: {n_train:,}  freeze_encoder={freeze}",
          flush=True)

    best_val, best_state = -1.0, None
    for epoch in range(epochs):
        tl = train_link_epoch(source, encoder, predictor, probe, train_loader,
                              opt, history_size, device, neg_per_pos=neg_per_pos)
        vm = eval_link_pred(source, encoder, predictor, probe, val_loader,
                            history_size, device)
        val_mrr = _mean_mrr(vm)
        print(f"  [link:{source} epoch {epoch+1}/{epochs}] bce={tl:.4f} "
              f"val_MRR={val_mrr:.4f}", flush=True)
        if val_mrr > best_val:
            best_val = val_mrr
            best_state = {
                "probe": copy.deepcopy(probe.state_dict()),
                "encoder": copy.deepcopy(encoder.state_dict()) if not freeze else None,
            }

    if best_state is not None:
        probe.load_state_dict(best_state["probe"])
        if best_state["encoder"] is not None:
            encoder.load_state_dict(best_state["encoder"])
    return eval_link_pred(source, encoder, predictor, probe, test_loader,
                          history_size, device)


# ===========================================================================
# Driver
# ===========================================================================
def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


def _resolve_arch(args, device):
    """Resolve encoder/predictor architecture from ckpts (JEPA > encoder > CLI)
    and return (enc_kind, enc_hidden, node_dim, enc_bidir, enc_ckpt, jepa_ckpt)."""
    enc_kind, enc_hidden, node_dim, enc_bidir = (
        args.encoder_kind, args.encoder_hidden, args.node_dim, True,
    )
    enc_ckpt = None
    if args.encoder_ckpt and Path(args.encoder_ckpt).exists():
        enc_ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
        enc_kind = enc_ckpt.get("encoder_kind", enc_kind)
        enc_hidden = enc_ckpt.get("encoder_hidden", enc_hidden)
        node_dim = enc_ckpt.get("node_dim", node_dim)
        enc_bidir = enc_ckpt.get("bidirectional", enc_bidir)
    jepa_ckpt = None
    if args.jepa_ckpt and Path(args.jepa_ckpt).exists():
        jepa_ckpt = torch.load(args.jepa_ckpt, map_location=device, weights_only=False)
        jargs = jepa_ckpt.get("args", {}) or {}
        enc_kind = jargs.get("encoder_kind", enc_kind)
        enc_hidden = jargs.get("encoder_hidden", enc_hidden)
        node_dim = jargs.get("node_dim", node_dim)
        enc_bidir = jargs.get("bidirectional", enc_bidir)
    return enc_kind, enc_hidden, node_dim, enc_bidir, enc_ckpt, jepa_ckpt


def build_models(args, feature_dim, num_nodes, device):
    """Build the frozen encoder + predictor used by both analyses."""
    (enc_kind, enc_hidden, node_dim, enc_bidir,
     enc_ckpt, jepa_ckpt) = _resolve_arch(args, device)

    encoder = build_encoder(
        enc_kind, in_dim=feature_dim, hidden_dim=enc_hidden, out_dim=node_dim,
        freeze=True, bidirectional=enc_bidir,
    ).to(device)
    weight_source = None
    if jepa_ckpt is not None and "encoder" in jepa_ckpt:
        encoder.load_state_dict(jepa_ckpt["encoder"])
        weight_source = f"jepa_ckpt ({args.jepa_ckpt})"
    elif enc_ckpt is not None:
        encoder.load_state_dict(enc_ckpt["encoder"])
        weight_source = args.encoder_ckpt
    print(f"[encoder] {enc_kind} bidir={enc_bidir} "
          f"{'from ' + weight_source if weight_source else 'random init (no ckpt)'}",
          flush=True)

    pred_args = dict(
        num_nodes=num_nodes, node_dim=node_dim,
        history_frames=args.history_size, pred_frames=args.num_preds,
        num_masked_nodes=args.num_masked_nodes,
        depth=args.depth, heads=args.heads,
        dim_head=node_dim // args.heads, mlp_dim=args.mlp_dim,
        use_attention_mask=False,
    )
    if jepa_ckpt is not None:
        saved = jepa_ckpt.get("args", {}) or {}
        for key in ("num_masked_nodes", "depth", "heads", "mlp_dim"):
            if key in saved:
                pred_args[key] = saved[key]
        if "no_attention_mask" in saved:
            pred_args["use_attention_mask"] = not saved["no_attention_mask"]
        if "heads" in saved:
            pred_args["dim_head"] = node_dim // saved["heads"]
        # residual_head / latent_norm have no parameters, so load_state_dict
        # can't catch a mismatch, and the edge head must exist for its weights
        # to load — restore all of them from the training args.
        pred_args.update(predictor_flags_from_args(saved))
    predictor = MaskedNodePredictor(**pred_args).to(device)
    if jepa_ckpt is not None and "predictor" in jepa_ckpt:
        predictor.load_state_dict(jepa_ckpt["predictor"])
        print(f"[jepa] loaded predictor from {args.jepa_ckpt} "
              f"(epoch={jepa_ckpt.get('epoch', '?')}, "
              f"val_loss={jepa_ckpt.get('val_loss', '?')})", flush=True)
    else:
        print("[jepa] random init predictor (no ckpt -- null baseline)", flush=True)

    def fresh_encoder_fn():
        return build_encoder(
            enc_kind, in_dim=feature_dim, hidden_dim=enc_hidden, out_dim=node_dim,
            freeze=False, bidirectional=enc_bidir,
        ).to(device)

    return encoder, predictor, node_dim, fresh_encoder_fn


def print_link_table(link_results):
    print("\n" + "=" * 72)
    print("DOWNSTREAM LINK PREDICTION (MRR / Hits@k per future horizon)")
    print("  'all' = every positive edge;  'new' = edges unseen earlier in window")
    print("=" * 72)
    for source, metrics in link_results.items():
        # metrics is {"all": {tp: {...}}, "new": {tp: {...}}}; tolerate the
        # legacy flat {tp: {...}} shape for older result blobs.
        if not metrics:
            print(f"[{source}] (no rankable edges)")
            continue
        subsets = metrics if ("all" in metrics or "new" in metrics) \
            else {"all": metrics}
        print(f"[{source}]")
        for subset in ("all", "new"):
            per_horizon = subsets.get(subset, {})
            if not per_horizon:
                print(f"  {subset:<4}: (no rankable edges)")
                continue
            for tp, m in sorted(per_horizon.items()):
                line = (f"  {subset:<4} +{tp+1:>2}: "
                        + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
                print(line)
    print("=" * 72, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Data (dataset-agnostic: any --data_source via the shared factory).
    p.add_argument("--data_path", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic_nodes", type=int, default=100)
    p.add_argument("--synthetic_years", type=int, default=30)
    p.add_argument("--synthetic_kind", choices=["dblp", "sbm"], default="dblp")
    p.add_argument("--synthetic_blocks", type=int, default=8)
    add_dataset_cli_flags(p)
    p.add_argument("--history_size", type=int, default=3)
    p.add_argument("--num_preds", type=int, default=1)
    p.add_argument("--split", choices=["val", "test"], default="test",
                   help="Split used for the representation diagnostic.")
    # Checkpoints.
    p.add_argument("--encoder_ckpt", default=None,
                   help="Encoder pretraining ckpt. If absent (and no jepa_ckpt "
                        "encoder), the encoder is randomly initialised.")
    p.add_argument("--jepa_ckpt", default=None,
                   help="DELOS best.ckpt. If absent, the predictor is random "
                        "(null baseline).")
    # Encoder / predictor arch (overridden by ckpt args when present).
    p.add_argument("--encoder_kind", default="tgcn",
                   choices=sorted(ENCODER_REGISTRY))
    p.add_argument("--encoder_hidden", type=int, default=128)
    p.add_argument("--node_dim", type=int, default=128)
    p.add_argument("--num_masked_nodes", type=int, default=30)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp_dim", type=int, default=512)
    # Which analyses to run.
    p.add_argument("--skip_diagnostic", action="store_true",
                   help="Skip the structural representation diagnostic (part 1).")
    p.add_argument("--skip_linkpred", action="store_true",
                   help="Skip the downstream link-prediction ablation (part 2).")
    p.add_argument("--link_sources", nargs="+", default=list(LINK_SOURCES),
                   choices=list(LINK_SOURCES),
                   help="Representation sources for the link-pred ablation.")
    p.add_argument("--block_probe", choices=["none", "louvain", "dataset"],
                   default="none",
                   help="Add a trained multiclass community/block-id probe to "
                        "the diagnostic.")
    # Probe / training hyper-params.
    p.add_argument("--probe_epochs", type=int, default=200,
                   help="Epochs for the trained structural diagnostic probes.")
    p.add_argument("--probe_lr", type=float, default=1e-2)
    p.add_argument("--probe_hidden", type=int, default=0,
                   help="Hidden dim for diagnostic probes (0 = linear readout).")
    p.add_argument("--probe_split", choices=["test", "val"], default="test",
                   help="Split the link-prediction sources are SCORED on. "
                        "Use 'val' for hyperparameter search so no "
                        "configuration is ranked on the split the final "
                        "table reports; 'test' (default) for that table.")
    p.add_argument("--link_epochs", type=int, default=30,
                   help="Epochs for the link-prediction probe / supervised run.")
    p.add_argument("--link_lr", type=float, default=1e-3)
    p.add_argument("--link_probe_hidden", type=int, default=64)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--neg_per_pos", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # Runtime.
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="outputs/analyze_jepa")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    eval_ds = build_dynamic_dataset(args, split=args.split)
    print(f"[data] {args.split}: {len(eval_ds)} windows, N={eval_ds.num_nodes}, "
          f"F={eval_ds.feature_dim}", flush=True)
    encoder, predictor, node_dim, fresh_encoder_fn = build_models(
        args, eval_ds.feature_dim, eval_ds.num_nodes, device,
    )

    combined = {"args": vars(args)}

    # ---- Part 1: representation diagnostic ----
    if not args.skip_diagnostic:
        eval_loader = make_loader(eval_ds, args.batch_size, False, args.num_workers)
        data = collect_representations(
            encoder, predictor, eval_loader, args.history_size, device,
        )
        block_labels = None
        if args.block_probe != "none":
            try:
                block_labels = compute_block_labels(eval_ds, source=args.block_probe)
            except Exception as e:
                print(f"[warn] block_probe={args.block_probe} failed ({e}); skipping",
                      flush=True)
        combined["diagnostic"] = run_diagnostic(
            data, out_dir, block_labels=block_labels, device=device,
            probe_epochs=args.probe_epochs, probe_lr=args.probe_lr,
            probe_hidden=args.probe_hidden,
        )

    # ---- Part 2: downstream link-prediction ablation ----
    if not args.skip_linkpred:
        train_ds = build_dynamic_dataset(args, split="train")
        val_ds = build_dynamic_dataset(args, split="val")
        test_ds = build_dynamic_dataset(args, split="test")
        print(f"[data] link-pred train={len(train_ds)} val={len(val_ds)} "
              f"test={len(test_ds)}", flush=True)
        train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
        val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
        test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

        link_results = {}
        for source in args.link_sources:
            print(f"\n--- link-pred source: {source} ---", flush=True)
            try:
                link_results[source] = run_link_source(
                    source, base_encoder=encoder, predictor=predictor,
                    fresh_encoder_fn=fresh_encoder_fn, node_dim=node_dim,
                    probe_hidden=args.link_probe_hidden,
                    train_loader=train_loader, val_loader=val_loader,
                    # Scoring loader. run_link_source names its scoring loader
                    # `test_loader`; handing it the val loader is what keeps a
                    # sweep off the test split.
                    test_loader=(val_loader if args.probe_split == "val"
                                 else test_loader),
                    history_size=args.history_size,
                    device=device, epochs=args.link_epochs, lr=args.link_lr,
                    weight_decay=args.weight_decay, neg_per_pos=args.neg_per_pos,
                )
            except Exception as e:
                print(f"[warn] link source {source} failed ({e}); skipping",
                      flush=True)
                link_results[source] = {}
        if not any(link_results.values()):
            print("[warn] link-prediction produced no rankable edges -- this "
                  "dataset may not be link-prediction-style; only the diagnostic "
                  "is meaningful here.", flush=True)
        combined["link_pred"] = link_results
        print_link_table(link_results)

    _suffix = "" if args.probe_split == "test" else "_val"
    with open(out_dir / f"analysis{_suffix}.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n[done] wrote {out_dir}/analysis{_suffix}.json", flush=True)


if __name__ == "__main__":
    main()
