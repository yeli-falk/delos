"""Run Louvain-community node classification on existing trained checkpoints.

This is the cheap pivot: instead of re-running the full structural diagnostic
(eff_rank, CKA, degree/PageRank/k-core probes), we only fit a multi-class
linear probe for community membership on the encoder and JEPA latents from
already-trained checkpoints. Output goes to `<ckpt_dir>/block_probe.json`.

The question this answers: "does the rank>1 representation produced by
feature-mask pretraining (post-normalisation) actually help when the
downstream task is multi-class node classification, not link prediction?"

If encoder + multi-class probe and JEPA + multi-class probe both score low,
the rank story doesn't translate. If JEPA scores higher, the linear-probe
link-prediction eval was the bottleneck all along.

Usage:
    # one checkpoint
    python scripts/dblp/run_block_probe.py \
        --jepa_ckpt outputs/reddit_body_featmask_k1000_s42_cjepa/best.ckpt

    # sweep over many checkpoints (one block_probe.json per ckpt dir)
    python scripts/dblp/run_block_probe.py \
        --pattern 'outputs/reddit_body_featmask_*_cjepa'
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.delos_predictor import build_predictor, predictor_flags_from_args
from src.custom_codes.dataset_factory import build_dynamic_dataset
from src.train.analyze_jepa import (
    compute_block_labels,
    ndcg_at_k,
    train_node_probe,
    _train_test_split_by_window,
)
from src.world_models.delos_encoder import build_encoder, normalize_adjacency


# Probing the VAL split writes to its own file so a hyperparameter search can
# rank configurations without ever reading — or overwriting — the test result.
# Selecting on state_probe.json would be selecting on test.
_SPLIT_SUFFIX = {"test": "", "val": "_val"}


def _make_args_namespace(saved_args: dict) -> types.SimpleNamespace:
    """Reconstruct an args.Namespace-like object from the args.json dict
    so build_dynamic_dataset() can use it. Pads in defaults for fields
    that may have been added since the checkpoint was written."""
    ns = types.SimpleNamespace(**saved_args)
    # Defaults for any fields the factory expects but old ckpts might miss.
    for k, default in [
        ("data_source", "dblp"),
        ("data_path", None),
        ("synthetic", False),
        ("synthetic_nodes", 500),
        ("synthetic_years", 30),
        ("synthetic_kind", "dblp"),
        ("synthetic_blocks", 8),
        ("history_size", 3),
        ("num_preds", 1),
        ("seed", 42),
        ("jodie_data_path", None),
        ("jodie_num_bins", 30),
        ("reddit_data_path", None),
        ("reddit_num_bins", 36),
        ("reddit_top_k", 1000),
        ("with_edge_features", False),
        ("normalize_edge_features", True),
        ("with_state_labels", False),
        ("jodie_state_label_mode", "ever"),
        ("cldg_data_dir", "dataset/cldg"),
        ("cldg_num_bins", 30),
        ("cldg_binning", "equal_freq"),
        ("cldg_top_k", 0),
        ("cldg_file_features", True),
        ("tgbn_data_dir", "dataset/tgb/tgbn-trade"),
        ("tgbn_weight_features", True),
    ]:
        if not hasattr(ns, k):
            setattr(ns, k, default)
    return ns


@torch.no_grad()
def collect_test_latents(encoder, predictor, loader, history_size, device):
    """Extract per-(node, window, future-step) latents for encoder + JEPA.
    Slim version of analyze_jepa.collect_representations — drops the
    expensive PageRank / k-core computation since we only need the latents.

    Also collects per-row degree at the target timestep so callers can fit
    a degree-only baseline probe alongside the learned-representation
    probes (essential on the binary state-label task, where banned users
    correlate with activity and we want to subtract out that confound).

    Three latent columns per row:
      Z_enc  — encoder at the TARGET timestep with the full window visible:
               an ORACLE upper bound (it has seen the answer's snapshot).
      Z_last — encoder at the LAST HISTORY step, held static across future
               horizons: the fair copy-last baseline the JEPA forecast has
               to beat.
      Z_pred — the predictor's forecast latent at the target step.

    Besides the single target-step degree, also collects deg_hist — the
    node's degree at every history bin (rows, T_hist). A probe on deg_hist
    is the activity-trend baseline: it sees the same raw temporal signal
    the predictor does, minus any learned structure. The forecast has to
    beat THIS, not just the static degree, to show learned dynamics.
    """
    enc_rows, last_rows, pred_rows, win_rows, node_rows = [], [], [], [], []
    prev_rows = []                        # latent at the second-to-last history step
    deg_rows, dhist_rows, bin_rows, hor_rows = [], [], [], []
    encoder.eval(); predictor.eval()
    for wid, batch in enumerate(loader):
        x_hist = batch["x_hist"].to(device)
        x_fut = batch["x_fut"].to(device)
        adj = batch["adj"].to(device)
        active = batch["active"].to(device)
        year = batch["year"]                  # (B,) absolute window start index

        adj_norm = normalize_adjacency(adj)
        x_full = torch.cat([x_hist, x_fut], dim=1)
        z_enc_full = encoder(x_full, adj_norm)
        z_hist = encoder(x_hist, adj_norm[:, :history_size])
        # Two-level runs (--hier_ckpt): the frozen coarse model inside the
        # predictor turns the raw history into the top-down hint.
        make_cond = getattr(predictor, "make_cond", None)
        cond = make_cond(x_hist, adj, active, x_fut=x_fut) if make_cond else None
        z_pred_fut = predictor.inference(
            z_hist, adj, active, **({"cond": cond} if cond is not None else {}))

        B, T_total, N = active.shape
        T_pred = z_pred_fut.shape[1]
        for b in range(B):
            for tp in range(T_pred):
                t_global = history_size + tp
                act = active[b, t_global]
                if act.sum() < 2:
                    continue
                idx = act.nonzero(as_tuple=True)[0]
                # Per-row degree at the target timestep, for the baseline probe.
                deg_b = adj[b, t_global].float().sum(dim=-1)[idx].cpu()
                # Degree at every HISTORY bin (activity-trend baseline).
                dhist_b = adj[b, :history_size].float().sum(dim=-1)[:, idx].T.cpu()
                enc_rows.append(z_enc_full[b, t_global, idx].cpu())
                last_rows.append(z_hist[b, -1, idx].cpu())
                prev_rows.append(z_hist[b, -2 if z_hist.shape[1] > 1 else -1, idx].cpu())
                pred_rows.append(z_pred_fut[b, tp, idx].cpu())
                dhist_rows.append(dhist_b)
                win_rows.append(torch.full((idx.numel(),), wid * 1000 + b))
                node_rows.append(idx.cpu().long())
                deg_rows.append(deg_b)
                # Absolute snapshot index of this target step, for dynamic
                # per-(bin, node) labels.
                abs_bin = int(year[b]) + t_global
                bin_rows.append(torch.full((idx.numel(),), abs_bin))
                hor_rows.append(torch.full((idx.numel(),), tp))

    Z_enc = torch.cat(enc_rows, dim=0).numpy().astype(np.float64)
    Z_last = torch.cat(last_rows, dim=0).numpy().astype(np.float64)
    Z_prev = torch.cat(prev_rows, dim=0).numpy().astype(np.float64)
    Z_pred = torch.cat(pred_rows, dim=0).numpy().astype(np.float64)
    window = torch.cat(win_rows, dim=0).numpy()
    node_id = torch.cat(node_rows, dim=0).numpy()
    degree = torch.cat(deg_rows, dim=0).numpy().astype(np.float64)
    deg_hist = torch.cat(dhist_rows, dim=0).numpy().astype(np.float64)
    abs_bin = torch.cat(bin_rows, dim=0).numpy()
    horizon = torch.cat(hor_rows, dim=0).numpy()
    return (Z_enc, Z_last, Z_pred, window, node_id, degree, deg_hist,
            abs_bin, horizon, Z_prev)


@torch.no_grad()
def collect_encoder_latents(encoder, loader, history_size, device):
    """Encoder-only variant of collect_test_latents for BARE encoder
    checkpoints (baselines CLDG / DDGCL / IDOL, or any save_encoder_ckpt
    output) that have no JEPA predictor. Extracts exactly the `encoder`
    latents at the target (future) timesteps that DELOS's `encoder`
    column uses, so the state probe compares the same representation across
    every method. Also returns per-row degree for the degree-only baseline
    and the copy-last column Z_last (last-history-step latent, the fair
    no-future baseline).
    """
    enc_rows, last_rows, win_rows, node_rows = [], [], [], []
    prev_rows = []                        # latent at the second-to-last history step
    deg_rows, dhist_rows, bin_rows, hor_rows = [], [], [], []
    encoder.eval()
    for wid, batch in enumerate(loader):
        x_hist = batch["x_hist"].to(device)
        x_fut = batch["x_fut"].to(device)
        adj = batch["adj"].to(device)
        active = batch["active"].to(device)
        year = batch["year"]                  # (B,) absolute window start index

        adj_norm = normalize_adjacency(adj)
        x_full = torch.cat([x_hist, x_fut], dim=1)
        z_enc_full = encoder(x_full, adj_norm)
        z_hist = encoder(x_hist, adj_norm[:, :history_size])

        B, T_total, N = active.shape
        T_pred = x_fut.shape[1]
        for b in range(B):
            for tp in range(T_pred):
                t_global = history_size + tp
                act = active[b, t_global]
                if act.sum() < 2:
                    continue
                idx = act.nonzero(as_tuple=True)[0]
                deg_b = adj[b, t_global].float().sum(dim=-1)[idx].cpu()
                dhist_b = adj[b, :history_size].float().sum(dim=-1)[:, idx].T.cpu()
                enc_rows.append(z_enc_full[b, t_global, idx].cpu())
                last_rows.append(z_hist[b, -1, idx].cpu())
                prev_rows.append(z_hist[b, -2 if z_hist.shape[1] > 1 else -1, idx].cpu())
                dhist_rows.append(dhist_b)
                win_rows.append(torch.full((idx.numel(),), wid * 1000 + b))
                node_rows.append(idx.cpu().long())
                deg_rows.append(deg_b)
                abs_bin = int(year[b]) + t_global
                bin_rows.append(torch.full((idx.numel(),), abs_bin))
                hor_rows.append(torch.full((idx.numel(),), tp))

    Z_enc = torch.cat(enc_rows, dim=0).numpy().astype(np.float64)
    Z_last = torch.cat(last_rows, dim=0).numpy().astype(np.float64)
    Z_prev = torch.cat(prev_rows, dim=0).numpy().astype(np.float64)
    window = torch.cat(win_rows, dim=0).numpy()
    node_id = torch.cat(node_rows, dim=0).numpy()
    degree = torch.cat(deg_rows, dim=0).numpy().astype(np.float64)
    deg_hist = torch.cat(dhist_rows, dim=0).numpy().astype(np.float64)
    abs_bin = torch.cat(bin_rows, dim=0).numpy()
    horizon = torch.cat(hor_rows, dim=0).numpy()
    return Z_enc, Z_last, window, node_id, degree, deg_hist, abs_bin, horizon, Z_prev


def _true_velocity(ds, node_id, last_bin):
    """Hidden velocity at each row's LAST OBSERVED bin, from the generator's own
    recursion z(t) = z(t-1) + v(t) + theta * (a_{c(t)} - z(t-1)):
    v(t) = z(t) - z(t-1) - theta * (a_{c(t)} - z(t-1)). Position and the
    mean-reversion pull are functions of the current snapshot; v is the part of
    the dynamics only the trajectory reveals, so R^2 to it measures how much
    HIDDEN STATE a representation carries, independent of horizon. Returns
    (v, valid) with valid False where t-1 < 0."""
    lat = getattr(ds, "latents", None)
    anchors = getattr(ds, "anchors", None)
    comm = getattr(ds, "communities", None)
    if lat is None or anchors is None or comm is None:
        return None, None
    theta = float(getattr(ds, "ou_theta", 0.0))
    t = np.clip(last_bin, 1, lat.shape[0] - 1)
    z_t, z_p = lat[t, node_id].astype(np.float64), lat[t - 1, node_id].astype(np.float64)
    a = anchors[comm[t, node_id]].astype(np.float64)
    return z_t - z_p - theta * (a - z_p), last_bin >= 1


def _fit_per_horizon(columns: dict, y, train_v, test_v, horizon, task: str, *,
                     per_row_prev=None, z_tgt=None, z_delta=None, z_vel=None,
                     vel_ok=None):
    """One probe PER FORECAST HORIZON for every column, written into each
    column's dict as ``subset_h<k>`` (k = 0 is one step past the last observed
    bin). Only for multi-horizon runs (num_preds > 1); a no-op otherwise.

    Why fit rather than score: a single probe pooled over all horizons is fed
    the SAME copy-last latent for every k and cannot even tell which horizon it
    is answering -- a handicap for the encoder-only baselines, whose only way
    to forecast k steps ahead is a head trained for k steps ahead ("direct"
    multi-step forecasting). Fitting per horizon gives every column, forecast
    and copy-last alike, a head trained on exactly the rows it is scored on,
    so the horizon curve compares representations, not head budgets.

    Per horizon and column: the task's headline metrics on that horizon's test
    rows, the switch / stay (multiclass) or onset (binary) subsets defined
    against the LAST OBSERVED label, the change alarm (multiclass) and, when
    the true latent is available, the ridge R^2 to it and to its change.
    """
    if horizon is None or len(np.unique(horizon)) < 2:
        return
    kw = dict(device=torch.device("cpu"))
    has_prev = None if per_row_prev is None else per_row_prev >= 0
    for k in np.unique(horizon):
        rows = horizon == k
        tr, te = train_v & rows, test_v & rows
        subs = {}
        if has_prev is not None:
            if task == "multiclass":
                subs = {"switch": has_prev & (per_row_prev != y) & rows,
                        "stay": has_prev & (per_row_prev == y) & rows}
            else:
                subs = {"onset": (per_row_prev == 0) & rows}
        for name, (probe, Z) in columns.items():
            r = train_node_probe(Z, y, tr, te, task, test_subsets=subs or None, **kw)
            flat = {"n": int(te.sum())}
            if task == "ranking":
                flat["ndcg10_test"] = r["ndcg10_test"]
            elif task == "multiclass":
                flat.update({"acc_test": r["acc_test"], "macro_f1_test": r["macro_f1_test"]})
                for sub in ("switch", "stay"):
                    if f"subset_{sub}" in r:
                        flat[f"{sub}_acc_test"] = r[f"subset_{sub}"]["acc_test"]
                        flat[f"n_{sub}"] = r[f"subset_{sub}"]["n"]
                if per_row_prev is not None:
                    flat.update(_change_alarm(r, y, per_row_prev))
                r.pop("_test_rows", None); r.pop("_pred_test", None)
            else:
                flat.update({"auroc_test": r["auroc_test"], "auprc_test": r["auprc_test"],
                             "n_pos": r["n_pos_test"]})
                if "subset_onset" in r:
                    flat["onset_auroc_test"] = r["subset_onset"]["auroc_test"]
                    flat["onset_auprc_test"] = r["subset_onset"]["auprc_test"]
                    flat["n_onset"] = r["subset_onset"]["n"]
            if z_tgt is not None and te.sum() > 1:
                flat["latent_r2_test"] = _ridge_r2(Z, z_tgt, tr, te)
                flat["delta_r2_test"] = _ridge_r2(Z, z_delta, tr, te)
            if z_vel is not None and (te & vel_ok).sum() > 1:
                flat["vel_r2_test"] = _ridge_r2(Z, z_vel, tr & vel_ok, te & vel_ok)
            probe[f"subset_h{int(k)}"] = flat


def _binary_state_result(Z_enc, Z_last, degree, per_row_block, train, test,
                         node_id, label_source, device, Z_pred=None,
                         per_row_prev=None, deg_hist=None, horizon=None,
                         Z_prev=None):
    """Fit the binary state-label (ban prediction) probe on each latent
    column and the degree-only baseline. Shared by the full-ckpt (run_one)
    and bare-encoder (run_one_encoder) paths so both report identical fields.
    Filters the -1 rows (users outside the top-K subset, which carry no valid
    label) and reports AUROC + AUPRC.

    Columns:
      encoder_oracle — encoder at the target step, full window visible
                       (upper bound: it has seen the answer's snapshot).
      encoder_last   — encoder at the last history step (fair copy-last
                       baseline; what the JEPA forecast has to beat).
      jepa           — the predictor's forecast latent (full ckpts only).
      degree_baseline— degree-only confound control (target-step degree).
      degree_history_baseline — probe on the node's degree at every history
                       bin: the activity-TREND control. The forecast must
                       beat this to show learned dynamics rather than a raw
                       activity-decline readout.

    per_row_prev (dynamic label modes only): each row's label at the
    PREVIOUS snapshot. Enables the 'onset' test subset — rows whose previous
    label is 0 — where label-persistence carries no signal and the question
    is genuinely "who transitions next?".
    """
    valid_rows = per_row_block >= 0
    if valid_rows.sum() == 0:
        raise ValueError("No valid binary labels in test rows — did the "
                         "dataset expose ds.node_labels?")
    train_v = train & valid_rows
    test_v = test & valid_rows
    y = per_row_block.astype(np.int64)
    subsets = {}
    if per_row_prev is not None:
        subsets["onset"] = (per_row_prev == 0) & valid_rows
    # Per-horizon results (multi-horizon runs only) are FITTED per horizon by
    # _fit_per_horizon below, not scored from this pooled probe -- see there.
    # Fit every probe on CPU: GPU AdamW kernels are nondeterministic, so on
    # cuda the SAME frozen-encoder latents gave a different encoder_last per
    # run (~0.005 AUROC jitter — the size of the effect we measure). On CPU the
    # fit is byte-reproducible, so the four heads' shared encoder yields an
    # identical encoder_last. The probes are tiny; the CPU cost is negligible.
    # (`device` — the latent-collection device — is intentionally not the fit
    # device.)
    kw = dict(device=torch.device("cpu"), test_subsets=subsets or None)
    enc_probe = train_node_probe(Z_enc, y, train_v, test_v, "binary", **kw)
    last_probe = train_node_probe(Z_last, y, train_v, test_v, "binary", **kw)
    deg_baseline = train_node_probe(degree.reshape(-1, 1), y, train_v, test_v,
                                    "binary", **kw)
    result = {
        "task": "binary",
        "label_source": label_source,
        "n_rows": int(len(node_id)),
        "n_valid_rows": int(valid_rows.sum()),
        "n_pos_train": int((y[train_v] == 1).sum()),
        "n_pos_test": int((y[test_v] == 1).sum()),
        "encoder": enc_probe,                 # legacy key = oracle column
        "encoder_oracle": enc_probe,
        "encoder_last": last_probe,
        "degree_baseline": deg_baseline,
    }
    if deg_hist is not None:
        result["degree_history_baseline"] = train_node_probe(
            deg_hist, y, train_v, test_v, "binary", **kw)
    if Z_pred is not None:
        result["jepa"] = train_node_probe(Z_pred, y, train_v, test_v,
                                          "binary", **kw)
    columns = {"encoder_oracle": (enc_probe, Z_enc), "encoder_last": (last_probe, Z_last),
               "degree_baseline": (deg_baseline, degree.reshape(-1, 1))}
    if Z_prev is not None:
        Z_diff = _velocity_feature(Z_last, Z_prev)
        result["encoder_diff"] = train_node_probe(Z_diff, y, train_v, test_v, "binary", **kw)
        columns["encoder_diff"] = (result["encoder_diff"], Z_diff)
    if deg_hist is not None:
        columns["degree_history_baseline"] = (result["degree_history_baseline"], deg_hist)
    if Z_pred is not None:
        columns["jepa"] = (result["jepa"], Z_pred)
    _fit_per_horizon(columns, y, train_v, test_v, horizon, "binary",
                     per_row_prev=per_row_prev)
    return result


def _ridge_r2(Z, Y, train_v, test_v, lam: float = 1e-2) -> float:
    """Closed-form ridge from representation Z (rows, D) to a multi-dim target
    Y (rows, d), standardised on TRAIN rows; returns the pooled test R^2
    (1 - SSE/SST over all target dims). Deterministic, no GPU."""
    mu, sd = Z[train_v].mean(0), Z[train_v].std(0) + 1e-6
    X = (Z - mu) / sd
    ymu = Y[train_v].mean(0)
    Xtr = np.c_[X[train_v], np.ones(train_v.sum())]
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ (Y[train_v] - ymu))
    pred = np.c_[X[test_v], np.ones(test_v.sum())] @ W + ymu
    sse = float(((Y[test_v] - pred) ** 2).sum())
    sst = float(((Y[test_v] - Y[test_v].mean(0)) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-12)


def _change_alarm(probe: dict, y: np.ndarray, prev: np.ndarray) -> dict:
    """Treat 'predicted state != current state' as a CHANGE ALARM and score it
    against the true change (y != prev) on the probe's test rows. One number
    for the trade-off the two accuracies pull apart: recall is anticipation,
    precision is the false-alarm side. Rows with no previous label are
    skipped. Pops the private per-row keys from the probe dict."""
    rows = probe.pop("_test_rows", None)
    pred = probe.pop("_pred_test", None)
    if rows is None:
        return {}
    keep = prev[rows] >= 0
    rows, pred = rows[keep], pred[keep]
    alarm = pred != prev[rows]
    truth = y[rows] != prev[rows]
    tp = int((alarm & truth).sum()); fp = int((alarm & ~truth).sum()); fn = int((~alarm & truth).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (tp + fp) and (tp + fn) and (prec + rec) > 0 else 0.0
    return {"change_precision": prec, "change_recall": rec, "change_f1": f1,
            "alarm_rate": float(alarm.mean()) if len(alarm) else float("nan"),
            "true_change_rate": float(truth.mean()) if len(truth) else float("nan")}


def _velocity_feature(Z_last, Z_prev):
    """Hand-built velocity feature from the SAME encoder: [z_last, z_last - z_prev].
    The `encoder_diff` column. If a linear head on this recovers the hidden
    velocity (vel_r2_test) far better than z_last alone, the information IS in
    the encoder's latents and a predictor that fails to use it is the
    bottleneck; if not, the encoder is."""
    return np.concatenate([Z_last, Z_last - Z_prev], axis=1)


def _multiclass_state_result(Z_enc, Z_last, per_row, per_row_prev, train, test,
                             node_id, label_source, Z_pred=None, deg_hist=None,
                             horizon=None, latents=None, abs_bin=None, ds=None,
                             Z_prev=None):
    """Dense multiclass node-STATE probe (synthetic_dyn --synth_state_target
    community). Same columns as the binary path so the classification
    aggregator and slurm scripts work unchanged, plus:

      label_persistence  predict each row's PREVIOUS label. Not a probe: the
                         exact copy-last answer in label space, the floor every
                         column has to beat.
      subset_switch      rows whose label differs from the previous step.
                         Persistence scores 0 here by construction, so this is
                         the pure anticipation measure.
      subset_stay        rows whose label did not change: the false-alarm side.
      change_*           on every probed column: 'predicted != current' as a
                         change alarm scored against the true change --
                         precision (false alarms), recall (anticipation), F1.
      latent_r2_test     synthetic only (needs ds.latents): pooled R^2 of a
                         ridge from the column's representation to the TRUE
                         causal latent z at the target step. How much of the
                         cause the representation carries.
      vel_r2_test       synthetic only: R^2 to the HIDDEN velocity at the last
                        observed step (see _true_velocity) -- the part of the
                        dynamics no single snapshot can read.
      delta_r2_test      same, for z(target) - z(last observed step): the
                         CHANGE of the latent. Position alone predicts some of
                         it (mean reversion); the momentum part needs velocity,
                         which only a forecast can carry. Dense over all rows,
                         so no 5%-of-rows statistics problem.

    Rows with no previous label (bin 0) are kept for the overall numbers but
    excluded from the subsets.
    """
    valid = per_row >= 0
    if valid.sum() == 0:
        raise ValueError("No valid multiclass state labels in the probe rows.")
    train_v, test_v = train & valid, test & valid
    y = per_row.astype(np.int64)
    has_prev = per_row_prev >= 0
    subsets = {"switch": has_prev & (per_row_prev != y) & valid,
               "stay": has_prev & (per_row_prev == y) & valid}
    # Per-horizon results are FITTED per horizon (_fit_per_horizon), not
    # scored from this pooled probe.
    # CPU fit for byte-reproducibility (see _binary_state_result).
    kw = dict(device=torch.device("cpu"), test_subsets=subsets)
    enc_probe = train_node_probe(Z_enc, y, train_v, test_v, "multiclass", **kw)
    last_probe = train_node_probe(Z_last, y, train_v, test_v, "multiclass", **kw)

    te_prev = test_v & has_prev
    persist = {
        "acc_test": float((per_row_prev[te_prev] == y[te_prev]).mean()),
        "macro_f1_test": float("nan"),
        "n": int(te_prev.sum()),
        "subset_switch": {"acc_test": 0.0, "n": int((subsets["switch"] & test_v).sum())},
        "subset_stay": {"acc_test": 1.0, "n": int((subsets["stay"] & test_v).sum())},
    }
    persist.update({"change_precision": float("nan"), "change_recall": 0.0,
                    "change_f1": 0.0, "alarm_rate": 0.0})
    columns = {"encoder_oracle": (enc_probe, Z_enc), "encoder_last": (last_probe, Z_last)}
    if Z_prev is not None:
        Z_diff = _velocity_feature(Z_last, Z_prev)
        columns["encoder_diff"] = (
            train_node_probe(Z_diff, y, train_v, test_v, "multiclass", **kw), Z_diff)
    if deg_hist is not None:
        columns["degree_history_baseline"] = (
            train_node_probe(deg_hist, y, train_v, test_v, "multiclass", **kw), deg_hist)
    if Z_pred is not None:
        columns["jepa"] = (train_node_probe(Z_pred, y, train_v, test_v, "multiclass", **kw), Z_pred)
        if Z_prev is not None:
            # jepa_plus_diff (2026-10-05): the rollout latent READ TOGETHER with the
            # same encoder's [z_last, z_last - z_prev]. The Δlatent control beats the
            # rollout at long horizons on the synthetic task even when the rollout
            # embeds the same linear forecast (ridge-initialised skip), i.e. an
            # MSE-optimal latent forecast loses label information that a head on the
            # raw history keeps. This column asks whether the rollout adds anything
            # BEYOND that linear information: jepa_plus_diff > encoder_diff = yes.
            Z_pd = np.concatenate([Z_pred, _velocity_feature(Z_last, Z_prev)], axis=1)
            columns["jepa_plus_diff"] = (
                train_node_probe(Z_pd, y, train_v, test_v, "multiclass", **kw), Z_pd)

    # Ground-truth latent targets (synthetic generator only).
    z_tgt = z_delta = z_vel = vel_ok = None
    if latents is not None and abs_bin is not None:
        T = latents.shape[0]
        tgt_bin = np.clip(abs_bin, 0, T - 1)
        hor = np.zeros_like(abs_bin) if horizon is None else horizon
        last_bin = np.clip(abs_bin - hor - 1, 0, T - 1)
        z_tgt = latents[tgt_bin, node_id].astype(np.float64)
        z_delta = z_tgt - latents[last_bin, node_id].astype(np.float64)
        if ds is not None:
            z_vel, vel_ok = _true_velocity(ds, node_id, abs_bin - hor - 1)

    for name, (probe, Z) in columns.items():
        probe.update(_change_alarm(probe, y, per_row_prev))
        if z_tgt is not None:
            probe["latent_r2_test"] = _ridge_r2(Z, z_tgt, train_v, test_v)
            probe["delta_r2_test"] = _ridge_r2(Z, z_delta, train_v, test_v)
        if z_vel is not None:
            # vel_r2_test: R^2 to the HIDDEN velocity at the last observed
            # step -- horizon-independent, the sharpest "carries the
            # dynamics" measure (delta_r2 grows with h through mean reversion,
            # which position alone predicts).
            probe["vel_r2_test"] = _ridge_r2(Z, z_vel, train_v & vel_ok, test_v & vel_ok)

    # Horizon curve (num_preds > 1): a head per horizon on every column, and
    # the persistence floor at each horizon = "the label at the last observed
    # bin still holds" (0 on switch rows, 1 on stay rows by construction).
    _fit_per_horizon(columns, y, train_v, test_v, horizon, "multiclass",
                     per_row_prev=per_row_prev, z_tgt=z_tgt, z_delta=z_delta,
                     z_vel=z_vel, vel_ok=vel_ok)
    if horizon is not None and len(np.unique(horizon)) > 1:
        for k in np.unique(horizon):
            rows = (horizon == k) & te_prev
            n_sw = int((subsets["switch"] & test_v & rows).sum())
            persist[f"subset_h{int(k)}"] = {
                "acc_test": float((per_row_prev[rows] == y[rows]).mean()) if rows.any() else float("nan"),
                "switch_acc_test": 0.0, "stay_acc_test": 1.0,
                "n": int(rows.sum()), "n_switch": n_sw,
                "switch_rate": n_sw / max(int(rows.sum()), 1),
            }

    result = {
        "task": "multiclass_state",
        "label_source": label_source,
        "n_rows": int(len(node_id)),
        "n_valid_rows": int(valid.sum()),
        "n_classes_seen": int(enc_probe["n_classes"]),
        "n_switch_test": int((subsets["switch"] & test_v).sum()),
        "switch_rate_test": float((subsets["switch"] & test_v).sum() / max(te_prev.sum(), 1)),
        "encoder": enc_probe,
        "label_persistence": persist,
        **{k: v[0] for k, v in columns.items()},
    }
    return result


def _train_test_split_by_node(node_id: np.ndarray, seed: int = 0,
                              test_frac: float = 0.3):
    """Hold out whole NODES. For a static per-node label the window split is
    wrong: the same node carries the same label in every window it appears
    in, so a probe fitted on its train-window rows would simply memorise it
    and score it again on its test-window rows. Holding nodes out makes the
    test rows nodes the head has never seen -- transductive node
    classification, the CLDG / DDGCL protocol."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(node_id)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_mask = np.isin(node_id, uniq[:n_test])
    return ~test_mask, test_mask


def _multiclass_static_result(Z_enc, Z_last, per_row, node_id, label_source,
                              Z_pred=None, deg_hist=None, horizon=None,
                              Z_prev=None):
    """STATIC multiclass node-classification probe (CLDG-format dblp / patent:
    one class per node for the whole timeline). Same column names as the
    state probes so the classification launcher / aggregator / checkpoint
    selection read it unchanged:

      encoder_oracle  encoder at the target step, full window visible
      encoder_last    encoder at the last history step (copy-last)
      jepa            the predictor's forecast latent (full ckpts only)
      degree_history_baseline  degree at every history bin (trend control)

    There is no label_persistence floor and no switch / stay subset: a static
    label never changes, so those would be 1.0 / empty by construction. The
    train/test split is over NODES (see _train_test_split_by_node); every
    window of the probe split contributes rows, and a node's rows all fall on
    one side.
    """
    valid = per_row >= 0
    if valid.sum() == 0:
        raise ValueError("No valid static multiclass labels in the probe rows "
                         "-- does the dataset expose ds.node_labels?")
    train, test = _train_test_split_by_node(node_id, seed=0)
    train_v, test_v = train & valid, test & valid
    y = per_row.astype(np.int64)
    kw = dict(device=torch.device("cpu"))       # CPU fit: byte-reproducible
    enc_probe = train_node_probe(Z_enc, y, train_v, test_v, "multiclass", **kw)
    last_probe = train_node_probe(Z_last, y, train_v, test_v, "multiclass", **kw)
    columns = {"encoder_oracle": (enc_probe, Z_enc), "encoder_last": (last_probe, Z_last)}
    if Z_prev is not None:
        Z_diff = _velocity_feature(Z_last, Z_prev)
        columns["encoder_diff"] = (
            train_node_probe(Z_diff, y, train_v, test_v, "multiclass", **kw), Z_diff)
    if deg_hist is not None:
        columns["degree_history_baseline"] = (
            train_node_probe(deg_hist, y, train_v, test_v, "multiclass", **kw), deg_hist)
    if Z_pred is not None:
        columns["jepa"] = (train_node_probe(Z_pred, y, train_v, test_v, "multiclass", **kw), Z_pred)
    for probe, _ in columns.values():           # private per-row keys
        probe.pop("_test_rows", None); probe.pop("_pred_test", None)
    _fit_per_horizon(columns, y, train_v, test_v, horizon, "multiclass")
    return {
        "task": "multiclass_static",
        "label_source": label_source,
        "split": "node",
        "n_rows": int(len(node_id)),
        "n_valid_rows": int(valid.sum()),
        "n_nodes_train": int(len(np.unique(node_id[train_v]))),
        "n_nodes_test": int(len(np.unique(node_id[test_v]))),
        "n_classes_seen": int(enc_probe["n_classes"]),
        "encoder": enc_probe,
        **{k: v[0] for k, v in columns.items()},
    }


def _per_row_label_vectors(ds, node_id, abs_bin, horizon=None, history_size=None):
    """Ranking labels (tgbn_*): each row's label DISTRIBUTION at its own
    snapshot, ``ds.node_label_dist[abs_bin, node]`` (rows, C), with ``valid``
    where the row has a label, plus two label-only floors defined against
    what a forecaster could see, as for the state tasks:

      Yprev / prev_valid  the distribution at the LAST OBSERVED bin
                          (``abs_bin - 1 - horizon``): TGB's persistence.
      Yavg / avg_valid    the MEAN of the valid label vectors over the
                          ``history_size`` observed bins ending there: TGB's
                          moving average, the stronger floor on the weekly
                          interaction datasets (genre / reddit / token), and
                          what a window encoder's copy-last can represent.
    """
    dist = getattr(ds, "node_label_dist", None)
    if dist is None:
        raise ValueError("--task ranking needs a dataset exposing "
                         "ds.node_label_dist (tgbn_trade / tgbn_genre / "
                         "tgbn_reddit / tgbn_token).")
    T = dist.shape[0]
    tgt = np.clip(abs_bin, 0, T - 1)
    Y = dist[tgt, node_id]
    valid = ds.label_valid[tgt, node_id]
    hor = np.zeros_like(tgt) if horizon is None else np.asarray(horizon)
    prev_bin = tgt - 1 - hor
    prev_ok = prev_bin >= 0
    Yprev = dist[np.clip(prev_bin, 0, None), node_id]
    prev_valid = prev_ok & ds.label_valid[np.clip(prev_bin, 0, None), node_id]
    Yprev = Yprev * prev_valid[:, None]
    H = int(history_size or getattr(ds, "history_size", 1) or 1)
    Yavg = np.zeros_like(Y, dtype=np.float64)
    n_avg = np.zeros(len(tgt), dtype=np.float64)
    for j in range(H):
        b = prev_bin - j
        ok = b >= 0
        v = ok & ds.label_valid[np.clip(b, 0, None), node_id]
        Yavg += dist[np.clip(b, 0, None), node_id] * v[:, None]
        n_avg += v
    avg_valid = n_avg > 0
    Yavg = np.divide(Yavg, n_avg[:, None], out=np.zeros_like(Yavg), where=n_avg[:, None] > 0)
    return Y, valid, Yprev, prev_valid, Yavg.astype(Y.dtype), avg_valid


def _ranking_state_result(Z_enc, Z_last, Y, valid, Yprev, prev_valid, train,
                          test, node_id, label_source, Z_pred=None,
                          deg_hist=None, horizon=None, Yavg=None, avg_valid=None,
                          **_ignored):
    """Per-step RANKING probe (tgbn-trade: predict each nation's next export
    share vector, scored with NDCG@10 as in TGB). Same columns as the state
    probes so the classification tooling reads it unchanged:

      encoder_oracle / encoder_last / jepa / degree_history_baseline
                         linear head from the latent to C scores, trained
                         with the soft-target cross-entropy on the train
                         windows' rows, NDCG@10 on the test windows' rows.
      label_persistence  not a probe: the LAST OBSERVED label vector used as
                         the scores -- TGB's persistence baseline. Rows
                         without a previous label are skipped for it.
      label_moving_average
                         not a probe: the mean of the label vectors over the
                         history window -- TGB's moving-average baseline,
                         which beats persistence and every learned model on
                         the weekly interaction datasets; the floor to beat.

    The label changes every step, so the window split of the dynamic tasks
    applies (a node's rows at different years are different questions).
    """
    if valid.sum() == 0:
        raise ValueError("No labelled rows for the ranking probe.")
    train_v, test_v = train & valid, test & valid
    kw = dict(device=torch.device("cpu"))
    enc_probe = train_node_probe(Z_enc, Y, train_v, test_v, "ranking", **kw)
    last_probe = train_node_probe(Z_last, Y, train_v, test_v, "ranking", **kw)
    columns = {"encoder_oracle": (enc_probe, Z_enc), "encoder_last": (last_probe, Z_last)}
    if deg_hist is not None:
        columns["degree_history_baseline"] = (
            train_node_probe(deg_hist, Y, train_v, test_v, "ranking", **kw), deg_hist)
    if Z_pred is not None:
        columns["jepa"] = (train_node_probe(Z_pred, Y, train_v, test_v, "ranking", **kw), Z_pred)

    te_prev = test_v & prev_valid
    persist = {"ndcg10_test": float(ndcg_at_k(Y[te_prev], Yprev[te_prev], 10).mean())
               if te_prev.any() else float("nan"),
               "n": int(te_prev.sum())}
    movavg = None
    if Yavg is not None and avg_valid is not None:
        te_avg = test_v & avg_valid
        movavg = {"ndcg10_test": float(ndcg_at_k(Y[te_avg], Yavg[te_avg], 10).mean())
                  if te_avg.any() else float("nan"),
                  "n": int(te_avg.sum())}
    _fit_per_horizon(columns, Y, train_v, test_v, horizon, "ranking")
    if horizon is not None and len(np.unique(horizon)) > 1:
        for k in np.unique(horizon):
            rows = (horizon == k) & te_prev
            persist[f"subset_h{int(k)}"] = {
                "ndcg10_test": float(ndcg_at_k(Y[rows], Yprev[rows], 10).mean()) if rows.any() else float("nan"),
                "n": int(rows.sum())}
            if movavg is not None:
                rows = (horizon == k) & te_avg
                movavg[f"subset_h{int(k)}"] = {
                    "ndcg10_test": float(ndcg_at_k(Y[rows], Yavg[rows], 10).mean()) if rows.any() else float("nan"),
                    "n": int(rows.sum())}
    out = {
        "task": "ranking",
        "label_source": label_source,
        "metric": "ndcg@10",
        "n_rows": int(len(node_id)),
        "n_valid_rows": int(valid.sum()),
        "n_classes": int(Y.shape[1]),
        "encoder": enc_probe,
        "label_persistence": persist,
        **{k: v[0] for k, v in columns.items()},
    }
    if movavg is not None:
        out["label_moving_average"] = movavg
    return out


def _state_probe_name(eff_source: str, split: str) -> str:
    """state_probe.json, or switch_probe.json when the binary target was DERIVED
    from a multiclass-state dataset -- so re-probing a state checkpoint for the
    switch task never overwrites its multiclass result."""
    stem = "switch_probe" if "switch:derived" in eff_source else "state_probe"
    return f"{stem}{_SPLIT_SUFFIX[split]}.json"


def _per_row_labels(ds, node_id, abs_bin, label_source, task, horizon=None):
    """Resolve the per-row label vector for the probe.

    When the dataset exposes a DYNAMIC per-(bin, node) matrix (JODIE
    --jodie_state_label_mode event/state), each row is labelled at *its own*
    snapshot via node_labels_dyn[abs_bin, node] — that's the whole point of
    relabelling. Otherwise falls back to the static per-node
    compute_block_labels. Returns (per_row, block_labels_or_None, eff_source);
    block_labels is None in the dynamic case (multiclass never uses it there).

    Also returns per_row_prev — each row's label at the LAST OBSERVED snapshot
    (dynamic modes only, else None): abs_bin - 1 - horizon, i.e. the final
    history bin of the row's window. For a single-horizon probe (horizon 0)
    that is the previous bin, as before. For multi-horizon runs it is what a
    forecaster can actually see, so label persistence, the onset / switch /
    stay subsets and the change alarm are all defined against the same
    observable state at every horizon. (Using abs_bin - 1 there would hand the
    persistence floor the label ONE step before the target -- a future value
    at any horizon beyond the first.) Rows whose last observed bin is < 0 get
    -1, which excludes them from those subsets downstream.
    """
    dyn = getattr(ds, "node_labels_dyn", None)
    mode = str(getattr(ds, "state_label_mode", ""))
    # A synthetic dataset built with the multiclass community target still
    # carries the binary switch-event labels (ds.switch_labels). Probing it as
    # --task binary must use THOSE, not treat "community == 1" as the positive
    # class -- so one trained checkpoint serves both tasks.
    if task == "binary" and mode.startswith("community") and getattr(ds, "switch_labels", None) is not None:
        dyn = ds.switch_labels
        mode = f"switch:derived_from_{mode}"
    # Dynamic multiclass state: the synthetic community target, or a real
    # dataset that declares it (CHASE music, 5 genres per month).
    dyn_multiclass = dyn is not None and (
        mode.startswith("community") or getattr(ds, "dyn_label_task", None) == "multiclass")
    if dyn is not None and (task == "binary" or (task == "multiclass" and dyn_multiclass)):
        n_bins = dyn.shape[0]
        clipped = np.clip(abs_bin, 0, n_bins - 1)
        per_row = dyn[clipped, node_id]
        hor = np.zeros_like(clipped) if horizon is None else np.asarray(horizon)
        prev_bin = clipped - 1 - hor
        per_row_prev = np.where(prev_bin >= 0,
                                dyn[np.clip(prev_bin, 0, None), node_id], -1)
        return per_row, None, f"dataset_dynamic:{mode}", per_row_prev
    block_labels = compute_block_labels(ds, source=label_source)
    return block_labels[node_id], block_labels, label_source, None


def run_one_encoder(encoder_ckpt: Path, device: str = "cuda",
                    batch_size: int = 4, num_workers: int = 2,
                    label_source: str = "dataset",
                    task: str = "binary",
                    with_state_labels: bool = True,
                    state_label_mode: str = "ever",
                    split: str = "test") -> tuple:
    """State-probe a BARE encoder checkpoint (no JEPA predictor).

    Handles the baseline encoders (CLDG / DDGCL / IDOL) and any other
    save_encoder_ckpt output, so node classification can be compared against
    DELOS on the same footing. Only the encoder-latent column is probed
    (plus the degree-only baseline). Writes <ckpt_dir>/state_probe.json
    (binary) or <ckpt_dir>/block_probe.json (multiclass)."""
    ckpt = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
    if "encoder" not in ckpt:
        raise ValueError(f"{encoder_ckpt} has no 'encoder' field — is this a "
                         "save_encoder_ckpt output?")
    saved_args = ckpt.get("args", {})
    if not saved_args:
        raise ValueError(f"{encoder_ckpt} has no 'args' — refusing to guess "
                         "the dataset configuration.")
    args = _make_args_namespace(saved_args)
    args.device = device
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.split = split
    if with_state_labels:
        args.with_state_labels = True
    args.jodie_state_label_mode = state_label_mode
    device = torch.device(args.device)

    ds = build_dynamic_dataset(args, split=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # Encoder metadata lives at the top level of the save_encoder_ckpt schema,
    # falling back to the training args for older checkpoints.
    enc_kind = ckpt.get("encoder_kind", saved_args.get("encoder_kind", "gcn"))
    enc_hidden = ckpt.get("encoder_hidden", saved_args.get("encoder_hidden", 128))
    node_dim = ckpt.get("node_dim", saved_args.get("node_dim", 128))
    enc_bidir = ckpt.get("bidirectional", saved_args.get("bidirectional", True))
    encoder = build_encoder(
        enc_kind, in_dim=ds.feature_dim, hidden_dim=enc_hidden,
        out_dim=node_dim, freeze=True, bidirectional=enc_bidir,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])

    Z_enc, Z_last, window, node_id, degree, deg_hist, abs_bin, horizon, Z_prev = \
        collect_encoder_latents(
            encoder, loader,
            history_size=saved_args.get("history_size", 3), device=device,
        )
    train, test = _train_test_split_by_window(window, seed=0)
    if task == "ranking":
        Y, valid, Yprev, prev_valid, Yavg, avg_valid = _per_row_label_vectors(
            ds, node_id, abs_bin, horizon, history_size=saved_args.get("history_size", 3))
        result = _ranking_state_result(Z_enc, Z_last, Y, valid, Yprev, prev_valid,
                                       train, test, node_id, "dataset",
                                       deg_hist=deg_hist, horizon=horizon,
                                       Yavg=Yavg, avg_valid=avg_valid)
        out_path = Path(encoder_ckpt).parent / f"state_probe{_SPLIT_SUFFIX[split]}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return result, out_path
    per_row_block, block_labels, eff_source, per_row_prev = _per_row_labels(
        ds, node_id, abs_bin, label_source, task, horizon=horizon)

    if task == "binary":
        result = _binary_state_result(Z_enc, Z_last, degree, per_row_block,
                                      train, test, node_id, eff_source,
                                      device, per_row_prev=per_row_prev,
                                      deg_hist=deg_hist, horizon=horizon,
                                      Z_prev=Z_prev)
        out_path = Path(encoder_ckpt).parent / _state_probe_name(eff_source, split)
    elif block_labels is None:
        result = _multiclass_state_result(Z_enc, Z_last, per_row_block,
                                          per_row_prev, train, test, node_id,
                                          eff_source, deg_hist=deg_hist,
                                          horizon=horizon,
                                          latents=getattr(ds, "latents", None),
                                          abs_bin=abs_bin, ds=ds, Z_prev=Z_prev)
        out_path = Path(encoder_ckpt).parent / f"state_probe{_SPLIT_SUFFIX[split]}.json"
    elif eff_source == "dataset":
        # Static per-node classes (cldg_dblp / cldg_patent): node-level split,
        # same file and columns as the state probes.
        result = _multiclass_static_result(Z_enc, Z_last, per_row_block, node_id,
                                           eff_source, deg_hist=deg_hist,
                                           horizon=horizon, Z_prev=Z_prev)
        out_path = Path(encoder_ckpt).parent / f"state_probe{_SPLIT_SUFFIX[split]}.json"
    else:
        enc_probe = train_node_probe(Z_enc, per_row_block, train, test,
                                     "multiclass", device=torch.device("cpu"))
        enc_probe.pop("_test_rows", None); enc_probe.pop("_pred_test", None)
        result = {
            "task": "multiclass",
            "label_source": eff_source,
            "encoder": enc_probe,
            "n_classes_seen": int(enc_probe["n_classes"]),
            "n_rows": int(len(node_id)),
        }
        out_path = Path(encoder_ckpt).parent / f"block_probe{_SPLIT_SUFFIX[split]}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result, out_path


def run_one_jodie(jodie_ckpt: Path, device: str = "cuda",
                  batch_size: int = 4, num_workers: int = 2,
                  task: str = "binary",
                  state_label_mode: str = "ever",
                  num_preds_override: "int | None" = None,
                  split: str = "test") -> tuple:
    """State-probe a JODIE baseline checkpoint (src/baselines/jodie).

    JODIE is an event-stream model, so instead of running a snapshot encoder
    it reads the per-bin embedding banks dumped by train_jodie.py (dynamic
    embeddings only -- statics are one-hot identity and would let the probe
    memorise users). The standard columns map onto JODIE's own mechanisms:

      encoder_oracle : embedding at the END of the target bin (saw its events)
      encoder_last   : embedding at the end of the LAST HISTORY bin
                       (persistence -- the copy-last baseline)
      jepa           : the PROJECTION of that embedding to the target bin's
                       midpoint -- JODIE's forecasting operator, i.e. its
                       genuine analogue of the DELOS predictor column.

    Rows (windows, active-node masks, degree controls, labels, split) are
    built from the SAME dataset object as every other method, so the metrics
    are directly comparable. Binary task only.
    """
    if task != "binary":
        raise ValueError("run_one_jodie supports --task binary only")
    ckpt = torch.load(jodie_ckpt, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    if not saved_args:
        raise ValueError(f"{jodie_ckpt} has no 'args'.")
    args = _make_args_namespace(saved_args)
    args.device = device
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.split = split
    args.with_state_labels = True
    args.jodie_state_label_mode = state_label_mode
    # Re-probe at more horizons WITHOUT retraining: all per-bin embeddings are
    # stored, so the projection operator can forecast any step ahead. This only
    # changes the probe window (history + num_preds); the model is untouched.
    if num_preds_override is not None:
        args.num_preds = num_preds_override

    ds = build_dynamic_dataset(args, split=split)
    n_users, n_items = ckpt["n_users"], ckpt["n_items"]
    if ds.num_nodes != n_users + n_items:
        raise ValueError(f"node-count mismatch: dataset {ds.num_nodes} vs "
                         f"jodie ckpt {n_users}+{n_items} — different subset?")
    T = ckpt["num_bins"]
    t_lo, span = ckpt["t_lo"], ckpt["span"]
    gap_u = max(ckpt["mean_gap_user"], 1e-9)
    emb_bins = np.concatenate(
        [ckpt["user_emb_bins"], ckpt["item_emb_bins"]], axis=1)   # (T, N, D)
    last_t_u = ckpt["last_t_user_bins"]                           # (T, n_users)

    from src.baselines.jodie.jodie import JODIE                   # lazy import
    model = JODIE(n_users, n_items, emb_dim=ckpt["emb_dim"],
                  feat_dim=ckpt["feat_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    history_size = saved_args.get("history_size", 3)

    or_rows, last_rows, pred_rows, prev_rows = [], [], [], []
    win_rows, node_rows, deg_rows, dhist_rows, bin_rows, hor_rows = \
        [], [], [], [], [], []
    for wid, batch in enumerate(loader):
        adj = batch["adj"]
        active = batch["active"]
        year = batch["year"]
        B, T_total, N = active.shape
        T_pred = T_total - history_size
        for b in range(B):
            for tp in range(T_pred):
                t_global = history_size + tp
                act = active[b, t_global]
                if act.sum() < 2:
                    continue
                idx = act.nonzero(as_tuple=True)[0].numpy()
                abs_bin = min(int(year[b]) + t_global, T - 1)
                last_abs = min(int(year[b]) + history_size - 1, T - 1)

                Z_last_np = emb_bins[last_abs, idx]
                Z_prev_np = emb_bins[max(last_abs - 1, 0), idx]   # second-to-last history bin
                Z_or_np = emb_bins[abs_bin, idx]
                Z_pred_np = Z_last_np.copy()
                um = idx < n_users
                if um.any():                     # project users to target bin
                    uu = idx[um]
                    t_mid = t_lo + (abs_bin + 0.5) * span / T
                    dt = (t_mid - last_t_u[last_abs, uu]) / gap_u
                    with torch.no_grad():
                        Z_pred_np[um] = model.project(
                            torch.from_numpy(emb_bins[last_abs, uu]).float(),
                            torch.from_numpy(dt).float()).numpy()

                or_rows.append(Z_or_np)
                last_rows.append(Z_last_np)
                prev_rows.append(Z_prev_np)
                pred_rows.append(Z_pred_np)
                deg_rows.append(
                    adj[b, t_global].float().sum(dim=-1)[idx].numpy())
                dhist_rows.append(
                    adj[b, :history_size].float().sum(dim=-1)[:, idx].T.numpy())
                win_rows.append(np.full(len(idx), wid * 1000 + b))
                node_rows.append(idx.astype(np.int64))
                bin_rows.append(np.full(len(idx), int(year[b]) + t_global))
                hor_rows.append(np.full(len(idx), tp))

    Z_enc = np.concatenate(or_rows).astype(np.float64)
    Z_last = np.concatenate(last_rows).astype(np.float64)
    Z_prev = np.concatenate(prev_rows).astype(np.float64)
    Z_pred = np.concatenate(pred_rows).astype(np.float64)
    window = np.concatenate(win_rows)
    node_id = np.concatenate(node_rows)
    degree = np.concatenate(deg_rows).astype(np.float64)
    deg_hist = np.concatenate(dhist_rows).astype(np.float64)
    abs_bin = np.concatenate(bin_rows)
    horizon = np.concatenate(hor_rows)

    per_row_block, _, eff_source, per_row_prev = _per_row_labels(
        ds, node_id, abs_bin, "dataset", task, horizon=horizon)
    train, test = _train_test_split_by_window(window, seed=0)
    result = _binary_state_result(Z_enc, Z_last, degree, per_row_block,
                                  train, test, node_id, eff_source,
                                  torch.device("cpu"), Z_pred=Z_pred,
                                  per_row_prev=per_row_prev,
                                  deg_hist=deg_hist, horizon=horizon,
                                      Z_prev=Z_prev)
    out_path = Path(jodie_ckpt).parent / f"state_probe{_SPLIT_SUFFIX[split]}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result, out_path


def run_one(jepa_ckpt: Path, device: str = "cuda",
            batch_size: int = 4, num_workers: int = 2,
            label_source: str = "louvain",
            task: str = "multiclass",
            with_state_labels: bool = False,
            state_label_mode: str = "ever",
            split: str = "test") -> dict:
    """Block-probe one trained-cjepa directory. Returns the result dict and
    also writes it to <ckpt_dir>/block_probe.json (multiclass) or
    state_probe.json (binary).

    Args:
      task: 'multiclass' (Louvain communities) or 'binary' (state_label,
            ban prediction). Binary uses a trained binary probe which reports
            AUROC — necessary because state_label is severely imbalanced.
      with_state_labels: if True, overrides args.with_state_labels so the
            dataset exposes ds.node_labels even if the ckpt didn't enable it.
    """
    ckpt = torch.load(jepa_ckpt, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    if not saved_args:
        raise ValueError(f"{jepa_ckpt} has no 'args' — refusing to guess "
                         "the dataset configuration.")
    args = _make_args_namespace(saved_args)
    args.device = device
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.split = split
    # State labels are typically not enabled at training time — turn them on
    # at probe time so ds.node_labels(_dyn) is populated. state_label_mode
    # 'event'/'state' builds the dynamic per-(bin, node) matrix.
    if with_state_labels:
        args.with_state_labels = True
    args.jodie_state_label_mode = state_label_mode
    device = torch.device(args.device)

    # Build dataset (handles all the synthetic / jodie / reddit_body cases)
    ds = build_dynamic_dataset(args, split=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # Encoder — prefer the JEPA-saved weights (post fine-tuning) over the
    # raw pretrain ckpt, matching what we do in analyze_jepa.
    enc_kind = saved_args.get("encoder_kind", "tgcn")
    enc_hidden = saved_args.get("encoder_hidden", 128)
    node_dim = saved_args.get("node_dim", 128)
    enc_bidir = saved_args.get("bidirectional", True)
    encoder = build_encoder(
        enc_kind, in_dim=ds.feature_dim, hidden_dim=enc_hidden,
        out_dim=node_dim, freeze=True, bidirectional=enc_bidir,
    ).to(device)
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    else:
        raise ValueError(f"{jepa_ckpt} has no 'encoder' field.")

    # Dispatch on the saved mask_scheme so the reconstructed predictor matches
    # the trained one — node_mask/future_forecast rebuild the transformer,
    # mlp_forecast/gru_forecast the graph-free PlainForecaster (Tier-2 control).
    predictor = build_predictor(
        saved_args.get("mask_scheme", "node_mask"),
        num_nodes=ds.num_nodes, node_dim=node_dim,
        history_frames=saved_args.get("history_size", 3),
        pred_frames=saved_args.get("num_preds", 1),
        num_masked_nodes=saved_args.get("num_masked_nodes", 30),
        depth=saved_args.get("depth", 4),
        heads=saved_args.get("heads", 4),
        dim_head=node_dim // saved_args.get("heads", 4),
        mlp_dim=saved_args.get("mlp_dim", 512),
        dropout=saved_args.get("dropout", 0.1),
        use_attention_mask=not saved_args.get("no_attention_mask", True),
        # residual_head / latent_norm have no parameters, so load_state_dict
        # can't catch a mismatch, and the edge head must exist for its weights
        # to load — restore all of them from the training args.
        **predictor_flags_from_args(saved_args),
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])

    # Latents (+ per-row degree for the baseline probe, + absolute bin for
    # dynamic labels).
    (Z_enc, Z_last, Z_pred, window, node_id, degree, deg_hist,
     abs_bin, horizon, Z_prev) = collect_test_latents(
        encoder, predictor, loader,
        history_size=saved_args.get("history_size", 3),
        device=device,
    )
    train, test = _train_test_split_by_window(window, seed=0)
    if task == "ranking":
        # Per-step label distribution (tgbn_*), NDCG@10 per row.
        Y, valid, Yprev, prev_valid, Yavg, avg_valid = _per_row_label_vectors(
            ds, node_id, abs_bin, horizon, history_size=saved_args.get("history_size", 3))
        result = _ranking_state_result(Z_enc, Z_last, Y, valid, Yprev, prev_valid,
                                       train, test, node_id, "dataset",
                                       Z_pred=Z_pred, deg_hist=deg_hist,
                                       horizon=horizon, Yavg=Yavg, avg_valid=avg_valid)
        out_path = Path(jepa_ckpt).parent / f"state_probe{_SPLIT_SUFFIX[split]}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        return result, out_path
    # Labels: dynamic per-(bin, node) when available, else static per-node.
    per_row_block, block_labels, eff_source, per_row_prev = _per_row_labels(
        ds, node_id, abs_bin, label_source, task, horizon=horizon)

    if task == "binary":
        # State-label probe: per-user binary (ban / no ban). Use AUROC +
        # AUPRC since accuracy is uninformative on severely imbalanced
        # labels. The degree-only baseline subtracts out the "banned users
        # are more active" confound; encoder_last is the fair copy-last
        # baseline; the jepa column is the actual forecast.
        result = _binary_state_result(Z_enc, Z_last, degree, per_row_block,
                                      train, test, node_id, eff_source,
                                      device, Z_pred=Z_pred,
                                      per_row_prev=per_row_prev,
                                      deg_hist=deg_hist, horizon=horizon,
                                      Z_prev=Z_prev)
        out_path = Path(jepa_ckpt).parent / _state_probe_name(eff_source, split)
    elif block_labels is None:
        # Dynamic multiclass state (synthetic_dyn community labels): every row
        # is labelled at its own step, so this is a forecast task with copy-last
        # as the floor — same file/columns as the binary state probe.
        result = _multiclass_state_result(Z_enc, Z_last, per_row_block,
                                          per_row_prev, train, test, node_id,
                                          eff_source, Z_pred=Z_pred,
                                          deg_hist=deg_hist, horizon=horizon,
                                          latents=getattr(ds, "latents", None),
                                          abs_bin=abs_bin, ds=ds, Z_prev=Z_prev)
        out_path = Path(jepa_ckpt).parent / f"state_probe{_SPLIT_SUFFIX[split]}.json"
    elif eff_source == "dataset":
        # Static per-node classes (cldg_dblp / cldg_patent): node-level split,
        # same file and columns as the state probes, so the forecast column
        # is compared with copy-last exactly as on the dynamic tasks.
        result = _multiclass_static_result(Z_enc, Z_last, per_row_block, node_id,
                                           eff_source, Z_pred=Z_pred,
                                           deg_hist=deg_hist, horizon=horizon,
                                           Z_prev=Z_prev)
        out_path = Path(jepa_ckpt).parent / f"state_probe{_SPLIT_SUFFIX[split]}.json"
    else:
        n_classes = int(block_labels.max() + 1) if (block_labels >= 0).any() else 0
        enc_probe = train_node_probe(Z_enc, per_row_block, train, test,
                                     "multiclass", device=torch.device("cpu"))
        jep_probe = train_node_probe(Z_pred, per_row_block, train, test,
                                     "multiclass", device=torch.device("cpu"))
        for pr in (enc_probe, jep_probe):
            pr.pop("_test_rows", None); pr.pop("_pred_test", None)
        result = {
            "task": "multiclass",
            "label_source": label_source,
            "n_classes_total": n_classes,
            "n_classes_seen": int(enc_probe["n_classes"]),
            "encoder": enc_probe,
            "jepa": jep_probe,
            "n_rows": int(len(node_id)),
        }
        out_path = Path(jepa_ckpt).parent / f"block_probe{_SPLIT_SUFFIX[split]}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result, out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jepa_ckpt", default=None,
                   help="A single best.ckpt to probe.")
    p.add_argument("--encoder_ckpt", default=None,
                   help="A bare encoder checkpoint (save_encoder_ckpt schema — "
                        "e.g. a baseline CLDG/DDGCL/IDOL encoder.ckpt). Probes "
                        "ONLY the encoder latents (no JEPA predictor) and writes "
                        "state_probe.json next to the ckpt, so baselines are "
                        "directly comparable to DELOS's encoder column.")
    p.add_argument("--encoder_pattern", default=None,
                   help="Glob of directories each holding an encoder.ckpt; "
                        "bare-encoder-probes each one.")
    p.add_argument("--jodie_ckpt", default=None,
                   help="A JODIE baseline checkpoint (src/baselines/jodie/"
                        "train_jodie.py output). Probes its per-bin dynamic "
                        "embeddings: last-history-bin embedding = the "
                        "encoder_last column, its temporal PROJECTION to the "
                        "target bin = the jepa column, end-of-target-bin "
                        "embedding = the oracle. Writes state_probe.json "
                        "next to the ckpt. --task binary only.")
    p.add_argument("--num_preds", type=int, default=None,
                   help="JODIE probe only: override the forecast horizon count, "
                        "scoring steps 1..num_preds. Re-probes an existing ckpt "
                        "at more horizons WITHOUT retraining (all per-bin "
                        "embeddings are stored). Per-horizon subsets h0..h{n-1} "
                        "are written into state_probe.json.")
    p.add_argument("--pattern", default=None,
                   help="Glob of *_cjepa directories; probes each one.")
    p.add_argument("--probe_split", choices=["test", "val"], default="test",
                   help="Which split the probe rows come from. 'test' (default) "
                        "is the reported number. 'val' writes state_probe_val."
                        "json instead — use it to RANK hyperparameter "
                        "configurations, so selection never touches test.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--label_source", choices=["louvain", "dataset"],
                   default="louvain",
                   help="Where multiclass labels come from: Louvain "
                        "communities (default, diagnostic) or the dataset's "
                        "own per-node classes (cldg_dblp / cldg_patent: static "
                        "node classification with a NODE-level split, written "
                        "to state_probe.json).")
    p.add_argument("--task", choices=["multiclass", "binary", "ranking"],
                   default="multiclass",
                   help="'ranking' (tgbn_trade): each row's label is a share "
                        "vector over the nations, a linear head to C scores is "
                        "fitted and scored with NDCG@10 (TGB's metric); "
                        "label_persistence = last observed vector. "
                        "'multiclass' uses Louvain communities by default and "
                        "reports accuracy + macro-F1; on a dataset exposing "
                        "DYNAMIC multiclass labels (synthetic_dyn "
                        "--synth_state_target community) it becomes the "
                        "node-STATE forecast probe and writes state_probe.json "
                        "with copy-last / switch-row columns. 'binary' uses the "
                        "dataset's per-node binary labels (JODIE state_label) "
                        "and reports AUROC + acc — required for the JODIE "
                        "ban-prediction task where class imbalance is severe.")
    p.add_argument("--with_state_labels", action="store_true",
                   help="Force the dataset to expose ds.node_labels = "
                        "per-user state_label, overriding any default. "
                        "Implied (and required) for --task binary.")
    p.add_argument("--state_label_mode", choices=["ever", "event", "state"],
                   default="ever",
                   help="JODIE state-label semantics for --task binary. 'ever' "
                        "= static per-user 'ever banned' (label constant across "
                        "time). 'event'/'state' = DYNAMIC per-(bin, node) label "
                        "so each row is scored at its own snapshot — 'event' "
                        "marks the exact ban snapshot (sparse), 'state' marks "
                        "every snapshot from the ban onward (denser). The "
                        "dynamic modes are the task the JEPA forecast is built "
                        "for; expect the jepa column to close/beat the encoder.")
    args = p.parse_args()
    # Sensible defaults: binary task implies state-label loading and dataset
    # source; the user shouldn't have to specify all three.
    if args.task == "binary":
        args.with_state_labels = True
        if args.label_source != "dataset":
            args.label_source = "dataset"

    # Two kinds of target: full DELOS ckpts (encoder + predictor) go
    # through run_one; bare encoder ckpts (baselines) through run_one_encoder.
    full_targets, enc_targets = [], []
    if args.jepa_ckpt:
        full_targets.append(Path(args.jepa_ckpt))
    if args.pattern:
        for d in sorted(_glob.glob(args.pattern)):
            ckpt = Path(d) / "best.ckpt"
            if ckpt.exists():
                full_targets.append(ckpt)
            else:
                print(f"[skip] {d} has no best.ckpt", file=sys.stderr)
    if args.encoder_ckpt:
        enc_targets.append(Path(args.encoder_ckpt))
    if args.encoder_pattern:
        for d in sorted(_glob.glob(args.encoder_pattern)):
            ckpt = Path(d) / "encoder.ckpt"
            if ckpt.exists():
                enc_targets.append(ckpt)
            else:
                print(f"[skip] {d} has no encoder.ckpt", file=sys.stderr)

    jodie_targets = [Path(args.jodie_ckpt)] if args.jodie_ckpt else []

    if not full_targets and not enc_targets and not jodie_targets:
        print("Pass --jepa_ckpt / --pattern (full ckpt), --encoder_ckpt / "
              "--encoder_pattern (bare encoder), or --jodie_ckpt.",
              file=sys.stderr)
        sys.exit(1)

    n = len(full_targets) + len(enc_targets) + len(jodie_targets)
    print(f"[block_probe] {n} checkpoint(s) to probe  task={args.task} "
          f"({len(full_targets)} full, {len(enc_targets)} bare-encoder, "
          f"{len(jodie_targets)} jodie)")

    def _print_state(tag, result, out, has_jepa):
        def _col(name):
            r = result.get(name)
            if r is None:
                return ""
            s = f"{name} acc={r['acc_test']:.4f}"
            sw = r.get("subset_switch")
            if sw and sw.get("n", 0) > 0:
                s += f"/switch={sw['acc_test']:.3f}"
            st = r.get("subset_stay")
            if st and st.get("n", 0) > 0:
                s += f"/stay={st['acc_test']:.3f}"
            if "change_f1" in r:
                s += f" chgF1={r['change_f1']:.3f}"
            if "delta_r2_test" in r:
                s += f" dR2={r['delta_r2_test']:.3f}"
            return s + "  "
        line = (_col("label_persistence") + _col("encoder_oracle")
                + _col("encoder_last") + (_col("jepa") if has_jepa else "")
                + _col("degree_history_baseline"))
        print(f"  [{tag}] {line}(switch rows={result['n_switch_test']}, "
              f"{100*result['switch_rate_test']:.2f}%)  -> {out}")

    def _print_ranking(tag, result, out, has_jepa):
        def _col(name):
            r = result.get(name)
            return "" if r is None else f"{name} ndcg@10={r['ndcg10_test']:.4f}  "
        line = (_col("label_persistence") + _col("label_moving_average")
                + _col("encoder_oracle") + _col("encoder_last")
                + (_col("jepa") if has_jepa else "") + _col("degree_history_baseline"))
        print(f"  [{tag}] {line}(rows={result['n_valid_rows']})  -> {out}")

    def _print_static(tag, result, out, has_jepa):
        def _col(name):
            r = result.get(name)
            if r is None:
                return ""
            return f"{name} acc={r['acc_test']:.4f}/f1={r['macro_f1_test']:.4f}  "
        line = (_col("encoder_oracle") + _col("encoder_last")
                + (_col("jepa") if has_jepa else "")
                + _col("degree_history_baseline"))
        print(f"  [{tag}] {line}(classes={result['n_classes_seen']}, "
              f"test nodes={result['n_nodes_test']}, rows={result['n_valid_rows']})"
              f"  -> {out}")

    def _print_binary(tag, result, out, has_jepa):
        def _col(name):
            r = result.get(name)
            if r is None:
                return ""
            s = f"{name} auroc={r['auroc_test']:.4f}/ap={r['auprc_test']:.4f}"
            on = r.get("subset_onset")
            if on and on["n"] > 0:
                s += f" (onset ap={on['auprc_test']:.4f}, n_pos={on['n_pos']})"
            return s + "  "
        line = (_col("encoder_oracle") + _col("encoder_last")
                + (_col("jepa") if has_jepa else "")
                + _col("degree_baseline") + _col("degree_history_baseline"))
        print(f"  [{tag}] {line}(pos_test={result['n_pos_test']} / "
              f"valid={result['n_valid_rows']})  -> {out}")

    for ckpt in full_targets:
        tag = ckpt.parent.name
        try:
            result, out = run_one(
                ckpt, device=args.device,
                batch_size=args.batch_size, num_workers=args.num_workers,
                label_source=args.label_source, task=args.task,
                with_state_labels=args.with_state_labels,
                state_label_mode=args.state_label_mode,
                split=args.probe_split,
            )
            if args.task == "binary":
                _print_binary(tag, result, out, has_jepa=True)
            elif result.get("task") == "multiclass_state":
                _print_state(tag, result, out, has_jepa=True)
            elif result.get("task") == "multiclass_static":
                _print_static(tag, result, out, has_jepa=True)
            elif result.get("task") == "ranking":
                _print_ranking(tag, result, out, has_jepa=True)
            else:
                print(f"  [{tag}] enc acc={result['encoder']['acc_test']:.4f}  "
                      f"jepa acc={result['jepa']['acc_test']:.4f}  "
                      f"(n_classes={result['n_classes_seen']}, "
                      f"rows={result['n_rows']})  -> {out}")
        except Exception as e:
            print(f"  [{tag}] FAILED: {type(e).__name__}: {e}", flush=True)

    for ckpt in jodie_targets:
        tag = ckpt.parent.name
        try:
            result, out = run_one_jodie(
                ckpt, device=args.device,
                batch_size=args.batch_size, num_workers=args.num_workers,
                task=args.task, state_label_mode=args.state_label_mode,
                num_preds_override=args.num_preds,
                split=args.probe_split,
            )
            _print_binary(tag, result, out, has_jepa=True)
        except Exception as e:
            print(f"  [{tag}] FAILED: {type(e).__name__}: {e}", flush=True)

    for ckpt in enc_targets:
        tag = ckpt.parent.name
        try:
            result, out = run_one_encoder(
                ckpt, device=args.device,
                batch_size=args.batch_size, num_workers=args.num_workers,
                label_source=args.label_source, task=args.task,
                with_state_labels=True,
                state_label_mode=args.state_label_mode,
                split=args.probe_split,
            )
            if args.task == "binary":
                _print_binary(tag, result, out, has_jepa=False)
            elif result.get("task") == "multiclass_state":
                _print_state(tag, result, out, has_jepa=False)
            elif result.get("task") == "multiclass_static":
                _print_static(tag, result, out, has_jepa=False)
            elif result.get("task") == "ranking":
                _print_ranking(tag, result, out, has_jepa=False)
            else:
                print(f"  [{tag}] enc acc={result['encoder']['acc_test']:.4f}  "
                      f"(n_classes={result['n_classes_seen']}, "
                      f"rows={result['n_rows']})  -> {out}")
        except Exception as e:
            print(f"  [{tag}] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
