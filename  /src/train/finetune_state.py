"""Fine-tune encoder + predictor + head on the dynamic node-STATE label.

The DDGCL-style protocol (Tian et al., CIKM'21): self-supervised weights are
only an INITIALISATION; the whole network is then trained end-to-end on labels.
This is the experiment the frozen probe cannot run -- "does the pretraining
transfer into a supervised model?" -- and its natural form is a label-efficiency
curve: fine-tune from each initialisation with 1 / 5 / 10 / 25 / 100 % of the
train-period positive users and watch where the curves separate.

Every row uses the SAME architecture, split, schedule and head; only the
starting weights differ (``--init``):

  random    encoder and predictor from scratch      (the supervised floor)
  encoder   encoder from --encoder_ckpt, predictor from scratch
            (edge-mask pretraining alone, or a CLDG / DDGCL / IDOL encoder)
  jepa      encoder AND predictor from --jepa_ckpt   (the full DyG-JEPA init)

``--jepa_ckpt`` is read for the predictor ARCHITECTURE under every init, so the
rows share depth / heads / masking / residual head exactly; its weights are
loaded only for ``--init jepa``.

Two fine-tuning regimes, mirroring DDGCL's two schemes:

  --aux_jepa_weight 0    pure fine-tuning: BCE only (DDGCL ``fune_train``).
                         The supervised loss may overwrite the forecast.
  --aux_jepa_weight w>0  multi-task: BCE + w * JEPA latent loss (DDGCL
                         ``mtl_train``). Keeps the predictor honest as a
                         forecaster while the head is trained.

Rows are (target step, active user) exactly as in the frozen state probe
(``scripts/dblp/run_block_probe.py``): the head reads the predictor's forecast
latent at the target step, and a row counts iff the node is active there and
carries a valid label. Split is the dataset's CHRONOLOGICAL one: fit on train
windows, select on val AUROC, report test.

Run:
    python src/train/finetune_state.py --data_source jodie_mooc \
        --with_edge_features \
        --jepa_ckpt outputs/cls_mooc_full/jepa_frozen_s42/best.ckpt \
        --init jepa --label_fraction 0.1 --aux_jepa_weight 1.0 \
        --out_dir outputs/cls_mooc_full/finetune/jepa_aux1_f0.1_s42
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cjepa_graph_predictor import build_predictor, compute_graph_jepa_loss
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.analyze_jepa import _auprc, _auroc
from src.train.engine import (
    cosine_warmup_lr,
    make_loader,
    set_deterministic,
    setup_output_dir,
)
from src.train.jepa import EMAEncoder
from src.world_models.graph_jepa_encoder import build_encoder
from src.world_models.probes import graph_inputs, hist_adj


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------
class StateHead(nn.Module):
    """Per-node binary readout on the forecast latent: MLP (or linear when
    ``hidden_dim == 0``). Kept separate from the probes so the fine-tuning
    head's capacity is an explicit, reported knob."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:      # (..., D) -> (...)
        return self.net(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def dynamic_label_matrix(ds) -> tuple:
    """(Y (T, N) int64 with -1 = no label, mode string) for a binary state task.

    JODIE exposes ``node_labels_dyn`` under --jodie_state_label_mode
    event/state. synthetic_dyn built for the multiclass community target still
    carries the binary switch matrix on ``switch_labels``; use that so one
    generator config serves both the probe and this stage (same rule as
    run_block_probe._per_row_labels).
    """
    dyn = getattr(ds, "node_labels_dyn", None)
    mode = str(getattr(ds, "state_label_mode", ""))
    sw = getattr(ds, "switch_labels", None)
    if sw is not None and mode.startswith("community"):
        dyn, mode = sw, f"switch:derived_from_{mode}"
    if dyn is None:
        raise SystemExit(
            "finetune_state needs a dynamic per-(step, node) binary label. For "
            "JODIE pass --with_state_labels --jodie_state_label_mode state|event; "
            "for synthetic_dyn any --synth_state_target works.")
    Y = np.asarray(dyn).astype(np.int64)
    if Y.ndim != 2:
        raise SystemExit(f"expected a (T, N) label matrix, got shape {Y.shape}")
    return Y, mode


def target_labels(Y_t: torch.Tensor, year: torch.Tensor, H: int, P: int):
    """Labels and previous-step labels at the P target steps of each window.

    Returns (lab (B, P, N), prev (B, P, N)); prev is -1 where there is no
    previous step. ``year`` is the window's absolute start index.
    """
    T = Y_t.shape[0]
    idx = year.to(Y_t.device).view(-1, 1) + H + torch.arange(P, device=Y_t.device).view(1, -1)
    idx = idx.clamp(max=T - 1)
    lab = Y_t[idx]                                            # (B, P, N)
    pidx = idx - 1
    prev = torch.where((pidx >= 0).unsqueeze(-1), Y_t[pidx.clamp(min=0)],
                       torch.full_like(lab, -1))
    return lab, prev


@torch.no_grad()
def scan_train_rows(loader, Y_t, H, P, device):
    """One pass over the train windows: which users ever carry a row, and
    which of those are positive at some train target step. Also the row
    counts the class weight needs. Rows = active at the target step with a
    valid label, the probe's definition."""
    N = Y_t.shape[1]
    has_row = torch.zeros(N, dtype=torch.bool, device=device)
    is_pos = torch.zeros(N, dtype=torch.bool, device=device)
    for batch in loader:
        active = batch["active"].to(device)
        lab, _ = target_labels(Y_t, batch["year"], H, P)
        row = active[:, H:H + P] & (lab >= 0)
        has_row |= row.any(dim=(0, 1))
        is_pos |= (row & (lab == 1)).any(dim=(0, 1))
    return has_row.cpu().numpy(), is_pos.cpu().numpy()


def choose_label_users(has_row, is_pos, fraction, seed):
    """Stratified subsample of the labelled users: ``fraction`` of the
    positive users and ``fraction`` of the negative ones. Returns a (N,) bool
    mask; rows of unselected users are dropped from the TRAINING loss only."""
    pos = np.flatnonzero(has_row & is_pos)
    neg = np.flatnonzero(has_row & ~is_pos)
    if fraction >= 1.0:
        sel_pos, sel_neg = pos, neg
    else:
        rng = np.random.default_rng(seed)
        n_pos = max(1, int(round(fraction * len(pos)))) if len(pos) else 0
        n_neg = max(1, int(round(fraction * len(neg)))) if len(neg) else 0
        sel_pos = rng.choice(pos, size=n_pos, replace=False) if n_pos else pos[:0]
        sel_neg = rng.choice(neg, size=n_neg, replace=False) if n_neg else neg[:0]
    mask = np.zeros(len(has_row), dtype=bool)
    mask[sel_pos] = True
    mask[sel_neg] = True
    return mask, int(len(sel_pos)), int(len(sel_neg)), int(len(pos)), int(len(neg))


@torch.no_grad()
def count_train_rows(loader, Y_t, user_mask_t, H, P, device):
    n_pos = n_neg = 0
    for batch in loader:
        active = batch["active"].to(device)
        lab, _ = target_labels(Y_t, batch["year"], H, P)
        row = active[:, H:H + P] & (lab >= 0) & user_mask_t.view(1, 1, -1)
        n_pos += int((row & (lab == 1)).sum())
        n_neg += int((row & (lab == 0)).sum())
    return n_pos, n_neg


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_models(args, feature_dim, num_nodes, device):
    """Encoder / predictor / head under ``--init``. Returns
    (encoder, predictor, head, info) with info recording where each part's
    weights came from."""
    jepa_ck = None
    saved = {}
    if args.jepa_ckpt:
        if not os.path.exists(args.jepa_ckpt):
            raise FileNotFoundError(f"--jepa_ckpt {args.jepa_ckpt!r} does not exist")
        jepa_ck = torch.load(args.jepa_ckpt, map_location=device, weights_only=False)
        saved = jepa_ck.get("args", {})
    enc_ck = None
    if args.init == "encoder":
        if not args.encoder_ckpt:
            raise SystemExit("--init encoder needs --encoder_ckpt")
        if not os.path.exists(args.encoder_ckpt):
            raise FileNotFoundError(f"--encoder_ckpt {args.encoder_ckpt!r} does not exist")
        enc_ck = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    if args.init == "jepa" and jepa_ck is None:
        raise SystemExit("--init jepa needs --jepa_ckpt")

    # Encoder geometry: from the checkpoint that supplies the weights, else
    # from the JEPA ckpt (architecture parity), else the CLI.
    meta = enc_ck if enc_ck is not None else saved
    enc_kind = meta.get("encoder_kind", args.encoder_kind)
    enc_hidden = meta.get("encoder_hidden", args.encoder_hidden)
    node_dim = meta.get("node_dim", args.node_dim)
    enc_bidir = meta.get("bidirectional", args.bidirectional)
    if saved and saved.get("node_dim", node_dim) != node_dim:
        raise SystemExit(f"encoder node_dim {node_dim} != predictor node_dim "
                         f"{saved['node_dim']} from --jepa_ckpt")
    encoder = build_encoder(enc_kind, in_dim=feature_dim, hidden_dim=enc_hidden,
                            out_dim=node_dim, freeze=False,
                            bidirectional=enc_bidir).to(device)
    args.node_dim = node_dim

    # Predictor architecture: from the JEPA ckpt when given, else the CLI.
    g = (lambda k, d: saved.get(k, d)) if saved else (lambda k, d: d)
    heads = g("heads", args.heads)
    predictor = build_predictor(
        g("mask_scheme", args.mask_scheme),
        num_nodes=num_nodes, node_dim=node_dim,
        history_frames=args.history_size, pred_frames=args.num_preds,
        num_masked_nodes=g("num_masked_nodes", args.num_masked_nodes),
        seed=args.seed, depth=g("depth", args.depth), heads=heads,
        dim_head=node_dim // heads, mlp_dim=g("mlp_dim", args.mlp_dim),
        dropout=g("dropout", args.dropout),
        allow_temporal_neighbors=g("allow_temporal_neighbors",
                                   args.allow_temporal_neighbors),
        use_attention_mask=not g("no_attention_mask", args.no_attention_mask),
        use_edge_head=False,
        residual_head=g("residual_head", args.residual_head),
        latent_norm=g("target_norm", False),
    ).to(device)
    if saved and saved.get("history_size", args.history_size) != args.history_size:
        raise SystemExit("--history_size differs from the --jepa_ckpt's; the "
                         "predictor's token grid would not match")

    info = {"encoder_from": "random", "predictor_from": "random",
            "encoder_kind": enc_kind, "node_dim": node_dim}
    if args.init == "jepa":
        encoder.load_state_dict(jepa_ck["encoder"])
        predictor.load_state_dict(jepa_ck["predictor"])
        info["encoder_from"] = info["predictor_from"] = args.jepa_ckpt
    elif args.init == "encoder":
        encoder.load_state_dict(enc_ck["encoder"])
        info["encoder_from"] = args.encoder_ckpt
    head = StateHead(node_dim, hidden_dim=args.head_hidden,
                     dropout=args.head_dropout).to(device)
    print(f"[init={args.init}] encoder<-{info['encoder_from']}  "
          f"predictor<-{info['predictor_from']}  ({enc_kind}, D={node_dim})",
          flush=True)
    return encoder, predictor, head, info


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def forecast_latents(encoder, predictor, batch, loader, H, P, device,
                     train_encoder):
    """(z_hist, z_pred (B,P,N,D), adj, adj_norm, active) for one batch."""
    x_hist = batch["x_hist"].to(device, non_blocking=True)
    x_fut = batch["x_fut"].to(device, non_blocking=True)
    adj, adj_norm, active = graph_inputs(batch, loader.dataset, device, H, P)
    if train_encoder:
        z_hist = encoder(x_hist, hist_adj(adj_norm, H))
    else:
        with torch.no_grad():
            z_hist = encoder(x_hist, hist_adj(adj_norm, H))
    z_pred = predictor.inference(z_hist, adj, active)          # (B, P, N, D)
    return x_hist, x_fut, z_hist, z_pred, adj, adj_norm, active


def train_epoch(encoder, predictor, head, loader, opt, sched, Y_t, user_mask_t,
                pos_weight, args, device, ema=None, max_steps=None):
    H, P = args.history_size, args.num_preds
    train_enc = not args.freeze_encoder
    encoder.train(train_enc)
    predictor.train(not args.freeze_predictor)
    head.train()
    tot = {"loss": 0.0, "loss_cls": 0.0, "loss_jepa": 0.0, "n": 0}
    t0 = time.time()
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x_hist, x_fut, z_hist, z_pred, adj, adj_norm, active = forecast_latents(
            encoder, predictor, batch, loader, H, P, device, train_enc)
        lab, _ = target_labels(Y_t, batch["year"], H, P)
        row = active[:, H:H + P] & (lab >= 0) & user_mask_t.view(1, 1, -1)
        if row.sum() == 0:
            continue
        logits = head(z_pred)                                  # (B, P, N)
        loss_cls = F.binary_cross_entropy_with_logits(
            logits[row], lab[row].float(), pos_weight=pos_weight)
        loss = loss_cls
        loss_jepa = torch.zeros((), device=device)
        if args.aux_jepa_weight > 0:
            x_full = torch.cat([x_hist, x_fut], dim=1)
            with torch.no_grad():
                z_target = (ema(x_full, adj_norm) if ema is not None
                            else encoder(x_full, adj_norm)).detach()
            pred, masked_idx, _ = predictor(z_hist, adj, active)
            loss_jepa = compute_graph_jepa_loss(
                pred, z_target, active, masked_idx, H,
                change_weight=args.change_weight,
                target_norm=getattr(predictor, "latent_norm", False))["loss"]
            loss = loss + args.aux_jepa_weight * loss_jepa
        if not torch.isfinite(loss):
            print(f"  [step {step+1}] non-finite loss, skipping", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Clip each part (head / predictor / encoder) on its OWN norm. One
        # global norm let the encoder's JEPA gradient set the clip factor for
        # everything, which (a) scaled the head's BCE gradient to ~0 and (b)
        # made the aux weight irrelevant: after clipping, w * g_jepa has unit
        # norm whatever w is. Pre-clip norms are logged per group so the
        # weight sweep can be read off the balance between the terms.
        max_norm = args.clip_norm if args.clip_norm > 0 else float("inf")
        for grp in opt.param_groups:
            gn = torch.nn.utils.clip_grad_norm_(grp["params"], max_norm)
            tot[f"gn_{grp['name']}"] = tot.get(f"gn_{grp['name']}", 0.0) + float(gn) * x_hist.shape[0]
        opt.step()
        if sched is not None:
            sched.step()
        if ema is not None and train_enc:
            ema.update(encoder)
        bs = x_hist.shape[0]
        tot["loss"] += loss.item() * bs
        tot["loss_cls"] += loss_cls.item() * bs
        tot["loss_jepa"] += loss_jepa.item() * bs
        tot["n"] += bs
    n = max(tot["n"], 1)
    out = {k: v / n for k, v in tot.items() if k != "n"}
    out["time_s"] = time.time() - t0
    return out


@torch.no_grad()
def evaluate(encoder, predictor, head, loader, Y_t, args, device):
    """AUROC / AUPRC over every valid row of the split (no user subsampling:
    the label fraction only restricts what the loss sees). Also the onset
    subset (previous-step label 0: where label persistence carries nothing)
    and, for multi-step horizons, one entry per forecast step."""
    H, P = args.history_size, args.num_preds
    encoder.eval(); predictor.eval(); head.eval()
    scores, labels, prevs, hors = [], [], [], []
    for batch in loader:
        _, _, _, z_pred, _, _, active = forecast_latents(
            encoder, predictor, batch, loader, H, P, device, False)
        lab, prev = target_labels(Y_t, batch["year"], H, P)
        row = active[:, H:H + P] & (lab >= 0)
        if row.sum() == 0:
            continue
        logits = head(z_pred)
        hor = torch.arange(P, device=device).view(1, P, 1).expand_as(lab)
        scores.append(logits[row].float().cpu())
        labels.append(lab[row].cpu())
        prevs.append(prev[row].cpu())
        hors.append(hor[row].cpu())
    if not scores:
        return {"auroc": float("nan"), "auprc": float("nan"), "n": 0, "n_pos": 0}
    s = torch.cat(scores).numpy(); y = torch.cat(labels).numpy()
    pv = torch.cat(prevs).numpy(); hr = torch.cat(hors).numpy()

    def _m(mask):
        ss, yy = s[mask], y[mask]
        return {"auroc": _auroc(ss, yy), "auprc": _auprc(ss, yy),
                "n": int(mask.sum()), "n_pos": int((yy == 1).sum())}
    out = _m(np.ones_like(y, dtype=bool))
    out["subset_onset"] = _m(pv == 0)
    if P > 1:
        for tp in range(P):
            out[f"h{tp}"] = _m(hr == tp)
    return out


def _fmt(m):
    on = m.get("subset_onset")
    s = f"auroc={m['auroc']:.4f} auprc={m['auprc']:.4f} (n={m['n']}, pos={m['n_pos']})"
    if on:
        s += f"  onset: auroc={on['auroc']:.4f} auprc={on['auprc']:.4f} (pos={on['n_pos']})"
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_dblp_synthetic_flags(p)
    add_dataset_cli_flags(p)
    cli.add_window_flags(p)
    cli.add_encoder_flags(p, default_kind="gcn", node_dim=128, encoder_hidden=128)
    cli.add_optim_flags(p, lr=1e-4, epochs=100, batch_size=4)
    cli.add_run_flags(p, out_dir="outputs/finetune_state")
    cli.add_early_stop_flags(p, patience=15, min_delta=0.0)
    # initialisation
    p.add_argument("--init", choices=["random", "encoder", "jepa"], required=True)
    p.add_argument("--jepa_ckpt", default=None,
                   help="DyG-JEPA best.ckpt. Supplies the predictor ARCHITECTURE "
                        "under every --init; its weights only under --init jepa.")
    p.add_argument("--encoder_ckpt", default=None,
                   help="Encoder checkpoint for --init encoder (pretrain_encoder "
                        "or a CLDG/DDGCL/IDOL encoder.ckpt).")
    # predictor architecture (used only when no --jepa_ckpt is given)
    p.add_argument("--mask_scheme", choices=["node_mask", "future_forecast"],
                   default="node_mask")
    p.add_argument("--num_masked_nodes", type=int, default=30)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_attention_mask", action="store_true")
    p.add_argument("--allow_temporal_neighbors", action="store_true")
    p.add_argument("--residual_head", action="store_true")
    # labels
    p.add_argument("--label_fraction", type=float, default=1.0,
                   help="Fraction of the train-period labelled users (stratified "
                        "by positive / negative) whose rows enter the loss.")
    p.add_argument("--label_seed", type=int, default=0,
                   help="Seed for the user subsample (separate from --seed so "
                        "the same users are drawn across inits).")
    p.add_argument("--state_label_mode", choices=["state", "event"], default="state",
                   help="JODIE label semantics (forwarded to the loader).")
    p.add_argument("--pos_weight", default="auto",
                   help="BCE positive-class weight: 'auto' = n_neg/n_pos over the "
                        "training rows, a float, or 0 for none.")
    # what trains
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--freeze_predictor", action="store_true")
    p.add_argument("--encoder_lr", type=float, default=None,
                   help="Encoder LR (default: --lr).")
    p.add_argument("--head_lr", type=float, default=None,
                   help="Head LR (default: --lr). The frozen probe fits its head "
                        "at 1e-2; a from-scratch head at the backbone's 1e-4 "
                        "barely moves off chance.")
    p.add_argument("--clip_norm", type=float, default=1.0,
                   help="Per-group (head / predictor / encoder) gradient-norm "
                        "clip. 0 = no clipping (norms are still logged).")
    p.add_argument("--head_hidden", type=int, default=64)
    p.add_argument("--head_dropout", type=float, default=0.1)
    # multi-task regime
    p.add_argument("--aux_jepa_weight", type=float, default=0.0,
                   help="0 = pure fine-tuning (BCE only). >0 adds w * JEPA latent "
                        "loss (DDGCL's multi-task scheme).")
    p.add_argument("--aux_target", choices=["ema", "online"], default="ema",
                   help="Target encoder for the auxiliary JEPA loss. 'ema' "
                        "(I-JEPA) unless the encoder is frozen, where the "
                        "online copy is already fixed.")
    p.add_argument("--ema_momentum", type=float, default=0.996)
    p.add_argument("--change_weight", type=float, default=2.0,
                   help="Transition up-weighting for the aux JEPA loss (the "
                        "state recipe's value).")
    p.add_argument("--warmup_epochs", type=float, default=1.0)
    p.add_argument("--lr_min_ratio", type=float, default=0.05)
    args = p.parse_args()

    set_deterministic(args.seed)
    out_dir = setup_output_dir(args.out_dir, args)
    device = torch.device(args.device)
    # The loader must expose the dynamic label matrix.
    args.with_state_labels = True
    args.jodie_state_label_mode = args.state_label_mode
    print(f"[setup] finetune_state init={args.init} frac={args.label_fraction} "
          f"aux={args.aux_jepa_weight} lr={args.lr} head_lr={args.head_lr} "
          f"enc_lr={args.encoder_lr} clip={args.clip_norm} data={args.data_source} "
          f"device={device}", flush=True)

    train_ds = build_dynamic_dataset(args, "train")
    val_ds = build_dynamic_dataset(args, "val")
    test_ds = build_dynamic_dataset(args, "test")
    H, P = args.history_size, args.num_preds
    Y, label_mode = dynamic_label_matrix(train_ds)
    Y_t = torch.from_numpy(Y).to(device)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"N={train_ds.num_nodes} F={train_ds.feature_dim} labels={label_mode} "
          f"Y={tuple(Y.shape)}", flush=True)
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    # Label budget: stratified subsample of the train-period labelled users.
    has_row, is_pos = scan_train_rows(train_loader, Y_t, H, P, device)
    user_mask, n_sel_pos, n_sel_neg, n_pos_u, n_neg_u = choose_label_users(
        has_row, is_pos, args.label_fraction, args.label_seed)
    user_mask_t = torch.from_numpy(user_mask).to(device)
    n_pos_rows, n_neg_rows = count_train_rows(train_loader, Y_t, user_mask_t,
                                              H, P, device)
    if args.pos_weight == "auto":
        pw = n_neg_rows / max(n_pos_rows, 1)
    else:
        pw = float(args.pos_weight)
    pos_weight = torch.tensor(pw, device=device) if pw > 0 else None
    print(f"[labels] train users pos={n_pos_u} neg={n_neg_u} -> selected "
          f"pos={n_sel_pos} neg={n_sel_neg} (fraction={args.label_fraction}, "
          f"label_seed={args.label_seed}); train rows pos={n_pos_rows} "
          f"neg={n_neg_rows}; pos_weight={pw:.2f}", flush=True)
    if n_pos_rows == 0:
        raise SystemExit("no positive training rows under this label budget")

    encoder, predictor, head, init_info = build_models(
        args, train_ds.feature_dim, train_ds.num_nodes, device)

    # Optimiser: one NAMED group per part so the head and the encoder can run
    # at their own LR and each part is gradient-clipped on its own norm.
    groups = [{"name": "head", "params": list(head.parameters()),
               "lr": args.head_lr if args.head_lr is not None else args.lr}]
    if not args.freeze_predictor:
        groups.append({"name": "pred", "params": list(predictor.parameters()),
                       "lr": args.lr})
    else:
        for q in predictor.parameters():
            q.requires_grad_(False)
    if not args.freeze_encoder:
        groups.append({"name": "enc", "params": list(encoder.parameters()),
                       "lr": args.encoder_lr if args.encoder_lr is not None else args.lr})
    else:
        for q in encoder.parameters():
            q.requires_grad_(False)
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total = max(1, args.epochs * steps_per_epoch)
    warmup = int(args.warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_warmup_lr(s, warmup, total, args.lr_min_ratio))
    ema = None
    if args.aux_jepa_weight > 0 and args.aux_target == "ema" and not args.freeze_encoder:
        ema = EMAEncoder(encoder, momentum=args.ema_momentum)
        ema.target.to(device)

    if args.smoke:
        tr = train_epoch(encoder, predictor, head, train_loader, opt, sched, Y_t,
                         user_mask_t, pos_weight, args, device, ema=ema, max_steps=2)
        vm = evaluate(encoder, predictor, head, val_loader, Y_t, args, device)
        print(f"[smoke] train {tr}  val {_fmt(vm)}", flush=True)
        with open(out_dir / "finetune.json", "w") as f:
            json.dump({"smoke": True, "val": vm, "train": tr}, f, indent=2)
        print("[smoke] ok"); return

    best_auc, best_epoch, no_improve, best_state = -1.0, -1, 0, None
    history = []
    for epoch in range(args.epochs):
        tr = train_epoch(encoder, predictor, head, train_loader, opt, sched, Y_t,
                         user_mask_t, pos_weight, args, device, ema=ema)
        vm = evaluate(encoder, predictor, head, val_loader, Y_t, args, device)
        improved = (not math.isnan(vm["auroc"])
                    and vm["auroc"] > best_auc + args.early_stop_min_delta)
        if improved:
            best_auc, best_epoch, no_improve = vm["auroc"], epoch + 1, 0
            best_state = {
                "encoder": copy.deepcopy(encoder.state_dict()),
                "predictor": copy.deepcopy(predictor.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
            }
        else:
            no_improve += 1
        history.append({"epoch": epoch + 1, **{f"train_{k}": v for k, v in tr.items()},
                        "val_auroc": vm["auroc"], "val_auprc": vm["auprc"],
                        "best": improved})
        gn = " ".join(f"{k[3:]}={v:.2f}" for k, v in tr.items() if k.startswith("gn_"))
        print(f"[epoch {epoch+1}/{args.epochs}] loss={tr['loss']:.4f} "
              f"(cls={tr['loss_cls']:.4f} jepa={tr['loss_jepa']:.4f}) "
              f"gnorm[{gn}] val {_fmt(vm)}{'  (best)' if improved else ''} "
              f"({tr['time_s']:.0f}s)", flush=True)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"  early stopping: no val improvement for {no_improve} epochs",
                  flush=True)
            break

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        predictor.load_state_dict(best_state["predictor"])
        head.load_state_dict(best_state["head"])
        torch.save({**best_state, "epoch": best_epoch, "val_auroc": best_auc,
                    "args": vars(args)}, out_dir / "best.ckpt")
    vm = evaluate(encoder, predictor, head, val_loader, Y_t, args, device)
    tm = evaluate(encoder, predictor, head, test_loader, Y_t, args, device)
    print(f"[best epoch {best_epoch}] val  {_fmt(vm)}", flush=True)
    print(f"[best epoch {best_epoch}] test {_fmt(tm)}", flush=True)
    result = {
        "init": args.init, "label_fraction": args.label_fraction,
        "label_seed": args.label_seed, "aux_jepa_weight": args.aux_jepa_weight,
        "aux_target": args.aux_target if args.aux_jepa_weight > 0 else None,
        "freeze_encoder": args.freeze_encoder, "freeze_predictor": args.freeze_predictor,
        "label_mode": label_mode, "pos_weight": pw,
        "label_users": {"pos_total": n_pos_u, "neg_total": n_neg_u,
                        "pos_selected": n_sel_pos, "neg_selected": n_sel_neg},
        "train_rows": {"pos": n_pos_rows, "neg": n_neg_rows},
        "best_epoch": best_epoch, "epochs_run": len(history),
        "val": vm, "test": tm, "init_info": init_info, "seed": args.seed,
    }
    with open(out_dir / "finetune.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] -> {out_dir / 'finetune.json'}", flush=True)


if __name__ == "__main__":
    main()
