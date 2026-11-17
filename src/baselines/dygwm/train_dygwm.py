"""Train the DyG-WM SSL baseline, then save a standard encoder checkpoint.

DyG-WM (see ``dygwm.py``) trains its context encoder (``--encoder_kind dygwm``:
shared GCN + node / neighbourhood trajectory GRUs + gated fusion) to predict an
EMA target encoder's representation of the NEXT snapshot's nodes and observed
relations from the preceding window, with variance / covariance
regularisation. SSL-only: the paper's supervised link and ranking losses are
not used, so the row is the paper's own frozen-encoder protocol (its node
classification setting). This script trains on the TRAIN portion of the
snapshot sequence, selects on the val-window loss, and saves ``encoder.ckpt``
in the ``pretrain_encoder.py`` schema so the encoder is scored through the
SAME frozen probes as every other baseline:

    # 1. pre-train (paper node setting: EMA 0.995, hidden 128, lr 1e-3, wd 1e-5)
    python -m src.baselines.dygwm.train_dygwm --data_source jodie_mooc \
        --encoder_kind dygwm --encoder_hidden 128 --node_dim 128 \
        --epochs 100 --device cuda --out_dir outputs/dygwm_mooc

    # 2. frozen-encoder probe, identical to every other baseline
    python -m src.train.analyze_jepa --skip_diagnostic --link_sources encoder \
        --data_source jodie_mooc --encoder_ckpt outputs/dygwm_mooc/encoder.ckpt

Windows: ``--window`` context snapshots (default = ``--history_size``, so the
encoder is pre-trained on the length it is probed on; the paper uses W = 3,
i.e. 2 context bins, for node tasks) followed by ONE target snapshot. The
best-val-loss epoch is restored, like every other trainer in the pipeline.
Supports the global dynamic-graph datasets (dblp / jodie_* / reddit_* / cldg_*
/ england_covid) and the fixed-topology signal datasets (metr_la) -- on a
constant graph the structural descriptors are constant and the objective
reduces to next-step latent prediction.
"""
from __future__ import annotations

import argparse
import contextlib
import time

import numpy as np
import torch

from src.baselines.common import BestValTracker, split_snapshots
from src.baselines.dygwm.dygwm import DyGWMSSL
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.engine import save_encoder_ckpt, set_deterministic, setup_output_dir
from src.world_models.graph_jepa_encoder import DyGWMEncoder, build_encoder

LOG_KEYS = ("loss", "node", "rel", "var", "cov")


def _window_length(T: int, want: int, what: str) -> int:
    # L context snapshots + 1 target must fit.
    L = min(want, T - 1)
    if L < want:
        print(f"[warn] {what}: only {T} snapshots; context window shrunk from "
              f"{want} to {L}.", flush=True)
    return L


def run_epoch(model, snaps, L, args, device, rng, n_steps, train: bool, opt=None):
    """``n_steps`` (context window of ``L`` snapshots, next snapshot) pairs
    sampled from ``snaps``. ``train=False`` scores the identical objective
    under ``no_grad`` with the caller's fixed-seed ``rng`` and no EMA update,
    so the val estimate is comparable across epochs."""
    adj, active, feats = snaps
    T = adj.shape[0]
    agg = {k: 0.0 for k in LOG_KEYS}
    n = 0
    for _ in range(n_steps):
        s = int(rng.integers(0, T - L))                  # target index s+L <= T-1
        adj_ctx = np.stack([np.asarray(adj[t]) for t in range(s, s + L)])
        gen = torch.Generator(device=device).manual_seed(int(rng.integers(2**31 - 1)))
        with contextlib.nullcontext() if train else torch.no_grad():
            loss, parts = model.window_loss(
                np.asarray(feats[s:s + L]), adj_ctx, np.asarray(active[s:s + L]),
                np.asarray(feats[s + L]), np.asarray(adj[s + L]),
                np.asarray(active[s + L]), device, gen,
                node_batch=args.node_batch_size, pair_batch=args.pair_batch_size)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.ema_update()
        agg["loss"] += float(loss.detach())
        for k in LOG_KEYS[1:]:
            agg[k] += parts[k]
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def train_dygwm(model, train_snaps, val_snaps, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    want = args.window if args.window > 0 else args.history_size
    if train_snaps[0].shape[0] < 2 or val_snaps[0].shape[0] < 2:
        raise SystemExit("DyG-WM needs >= 2 train and >= 2 val snapshots (a "
                         "context bin and a target bin); bin the timeline "
                         "into more steps.")
    L_tr = _window_length(train_snaps[0].shape[0], want, "train")
    L_va = _window_length(val_snaps[0].shape[0], want, "val")
    tracker = BestValTracker(args.early_stop_patience, args.early_stop_min_delta)

    n_epochs = 1 if args.smoke else args.epochs
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        n_steps = 3 if args.smoke else args.steps_per_epoch
        tr = run_epoch(model, train_snaps, L_tr, args, device, rng, n_steps,
                       train=True, opt=opt)
        if args.smoke:
            print(f"[smoke] ok {tr}"); return float("inf")

        model.eval()
        va = run_epoch(model, val_snaps, L_va, args, device,
                       np.random.default_rng(args.seed + 10_000), args.val_steps,
                       train=False)
        improved, stop = tracker.update(va["loss"], model.encoder)
        print(f"[epoch {epoch+1}/{n_epochs}] loss={tr['loss']:.4f} "
              f"(node={tr['node']:.3f} rel={tr['rel']:.3f} var={tr['var']:.3f} "
              f"cov={tr['cov']:.3f}) val_loss={va['loss']:.4f} "
              f"(node={va['node']:.3f} rel={va['rel']:.3f})"
              f"{'  (best)' if improved else ''} ({time.time()-t0:.1f}s)", flush=True)
        if stop:
            print(f"  early stopping: no val improvement for "
                  f"{args.early_stop_patience} epochs", flush=True)
            break

    if tracker.restore(model.encoder):
        print(f"[best] restored encoder from best val epoch "
              f"(val={tracker.best:.4f})", flush=True)
    return tracker.best


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_dblp_synthetic_flags(p)
    add_dataset_cli_flags(p)
    cli.add_window_flags(p)          # dataset construction + default window length
    cli.add_encoder_flags(p, default_kind="dygwm", node_dim=128, encoder_hidden=128)
    cli.add_run_flags(p, out_dir="outputs/dygwm")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    # DyG-WM-specific (paper Sec. 3.4 / App. A.2 defaults, node setting)
    p.add_argument("--ema_momentum", type=float, default=0.995,
                   help="EMA coefficient of the target encoder (paper: 0.995 "
                        "for node tasks, 0.999 for link prediction).")
    p.add_argument("--n_ctx", type=int, default=32,
                   help="Relation-context budget B: other nodes per pair (paper 32).")
    p.add_argument("--decay", type=float, default=0.8,
                   help="Temporal decay of the context-node recency score (paper 0.8).")
    p.add_argument("--cn_bonus", type=float, default=1.0,
                   help="Common-neighbour bonus in the context-node score (paper 1.0).")
    p.add_argument("--lam_node", type=float, default=1.0,
                   help="Weight of the node prediction term L^N (paper 0.02 next to "
                        "unit link losses; 1.0 here, SSL-only).")
    p.add_argument("--lam_rel", type=float, default=1.0,
                   help="Weight of the relation prediction term L^R (see --lam_node).")
    p.add_argument("--lam_var", type=float, default=0.25,
                   help="Variance regulariser weight (paper ratio 0.005 / 0.02).")
    p.add_argument("--lam_cov", type=float, default=0.025,
                   help="Covariance regulariser weight (paper ratio 0.0005 / 0.02).")
    p.add_argument("--gamma", type=float, default=1.0, help="Variance hinge floor.")
    p.add_argument("--pred_hidden", type=int, default=256,
                   help="Predictor hidden width (paper 256).")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout on the encoder's gated fusion branch.")
    p.add_argument("--window", type=int, default=0,
                   help="Context snapshots per window. 0 = --history_size, so "
                        "the encoder is pre-trained on the length it is probed on.")
    p.add_argument("--node_batch_size", type=int, default=256,
                   help="Target nodes per step (from the nodes active in the target bin).")
    p.add_argument("--pair_batch_size", type=int, default=256,
                   help="Observed target-bin edges per step for the relation term.")
    p.add_argument("--steps_per_epoch", type=int, default=20,
                   help="Training windows per epoch (same budget as the other baselines).")
    p.add_argument("--val_steps", type=int, default=10,
                   help="Val windows scored per epoch (fixed seed).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="AdamW learning rate (paper: 1e-3 for node tasks).")
    p.add_argument("--weight_decay", type=float, default=1e-5,
                   help="AdamW weight decay (paper 1e-5).")
    args = p.parse_args()

    set_deterministic(args.seed)
    out_dir = setup_output_dir(args.out_dir, args)
    device = torch.device(args.device)
    print(f"[setup] DyG-WM baseline (SSL-only)  data={args.data_source} "
          f"encoder={args.encoder_kind} ema={args.ema_momentum} n_ctx={args.n_ctx} "
          f"lam_node={args.lam_node} lam_rel={args.lam_rel} lam_var={args.lam_var} "
          f"lam_cov={args.lam_cov} window={args.window or args.history_size}", flush=True)

    train_ds = build_dynamic_dataset(args, split="train")
    train_snaps, val_snaps = split_snapshots(train_ds, "DyG-WM", args)
    print(f"[data] train snapshots={train_snaps[0].shape[0]} "
          f"val snapshots={val_snaps[0].shape[0]} N={train_ds.num_nodes} "
          f"F={train_ds.feature_dim}", flush=True)

    encoder = build_encoder(
        args.encoder_kind, in_dim=train_ds.feature_dim,
        hidden_dim=args.encoder_hidden, out_dim=args.node_dim, freeze=False,
        bidirectional=args.bidirectional).to(device)
    if isinstance(encoder, DyGWMEncoder):
        encoder.dropout.p = args.dropout
    else:
        print(f"[warn] DyG-WM's context encoder is 'dygwm' (shared GCN + trajectory "
              f"GRUs); got '{args.encoder_kind}' -- this is the DyG-WM objective on "
              f"that encoder, not DyG-WM. The EMA target is the whole encoder on "
              f"the target snapshot alone.", flush=True)
    model = DyGWMSSL(encoder, latent_dim=args.node_dim, n_ctx=args.n_ctx,
                     decay=args.decay, cn_bonus=args.cn_bonus,
                     ema_momentum=args.ema_momentum, pred_hidden=args.pred_hidden,
                     lam_node=args.lam_node, lam_rel=args.lam_rel,
                     lam_var=args.lam_var, lam_cov=args.lam_cov,
                     gamma=args.gamma).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[model] DyG-WM trainable={n_params:,}", flush=True)

    best = train_dygwm(model, train_snaps, val_snaps, args, device)
    if args.smoke:
        return
    ckpt_path = out_dir / "encoder.ckpt"
    save_encoder_ckpt(ckpt_path, model.encoder, args,
                      feature_dim=train_ds.feature_dim,
                      pretrain_objective="dygwm", best_loss=best, val_loss=best)
    print(f"[done] best val loss = {best:.4f}  -> {ckpt_path}\n"
          f"[eval] probe it with: python -m src.train.analyze_jepa "
          f"--skip_diagnostic --link_sources encoder --encoder_ckpt {ckpt_path} "
          f"<same --data_* flags>", flush=True)


if __name__ == "__main__":
    main()
