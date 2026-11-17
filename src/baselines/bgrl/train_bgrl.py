"""Train the BGRL baseline (STATIC bootstrapped graph latents), then save a
standard encoder checkpoint.

BGRL is encoder-only self-supervised learning on ONE static graph; the method
and its static per-snapshot protocol are described in ``bgrl.py``. The ONLINE
encoder is saved as ``encoder.ckpt`` in the same schema as
``pretrain_encoder.py`` so it evaluates through the identical probe, which
reads the encoder at the LAST history snapshot ("static SSL, last frame"):

    # 1. train the BGRL encoder
    python -m src.baselines.bgrl.train_bgrl --data_source dblp \
        --data_path dataset/dblp/edges.csv \
        --encoder_kind gcn --encoder_hidden 128 --node_dim 128 \
        --p_e1 0.2 --p_e2 0.3 --p_f1 0.2 --p_f2 0.1 --ema_momentum 0.99 \
        --epochs 100 --device cuda --out_dir outputs/bgrl_dblp

    # 2. evaluate with the SAME link-prediction probe as DELOS
    python -m src.train.analyze_jepa --skip_diagnostic --link_sources encoder \
        --data_source dblp --data_path dataset/dblp/edges.csv \
        --encoder_ckpt outputs/bgrl_dblp/encoder.ckpt

Checkpoint selection. ``--select val`` (default) is the rule every other
baseline uses: keep the epoch with the lowest val loss. For a bootstrap loss
that signal is weaker than for a contrastive or reconstruction loss, because
the target moves: a low loss can also mean the online and target networks
agree on little. ``--select last`` keeps the final epoch instead (the paper's
protocol: train for a fixed budget). The per-epoch ``spread`` column, shown
as a ratio to the untrained encoder's, is the collapse check to read before
trusting either.

Supports the global dynamic-graph datasets (dblp / jodie / reddit /
england_covid) and fixed-topology signal datasets (metr_la).
"""
from __future__ import annotations

import argparse
import contextlib
import math
import random
import time

import numpy as np
import torch

from src.baselines.bgrl.bgrl import BGRL, bgrl_loss, ema_momentum
from src.baselines.common import BestValTracker, split_snapshots
from src.baselines.grace.grace import snapshot_tensors
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.engine import save_encoder_ckpt, set_deterministic, setup_output_dir
from src.world_models.delos_encoder import build_encoder


def lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    """Reference schedule: linear warm-up, then cosine decay to 0."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    span = max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(step - warmup, span) / span))


def run_epoch(model, snaps, args, device, rng, n_steps, train: bool,
              opt=None, sched=None):
    """``n_steps`` BGRL steps, each on ONE snapshot drawn from ``snaps``.

    ``train=False`` scores the identical objective (current online vs current
    target) under ``no_grad`` on the val snapshots with a fixed-seed ``rng``.
    ``sched`` is the mutable ``{"step", "total", "warmup"}`` training clock
    that drives the LR and the EMA momentum. Returns (mean loss, mean spread).
    """
    adj, active, feats = snaps
    T = int(adj.shape[0])
    agg, spread, n = 0.0, 0.0, 0
    for _ in range(n_steps):
        t = rng.randrange(T)
        idx = np.flatnonzero(active[t])
        if idx.size < 2:
            continue
        idx_t = torch.from_numpy(idx).long().to(device)
        gen = torch.Generator(device=device)
        gen.manual_seed(rng.randrange(2 ** 31))
        x, a = snapshot_tensors(adj[t], feats[t], device)
        if train:
            for g in opt.param_groups:
                g["lr"] = lr_at(sched["step"], sched["total"], args.lr, sched["warmup"])
        with contextlib.nullcontext() if train else torch.no_grad():
            q1, y2, q2, y1 = model(x, a, gen)
            loss = bgrl_loss(q1[idx_t], y2[idx_t], q2[idx_t], y1[idx_t])
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()
            model.update_target(
                ema_momentum(sched["step"], sched["mom_total"], args.ema_momentum))
            sched["step"] += 1
        else:
            spread += model.embedding_spread(x, a, idx_t)
        agg += float(loss.detach()); n += 1
    return agg / max(n, 1), spread / max(n, 1)


def train_bgrl(model, train_snaps, val_snaps, args, device):
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    # --select last: patience 0 so the tracker never stops the run early.
    patience = args.early_stop_patience if args.select == "val" else 0
    tracker = BestValTracker(patience, args.early_stop_min_delta)

    n_epochs = 1 if args.smoke else args.epochs
    steps = 3 if args.smoke else args.steps_per_epoch
    total = n_epochs * steps
    mom_total = (args.ema_anneal_epochs * steps if args.ema_anneal_epochs > 0
                 else total)
    sched = {"step": 0, "total": total, "mom_total": mom_total,
             "warmup": int(round(args.warmup_frac * total))}
    val_loss = float("inf")
    # Collapse reference: the spread of the UNTRAINED online encoder on the
    # same fixed val draws. The absolute level depends on the features, so
    # every epoch is reported as a ratio to this.
    spread0 = 0.0
    if not args.smoke:
        model.eval()
        _, spread0 = run_epoch(model, val_snaps, args, device,
                               random.Random(args.seed + 10_000),
                               args.val_steps, train=False)
        print(f"[init] embedding spread of the untrained encoder = {spread0:.4f}",
              flush=True)
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        train_loss, _ = run_epoch(model, train_snaps, args, device, rng, steps,
                                  train=True, opt=opt, sched=sched)
        if args.smoke:
            print("[smoke] ok"); return float("inf")

        model.eval()
        val_loss, spread = run_epoch(model, val_snaps, args, device,
                                     random.Random(args.seed + 10_000),
                                     args.val_steps, train=False)
        improved, stop = tracker.update(val_loss, model.encoder)
        mom = ema_momentum(sched["step"], mom_total, args.ema_momentum)
        print(f"[epoch {epoch+1}/{n_epochs}] bgrl={train_loss:.4f} "
              f"val_bgrl={val_loss:.4f}{'  (best)' if improved else ''} "
              f"spread={spread:.4f} ({spread / max(spread0, 1e-12):.2f}x init) "
              f"tau={mom:.5f} ({time.time()-t0:.1f}s)", flush=True)
        if spread < args.collapse_ratio * spread0:
            print(f"  [warn] embedding spread fell below "
                  f"{args.collapse_ratio:.2f}x its initial value -- the online "
                  f"encoder may be collapsing; lower --lr or raise "
                  f"--ema_momentum", flush=True)
        if stop:
            print(f"  early stopping: no val improvement for "
                  f"{args.early_stop_patience} epochs", flush=True)
            break

    if args.select == "last":
        print(f"[select] --select last: keeping the final online encoder "
              f"(val={val_loss:.4f}; best val was {tracker.best:.4f})", flush=True)
        return val_loss
    if tracker.restore(model.encoder):
        print(f"[best] restored online encoder from best val epoch "
              f"(val={tracker.best:.4f})", flush=True)
    return tracker.best


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_dblp_synthetic_flags(p)
    add_dataset_cli_flags(p)
    cli.add_window_flags(p)                       # only used to construct the dataset
    cli.add_encoder_flags(p, default_kind="gcn", node_dim=128, encoder_hidden=128)
    cli.add_run_flags(p, out_dir="outputs/bgrl")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    # BGRL-specific (reference WikiCS config)
    p.add_argument("--p_e1", type=float, default=0.2,
                   help="Edge-removal probability, view 1.")
    p.add_argument("--p_e2", type=float, default=0.3,
                   help="Edge-removal probability, view 2.")
    p.add_argument("--p_f1", type=float, default=0.2,
                   help="Feature-dimension masking probability, view 1.")
    p.add_argument("--p_f2", type=float, default=0.1,
                   help="Feature-dimension masking probability, view 2.")
    p.add_argument("--ema_momentum", type=float, default=0.99,
                   help="Initial target momentum; cosine-annealed to 1 over "
                        "--ema_anneal_epochs (default: the whole run).")
    p.add_argument("--ema_anneal_epochs", type=int, default=0,
                   help="Epochs over which the momentum anneal runs (0 = the "
                        "whole --epochs budget, the reference). Under val "
                        "early stopping the whole-run horizon is rarely "
                        "reached, so set this below a typical run length.")
    p.add_argument("--pred_hidden", type=int, default=512,
                   help="Predictor MLP hidden width.")
    p.add_argument("--warmup_frac", type=float, default=0.1,
                   help="Share of the total steps spent in linear LR warm-up "
                        "(the reference's 1k of 10k steps), then cosine decay.")
    p.add_argument("--collapse_ratio", type=float, default=0.1,
                   help="Warn when the val embedding spread drops below this "
                        "fraction of the untrained encoder's.")
    p.add_argument("--select", choices=["val", "last"], default="val",
                   help="Checkpoint rule: best val loss (every other "
                        "baseline's rule) or the final epoch (the paper's "
                        "fixed-budget protocol).")
    p.add_argument("--steps_per_epoch", type=int, default=10,
                   help="Snapshots (one per step) visited per epoch.")
    p.add_argument("--val_steps", type=int, default=10,
                   help="Steps used to estimate the val loss each epoch "
                        "(fixed seed, so the estimate is comparable across "
                        "epochs).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    args = p.parse_args()

    set_deterministic(args.seed)
    out_dir = setup_output_dir(args.out_dir, args)
    device = torch.device(args.device)
    print(f"[setup] BGRL baseline (static SSL)  data={args.data_source} "
          f"encoder={args.encoder_kind} p_e=({args.p_e1},{args.p_e2}) "
          f"p_f=({args.p_f1},{args.p_f2}) tau0={args.ema_momentum} "
          f"select={args.select}", flush=True)

    train_ds = build_dynamic_dataset(args, split="train")
    train_snaps, val_snaps = split_snapshots(train_ds, "BGRL", args)
    print(f"[data] train snapshots={train_snaps[0].shape[0]} "
          f"val snapshots={val_snaps[0].shape[0]} N={train_ds.num_nodes} "
          f"F={train_ds.feature_dim}", flush=True)
    if train_ds.feature_dim <= 2 and max(args.p_f1, args.p_f2) > 0:
        print(f"[warn] F={train_ds.feature_dim}: feature-dimension masking "
              f"blanks a large share of the input per view; consider "
              f"--p_f1 0 --p_f2 0", flush=True)

    encoder = build_encoder(
        args.encoder_kind, in_dim=train_ds.feature_dim,
        hidden_dim=args.encoder_hidden, out_dim=args.node_dim, freeze=False,
        bidirectional=args.bidirectional).to(device)
    model = BGRL(encoder, latent_dim=args.node_dim, pred_hidden=args.pred_hidden,
                 p_e1=args.p_e1, p_e2=args.p_e2,
                 p_f1=args.p_f1, p_f2=args.p_f2).to(device)
    n_params = sum(p_.numel() for p_ in model.trainable_parameters())
    print(f"[model] BGRL trainable={n_params:,} (+ EMA target encoder)", flush=True)

    best = train_bgrl(model, train_snaps, val_snaps, args, device)
    if args.smoke:
        return
    save_encoder_ckpt(out_dir / "encoder.ckpt", model.encoder, args,
                      feature_dim=train_ds.feature_dim,
                      pretrain_objective="bgrl", best_loss=best,
                      val_loss=best)
    print(f"[done] selected val bgrl = {best:.4f}  -> {out_dir/'encoder.ckpt'}\n"
          f"[eval] probe it with: python -m src.train.analyze_jepa "
          f"--skip_diagnostic --link_sources encoder --encoder_ckpt {out_dir/'encoder.ckpt'} "
          f"<same --data_* flags>", flush=True)


if __name__ == "__main__":
    main()
