"""Train the CLDG baseline, then save a standard encoder checkpoint.

CLDG is encoder-only self-supervised learning (see ``cldg.py``). This script
trains the shared GNN encoder with the temporal contrastive objective on the
TRAIN portion of the snapshot sequence, then saves ``encoder.ckpt`` in the same
schema as ``pretrain_encoder.py`` so it evaluates through the identical probe:

    # 1. train the CLDG encoder
    python -m src.baselines.cldg.train_cldg --data_path dataset/dblp/edges.csv \
        --encoder_kind gcn --encoder_hidden 128 --node_dim 128 \
        --snapshots 4 --views 2 --strategy random --epochs 100 --device cuda \
        --out_dir outputs/cldg_dblp

    # 2. evaluate with the SAME link-prediction probe as DELOS
    python -m src.train.analyze_jepa --skip_diagnostic --link_sources encoder \
        --data_path dataset/dblp/edges.csv \
        --encoder_ckpt outputs/cldg_dblp/encoder.ckpt

Supports the global dynamic-graph datasets (dblp / jodie / reddit).
"""
from __future__ import annotations

import argparse
import contextlib
import random
import time

import numpy as np
import torch

from src.baselines.cldg.cldg import CLDG, nt_xent_loss, sampling_layer, view_arrays
from src.baselines.common import BestValTracker, split_snapshots
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.engine import save_encoder_ckpt, set_deterministic, setup_output_dir
from src.world_models.delos_encoder import build_encoder


def run_epoch(model, snaps, args, device, rng, n_steps, train: bool, opt=None):
    """One pass of ``n_steps`` contrastive steps over ``snaps``.

    ``train=False`` runs the identical objective under ``no_grad`` on the val
    snapshots — the selection signal, so it must be the same loss the training
    steps minimise (not a proxy).
    """
    adj, active, feats = snaps
    span = float(args.span) if args.span > 0 else float(adj.shape[0])
    # --view_length pins the view to a fixed number of snapshots; otherwise fall
    # back to the reference's "a fraction of the whole timeline" derivation.
    length = (float(args.view_length) if getattr(args, "view_length", 0) > 0
              else span / args.snapshots)
    agg, n = 0.0, 0
    for _ in range(n_steps):
        starts = sampling_layer(args.snapshots, args.views, span,
                                args.strategy, rng)
        with contextlib.nullcontext() if train else torch.no_grad():
            embs, common = [], None
            for st in starts:
                adj_v, active_v, feats_v = view_arrays(adj, active, feats, st, length)
                embs.append(model.embed_view(feats_v, adj_v, device))    # (N,D)
                common = active_v if common is None else (common & active_v)
            idx = np.flatnonzero(common)
            if idx.size < 2:
                continue
            if idx.size > args.node_batch_size:
                idx = np.array(rng.sample(idx.tolist(), args.node_batch_size))
            idx_t = torch.from_numpy(idx).long().to(device)

            loss, pairs = 0.0, 0
            for i in range(args.views):
                for j in range(i + 1, args.views):
                    loss = loss + nt_xent_loss(embs[i][idx_t], embs[j][idx_t], args.tau)
                    pairs += 1
            loss = loss / max(pairs, 1)

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        agg += float(loss.detach()); n += 1
    return agg / max(n, 1)


def train_cldg(model, train_snaps, val_snaps, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    tracker = BestValTracker(args.early_stop_patience, args.early_stop_min_delta)

    n_epochs = 1 if args.smoke else args.epochs
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        n_steps = 3 if args.smoke else args.steps_per_epoch
        train_loss = run_epoch(model, train_snaps, args, device, rng, n_steps,
                               train=True, opt=opt)
        if args.smoke:
            print("[smoke] ok"); return float("inf")

        # Val loss on the held-out snapshots. A fresh, fixed-seed RNG means every
        # epoch is scored on the SAME view samples, so the comparison across
        # epochs is a model difference and not a sampling difference.
        model.eval()
        val_loss = run_epoch(model, val_snaps, args, device,
                             random.Random(args.seed + 10_000), args.val_steps,
                             train=False)
        improved, stop = tracker.update(val_loss, model.encoder)
        print(f"[epoch {epoch+1}/{n_epochs}] nt_xent={train_loss:.4f} "
              f"val_nt_xent={val_loss:.4f}{'  (best)' if improved else ''} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if stop:
            print(f"  early stopping: no val improvement for "
                  f"{args.early_stop_patience} epochs", flush=True)
            break

    if tracker.restore(model.encoder):
        print(f"[best] restored encoder from best val epoch "
              f"(val={tracker.best:.4f})", flush=True)
    return tracker.best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cli.add_dblp_synthetic_flags(p)
    add_dataset_cli_flags(p)
    cli.add_window_flags(p)                       # only used to construct the dataset
    cli.add_encoder_flags(p, default_kind="gcn", node_dim=128, encoder_hidden=128)
    cli.add_run_flags(p, out_dir="outputs/cldg")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    # CLDG-specific (defaults from the reference repo)
    p.add_argument("--snapshots", type=int, default=4,
                   help="Number of evenly-spaced temporal windows to sample "
                        "from. Each view therefore spans span/snapshots "
                        "snapshots — see --view_length before using this on a "
                        "long timeline.")
    p.add_argument("--view_length", type=float, default=0.0,
                   help="View span in SNAPSHOT UNITS, overriding the "
                        "span/snapshots derivation. Prefer this whenever the "
                        "timeline length varies between datasets: the reference "
                        "default (--snapshots 4) means 'a quarter of the whole "
                        "timeline', which is ~10 days on England COVID's 42 "
                        "daily snapshots but ~20 DAYS on METR-LA's ~24k 5-minute "
                        "ones. Mean-pooling 20 days of traffic into one static "
                        "view erases the very temporal variation CLDG "
                        "contrasts, and the objective sits at chance. Setting "
                        "this to the forecast history length keeps the view a "
                        "comparable amount of real time on every dataset.")
    p.add_argument("--views", type=int, default=2,
                   help="Number of temporal views contrasted per step.")
    p.add_argument("--span", type=float, default=0.0,
                   help="Total span (in snapshot units) to sample views from. "
                        "0 = the full training time range.")
    p.add_argument("--strategy", choices=["random", "sequential"], default="random")
    p.add_argument("--tau", type=float, default=0.07, help="InfoNCE temperature.")
    p.add_argument("--node_batch_size", type=int, default=256,
                   help="Nodes sampled per contrastive step (from the "
                        "view-intersection).")
    p.add_argument("--steps_per_epoch", type=int, default=10)
    p.add_argument("--val_steps", type=int, default=10,
                   help="Contrastive steps used to estimate the val loss each "
                        "epoch (fixed seed, so the estimate is comparable "
                        "across epochs).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    args = p.parse_args()

    set_deterministic(args.seed)
    out_dir = setup_output_dir(args.out_dir, args)
    device = torch.device(args.device)
    print(f"[setup] CLDG baseline  data={args.data_source} "
          f"encoder={args.encoder_kind} snapshots={args.snapshots} "
          f"views={args.views} strategy={args.strategy} tau={args.tau}", flush=True)

    train_ds = build_dynamic_dataset(args, split="train")
    train_snaps, val_snaps = split_snapshots(train_ds, "CLDG", args)
    print(f"[data] train snapshots={train_snaps[0].shape[0]} "
          f"val snapshots={val_snaps[0].shape[0]} N={train_ds.num_nodes} "
          f"F={train_ds.feature_dim}", flush=True)
    if args.views > args.snapshots and args.strategy == "sequential":
        raise SystemExit("--views must be <= --snapshots for sequential strategy")

    encoder = build_encoder(
        args.encoder_kind, in_dim=train_ds.feature_dim,
        hidden_dim=args.encoder_hidden, out_dim=args.node_dim, freeze=False,
        bidirectional=args.bidirectional).to(device)
    model = CLDG(encoder, latent_dim=args.node_dim, tau=args.tau).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[model] CLDG trainable={n_params:,}", flush=True)

    best = train_cldg(model, train_snaps, val_snaps, args, device)
    if args.smoke:
        return
    save_encoder_ckpt(out_dir / "encoder.ckpt", model.encoder, args,
                      feature_dim=train_ds.feature_dim,
                      pretrain_objective="cldg", best_loss=best,
                      val_loss=best)
    print(f"[done] best val nt_xent = {best:.4f}  -> {out_dir/'encoder.ckpt'}\n"
          f"[eval] probe it with: python -m src.train.analyze_jepa "
          f"--skip_diagnostic --link_sources encoder --encoder_ckpt {out_dir/'encoder.ckpt'} "
          f"<same --data_* flags>", flush=True)


if __name__ == "__main__":
    main()
