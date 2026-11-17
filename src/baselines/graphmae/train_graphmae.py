"""Train the GraphMAE baseline (STATIC masked-feature autoencoder), then save
a standard encoder checkpoint.

GraphMAE is encoder-only self-supervised learning on ONE static graph; the
method and its static per-snapshot protocol are described in
``graphmae.py``. The encoder is saved as ``encoder.ckpt`` in the same schema
as ``pretrain_encoder.py`` so it evaluates through the identical probe, which
reads the encoder at the LAST history snapshot ("static SSL, last frame"):

    # 1. train the GraphMAE encoder
    python -m src.baselines.graphmae.train_graphmae --data_source dblp \
        --data_path dataset/dblp/edges.csv \
        --encoder_kind gcn --encoder_hidden 128 --node_dim 128 \
        --mask_rate 0.5 --replace_rate 0.05 --sce_gamma 2 \
        --epochs 100 --device cuda --out_dir outputs/graphmae_dblp

    # 2. evaluate with the SAME link-prediction probe as DELOS
    python -m src.train.analyze_jepa --skip_diagnostic --link_sources encoder \
        --data_source dblp --data_path dataset/dblp/edges.csv \
        --encoder_ckpt outputs/graphmae_dblp/encoder.ckpt

What is reconstructed: by default every input channel (the method as
published). ``--target semantic`` restricts the target to the channels past
``--graph_feat_dim`` (the pooled edge features that ``--with_edge_features``
appends), for the state tables where the 8 graph-derived channels are degree
statistics and per-snapshot time constants; it errors if there are none.

Supports the global dynamic-graph datasets (dblp / jodie / reddit /
england_covid) and fixed-topology signal datasets (metr_la).
"""
from __future__ import annotations

import argparse
import contextlib
import random
import time

import numpy as np
import torch

from src.baselines.common import BestValTracker, split_snapshots
from src.baselines.grace.grace import snapshot_tensors
from src.baselines.graphmae.graphmae import GraphMAE
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.engine import save_encoder_ckpt, set_deterministic, setup_output_dir
from src.world_models.delos_encoder import build_encoder


def run_epoch(model, snaps, args, device, rng, n_steps, train: bool, opt=None):
    """``n_steps`` GraphMAE steps, each on ONE snapshot drawn from ``snaps``.

    ``train=False`` runs the identical objective under ``no_grad`` on the val
    snapshots (fixed-seed ``rng`` at the call site, so every epoch is scored
    on the SAME snapshots and masks).
    """
    adj, active, feats = snaps
    T = int(adj.shape[0])
    agg, n = 0.0, 0
    for _ in range(n_steps):
        t = rng.randrange(T)
        act = torch.from_numpy(np.ascontiguousarray(active[t])).bool().to(device)
        if int(act.sum()) < 2:
            continue
        gen = torch.Generator(device=device)
        gen.manual_seed(rng.randrange(2 ** 31))
        x, a = snapshot_tensors(adj[t], feats[t], device)
        target = x[..., args.graph_feat_dim:] if args.target == "semantic" else None
        with contextlib.nullcontext() if train else torch.no_grad():
            loss = model(x, a, act, gen, target=target)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        agg += float(loss.detach()); n += 1
    return agg / max(n, 1)


def train_graphmae(model, train_snaps, val_snaps, args, device):
    # Reference optimiser: Adam, lr 1e-3, weight decay 2e-4 (Cora config).
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
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

        model.eval()
        val_loss = run_epoch(model, val_snaps, args, device,
                             random.Random(args.seed + 10_000), args.val_steps,
                             train=False)
        improved, stop = tracker.update(val_loss, model.encoder)
        print(f"[epoch {epoch+1}/{n_epochs}] sce={train_loss:.4f} "
              f"val_sce={val_loss:.4f}{'  (best)' if improved else ''} "
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_dblp_synthetic_flags(p)
    add_dataset_cli_flags(p)
    cli.add_window_flags(p)                       # only used to construct the dataset
    cli.add_encoder_flags(p, default_kind="gcn", node_dim=128, encoder_hidden=128)
    cli.add_run_flags(p, out_dir="outputs/graphmae")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    # GraphMAE-specific (reference Cora defaults)
    p.add_argument("--mask_rate", type=float, default=0.5,
                   help="Fraction of active nodes whose features are masked.")
    p.add_argument("--replace_rate", type=float, default=0.05,
                   help="Share of masked nodes that get a random node's "
                        "features instead of the [MASK] token.")
    p.add_argument("--sce_gamma", type=float, default=2.0,
                   help="Exponent of the scaled cosine error (>= 1).")
    p.add_argument("--target", choices=["all", "semantic"], default="all",
                   help="Reconstruct every input channel (the paper) or only "
                        "the channels past --graph_feat_dim.")
    p.add_argument("--graph_feat_dim", type=int, default=8,
                   help="Channels [0, this) are graph-derived; only read "
                        "under --target semantic.")
    p.add_argument("--steps_per_epoch", type=int, default=10,
                   help="Snapshots (one per step) visited per epoch.")
    p.add_argument("--val_steps", type=int, default=10,
                   help="Steps used to estimate the val loss each epoch "
                        "(fixed seed, so the estimate is comparable across "
                        "epochs).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=2e-4)
    args = p.parse_args()

    set_deterministic(args.seed)
    out_dir = setup_output_dir(args.out_dir, args)
    device = torch.device(args.device)
    print(f"[setup] GraphMAE baseline (static SSL)  data={args.data_source} "
          f"encoder={args.encoder_kind} mask_rate={args.mask_rate} "
          f"replace_rate={args.replace_rate} gamma={args.sce_gamma} "
          f"target={args.target}", flush=True)

    train_ds = build_dynamic_dataset(args, split="train")
    train_snaps, val_snaps = split_snapshots(train_ds, "GraphMAE", args)
    print(f"[data] train snapshots={train_snaps[0].shape[0]} "
          f"val snapshots={val_snaps[0].shape[0]} N={train_ds.num_nodes} "
          f"F={train_ds.feature_dim}", flush=True)
    if args.target == "semantic" and train_ds.feature_dim <= args.graph_feat_dim:
        raise SystemExit(
            f"--target semantic needs channels past --graph_feat_dim="
            f"{args.graph_feat_dim}, but F={train_ds.feature_dim}. Pass "
            f"--with_edge_features (state tables) or use --target all.")

    encoder = build_encoder(
        args.encoder_kind, in_dim=train_ds.feature_dim,
        hidden_dim=args.encoder_hidden, out_dim=args.node_dim, freeze=False,
        bidirectional=args.bidirectional).to(device)
    model = GraphMAE(encoder, in_dim=train_ds.feature_dim,
                     latent_dim=args.node_dim, mask_rate=args.mask_rate,
                     replace_rate=args.replace_rate,
                     gamma=args.sce_gamma).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[model] GraphMAE trainable={n_params:,}", flush=True)

    best = train_graphmae(model, train_snaps, val_snaps, args, device)
    if args.smoke:
        return
    save_encoder_ckpt(out_dir / "encoder.ckpt", model.encoder, args,
                      feature_dim=train_ds.feature_dim,
                      pretrain_objective="graphmae", best_loss=best,
                      val_loss=best)
    print(f"[done] best val sce = {best:.4f}  -> {out_dir/'encoder.ckpt'}\n"
          f"[eval] probe it with: python -m src.train.analyze_jepa "
          f"--skip_diagnostic --link_sources encoder --encoder_ckpt {out_dir/'encoder.ckpt'} "
          f"<same --data_* flags>", flush=True)


if __name__ == "__main__":
    main()
