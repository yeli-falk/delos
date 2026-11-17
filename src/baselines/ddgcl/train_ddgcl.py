"""Train the DDGCL baseline, then save a standard encoder checkpoint.

DDGCL is encoder-only self-supervised learning (see ``ddgcl.py``): time-aware
temporal positive pairs trained with the debiased contrastive loss. This script
trains the shared GNN encoder on the TRAIN portion of the snapshot sequence,
then saves ``encoder.ckpt`` in the same schema as ``pretrain_encoder.py`` so it
evaluates through the identical probe:

    # 1. train the DDGCL encoder
    python -m src.baselines.ddgcl.train_ddgcl --data_source dblp/jodie_mooc \
        --encoder_kind gcn --encoder_hidden 128 --node_dim 128 \
        --max_gap 4 --tau 0.1  --epochs 100 --device cuda \
    

    # 2. evaluate with the SAME link-prediction probe as DyG-JEPA
    python -m src.train.analyze_jepa --skip_diagnostic --link_sources encoder \
        --data_path dataset/dblp/edges.csv \
        --encoder_ckpt outputs/ddgcl_dblp/encoder.ckpt

The dynamic regime is realised by sampling, per step, a node's two temporal
views as two nearby TRAIN snapshots ``t1 < t2`` (gap up to ``--max_gap``); the
time gap conditions the time-aware projection. Supports the global dynamic-graph
datasets (dblp / jodie / reddit). The encoder consumes the pipeline's node
features, so its input matches DyG-JEPA exactly — only the SSL objective differs.
"""
from __future__ import annotations

import argparse
import contextlib
import time

import numpy as np
import torch

from src.baselines.common import BestValTracker, split_snapshots
from src.baselines.ddgcl.ddgcl import DDGCL
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.engine import save_encoder_ckpt, set_deterministic, setup_output_dir
from src.world_models.graph_jepa_encoder import build_encoder


def run_epoch(model, snaps, args, device, rng, n_steps, train: bool, opt=None):
    """One pass of ``n_steps`` time-aware contrastive steps over ``snaps``.

    ``train=False`` scores the identical objective under ``no_grad`` with
    ``update=False``, so evaluating on the val snapshots neither advances the
    momentum encoder nor pushes held-out keys into the negative queue.
    """
    adj, active, feats = snaps
    T = adj.shape[0]
    if T < 2:
        return float("nan")
    agg, n = 0.0, 0
    for _ in range(n_steps):
        # Two temporal views of the same nodes: snapshots t1 < t2, gap up
        # to max_gap (the "node at two nearby times" positive pair).
        t1 = int(rng.integers(0, T - 1))
        gap = int(rng.integers(1, min(args.max_gap, T - 1 - t1) + 1))
        t2 = t1 + gap
        common = active[t1] & active[t2]
        idx = np.flatnonzero(common)
        if idx.size < 2:
            continue
        if idx.size > args.node_batch_size:
            idx = rng.choice(idx, size=args.node_batch_size, replace=False)
        idx_t = torch.from_numpy(idx).long().to(device)

        with contextlib.nullcontext() if train else torch.no_grad():
            loss = model.contrastive_step(
                feats[t1], adj[t1], feats[t2], adj[t2], idx_t,
                t_query=t1, t_key=t2, device=device, update=train)

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        agg += float(loss.detach()); n += 1
    return agg / max(n, 1)


def train_ddgcl(model, train_snaps, val_snaps, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    if train_snaps[0].shape[0] < 2:
        raise SystemExit(
            f"DDGCL needs >=2 train snapshots; got {train_snaps[0].shape[0]}.")
    if val_snaps[0].shape[0] < 2:
        raise SystemExit(
            f"DDGCL needs >=2 val snapshots to form a temporal pair for model "
            f"selection; got {val_snaps[0].shape[0]}. Bin the timeline into "
            f"more steps.")
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
                             np.random.default_rng(args.seed + 10_000),
                             args.val_steps, train=False)
        improved, stop = tracker.update(val_loss, model.encoder)
        print(f"[epoch {epoch+1}/{n_epochs}] loss_cpc_debais={train_loss:.4f} "
              f"val_loss_cpc_debais={val_loss:.4f}"
              f"{'  (best)' if improved else ''} "
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
    add_dataset_cli_flags(p)
    cli.add_window_flags(p)                       # only used to construct the dataset
    cli.add_encoder_flags(p, default_kind="gcn", node_dim=128, encoder_hidden=128)
    cli.add_run_flags(p, out_dir="outputs/ddgcl")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    # DDGCL-specific (defaults from the reference repo)
    p.add_argument("--max_gap", type=int, default=4,
                   help="Max snapshot gap between the two temporal views.")
    p.add_argument("--time_dim", type=int, default=32,
                   help="Dimension of the Bochner/TGAT time-gap encoding (even).")
    p.add_argument("--tau", type=float, default=0.01,
                   help="Debias class-prior (reference's 1/100); 0 = no debias term.")
    p.add_argument("--beta", type=float, default=1.5,
                   help="Hard-negative reweighting concentration "
                        "(reference's beta); 0 = uniform negatives.")
    p.add_argument("--momentum", type=float, default=0.999,
                   help="MoCo key-encoder EMA coefficient; 0 disables the "
                        "momentum encoder + queue (in-batch negatives).")
    p.add_argument("--queue_size", type=int, default=512,
                   help="FIFO negative-queue size; 0 uses in-batch negatives.")
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
    print(f"[setup] DDGCL baseline  data={args.data_source} "
          f"encoder={args.encoder_kind} max_gap={args.max_gap} "
          f"tau={args.tau} beta={args.beta} momentum={args.momentum} "
          f"queue_size={args.queue_size}", flush=True)

    train_ds = build_dynamic_dataset(args, split="train")
    train_snaps, val_snaps = split_snapshots(train_ds, "DDGCL", args)
    print(f"[data] train snapshots={train_snaps[0].shape[0]} "
          f"val snapshots={val_snaps[0].shape[0]} N={train_ds.num_nodes} "
          f"F={train_ds.feature_dim}", flush=True)

    encoder = build_encoder(
        args.encoder_kind, in_dim=train_ds.feature_dim,
        hidden_dim=args.encoder_hidden, out_dim=args.node_dim, freeze=False,
        bidirectional=args.bidirectional).to(device)
    model = DDGCL(encoder, latent_dim=args.node_dim, time_dim=args.time_dim,
                  tau=args.tau, beta=args.beta, momentum=args.momentum,
                  queue_size=args.queue_size).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[model] DDGCL trainable={n_params:,}", flush=True)

    best = train_ddgcl(model, train_snaps, val_snaps, args, device)
    if args.smoke:
        return
    ckpt_path = out_dir / "encoder.ckpt"
    save_encoder_ckpt(ckpt_path, model.encoder, args,
                      feature_dim=train_ds.feature_dim,
                      pretrain_objective="ddgcl", best_loss=best,
                      val_loss=best)
    print(f"[done] best val loss_cpc_debais = {best:.4f}  -> {ckpt_path}\n"
          f"[eval] probe it with: python -m src.train.analyze_jepa "
          f"--skip_diagnostic --link_sources encoder --encoder_ckpt {ckpt_path} "
          f"<same --data_* flags>", flush=True)


if __name__ == "__main__":
    main()
