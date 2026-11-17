"""Train the GRACE baseline (STATIC graph SSL), then save a standard encoder
checkpoint.

GRACE is encoder-only self-supervised learning on ONE static graph; the
method and its static per-snapshot protocol are described in ``grace.py``.
The encoder is saved as ``encoder.ckpt`` in the same schema as
``pretrain_encoder.py`` so it evaluates through the identical probe, which
reads the encoder at the LAST history snapshot ("static SSL, last frame"):

    # 1. train the GRACE encoder
    python -m src.baselines.grace.train_grace --data_source dblp \
        --data_path dataset/dblp/edges.csv \
        --encoder_kind gcn --encoder_hidden 128 --node_dim 128 \
        --tau 0.4 --p_e1 0.2 --p_e2 0.4 --p_f1 0.3 --p_f2 0.4 \
        --epochs 100 --device cuda --out_dir outputs/grace_dblp

    # 2. evaluate with the SAME link-prediction probe as DELOS
    python -m src.train.analyze_jepa --skip_diagnostic --link_sources encoder \
        --data_source dblp --data_path dataset/dblp/edges.csv \
        --encoder_ckpt outputs/grace_dblp/encoder.ckpt

Supports the global dynamic-graph datasets (dblp / jodie / reddit /
england_covid) and fixed-topology signal datasets (metr_la: one shared
graph, so the edge-dropping view differs per step only through its random
mask while the feature view carries that step's readings).
"""
from __future__ import annotations

import argparse
import contextlib
import random
import time

import numpy as np
import torch

from src.baselines.common import BestValTracker, split_snapshots
from src.baselines.grace.grace import GRACE, grace_loss, snapshot_tensors
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.engine import save_encoder_ckpt, set_deterministic, setup_output_dir
from src.world_models.delos_encoder import build_encoder


def run_epoch(model, snaps, args, device, rng, n_steps, train: bool, opt=None):
    """``n_steps`` GRACE steps, each on ONE snapshot drawn from ``snaps``.

    ``train=False`` runs the identical objective under ``no_grad`` on the val
    snapshots (fixed-seed ``rng`` at the call site, so every epoch is scored
    on the SAME snapshots, node subsets and augmentation masks).
    """
    adj, active, feats = snaps
    T = int(adj.shape[0])
    agg, n = 0.0, 0
    for _ in range(n_steps):
        t = rng.randrange(T)
        idx = np.flatnonzero(active[t])
        if idx.size < 2:
            continue
        if 0 < args.node_batch_size < idx.size:
            idx = np.array(rng.sample(idx.tolist(), args.node_batch_size))
        idx_t = torch.from_numpy(idx).long().to(device)
        # Augmentation randomness is drawn from the epoch RNG so val masks are
        # reproducible across epochs (and across CPU / CUDA generators).
        gen = torch.Generator(device=device)
        gen.manual_seed(rng.randrange(2 ** 31))

        x, a = snapshot_tensors(adj[t], feats[t], device)
        with contextlib.nullcontext() if train else torch.no_grad():
            z1, z2 = model.two_views(x, a, gen)
            loss = grace_loss(z1[idx_t], z2[idx_t], args.tau)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        agg += float(loss.detach()); n += 1
    return agg / max(n, 1)


def train_grace(model, train_snaps, val_snaps, args, device):
    # Reference optimiser: Adam, lr 5e-4, weight decay 1e-5 (Cora config).
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
        print(f"[epoch {epoch+1}/{n_epochs}] grace={train_loss:.4f} "
              f"val_grace={val_loss:.4f}{'  (best)' if improved else ''} "
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
    cli.add_run_flags(p, out_dir="outputs/grace")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    # GRACE-specific (reference Cora defaults)
    p.add_argument("--tau", type=float, default=0.4,
                   help="Temperature of the contrastive objective.")
    p.add_argument("--p_e1", type=float, default=0.2,
                   help="Edge-removal probability, view 1.")
    p.add_argument("--p_e2", type=float, default=0.4,
                   help="Edge-removal probability, view 2.")
    p.add_argument("--p_f1", type=float, default=0.3,
                   help="Feature-dimension masking probability, view 1.")
    p.add_argument("--p_f2", type=float, default=0.4,
                   help="Feature-dimension masking probability, view 2.")
    p.add_argument("--proj_hidden", type=int, default=0,
                   help="Projection-head hidden width (0 = node_dim, the "
                        "reference's num_proj_hidden = num_hidden).")
    p.add_argument("--node_batch_size", type=int, default=256,
                   help="Nodes scored per step (sampled from the snapshot's "
                        "active nodes; negatives come from the same subset). "
                        "0 = every active node, the reference's full-graph "
                        "loss -- O(N^2) per step.")
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
    print(f"[setup] GRACE baseline (static SSL)  data={args.data_source} "
          f"encoder={args.encoder_kind} tau={args.tau} "
          f"p_e=({args.p_e1},{args.p_e2}) p_f=({args.p_f1},{args.p_f2})",
          flush=True)

    train_ds = build_dynamic_dataset(args, split="train")
    train_snaps, val_snaps = split_snapshots(train_ds, "GRACE", args)
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
    model = GRACE(encoder, latent_dim=args.node_dim,
                  proj_hidden=args.proj_hidden or None, tau=args.tau,
                  p_e1=args.p_e1, p_e2=args.p_e2,
                  p_f1=args.p_f1, p_f2=args.p_f2).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[model] GRACE trainable={n_params:,}", flush=True)

    best = train_grace(model, train_snaps, val_snaps, args, device)
    if args.smoke:
        return
    save_encoder_ckpt(out_dir / "encoder.ckpt", model.encoder, args,
                      feature_dim=train_ds.feature_dim,
                      pretrain_objective="grace", best_loss=best,
                      val_loss=best)
    print(f"[done] best val grace = {best:.4f}  -> {out_dir/'encoder.ckpt'}\n"
          f"[eval] probe it with: python -m src.train.analyze_jepa "
          f"--skip_diagnostic --link_sources encoder --encoder_ckpt {out_dir/'encoder.ckpt'} "
          f"<same --data_* flags>", flush=True)


if __name__ == "__main__":
    main()
