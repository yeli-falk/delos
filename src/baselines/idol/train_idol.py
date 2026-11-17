"""Train the IDOL baseline, then save a standard encoder checkpoint.

IDOL is encoder-only self-supervised learning (see ``idol.py``): PPR-propagated
features (the ``ppr_gcn`` encoder) trained with the GGD group-discrimination
objective. This script trains the encoder on the TRAIN portion of the snapshot
sequence, then saves ``encoder.ckpt`` in the same schema as
``pretrain_encoder.py`` so it evaluates through the identical probe:

    # 1. train the IDOL encoder
    python -m src.baselines.idol.train_idol --data_path dataset/dblp/edges.csv \
        --encoder_hidden 128 --node_dim 128 \
        --epochs 100 --device cuda --out_dir outputs/idol_dblp

    # 2. evaluate with the SAME link-prediction probe as DyG-JEPA
    python -m src.train.analyze_jepa --skip_diagnostic --link_sources encoder \
        --data_path dataset/dblp/edges.csv \
        --encoder_ckpt outputs/idol_dblp/encoder.ckpt

The "topology-monitorable" / dynamic regime is realised by training over the
sequence of TRAIN snapshots: each snapshot carries its own (incrementally
evolving) adjacency, and the encoder re-runs PPR propagation per snapshot — the
dense analogue of InstantGNN's incremental PPR maintenance. Supports the global
dynamic-graph datasets (dblp / jodie / reddit). The encoder consumes the
pipeline's node features, so its input matches DyG-JEPA exactly — only the SSL
objective differs. The encoder is ALWAYS ``ppr_gcn``: any ``--encoder_kind``
passed by a launcher is ignored (with a notice) so the shared ``$ENCODER_ARG``
cannot silently swap IDOL onto another architecture.
"""
from __future__ import annotations

import argparse
import contextlib
import time

import numpy as np
import torch

from src.baselines.common import BestValTracker, split_snapshots
from src.baselines.idol.idol import GGD, ggd_group_loss
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.engine import save_encoder_ckpt, set_deterministic, setup_output_dir
from src.world_models.graph_jepa_encoder import build_encoder

# IDOL is defined by PPR-propagated features + GGD; the encoder is not a knob.
IDOL_ENCODER_KIND = "ppr_gcn"


def run_epoch(model, snaps, args, device, rng, train: bool, opt=None,
              limit: int = 0):
    """One pass over the snapshots in ``snaps`` (shuffled when training).

    ``train=False`` runs the identical group-discrimination objective under
    ``no_grad`` on the val snapshots, in fixed snapshot order.
    """
    adj, active, feats = snaps
    T = adj.shape[0]
    order = rng.permutation(T) if train else np.arange(T)
    if limit:
        order = order[:limit]
    agg, n = 0.0, 0
    for t in order:
        adj_t, active_t, feats_t = adj[t], active[t], feats[t]
        idx = np.flatnonzero(active_t)
        if idx.size < 2:
            continue
        if idx.size > args.node_batch_size:
            idx = rng.choice(idx, size=args.node_batch_size, replace=False)
        idx_t = torch.from_numpy(idx).long().to(device)

        with contextlib.nullcontext() if train else torch.no_grad():
            s_pos, s_neg = model.group_scores(feats_t, adj_t, device, rng)
            loss = ggd_group_loss(s_pos[idx_t], s_neg[idx_t])

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        agg += float(loss.detach()); n += 1
    return agg / max(n, 1)


def train_idol(model, train_snaps, val_snaps, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    tracker = BestValTracker(args.early_stop_patience, args.early_stop_min_delta)

    n_epochs = 1 if args.smoke else args.epochs
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        train_loss = run_epoch(model, train_snaps, args, device, rng,
                               train=True, opt=opt,
                               limit=3 if args.smoke else args.steps_per_epoch)
        if args.smoke:
            print("[smoke] ok"); return float("inf")

        model.eval()
        val_loss = run_epoch(model, val_snaps, args, device,
                             np.random.default_rng(args.seed + 10_000),
                             train=False, limit=args.val_steps)
        improved, stop = tracker.update(val_loss, model.encoder)
        print(f"[epoch {epoch+1}/{n_epochs}] ggd_bce={train_loss:.4f} "
              f"val_ggd_bce={val_loss:.4f}{'  (best)' if improved else ''} "
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
    cli.add_encoder_flags(p, default_kind=IDOL_ENCODER_KIND, node_dim=128, encoder_hidden=128)
    cli.add_run_flags(p, out_dir="outputs/idol")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    # IDOL / GGD-specific (defaults from the reference repo)
    p.add_argument("--proj_layers", type=int, default=1,
                   help="Number of linear projection layers before the per-node "
                        "group score (GGD's --proj_layers).")
    p.add_argument("--steps_per_epoch", type=int, default=0,
                   help="Snapshots visited per training epoch (0 = ALL of "
                        "them). CLDG and DDGCL both cap this, so leaving IDOL "
                        "uncapped hands it a wildly different optimisation "
                        "budget at the same nominal --epochs: on England COVID "
                        "(42 train snapshots) that is 2x CLDG/DDGCL's 20 steps, "
                        "but on METR-LA (23,974) it is ~1200x. A table built "
                        "that way compares training budgets, not methods. Set "
                        "it to the same value the other two use.")
    p.add_argument("--val_steps", type=int, default=0,
                   help="Val snapshots scored per epoch (0 = all). METR-LA has "
                        "3,425, so an uncapped val pass dominates the epoch.")
    p.add_argument("--node_batch_size", type=int, default=256,
                   help="Active nodes scored per snapshot step.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    args = p.parse_args()

    # IDOL's encoder is ALWAYS its native PPR-propagation GCN, regardless of
    # the --encoder_kind the launcher passes.
    if args.encoder_kind != IDOL_ENCODER_KIND:
        print(f"[setup] --encoder_kind {args.encoder_kind!r} ignored: IDOL "
              f"always uses its native {IDOL_ENCODER_KIND!r} encoder.", flush=True)
        args.encoder_kind = IDOL_ENCODER_KIND

    set_deterministic(args.seed)
    out_dir = setup_output_dir(args.out_dir, args)
    device = torch.device(args.device)
    print(f"[setup] IDOL baseline  data={args.data_source} "
          f"encoder={args.encoder_kind} proj_layers={args.proj_layers}", flush=True)

    train_ds = build_dynamic_dataset(args, split="train")
    train_snaps, val_snaps = split_snapshots(train_ds, "IDOL", args)
    if not args.steps_per_epoch and train_snaps[0].shape[0] > 100:
        print(f"[WARN] --steps_per_epoch 0 on {train_snaps[0].shape[0]} train "
              f"snapshots: IDOL will take that many gradient steps per epoch "
              f"while CLDG/DDGCL take ~20. Do not put this run in a comparison "
              f"table without capping it.", flush=True)
    print(f"[data] train snapshots={train_snaps[0].shape[0]} "
          f"val snapshots={val_snaps[0].shape[0]} N={train_ds.num_nodes} "
          f"F={train_ds.feature_dim}", flush=True)

    encoder = build_encoder(
        IDOL_ENCODER_KIND, in_dim=train_ds.feature_dim,
        hidden_dim=args.encoder_hidden, out_dim=args.node_dim, freeze=False,
        bidirectional=args.bidirectional).to(device)
    model = GGD(encoder, latent_dim=args.node_dim, proj_layers=args.proj_layers).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[model] IDOL/GGD trainable={n_params:,}", flush=True)

    best = train_idol(model, train_snaps, val_snaps, args, device)
    if args.smoke:
        return
    save_encoder_ckpt(out_dir / "encoder.ckpt", model.encoder, args,
                      feature_dim=train_ds.feature_dim,
                      pretrain_objective="idol", best_loss=best,
                      val_loss=best)
    print(f"[done] best val ggd_bce = {best:.4f}  -> {out_dir/'encoder.ckpt'}\n"
          f"[eval] probe it with: python -m src.train.analyze_jepa "
          f"--skip_diagnostic --link_sources encoder --encoder_ckpt {out_dir/'encoder.ckpt'} "
          f"<same --data_* flags>", flush=True)


if __name__ == "__main__":
    main()
