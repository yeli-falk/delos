"""Train the DySubC baseline, then save a standard encoder checkpoint.

DySubC is encoder-only self-supervised learning (see ``dysubc.py``): each node
is contrasted with the influence-weighted summary of its temporal subgraph
(structural negative: another node's subgraph; temporal negative: the same
subgraph encoded without its time weights). This script trains the shared GNN
encoder on the TRAIN portion of the snapshot sequence, then saves
``encoder.ckpt`` in the same schema as ``pretrain_encoder.py`` so it evaluates
through the identical probe:

    # 1. train the DySubC encoder
    python -m src.baselines.dysubc.train_dysubc --data_source dblp \
        --data_path dataset/dblp/edges.csv \
        --encoder_kind gcn --encoder_hidden 128 --node_dim 128 \
        --subgraph_size 20 --alpha 5 --beta 1 --epochs 100 --device cuda \
        --out_dir outputs/dysubc_dblp

    # 2. evaluate with the SAME link-prediction probe as DELOS
    python -m src.train.analyze_jepa --skip_diagnostic --link_sources encoder \
        --data_source dblp --data_path dataset/dblp/edges.csv \
        --encoder_ckpt outputs/dysubc_dblp/encoder.ckpt

The dynamic regime: each step samples an anchor snapshot ``t`` from the range,
builds the time-weighted temporal graph from the ``--lookback`` snapshots
ending at ``t`` (default: the probe's ``--history_size``, so the encoder is
trained on the time structure of exactly the window it is probed on), samples
``--node_batch_size`` central nodes present in that window and scores their
temporal subgraphs. Supports the global dynamic-graph datasets (dblp / jodie /
reddit / england_covid) and fixed-topology signal datasets (metr_la; there
every edge shares the latest time, so the sampling score reduces to degree and
the readout weights to ``1 + beta / dist``). The encoder consumes the
pipeline's node features, so its input matches DELOS exactly — only the
SSL objective differs.
"""
from __future__ import annotations

import argparse
import contextlib
import time

import numpy as np
import torch

from src.baselines.common import BestValTracker, split_snapshots
from src.baselines.dysubc.dysubc import DySubC, build_temporal_graph
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.engine import save_encoder_ckpt, set_deterministic, setup_output_dir
from src.world_models.delos_encoder import build_encoder


def run_epoch(model, snaps, args, device, rng, n_steps, train: bool, opt=None):
    """One pass of ``n_steps`` subgraph-contrast steps over ``snaps``.

    ``train=False`` scores the identical objective under ``no_grad`` on the
    val snapshots (fixed-seed ``rng`` at the call site, so every epoch is
    scored on the SAME anchors / centres / shuffles).
    """
    adj, active, feats = snaps
    T = adj.shape[0]
    lookback = max(1, min(args.lookback, T))
    agg, agg1, agg2, n = 0.0, 0.0, 0.0, 0
    for _ in range(n_steps):
        # Anchor snapshot with a full lookback window behind it.
        t = int(rng.integers(lookback - 1, T))
        g = build_temporal_graph(adj, t, lookback, device)
        present = np.flatnonzero(g.present.cpu().numpy())
        if present.size < 2:
            continue
        if present.size > args.node_batch_size:
            present = rng.choice(present, size=args.node_batch_size, replace=False)
        centres = torch.from_numpy(present).long().to(device)
        perm = torch.from_numpy(rng.permutation(present.size)).long().to(device)

        with contextlib.nullcontext() if train else torch.no_grad():
            loss, l1, l2 = model.contrastive_step(g, feats[t], centres, perm,
                                                  device)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        agg += float(loss.detach()); agg1 += float(l1.detach())
        agg2 += float(l2.detach()); n += 1
        del g
    d = max(n, 1)
    return agg / d, agg1 / d, agg2 / d


def train_dysubc(model, train_snaps, val_snaps, args, device):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    if train_snaps[0].shape[0] < 1:
        raise SystemExit("DySubC needs >=1 train snapshot.")
    if val_snaps[0].shape[0] < 1:
        raise SystemExit(
            "DySubC needs >=1 val snapshot for model selection; bin the "
            "timeline into more steps.")
    tracker = BestValTracker(args.early_stop_patience, args.early_stop_min_delta)

    n_epochs = 1 if args.smoke else args.epochs
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        n_steps = 3 if args.smoke else args.steps_per_epoch
        train_loss, tr1, tr2 = run_epoch(model, train_snaps, args, device, rng,
                                         n_steps, train=True, opt=opt)
        if args.smoke:
            print(f"[smoke] ok loss={train_loss:.4f}"); return float("inf")

        model.eval()
        val_loss, v1, v2 = run_epoch(model, val_snaps, args, device,
                                     np.random.default_rng(args.seed + 10_000),
                                     args.val_steps, train=False)
        improved, stop = tracker.update(val_loss, model.encoder)
        print(f"[epoch {epoch+1}/{n_epochs}] loss={train_loss:.4f} "
              f"(struct={tr1:.4f} time={tr2:.4f}) val_loss={val_loss:.4f} "
              f"(struct={v1:.4f} time={v2:.4f})"
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
    cli.add_dblp_synthetic_flags(p)
    add_dataset_cli_flags(p)
    cli.add_window_flags(p)                       # dataset construction + lookback default
    cli.add_encoder_flags(p, default_kind="gcn", node_dim=128, encoder_hidden=128)
    cli.add_run_flags(p, out_dir="outputs/dysubc")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    # DySubC-specific (defaults from the paper; code values noted)
    p.add_argument("--subgraph_size", type=int, default=20,
                   help="k: nodes per temporal subgraph incl. the centre "
                        "(paper: 20; >=20 saturates, 10 is clearly worse).")
    p.add_argument("--alpha", type=float, default=5.0,
                   help="Sampling score S = S_time + alpha * degree (Eq. 2). "
                        "Time is on a [0, 10] scale; the released code uses "
                        "alpha=5 on MOOC, 1 on Wikipedia, 10 on bitcoin-alpha.")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Readout influence Inf_j = tau_j + beta / Dist(i, j) "
                        "(Eq. 7). Released code fixes beta=1; the paper's "
                        "sweep peaks at 1.6 on fb-forum.")
    p.add_argument("--margin", type=float, default=0.75,
                   help="Margin of both triplet losses (paper: phi = varphi = "
                        "0.75; released code 0.5).")
    p.add_argument("--lam", type=float, default=0.5,
                   help="Weight of the temporal loss: L = L1 + lam * L2 (Eq. 13).")
    p.add_argument("--lookback", type=int, default=0,
                   help="Snapshots forming the temporal graph at each anchor "
                        "(edge weight = normalised latest snapshot). 0 = "
                        "--history_size, the window the probe hands the encoder.")
    p.add_argument("--node_batch_size", type=int, default=256,
                   help="Central nodes sampled per step (from the nodes with "
                        ">= 1 edge in the window). The reference is "
                        "full-batch (all nodes per step).")
    p.add_argument("--steps_per_epoch", type=int, default=10)
    p.add_argument("--val_steps", type=int, default=10,
                   help="Steps used to estimate the val loss each epoch "
                        "(fixed seed, so the estimate is comparable across "
                        "epochs).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate (paper: 0.001).")
    p.add_argument("--weight_decay", type=float, default=0.0)
    args = p.parse_args()
    if args.lookback <= 0:
        args.lookback = args.history_size

    set_deterministic(args.seed)
    out_dir = setup_output_dir(args.out_dir, args)
    device = torch.device(args.device)
    print(f"[setup] DySubC baseline  data={args.data_source} "
          f"encoder={args.encoder_kind} k={args.subgraph_size} "
          f"alpha={args.alpha} beta={args.beta} margin={args.margin} "
          f"lam={args.lam} lookback={args.lookback}", flush=True)

    train_ds = build_dynamic_dataset(args, split="train")
    train_snaps, val_snaps = split_snapshots(train_ds, "DySubC", args)
    print(f"[data] train snapshots={train_snaps[0].shape[0]} "
          f"val snapshots={val_snaps[0].shape[0]} N={train_ds.num_nodes} "
          f"F={train_ds.feature_dim}", flush=True)
    if args.lookback == 1:
        print("[note] lookback=1: every edge shares one time level, so the "
              "sampling score reduces to alpha*degree and the temporal "
              "negative differs from the positive only through E2's weights.",
              flush=True)

    def make_encoder():
        return build_encoder(
            args.encoder_kind, in_dim=train_ds.feature_dim,
            hidden_dim=args.encoder_hidden, out_dim=args.node_dim, freeze=False,
            bidirectional=args.bidirectional).to(device)

    # E1 (time-weighted; kept) and E2 (unweighted; temporal negative only) —
    # separate weights, as in Eq. 4.
    model = DySubC(make_encoder(), make_encoder(),
                   subgraph_size=args.subgraph_size, alpha=args.alpha,
                   beta=args.beta, margin_struct=args.margin,
                   margin_time=args.margin, lam=args.lam).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[model] DySubC trainable={n_params:,}", flush=True)

    best = train_dysubc(model, train_snaps, val_snaps, args, device)
    if args.smoke:
        return
    ckpt_path = out_dir / "encoder.ckpt"
    save_encoder_ckpt(ckpt_path, model.encoder, args,
                      feature_dim=train_ds.feature_dim,
                      pretrain_objective="dysubc", best_loss=best,
                      val_loss=best)
    print(f"[done] best val loss = {best:.4f}  -> {ckpt_path}\n"
          f"[eval] probe it with: python -m src.train.analyze_jepa "
          f"--skip_diagnostic --link_sources encoder --encoder_ckpt {ckpt_path} "
          f"<same --data_* flags>", flush=True)


if __name__ == "__main__":
    main()
