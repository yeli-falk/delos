"""Train the DyG2Vec SSL baseline, then save a standard encoder checkpoint.

DyG2Vec (see ``dyg2vec.py``) pre-trains its window-based temporal attention
encoder (``--encoder_kind dyg2vec``) with VICReg on two distorted views (edge
dropout + structural-feature masking) of each history window. This script
trains on the TRAIN portion of the snapshot sequence, selects on the val-window
loss, and saves ``encoder.ckpt`` in the ``pretrain_encoder.py`` schema so the
encoder is scored through the SAME frozen probes as every other baseline:

    # 1. pre-train (paper: p_d = 0.3, lambda = mu = 25, nu = 1, lr 1e-4, 100 epochs)
    python -m src.baselines.dyg2vec.train_dyg2vec --data_source jodie_mooc \
        --encoder_kind dyg2vec --encoder_hidden 128 --node_dim 128 \
        --epochs 100 --device cuda --out_dir outputs/dyg2vec_mooc

    # 2. frozen-encoder probe, identical to every other baseline
    python -m src.train.analyze_jepa --skip_diagnostic --link_sources encoder \
        --data_source jodie_mooc --encoder_ckpt outputs/dyg2vec_mooc/encoder.ckpt

Windows: ``--window`` consecutive snapshots form one history graph (default =
``--history_size``, so the encoder is pre-trained on windows of the length it
is probed on; the paper's SSL windows are 32K interactions with stride 200).
The paper keeps the LAST SSL epoch; here the best-val-loss epoch is restored,
like every other trainer in the pipeline. Supports the global dynamic-graph
datasets (dblp / jodie_* / reddit_* / england_covid) and the fixed-topology
signal datasets (metr_la) -- on a constant graph every snapshot has the same
interactions and the time encoding alone separates them.
"""
from __future__ import annotations

import argparse
import contextlib
import time

import numpy as np
import torch

from src.baselines.common import BestValTracker, split_snapshots
from src.baselines.dyg2vec.dyg2vec import DyG2VecSSL
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.engine import save_encoder_ckpt, set_deterministic, setup_output_dir
from src.world_models.delos_encoder import build_encoder


def _window_length(T: int, want: int, what: str) -> int:
    L = min(want, T)
    if L < want:
        print(f"[warn] {what}: only {T} snapshots; window shrunk from {want} "
              f"to {L}.", flush=True)
    return L


def run_epoch(model, snaps, L, args, device, rng, n_steps, train: bool, opt=None):
    """``n_steps`` windows of ``L`` snapshots sampled from ``snaps``.
    ``train=False`` scores the identical objective under ``no_grad`` with the
    caller's fixed-seed ``rng`` (same windows, same distortions), so the val
    estimate is comparable across epochs."""
    adj, active, feats = snaps
    T = adj.shape[0]
    agg = {"loss": 0.0, "inv": 0.0, "var": 0.0, "cov": 0.0}
    n = 0
    for _ in range(n_steps):
        s = int(rng.integers(0, T - L + 1))
        adj_w = np.stack([np.asarray(adj[t]) for t in range(s, s + L)])
        feats_w = np.asarray(feats[s:s + L])
        gen = torch.Generator(device=device).manual_seed(int(rng.integers(2**31 - 1)))
        with contextlib.nullcontext() if train else torch.no_grad():
            loss, inv, var, cov = model.window_loss(feats_w, adj_w, device, gen)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        for k, v in zip(agg, (loss, inv, var, cov)):
            agg[k] += float(v.detach())
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def train_dyg2vec(model, train_snaps, val_snaps, args, device):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    want = args.window if args.window > 0 else args.history_size
    if train_snaps[0].shape[0] < 1 or val_snaps[0].shape[0] < 1:
        raise SystemExit("DyG2Vec needs >= 1 train and >= 1 val snapshot; bin "
                         "the timeline into more steps.")
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
        print(f"[epoch {epoch+1}/{n_epochs}] vicreg={tr['loss']:.4f} "
              f"(inv={tr['inv']:.3f} var={tr['var']:.3f} cov={tr['cov']:.3f}) "
              f"val_vicreg={va['loss']:.4f} (inv={va['inv']:.3f} "
              f"var={va['var']:.3f} cov={va['cov']:.3f})"
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
    p = argparse.ArgumentParser(description=__doc__)
    cli.add_dblp_synthetic_flags(p)
    add_dataset_cli_flags(p)
    cli.add_window_flags(p)          # dataset construction + default window length
    cli.add_encoder_flags(p, default_kind="dyg2vec", node_dim=128, encoder_hidden=128)
    cli.add_run_flags(p, out_dir="outputs/dyg2vec")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    # DyG2Vec-specific (paper App. A.1 / A.3 defaults)
    p.add_argument("--p_edge", type=float, default=0.3,
                   help="Edge-dropout probability of each view (paper p_d = 0.3).")
    p.add_argument("--p_struct", type=float, default=0.3,
                   help="Masking probability of the structural edge encodings "
                        "z_p / c_p in each view (paper p_d = 0.3).")
    p.add_argument("--lam", type=float, default=25.0, help="VICReg invariance weight.")
    p.add_argument("--mu", type=float, default=25.0, help="VICReg variance weight.")
    p.add_argument("--nu", type=float, default=1.0, help="VICReg covariance weight.")
    p.add_argument("--gamma", type=float, default=1.0, help="VICReg variance floor.")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout inside the attention layers (paper 0.1).")
    p.add_argument("--window", type=int, default=0,
                   help="Snapshots per history window. 0 = --history_size, so "
                        "the encoder is pre-trained on the length it is probed on.")
    p.add_argument("--steps_per_epoch", type=int, default=20,
                   help="Training windows per epoch (same budget as the other baselines).")
    p.add_argument("--val_steps", type=int, default=10,
                   help="Val windows scored per epoch (fixed seed).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Adam learning rate (paper: constant 1e-4).")
    p.add_argument("--weight_decay", type=float, default=0.0)
    args = p.parse_args()

    set_deterministic(args.seed)
    out_dir = setup_output_dir(args.out_dir, args)
    device = torch.device(args.device)
    print(f"[setup] DyG2Vec baseline  data={args.data_source} "
          f"encoder={args.encoder_kind} p_edge={args.p_edge} p_struct={args.p_struct} "
          f"lam={args.lam} mu={args.mu} nu={args.nu} "
          f"window={args.window or args.history_size}", flush=True)
    if args.encoder_kind != "dyg2vec":
        print(f"[warn] DyG2Vec's encoder is 'dyg2vec' (window temporal attention); "
              f"got '{args.encoder_kind}' -- this is VICReg on that encoder, not "
              f"DyG2Vec. Structural-feature masking has no effect on it.", flush=True)

    train_ds = build_dynamic_dataset(args, split="train")
    train_snaps, val_snaps = split_snapshots(train_ds, "DyG2Vec", args)
    print(f"[data] train snapshots={train_snaps[0].shape[0]} "
          f"val snapshots={val_snaps[0].shape[0]} N={train_ds.num_nodes} "
          f"F={train_ds.feature_dim}", flush=True)

    encoder = build_encoder(
        args.encoder_kind, in_dim=train_ds.feature_dim,
        hidden_dim=args.encoder_hidden, out_dim=args.node_dim, freeze=False,
        bidirectional=args.bidirectional).to(device)
    for layer in getattr(encoder, "layers", []):
        if hasattr(layer, "drop"):
            layer.drop.p = args.dropout
    p_struct = args.p_struct if args.encoder_kind == "dyg2vec" else 0.0
    model = DyG2VecSSL(encoder, latent_dim=args.node_dim, p_edge=args.p_edge,
                       p_struct=p_struct, lam=args.lam, mu=args.mu, nu=args.nu,
                       gamma=args.gamma).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[model] DyG2Vec trainable={n_params:,}", flush=True)

    best = train_dyg2vec(model, train_snaps, val_snaps, args, device)
    if args.smoke:
        return
    ckpt_path = out_dir / "encoder.ckpt"
    save_encoder_ckpt(ckpt_path, model.encoder, args,
                      feature_dim=train_ds.feature_dim,
                      pretrain_objective="dyg2vec", best_loss=best, val_loss=best)
    print(f"[done] best val vicreg = {best:.4f}  -> {ckpt_path}\n"
          f"[eval] probe it with: python -m src.train.analyze_jepa "
          f"--skip_diagnostic --link_sources encoder --encoder_ckpt {ckpt_path} "
          f"<same --data_* flags>", flush=True)


if __name__ == "__main__":
    main()
