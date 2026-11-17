"""Train the SG-JEPA SSL baseline, then save a standard encoder checkpoint.

SG-JEPA (see ``sgjepa.py``) trains the shared GNN encoder through a spiking
context encoder that predicts the GNN's stop-gradient latents on the window's
LAST snapshot. This script trains on the TRAIN portion of the snapshot
sequence, selects on the val-window loss, and saves ``encoder.ckpt`` in the
``pretrain_encoder.py`` schema so the encoder is scored through the SAME
frozen probes as every other baseline:

    # 1. pre-train
    python -m src.baselines.sgjepa.train_sgjepa --data_source jodie_mooc \
        --encoder_kind gcn --encoder_hidden 128 --node_dim 128 \
        --epochs 100 --device cuda --out_dir outputs/sgjepa_mooc

    # 2. frozen-encoder probe, identical to every other baseline
    python -m src.train.analyze_jepa --skip_diagnostic --link_sources encoder \
        --data_source jodie_mooc --encoder_ckpt outputs/sgjepa_mooc/encoder.ckpt

Windows: ``--window`` context snapshots (default = ``--history_size``) followed
by ONE target snapshot, i.e. the paper's ``w = window + 1``. The best-val-loss
epoch is restored, like every other trainer in the pipeline. The encoder
should be per-snapshot (``gcn`` / ``gcn_root``): a window encoder that mixes
along time would let the target snapshot leak into the context latents.
Supports the global dynamic-graph datasets (dblp / jodie_* / reddit_* / cldg_*
/ england_covid) and the fixed-topology signal datasets (metr_la).
"""
from __future__ import annotations

import argparse
import contextlib
import time

import numpy as np
import torch

from src.baselines.common import BestValTracker, split_snapshots
from src.baselines.sgjepa.sgjepa import SGJEPA
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.engine import save_encoder_ckpt, set_deterministic, setup_output_dir
from src.world_models.delos_encoder import build_encoder

PER_SNAPSHOT_KINDS = ("gcn", "gcn_root", "linear", "ppr_gcn")


def _window_length(T: int, want: int, what: str) -> int:
    L = min(want, T - 1)                        # L context + 1 target must fit
    if L < want:
        print(f"[warn] {what}: only {T} snapshots; context window shrunk from "
              f"{want} to {L}.", flush=True)
    return L


def run_epoch(model, snaps, L, args, device, rng, n_steps, train: bool, opt=None):
    """``n_steps`` windows of ``L + 1`` snapshots sampled from ``snaps``.
    ``train=False`` scores the identical objective under ``no_grad`` with the
    caller's fixed-seed ``rng``, so the val estimate is comparable across
    epochs."""
    adj, active, feats = snaps
    T = adj.shape[0]
    agg = {"loss": 0.0, "spike_rate": 0.0}
    n = 0
    for _ in range(n_steps):
        s = int(rng.integers(0, T - L))
        adj_w = np.stack([np.asarray(adj[t]) for t in range(s, s + L + 1)])
        gen = torch.Generator(device=device).manual_seed(int(rng.integers(2**31 - 1)))
        with contextlib.nullcontext() if train else torch.no_grad():
            loss, parts = model.window_loss(
                np.asarray(feats[s:s + L + 1]), adj_w, np.asarray(active[s:s + L + 1]),
                device, gen, node_batch=args.node_batch_size)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        agg["loss"] += float(loss.detach())
        agg["spike_rate"] += parts["spike_rate"]
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def train_sgjepa(model, train_snaps, val_snaps, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    want = args.window if args.window > 0 else args.history_size
    if train_snaps[0].shape[0] < 2 or val_snaps[0].shape[0] < 2:
        raise SystemExit("SG-JEPA needs >= 2 train and >= 2 val snapshots (a "
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
        print(f"[epoch {epoch+1}/{n_epochs}] nce={tr['loss']:.4f} "
              f"(spike_rate={tr['spike_rate']:.3f}) val_nce={va['loss']:.4f} "
              f"(spike_rate={va['spike_rate']:.3f})"
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
    cli.add_encoder_flags(p, default_kind="gcn", node_dim=128, encoder_hidden=128)
    cli.add_run_flags(p, out_dir="outputs/sgjepa")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    # SG-JEPA-specific. The paper states neither its temperature, similarity
    # nor optimiser settings; these are the choices, all overridable.
    p.add_argument("--snn_layers", type=int, default=1,
                   help="Spiking layers (Linear + PLIF) in the context encoder.")
    p.add_argument("--tau", type=float, default=0.1,
                   help="InfoNCE temperature on cosine similarity (not given in the paper).")
    p.add_argument("--init_tau", type=float, default=2.0,
                   help="PLIF initial membrane time constant (Fang et al.: 2.0).")
    p.add_argument("--v_th", type=float, default=1.0, help="PLIF firing threshold.")
    p.add_argument("--no_matryoshka", action="store_true",
                   help="Train only the full-prefix prediction instead of every "
                        "prefix length.")
    p.add_argument("--no_current_norm", action="store_true",
                   help="Drop the LayerNorm on each spiking layer's synaptic "
                        "current. Without it the shared GCN's latents never "
                        "reach the threshold and no neuron fires.")
    p.add_argument("--window", type=int, default=0,
                   help="Context snapshots per window (paper w - 1). 0 = "
                        "--history_size, the length the encoder is probed on.")
    p.add_argument("--node_batch_size", type=int, default=256,
                   help="Target nodes per window (also the InfoNCE negatives).")
    p.add_argument("--steps_per_epoch", type=int, default=20,
                   help="Training windows per epoch (same budget as the other baselines).")
    p.add_argument("--val_steps", type=int, default=10,
                   help="Val windows scored per epoch (fixed seed).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    p.add_argument("--weight_decay", type=float, default=0.0)
    args = p.parse_args()

    set_deterministic(args.seed)
    out_dir = setup_output_dir(args.out_dir, args)
    device = torch.device(args.device)
    print(f"[setup] SG-JEPA baseline  data={args.data_source} "
          f"encoder={args.encoder_kind} snn_layers={args.snn_layers} tau={args.tau} "
          f"init_tau={args.init_tau} v_th={args.v_th} "
          f"matryoshka={not args.no_matryoshka} current_norm={not args.no_current_norm} "
          f"window={args.window or args.history_size}", flush=True)
    if args.encoder_kind not in PER_SNAPSHOT_KINDS:
        print(f"[warn] SG-JEPA's target is the LAST snapshot's latent under "
              f"stop-gradient; '{args.encoder_kind}' mixes along the window, so "
              f"the context latents can see the target snapshot. Use a "
              f"per-snapshot kind ({', '.join(PER_SNAPSHOT_KINDS)}) for the "
              f"paper's setting.", flush=True)

    train_ds = build_dynamic_dataset(args, split="train")
    train_snaps, val_snaps = split_snapshots(train_ds, "SG-JEPA", args)
    print(f"[data] train snapshots={train_snaps[0].shape[0]} "
          f"val snapshots={val_snaps[0].shape[0]} N={train_ds.num_nodes} "
          f"F={train_ds.feature_dim}", flush=True)

    encoder = build_encoder(
        args.encoder_kind, in_dim=train_ds.feature_dim,
        hidden_dim=args.encoder_hidden, out_dim=args.node_dim, freeze=False,
        bidirectional=args.bidirectional).to(device)
    model = SGJEPA(encoder, latent_dim=args.node_dim, snn_layers=args.snn_layers,
                   tau=args.tau, init_tau=args.init_tau, v_th=args.v_th,
                   matryoshka=not args.no_matryoshka,
                   current_norm=not args.no_current_norm).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[model] SG-JEPA trainable={n_params:,}", flush=True)

    best = train_sgjepa(model, train_snaps, val_snaps, args, device)
    if args.smoke:
        return
    ckpt_path = out_dir / "encoder.ckpt"
    save_encoder_ckpt(ckpt_path, model.encoder, args,
                      feature_dim=train_ds.feature_dim,
                      pretrain_objective="sgjepa", best_loss=best, val_loss=best)
    print(f"[done] best val nce = {best:.4f}  -> {ckpt_path}\n"
          f"[eval] probe it with: python -m src.train.analyze_jepa "
          f"--skip_diagnostic --link_sources encoder --encoder_ckpt {ckpt_path} "
          f"<same --data_* flags>", flush=True)


if __name__ == "__main__":
    main()
