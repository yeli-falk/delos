"""C-JEPA training on METR-LA (dynamic-graph, fixed-topology variant).

This is the simplest possible extension of C-JEPA to the spatiotemporal /
dynamic-graph setting:

  raw signals (B, T, N, F)
      |  frozen node encoder (linear MLP or 2-layer GCN)
      v
  slot tensor (B, T, N, D)
      |  MaskedNodePredictor with masked-history + forward-prediction loss
      v
  predicted slots (B, T, N, D) -> latent JEPA loss vs encoded targets
                              -> linear probe -> speed MAE/RMSE at 15/30/60 min

Usage:
    python src/train/train_cjepa_metr_la.py --data_dir dataset/metr_la
    python src/train/train_cjepa_metr_la.py --synthetic   # no data needed

The script intentionally avoids stable_pretraining/Lightning: METR-LA is
small enough that a plain PyTorch loop is cleaner to iterate on.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.cjepa_graph_predictor import MaskedNodePredictor, compute_graph_jepa_loss
from src.custom_codes.metr_la_dataset import MetrLADataset
from src.world_models.graph_jepa_encoder import (
    LinearProbe,
    build_encoder,
    normalize_adjacency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def encode_window(encoder, x, adj_arg):
    """Run the frozen encoder on a (B, T, N, F) tensor.

    `adj_arg` is the second positional arg the encoder expects: a normalised
    (N, N) adjacency for the GCN, ignored by the linear encoder.
    """
    with torch.no_grad():
        z = encoder(x, adj_arg)
    return z


def speed_metrics(pred_speed, true_speed, valid_mask):
    """MAE / RMSE / MAPE on de-normalised speeds, ignoring missing entries."""
    diff = pred_speed - true_speed
    mae = (diff.abs() * valid_mask).sum() / valid_mask.sum().clamp(min=1)
    rmse = (((diff ** 2) * valid_mask).sum() / valid_mask.sum().clamp(min=1)).sqrt()
    mape = (
        ((diff.abs() / true_speed.abs().clamp(min=1.0)) * valid_mask).sum()
        / valid_mask.sum().clamp(min=1)
    )
    return mae.item(), rmse.item(), mape.item()


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def cosine_warmup_lr(step, warmup_steps, total_steps, min_ratio=0.05):
    """Linear warmup from 0 to peak over `warmup_steps`, then cosine decay
    from peak to `min_ratio * peak` by `total_steps`. Returns the LR multiplier
    in [0, 1]."""
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine

def train_one_epoch(predictor, encoder, loader, optimizer, adj_dense, adj_norm,
                    history_size, device, log_every=50, max_steps=None,
                    scheduler=None):
    predictor.train()
    log = {"loss": 0.0, "loss_masked_history": 0.0, "loss_future": 0.0, "n": 0}
    t0 = time.time()

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x_hist = batch["x_hist"].to(device, non_blocking=True)
        x_fut = batch["x_fut"].to(device, non_blocking=True)
        full = torch.cat([x_hist, x_fut], dim=1)        # (B, T_total, N, F)

        z_hist = encode_window(encoder, x_hist, adj_norm)            # (B, T_hist, N, D)
        z_target = encode_window(encoder, full, adj_norm)            # (B, T_total, N, D)

        B, T_total, N, _ = z_target.shape
        adj_bool = adj_dense.bool().unsqueeze(0).unsqueeze(0).expand(B, T_total, N, N)
        active = torch.ones(B, T_total, N, dtype=torch.bool, device=device)

        pred, masked_idx, _ = predictor(z_hist, adj_bool, active)
        losses = compute_graph_jepa_loss(
            pred, z_target.detach(), active, masked_idx, history_size,
        )

        loss_value = losses["loss"]
        if not torch.isfinite(loss_value):
            print(f"  [step {step+1}] non-finite loss ({loss_value.item()}), skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            print(f"  [step {step+1}] non-finite grad norm, skipping step")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = x_hist.shape[0]
        for k in ("loss", "loss_masked_history", "loss_future"):
            log[k] += losses[k].item() * bs
        log["n"] += bs

        if (step + 1) % log_every == 0:
            n = log["n"]
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [step {step+1:>5}] loss={log['loss']/n:.4f} "
                f"mh={log['loss_masked_history']/n:.4f} "
                f"fut={log['loss_future']/n:.4f} "
                f"lr={cur_lr:.2e} "
                f"({(time.time()-t0):.1f}s)"
            )

    n = max(log["n"], 1)
    return {k: v / n for k, v in log.items() if k != "n"}


@torch.no_grad()
def evaluate_jepa(predictor, encoder, loader, adj_dense, adj_norm, history_size, device):
    predictor.eval()
    total = {"loss": 0.0, "loss_masked_history": 0.0, "loss_future": 0.0, "n": 0}
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        x_fut = batch["x_fut"].to(device)
        full = torch.cat([x_hist, x_fut], dim=1)
        z_hist = encoder(x_hist, adj_norm)
        z_target = encoder(full, adj_norm)
        B, T_total, N, _ = z_target.shape
        adj_bool = adj_dense.bool().unsqueeze(0).unsqueeze(0).expand(B, T_total, N, N)
        active = torch.ones(B, T_total, N, dtype=torch.bool, device=device)

        pred, masked_idx, _ = predictor(z_hist, adj_bool, active)
        losses = compute_graph_jepa_loss(pred, z_target, active, masked_idx, history_size)
        bs = x_hist.shape[0]
        for k in ("loss", "loss_masked_history", "loss_future"):
            total[k] += losses[k].item() * bs
        total["n"] += bs
    n = max(total["n"], 1)
    return {k: v / n for k, v in total.items() if k != "n"}


# ---------------------------------------------------------------------------
# Linear-probe eval: predicted-latents -> speed MAE
# ---------------------------------------------------------------------------
def fit_linear_probe(predictor, encoder, train_loader, adj_dense, adj_norm,
                     history_size, num_preds, device, epochs=3, lr=1e-3,
                     hidden_dim=0):
    probe = LinearProbe(in_dim=predictor.node_dim, hidden_dim=hidden_dim).to(device)
    n_probe = sum(p.numel() for p in probe.parameters())
    head = "linear" if hidden_dim == 0 else f"MLP h={hidden_dim}"
    print(f"  [probe] {head} ({n_probe:,} params)")
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    predictor.eval()

    mean = train_loader.dataset.mean
    std = train_loader.dataset.std
    for epoch in range(epochs):
        losses = 0.0
        n = 0
        for batch in train_loader:
            x_hist = batch["x_hist"].to(device)
            speeds = batch["speeds"].to(device)             # (B, T_total, N) un-normalised
            B = x_hist.shape[0]
            T_total = history_size + num_preds
            adj_bool = adj_dense.bool().unsqueeze(0).unsqueeze(0).expand(
                B, T_total, predictor.num_nodes, predictor.num_nodes,
            )
            active = torch.ones(
                B, T_total, predictor.num_nodes, dtype=torch.bool, device=device,
            )
            with torch.no_grad():
                z_hist = encoder(x_hist, adj_norm)
                pred = predictor.inference(z_hist, adj_bool, active)  # (B, T_pred, N, D)
            future_speeds_norm = (speeds[:, history_size:] - mean) / std
            pred_speed_norm = probe(pred)
            loss = F.mse_loss(pred_speed_norm, future_speeds_norm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses += loss.item() * B
            n += B
        print(f"  [probe epoch {epoch+1}] mse={losses/max(n,1):.4f}")
    return probe


@torch.no_grad()
def evaluate_speed(predictor, encoder, probe, loader, adj_dense, adj_norm,
                   history_size, num_preds, device, horizons=(2, 5, 11)):
    """Report MAE/RMSE/MAPE at several future horizons (in steps).

    Default horizons = (2, 5, 11) ~ (15, 30, 60) min on METR-LA at 5-min steps,
    counted from t = T_hist (so horizon h means step T_hist + h).
    """
    predictor.eval()
    probe.eval()
    mean = loader.dataset.mean
    std = loader.dataset.std

    sums = {h: {"abs": 0.0, "sq": 0.0, "ape": 0.0, "n": 0.0} for h in horizons}
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        speeds = batch["speeds"].to(device)
        B, T_hist, N, _ = x_hist.shape
        T_total = history_size + num_preds
        z_hist = encoder(x_hist, adj_norm)
        adj_bool = adj_dense.bool().unsqueeze(0).unsqueeze(0).expand(B, T_total, N, N)
        active = torch.ones(B, T_total, N, dtype=torch.bool, device=device)
        pred = predictor.inference(z_hist, adj_bool, active)            # (B, T_pred, N, D)
        pred_speed = probe(pred) * std + mean                            # (B, T_pred, N)

        for h in horizons:
            if h >= num_preds:
                continue
            true = speeds[:, history_size + h]                           # (B, N)
            est = pred_speed[:, h]
            valid = (true > 0).float()
            diff = est - true
            sums[h]["abs"] += (diff.abs() * valid).sum().item()
            sums[h]["sq"] += ((diff ** 2) * valid).sum().item()
            sums[h]["ape"] += (
                (diff.abs() / true.abs().clamp(min=1.0)) * valid
            ).sum().item()
            sums[h]["n"] += valid.sum().item()

    out = {}
    for h, s in sums.items():
        if s["n"] == 0:
            continue
        out[h] = {
            "MAE": s["abs"] / s["n"],
            "RMSE": (s["sq"] / s["n"]) ** 0.5,
            "MAPE": s["ape"] / s["n"],
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="dataset/metr_la")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--history_size", type=int, default=12)
    p.add_argument("--num_preds", type=int, default=12)
    p.add_argument("--node_dim", type=int, default=64)
    p.add_argument("--encoder_hidden", type=int, default=64)
    p.add_argument("--encoder_kind", choices=["linear", "gcn", "tgcn"], default="gcn")
    p.add_argument("--num_masked_nodes", type=int, default=30)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--allow_temporal_neighbors", action="store_true")
    p.add_argument("--no_attention_mask", action="store_true",
                   help="Skip the (B,T,N,T,N) structural mask. Recommended for "
                        "METR-LA: the 207-node × 24-step mask alone needs ~400MB "
                        "of VRAM at batch 16 and the structural prior is not "
                        "essential for first-pass JEPA training.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=float, default=1.0,
                   help="Linear warmup duration before cosine decay starts. "
                        "Set to 0 to disable warmup; set --no_lr_schedule to "
                        "disable scheduling entirely (constant LR).")
    p.add_argument("--lr_min_ratio", type=float, default=0.05,
                   help="End-of-training LR as a fraction of peak LR.")
    p.add_argument("--no_lr_schedule", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="outputs/metr_la_cjepa")
    p.add_argument("--probe_epochs", type=int, default=3)
    p.add_argument("--probe_hidden", type=int, default=0,
                   help="If >0, use a 2-layer MLP probe with this hidden size "
                        "instead of a 1-layer linear probe. Tests whether the "
                        "JEPA latents carry speed info that a linear readout "
                        "can't extract.")
    p.add_argument("--smoke", action="store_true",
                   help="One-batch dry run to verify wiring.")
    p.add_argument("--resume", default=None,
                   help="Path to a best.ckpt to resume predictor weights from.")
    p.add_argument("--encoder_ckpt", default=None,
                   help="Path to a pretrained encoder.ckpt produced by "
                        "pretrain_encoder_metr_la.py. Loaded before freezing.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[setup] device={args.device} encoder={args.encoder_kind} D={args.node_dim}")

    train_ds = MetrLADataset(
        args.data_dir, "train", args.history_size, args.num_preds,
        synthetic=args.synthetic,
    )
    val_ds = MetrLADataset(
        args.data_dir, "val", args.history_size, args.num_preds,
        synthetic=args.synthetic,
    )
    test_ds = MetrLADataset(
        args.data_dir, "test", args.history_size, args.num_preds,
        synthetic=args.synthetic,
    )
    # Share the train statistics so val/test are normalised consistently.
    val_ds.mean = train_ds.mean
    val_ds.std = train_ds.std
    test_ds.mean = train_ds.mean
    test_ds.std = train_ds.std
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"N={train_ds.num_nodes} F={train_ds.feature_dim}")

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    device = torch.device(args.device)
    adj_dense = torch.from_numpy(train_ds.adj).to(device)
    adj_norm = normalize_adjacency(adj_dense)

    # If --encoder_ckpt is supplied, peek at its metadata first so we build
    # the right architecture and dims regardless of the CLI flags. Otherwise
    # fall back to the CLI values.
    enc_kind = args.encoder_kind
    enc_hidden = args.encoder_hidden
    enc_out = args.node_dim
    ckpt = None
    if args.encoder_ckpt:
        ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
        enc_kind = ckpt.get("encoder_kind", enc_kind)
        enc_hidden = ckpt.get("encoder_hidden", enc_hidden)
        enc_out = ckpt.get("node_dim", enc_out)
        if enc_kind != args.encoder_kind:
            print(f"[encoder] overriding --encoder_kind={args.encoder_kind} "
                  f"with '{enc_kind}' from checkpoint")
        if enc_out != args.node_dim:
            print(f"[encoder] overriding --node_dim={args.node_dim} "
                  f"with {enc_out} from checkpoint")

    encoder = build_encoder(
        enc_kind,
        in_dim=train_ds.feature_dim,
        hidden_dim=enc_hidden,
        out_dim=enc_out,
        freeze=False,
    ).to(device)
    if ckpt is not None:
        encoder.load_state_dict(ckpt["encoder"])
        print(f"[encoder] loaded {enc_kind} weights from {args.encoder_ckpt} "
              f"(epoch={ckpt.get('epoch','?')} val_loss={ckpt.get('val_loss','?'):.4f})")
    else:
        print(f"[encoder] random init {enc_kind} (no --encoder_ckpt provided)")
    args.node_dim = enc_out  # downstream uses (predictor, probe) must match
    encoder.eval()
    for p_ in encoder.parameters():
        p_.requires_grad_(False)

    predictor = MaskedNodePredictor(
        num_nodes=train_ds.num_nodes,
        node_dim=args.node_dim,
        history_frames=args.history_size,
        pred_frames=args.num_preds,
        num_masked_nodes=args.num_masked_nodes,
        seed=args.seed,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.node_dim // args.heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        allow_temporal_neighbors=args.allow_temporal_neighbors,
        use_attention_mask=not args.no_attention_mask,
    ).to(device)

    n_pred_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    n_enc_params = sum(p.numel() for p in encoder.parameters())
    print(f"[model] predictor params (trainable)={n_pred_params:,} "
          f"encoder params (frozen)={n_enc_params:,}")

    opt = torch.optim.AdamW(
        predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    if args.no_lr_schedule or args.epochs == 0:
        scheduler = None
        print(f"[lr] constant peak={args.lr:.2e} (no schedule)")
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda s: cosine_warmup_lr(
                s, warmup_steps, total_steps, args.lr_min_ratio,
            ),
        )
        print(f"[lr] cosine schedule peak={args.lr:.2e} "
              f"warmup={warmup_steps} steps ({args.warmup_epochs} ep) "
              f"total={total_steps} steps min_ratio={args.lr_min_ratio}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        predictor.load_state_dict(ckpt["predictor"])
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        print(f"[resume] loaded predictor from {args.resume} "
              f"(epoch={ckpt.get('epoch','?')} val_loss={ckpt.get('val_loss','?')})")

    if args.smoke:
        print("[smoke] 5 train steps + 2 val batches")
        epoch_log = train_one_epoch(
            predictor, encoder,
            DataLoader(train_ds, batch_size=2, shuffle=False),
            opt, adj_dense, adj_norm, args.history_size, device,
            log_every=1, max_steps=5,
        )
        print(f"  smoke train loss = {epoch_log['loss']:.4f}")
        val_loader_smoke = DataLoader(val_ds, batch_size=2, shuffle=False)
        val_log = {"loss": 0.0, "n": 0}
        with torch.no_grad():
            predictor.eval()
            for i, batch in enumerate(val_loader_smoke):
                if i >= 2:
                    break
                x_hist = batch["x_hist"].to(device)
                x_fut = batch["x_fut"].to(device)
                full = torch.cat([x_hist, x_fut], dim=1)
                z_hist = encoder(x_hist, adj_norm)
                z_target = encoder(full, adj_norm)
                B, T_total, N, _ = z_target.shape
                adj_bool = adj_dense.bool().unsqueeze(0).unsqueeze(0).expand(B, T_total, N, N)
                active = torch.ones(B, T_total, N, dtype=torch.bool, device=device)
                pred, masked_idx, _ = predictor(z_hist, adj_bool, active)
                losses = compute_graph_jepa_loss(
                    pred, z_target, active, masked_idx, args.history_size,
                )
                val_log["loss"] += losses["loss"].item() * B
                val_log["n"] += B
        print(f"  smoke val loss   = {val_log['loss']/max(val_log['n'],1):.4f}")
        return

    best_val = float("inf")
    for epoch in range(args.epochs):
        print(f"[epoch {epoch+1}/{args.epochs}]")
        train_log = train_one_epoch(
            predictor, encoder, train_loader, opt,
            adj_dense, adj_norm, args.history_size, device,
            scheduler=scheduler,
        )
        val_log = evaluate_jepa(
            predictor, encoder, val_loader,
            adj_dense, adj_norm, args.history_size, device,
        )
        print(
            f"  train loss={train_log['loss']:.4f}  "
            f"val loss={val_log['loss']:.4f}  "
            f"(mh={val_log['loss_masked_history']:.4f} "
            f"fut={val_log['loss_future']:.4f})"
        )
        if val_log["loss"] < best_val:
            best_val = val_log["loss"]
            torch.save(
                {
                    "predictor": predictor.state_dict(),
                    "encoder": encoder.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "args": vars(args),
                },
                out_dir / "best.ckpt",
            )

    # Always evaluate the best-val checkpoint, not the (possibly diverged)
    # final-epoch predictor. Without this, a training run that bottoms at
    # epoch 4 and then climbs to epoch 20 reports the epoch-20 MAE, which is
    # essentially the random baseline.
    best_path = out_dir / "best.ckpt"
    if args.epochs > 0 and best_path.exists():
        ck = torch.load(best_path, map_location=device, weights_only=False)
        predictor.load_state_dict(ck["predictor"])
        print(f"[probe] reloaded best.ckpt for eval "
              f"(epoch={ck.get('epoch','?')} val_loss={ck.get('val_loss','?'):.4f})")

    print("[probe] fitting probe on train embeddings -> speed")
    probe = fit_linear_probe(
        predictor, encoder, train_loader, adj_dense, adj_norm,
        args.history_size, args.num_preds, device, epochs=args.probe_epochs,
        hidden_dim=args.probe_hidden,
    )
    metrics = evaluate_speed(
        predictor, encoder, probe, test_loader, adj_dense, adj_norm,
        args.history_size, args.num_preds, device,
    )
    print("[test] horizon-step  MAE  RMSE  MAPE")
    for h, m in metrics.items():
        print(f"  +{h:>2} steps   {m['MAE']:.3f}  {m['RMSE']:.3f}  {m['MAPE']:.3f}")
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
