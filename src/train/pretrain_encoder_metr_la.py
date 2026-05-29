"""GraphMAE-style encoder pretraining on METR-LA.

Trains the GCN (or Linear) node encoder + a small MLP decoder to reconstruct
randomly masked speed values from their spatial context. After pretraining,
only the encoder weights are saved; the decoder is discarded. The frozen
encoder is then loaded by `train_cjepa_metr_la.py` via `--encoder_ckpt`.

Why bother: the JEPA pipeline with a frozen *random* encoder caps at MAE ~8
on METR-LA. The MLP-probe ablation showed the readout is not the bottleneck
-- the encoder latents simply don't carry enough traffic-specific structure.
A few epochs of masked-speed reconstruction give the encoder that structure
without changing anything else downstream.

Run:
    python src/train/pretrain_encoder_metr_la.py --device cuda --epochs 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.custom_codes.metr_la_dataset import MetrLADataset
from src.world_models.graph_jepa_encoder import (
    EncoderPretrainer,
    build_encoder,
    normalize_adjacency,
)


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=shuffle,
    )


def sample_mask(shape, p, device):
    """Bernoulli mask: True = hidden, train target."""
    return torch.rand(shape, device=device) < p


def reconstruction_loss(pred, target, mask):
    """MSE on masked positions only."""
    diff2 = (pred - target).pow(2)
    weighted = diff2 * mask.float()
    return weighted.sum() / mask.float().sum().clamp(min=1)


def train_one_epoch(model, loader, opt, adj_norm, mask_rate, device, log_every=50):
    model.train()
    log = {"loss": 0.0, "n": 0}
    t0 = time.time()
    for step, batch in enumerate(loader):
        x_hist = batch["x_hist"].to(device, non_blocking=True)
        x_fut = batch["x_fut"].to(device, non_blocking=True)
        x = torch.cat([x_hist, x_fut], dim=1)               # (B, T_tot, N, F)
        speeds = x[..., 0]                                  # un-masked targets
        mask = sample_mask(speeds.shape, mask_rate, device)
        pred = model(x, adj_norm, mask)
        loss = reconstruction_loss(pred, speeds, mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        bs = x.shape[0]
        log["loss"] += loss.item() * bs
        log["n"] += bs
        if (step + 1) % log_every == 0:
            n = log["n"]
            print(f"  [step {step+1:>5}] loss={log['loss']/n:.4f} "
                  f"({(time.time()-t0):.1f}s)")
    return log["loss"] / max(log["n"], 1)


@torch.no_grad()
def evaluate(model, loader, adj_norm, mask_rate, device, seed=42):
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    total, n = 0.0, 0
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        x_fut = batch["x_fut"].to(device)
        x = torch.cat([x_hist, x_fut], dim=1)
        speeds = x[..., 0]
        mask = torch.rand(speeds.shape, generator=rng, device=device) < mask_rate
        pred = model(x, adj_norm, mask)
        total += reconstruction_loss(pred, speeds, mask).item() * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="dataset/metr_la")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--history_size", type=int, default=12)
    p.add_argument("--num_preds", type=int, default=12)
    p.add_argument("--encoder_kind", choices=["linear", "gcn", "tgcn"], default="gcn")
    p.add_argument("--node_dim", type=int, default=128)
    p.add_argument("--encoder_hidden", type=int, default=128)
    p.add_argument("--decoder_hidden", type=int, default=64)
    p.add_argument("--mask_rate", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="outputs/metr_la_encoder")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[setup] device={args.device} encoder={args.encoder_kind} "
          f"D={args.node_dim} mask_rate={args.mask_rate}")

    train_ds = MetrLADataset(args.data_dir, "train", args.history_size,
                             args.num_preds, synthetic=args.synthetic)
    val_ds = MetrLADataset(args.data_dir, "val", args.history_size,
                           args.num_preds, synthetic=args.synthetic)
    val_ds.mean = train_ds.mean
    val_ds.std = train_ds.std
    print(f"[data] train={len(train_ds)} val={len(val_ds)} "
          f"N={train_ds.num_nodes} F={train_ds.feature_dim}")

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)

    device = torch.device(args.device)
    adj_dense = torch.from_numpy(train_ds.adj).to(device)
    adj_norm = normalize_adjacency(adj_dense)

    encoder = build_encoder(
        args.encoder_kind,
        in_dim=train_ds.feature_dim,
        hidden_dim=args.encoder_hidden,
        out_dim=args.node_dim,
        freeze=False,                          # <- key difference vs JEPA path
    ).to(device)

    model = EncoderPretrainer(
        encoder=encoder,
        latent_dim=args.node_dim,
        decoder_hidden=args.decoder_hidden,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"[model] total trainable={n_params:,}  encoder={n_enc:,}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    if args.smoke:
        print("[smoke] 5 train steps + 1 val pass")
        # 5-step dry run
        for step, batch in enumerate(train_loader):
            if step >= 5:
                break
            x = torch.cat([batch["x_hist"], batch["x_fut"]], dim=1).to(device)
            speeds = x[..., 0]
            m = sample_mask(speeds.shape, args.mask_rate, device)
            pred = model(x, adj_norm, m)
            loss = reconstruction_loss(pred, speeds, m)
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"  [step {step+1}] loss={loss.item():.4f}")
        return

    best_val = float("inf")
    for epoch in range(args.epochs):
        print(f"[epoch {epoch+1}/{args.epochs}]")
        train_loss = train_one_epoch(
            model, train_loader, opt, adj_norm, args.mask_rate, device,
        )
        val_loss = evaluate(model, val_loader, adj_norm, args.mask_rate, device)
        print(f"  train loss={train_loss:.4f}  val loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "encoder_kind": args.encoder_kind,
                    "node_dim": args.node_dim,
                    "encoder_hidden": args.encoder_hidden,
                    "feature_dim": train_ds.feature_dim,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                out_dir / "encoder.ckpt",
            )
    print(f"[done] best val loss = {best_val:.4f}  -> {out_dir/'encoder.ckpt'}")


if __name__ == "__main__":
    main()
