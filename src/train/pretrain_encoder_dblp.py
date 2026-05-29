"""GraphMAE-style encoder pretraining on a DBLP-style yearly graph.

Mirror of pretrain_encoder_metr_la.py adapted to dynamic graphs:
  - per-timestep adjacency flows through the encoder
  - target reconstruction is the first feature channel (normalised degree)
    rather than speed; for DBLP this is what the encoder needs to learn
    locally before JEPA can give it a richer temporal signal.

After pretraining, the encoder weights are saved to
`outputs/dblp_encoder/encoder.ckpt` and consumed by `train_cjepa_dblp.py`
via `--encoder_ckpt`.

Run:
    python src/train/pretrain_encoder_dblp.py --synthetic --device cuda
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

from src.custom_codes.dblp_dataset import DBLPDynamicGraphDataset
from src.world_models.graph_jepa_encoder import (
    EncoderPretrainer,
    build_encoder,
    normalize_adjacency,
)


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=shuffle,
    )


def sample_mask(shape, p, device):
    return torch.rand(shape, device=device) < p


def reconstruction_loss(pred, target, mask):
    diff2 = (pred - target).pow(2) * mask.float()
    return diff2.sum() / mask.float().sum().clamp(min=1)


def train_one_epoch(model, loader, opt, mask_rate, device, log_every=50):
    model.train()
    log = {"loss": 0.0, "n": 0}
    t0 = time.time()
    for step, batch in enumerate(loader):
        x_hist = batch["x_hist"].to(device, non_blocking=True)
        x_fut = batch["x_fut"].to(device, non_blocking=True)
        adj = batch["adj"].to(device, non_blocking=True).float()
        active = batch["active"].to(device, non_blocking=True)

        x = torch.cat([x_hist, x_fut], dim=1)              # (B, T_tot, N, F)
        adj_norm = normalize_adjacency(adj)                # (B, T_tot, N, N)

        target = x[..., model.speed_channel].clone()
        # Don't supervise on inactive (node-doesn't-exist) positions.
        elig = active                                       # (B, T_tot, N)
        mask = sample_mask(target.shape, mask_rate, device) & elig
        pred = model(x, adj_norm, mask)
        loss = reconstruction_loss(pred, target, mask)

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
def evaluate(model, loader, mask_rate, device, seed=42):
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    total, n = 0.0, 0
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        x_fut = batch["x_fut"].to(device)
        adj = batch["adj"].to(device).float()
        active = batch["active"].to(device)
        x = torch.cat([x_hist, x_fut], dim=1)
        adj_norm = normalize_adjacency(adj)
        target = x[..., model.speed_channel].clone()
        mask = (torch.rand(target.shape, generator=rng, device=device) < mask_rate) & active
        pred = model(x, adj_norm, mask)
        total += reconstruction_loss(pred, target, mask).item() * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic_nodes", type=int, default=100)
    p.add_argument("--synthetic_years", type=int, default=60)
    p.add_argument("--history_size", type=int, default=3)
    p.add_argument("--num_preds", type=int, default=1)
    p.add_argument("--encoder_kind", choices=["linear", "gcn", "tgcn"], default="tgcn")
    p.add_argument("--node_dim", type=int, default=64)
    p.add_argument("--encoder_hidden", type=int, default=64)
    p.add_argument("--decoder_hidden", type=int, default=64)
    p.add_argument("--mask_rate", type=float, default=0.25)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="outputs/dblp_encoder")
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

    ds_kwargs = dict(
        data_path=args.data_path, history_size=args.history_size,
        num_preds=args.num_preds, synthetic=args.synthetic,
        synthetic_nodes=args.synthetic_nodes,
        synthetic_years=args.synthetic_years,
    )
    train_ds = DBLPDynamicGraphDataset(split="train", **ds_kwargs)
    val_ds = DBLPDynamicGraphDataset(split="val", **ds_kwargs)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} "
          f"N={train_ds.num_nodes} F={train_ds.feature_dim}")

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)

    device = torch.device(args.device)
    encoder = build_encoder(
        args.encoder_kind, in_dim=train_ds.feature_dim,
        hidden_dim=args.encoder_hidden, out_dim=args.node_dim, freeze=False,
    ).to(device)
    model = EncoderPretrainer(
        encoder=encoder, latent_dim=args.node_dim,
        decoder_hidden=args.decoder_hidden, speed_channel=0,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"[model] total trainable={n_params:,}  encoder={n_enc:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    if args.smoke:
        print("[smoke] 3 train steps")
        for step, batch in enumerate(train_loader):
            if step >= 3:
                break
            x = torch.cat([batch["x_hist"], batch["x_fut"]], dim=1).to(device)
            adj = batch["adj"].to(device).float()
            active = batch["active"].to(device)
            adj_norm = normalize_adjacency(adj)
            target = x[..., 0]
            m = sample_mask(target.shape, args.mask_rate, device) & active
            pred = model(x, adj_norm, m)
            loss = reconstruction_loss(pred, target, m)
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"  [step {step+1}] loss={loss.item():.4f}")
        return

    best_val = float("inf")
    for epoch in range(args.epochs):
        print(f"[epoch {epoch+1}/{args.epochs}]")
        train_loss = train_one_epoch(model, train_loader, opt, args.mask_rate, device)
        val_loss = evaluate(model, val_loader, args.mask_rate, device)
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
