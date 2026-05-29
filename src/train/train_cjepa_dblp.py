"""C-JEPA training on DBLP-style yearly co-authorship snapshots.

Mirror of train_cjepa_metr_la.py but for dynamic graphs:
  - per-timestep adjacency (not broadcast)
  - per-timestep node existence mask
  - link-prediction probe (MRR + Hits@k) instead of speed regression

Run:
    python src/train/train_cjepa_dblp.py --synthetic --device cuda
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
from src.custom_codes.dblp_dataset import DBLPDynamicGraphDataset
from src.world_models.graph_jepa_encoder import (
    LinkPredictionProbe,
    build_encoder,
    link_ranking_metrics,
    normalize_adjacency,
)


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=shuffle,
    )


def cosine_warmup_lr(step, warmup_steps, total_steps, min_ratio=0.05):
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def encode_window(encoder, x, adj_norm):
    with torch.no_grad():
        return encoder(x, adj_norm)


def train_one_epoch(predictor, encoder, loader, optimizer, history_size,
                    device, log_every=50, scheduler=None, max_steps=None):
    predictor.train()
    log = {"loss": 0.0, "loss_masked_history": 0.0, "loss_future": 0.0, "n": 0}
    t0 = time.time()
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x_hist = batch["x_hist"].to(device, non_blocking=True)
        x_fut = batch["x_fut"].to(device, non_blocking=True)
        adj = batch["adj"].to(device, non_blocking=True)             # (B, T, N, N) bool
        active = batch["active"].to(device, non_blocking=True)        # (B, T, N) bool

        adj_norm = normalize_adjacency(adj.float())
        z_hist = encode_window(encoder, x_hist, adj_norm[:, : history_size])
        x_full = torch.cat([x_hist, x_fut], dim=1)
        z_target = encode_window(encoder, x_full, adj_norm)

        pred, masked_idx, _ = predictor(z_hist, adj, active)
        losses = compute_graph_jepa_loss(
            pred, z_target.detach(), active, masked_idx, history_size,
        )

        loss = losses["loss"]
        if not torch.isfinite(loss):
            print(f"  [step {step+1}] non-finite loss, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        if not torch.isfinite(gnorm):
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
            print(f"  [step {step+1:>5}] loss={log['loss']/n:.4f} "
                  f"mh={log['loss_masked_history']/n:.4f} "
                  f"fut={log['loss_future']/n:.4f} "
                  f"lr={cur_lr:.2e} ({time.time()-t0:.1f}s)")
    n = max(log["n"], 1)
    return {k: v / n for k, v in log.items() if k != "n"}


@torch.no_grad()
def evaluate_jepa(predictor, encoder, loader, history_size, device):
    predictor.eval()
    total = {"loss": 0.0, "loss_masked_history": 0.0, "loss_future": 0.0, "n": 0}
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        x_fut = batch["x_fut"].to(device)
        adj = batch["adj"].to(device)
        active = batch["active"].to(device)
        adj_norm = normalize_adjacency(adj.float())
        z_hist = encoder(x_hist, adj_norm[:, : history_size])
        z_target = encoder(torch.cat([x_hist, x_fut], dim=1), adj_norm)
        pred, masked_idx, _ = predictor(z_hist, adj, active)
        losses = compute_graph_jepa_loss(pred, z_target, active, masked_idx, history_size)
        bs = x_hist.shape[0]
        for k in ("loss", "loss_masked_history", "loss_future"):
            total[k] += losses[k].item() * bs
        total["n"] += bs
    n = max(total["n"], 1)
    return {k: v / n for k, v in total.items() if k != "n"}


# ---------------------------------------------------------------------------
# Link-prediction probe
# ---------------------------------------------------------------------------
def sample_neg_edges(adj_t: torch.Tensor, active_t: torch.Tensor,
                     k_per_pos: int, seed: int = 0) -> torch.Tensor:
    """For each positive (i, j) edge at time t, sample k_per_pos negatives
    that share node i and target an active-but-not-connected j.

    Returns (P, k) long tensor of negative j-indices for each positive edge.
    Used at eval time to compute MRR.
    """
    # adj_t: (N, N) bool, active_t: (N,) bool
    g = torch.Generator(device=adj_t.device).manual_seed(seed)
    src, dst = adj_t.nonzero(as_tuple=True)
    P = src.shape[0]
    N = adj_t.shape[0]
    if P == 0:
        return torch.empty((0, k_per_pos), dtype=torch.long, device=adj_t.device)
    candidates = active_t.clone()
    neg = torch.zeros(P, k_per_pos, dtype=torch.long, device=adj_t.device)
    for p in range(P):
        valid = candidates & ~adj_t[src[p]]
        valid[src[p]] = False
        idx = valid.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            neg[p] = src[p]                              # fallback
        else:
            choice = torch.randint(0, idx.numel(), (k_per_pos,), generator=g, device=adj_t.device)
            neg[p] = idx[choice]
    return neg


def fit_link_probe(predictor, encoder, train_loader, history_size,
                   num_preds, device, epochs=3, lr=1e-3, hidden_dim=64,
                   neg_per_pos=4):
    z_dim = predictor.node_dim
    probe = LinkPredictionProbe(in_dim=z_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    predictor.eval()

    for epoch in range(epochs):
        losses, n = 0.0, 0
        for batch in train_loader:
            x_hist = batch["x_hist"].to(device)
            adj_b = batch["adj"].to(device)
            active_b = batch["active"].to(device)
            B, T_total, N = active_b.shape
            adj_norm = normalize_adjacency(adj_b.float())
            with torch.no_grad():
                z_hist = encoder(x_hist, adj_norm[:, : history_size])
                pred = predictor.inference(z_hist, adj_b, active_b)   # (B, T_pred, N, D)

            # Probe loss: predict edges at each future timestep from latents.
            loss = 0.0
            tot_pairs = 0
            for b in range(B):
                for tp in range(pred.shape[1]):
                    t_global = history_size + tp
                    adj_t = adj_b[b, t_global]
                    active_t = active_b[b, t_global]
                    if active_t.sum() < 2 or adj_t.sum() == 0:
                        continue
                    src, dst = adj_t.nonzero(as_tuple=True)
                    if src.numel() == 0:
                        continue
                    z = pred[b, tp]                                    # (N, D)
                    pos_logits = probe(z[src], z[dst])
                    neg_dst = torch.randint(
                        0, N, (src.numel(), neg_per_pos), device=device,
                    )
                    neg_z = z[neg_dst]
                    neg_logits = probe(
                        z[src].unsqueeze(1).expand_as(neg_z),
                        neg_z,
                    )
                    pos_y = torch.ones_like(pos_logits)
                    neg_y = torch.zeros_like(neg_logits)
                    loss = loss + F.binary_cross_entropy_with_logits(
                        torch.cat([pos_logits, neg_logits.reshape(-1)]),
                        torch.cat([pos_y, neg_y.reshape(-1)]),
                    )
                    tot_pairs += pos_logits.numel() + neg_logits.numel()
            if isinstance(loss, float):
                continue
            loss = loss / max(tot_pairs, 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses += loss.item() * B
            n += B
        print(f"  [probe epoch {epoch+1}] bce={losses/max(n,1):.6f}")
    return probe


@torch.no_grad()
def evaluate_link_prediction(predictor, encoder, probe, loader, history_size,
                             device, k_per_pos=99):
    predictor.eval()
    probe.eval()
    per_horizon: dict = {}
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        adj_b = batch["adj"].to(device)
        active_b = batch["active"].to(device)
        B = x_hist.shape[0]
        adj_norm = normalize_adjacency(adj_b.float())
        z_hist = encoder(x_hist, adj_norm[:, : history_size])
        pred = predictor.inference(z_hist, adj_b, active_b)         # (B, T_pred, N, D)

        for b in range(B):
            for tp in range(pred.shape[1]):
                t_global = history_size + tp
                adj_t = adj_b[b, t_global]
                active_t = active_b[b, t_global]
                if adj_t.sum() == 0 or active_t.sum() < 2:
                    continue
                src, dst = adj_t.nonzero(as_tuple=True)
                z = pred[b, tp]
                pos_scores = probe(z[src], z[dst])                  # (P,)
                neg_dst = sample_neg_edges(adj_t, active_t, k_per_pos=k_per_pos, seed=tp + b)
                if neg_dst.numel() == 0:
                    continue
                neg_z = z[neg_dst]
                neg_scores = probe(
                    z[src].unsqueeze(1).expand_as(neg_z),
                    neg_z,
                )                                                    # (P, k)
                m = link_ranking_metrics(pos_scores, neg_scores)
                per_horizon.setdefault(tp, []).append(m)
    out = {}
    for tp, ms in per_horizon.items():
        keys = ms[0].keys()
        out[tp] = {k: float(np.mean([m[k] for m in ms])) for k in keys}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=None,
                   help="CSV edge list `src,dst,year`. If absent, --synthetic.")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic_nodes", type=int, default=100)
    p.add_argument("--synthetic_years", type=int, default=30)
    p.add_argument("--history_size", type=int, default=3)
    p.add_argument("--num_preds", type=int, default=1)
    p.add_argument("--node_dim", type=int, default=64)
    p.add_argument("--encoder_hidden", type=int, default=64)
    p.add_argument("--encoder_kind", choices=["linear", "gcn", "tgcn"], default="tgcn")
    p.add_argument("--num_masked_nodes", type=int, default=10)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_attention_mask", action="store_true", default=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--encoder_ckpt", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--out_dir", default="outputs/dblp_cjepa")
    p.add_argument("--probe_epochs", type=int, default=3)
    p.add_argument("--probe_hidden", type=int, default=64)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[setup] device={args.device} encoder={args.encoder_kind} D={args.node_dim}")
    ds_kwargs = dict(
        data_path=args.data_path, history_size=args.history_size,
        num_preds=args.num_preds, synthetic=args.synthetic,
        synthetic_nodes=args.synthetic_nodes,
        synthetic_years=args.synthetic_years,
    )
    train_ds = DBLPDynamicGraphDataset(split="train", **ds_kwargs)
    val_ds = DBLPDynamicGraphDataset(split="val", **ds_kwargs)
    test_ds = DBLPDynamicGraphDataset(split="test", **ds_kwargs)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"N={train_ds.num_nodes} F={train_ds.feature_dim}")

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    device = torch.device(args.device)

    # Build encoder (with optional pretrained ckpt auto-detect).
    enc_kind, enc_hidden, enc_out = args.encoder_kind, args.encoder_hidden, args.node_dim
    ckpt = None
    if args.encoder_ckpt:
        ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
        enc_kind = ckpt.get("encoder_kind", enc_kind)
        enc_hidden = ckpt.get("encoder_hidden", enc_hidden)
        enc_out = ckpt.get("node_dim", enc_out)

    encoder = build_encoder(
        enc_kind, in_dim=train_ds.feature_dim, hidden_dim=enc_hidden,
        out_dim=enc_out, freeze=False,
    ).to(device)
    if ckpt is not None:
        encoder.load_state_dict(ckpt["encoder"])
        print(f"[encoder] loaded {enc_kind} from {args.encoder_ckpt}")
    else:
        print(f"[encoder] random init {enc_kind}")
    args.node_dim = enc_out
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
        use_attention_mask=not args.no_attention_mask,
    ).to(device)

    n_pred = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in encoder.parameters())
    print(f"[model] predictor={n_pred:,} encoder (frozen)={n_enc:,}")

    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, args.epochs * steps_per_epoch)
    if args.epochs > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: cosine_warmup_lr(s, steps_per_epoch, total_steps, 0.05),
        )
    else:
        scheduler = None

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        predictor.load_state_dict(ck["predictor"])
        if "encoder" in ck:
            encoder.load_state_dict(ck["encoder"])
        print(f"[resume] loaded predictor from {args.resume} "
              f"(epoch={ck.get('epoch','?')} val_loss={ck.get('val_loss','?')})")

    if args.smoke:
        print("[smoke] 3 train steps")
        train_one_epoch(
            predictor, encoder, train_loader, opt, args.history_size, device,
            log_every=1, max_steps=3, scheduler=scheduler,
        )
        return

    best_val = float("inf")
    for epoch in range(args.epochs):
        print(f"[epoch {epoch+1}/{args.epochs}]")
        train_log = train_one_epoch(
            predictor, encoder, train_loader, opt, args.history_size, device,
            scheduler=scheduler,
        )
        val_log = evaluate_jepa(
            predictor, encoder, val_loader, args.history_size, device,
        )
        print(f"  train loss={train_log['loss']:.4f}  val loss={val_log['loss']:.4f}")
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

    best_path = out_dir / "best.ckpt"
    if args.epochs > 0 and best_path.exists():
        ck = torch.load(best_path, map_location=device, weights_only=False)
        predictor.load_state_dict(ck["predictor"])
        print(f"[probe] reloaded best.ckpt (epoch={ck.get('epoch','?')} "
              f"val_loss={ck.get('val_loss','?'):.4f})")

    print("[probe] fitting link-prediction probe")
    probe = fit_link_probe(
        predictor, encoder, train_loader, args.history_size, args.num_preds,
        device, epochs=args.probe_epochs, hidden_dim=args.probe_hidden,
    )
    metrics = evaluate_link_prediction(
        predictor, encoder, probe, test_loader, args.history_size, device,
    )
    print("[test] link prediction (per horizon step)")
    for tp, m in sorted(metrics.items()):
        print(f"  +{tp+1:>2} year: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
