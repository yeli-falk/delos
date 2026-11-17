"""Unified DELOS encoder pretraining (Block 2).

Pick the encoder (``--encoder_kind``, registry-backed), the dataset
(``--data_source``), and the pretext task (``--pretext``):

  - ``edge_mask``     hide a fraction of real edges, predict their existence
                      vs sampled negatives (BCE). Canonical DTDG pretext.
  - ``feature_mask``  hide the semantic node-feature channels (index
                      ``--graph_feat_dim`` onward) and reconstruct (MSE).
                      Requires --with_edge_features.
  - ``masked_signal`` hide channel-0 values (e.g. METR-LA speeds) and
                      reconstruct from spatial context (MSE).

Fixed-topology datasets (METR-LA) carry no per-snapshot ``adj`` in the batch;
the static adjacency is taken from the dataset and normalised once. Dynamic
graphs normalise the per-batch adjacency.

After training, only the encoder weights + rebuild metadata are saved to
``<out_dir>/encoder_<data_source>_<encoder_kind>_<pretext>.ckpt`` (the name is
templated on the run config so multiple encoders can coexist in one out_dir;
schema unchanged, so existing ``--encoder_ckpt`` consumers keep loading — but
they must be pointed at that templated filename, NOT a plain ``encoder.ckpt``).

Run:
    python -m src.train.pretrain_encoder --data_path dataset/dblp/edges.csv \
    --encoder_kind tgcn --bidirectional false --encoder_hidden 128 --node_dim 128 \
    --decoder_hidden 64 --mask_rate 0.25 --pretext edge_mask --batch_size 4 --num_workers 8 --device cuda --epochs 300 --lr 1e-3 --early_stop_patience 30 --seed 0 --out_dir outputs/dblp_encoder --plot_losses
    python src/train/pretrain_encoder.py --data_source metr_la \
        --metr_la_synthetic --encoder_kind gcn --pretext masked_signal
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.engine import (
    make_loader,
    save_encoder_ckpt,
    set_deterministic,
    setup_output_dir,
)
from src.world_models.delos_encoder import build_encoder, normalize_adjacency
from src.world_models.pretrainers import (
    EdgePretrainer,
    EncoderPretrainer,
    FeatureMaskPretrainer,
    sample_feature_mask,
    sample_masked_edges,
)


# ---------------------------------------------------------------------------
# adjacency handling: dynamic (per-batch) vs fixed (dataset-level)
# ---------------------------------------------------------------------------
def _full_x(batch, device):
    return torch.cat([batch["x_hist"], batch["x_fut"]], dim=1).to(
        device, non_blocking=True)


def _batch_adj_norm(batch, device, fixed_adj_norm):
    """Normalised adjacency for a batch. Uses the fixed graph if the dataset
    has no per-snapshot adjacency (METR-LA), else the batch adjacency."""
    if fixed_adj_norm is not None:
        return fixed_adj_norm
    return normalize_adjacency(batch["adj"].to(device, non_blocking=True))


# ---------------------------------------------------------------------------
# pretext losses / metrics
# ---------------------------------------------------------------------------
def edge_bce_loss(pos_logits, neg_logits):
    if pos_logits.numel() == 0:
        return torch.tensor(0.0, device=pos_logits.device, requires_grad=True)
    logits = torch.cat([pos_logits, neg_logits])
    targets = torch.cat([torch.ones_like(pos_logits),
                         torch.zeros_like(neg_logits)])
    return F.binary_cross_entropy_with_logits(logits, targets)


@torch.no_grad()
def edge_accuracy(pos_logits, neg_logits):
    if pos_logits.numel() == 0:
        return {"acc": 0.0, "margin": 0.0}
    pos_correct = (pos_logits > 0).float().mean().item()
    neg_correct = (neg_logits < 0).float().mean().item()
    margin = (pos_logits.mean() - neg_logits.mean()).item()
    return {"acc": 0.5 * (pos_correct + neg_correct), "margin": margin}


def signal_recon_loss(pred, target, mask):
    diff2 = (pred - target).pow(2) * mask.float()
    return diff2.sum() / mask.float().sum().clamp(min=1)


# ---------------------------------------------------------------------------
# one-epoch drivers per pretext (train + eval share the body via `gen`)
# ---------------------------------------------------------------------------
def run_epoch(model, loader, opt, args, device, fixed_adj_norm, train: bool,
              gen=None, log_every=50):
    model.train(train)
    pretext = args.pretext
    agg = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}
    t0 = time.time()
    for step, batch in enumerate(loader):
        x = _full_x(batch, device)
        if pretext == "edge_mask":
            adj = batch["adj"].to(device); active = batch["active"].to(device)
            adj_m, pos, neg = sample_masked_edges(
                adj, active, args.mask_rate, args.neg_per_pos, generator=gen)
            if pos.numel() == 0:
                continue
            pl, nl = model(x, normalize_adjacency(adj_m), pos, neg)
            loss = edge_bce_loss(pl, nl)
            m = edge_accuracy(pl.detach(), nl.detach())
        elif pretext == "feature_mask":
            active = batch["active"].to(device)
            mask = sample_feature_mask(active, args.mask_rate, generator=gen)
            if not mask.any():
                continue
            adj_norm = _batch_adj_norm(batch, device, fixed_adj_norm)
            pred, target = model(x, adj_norm, mask)
            loss = F.mse_loss(pred, target)
            m = {"acc": 0.0, "margin": 0.0}
        else:  # masked_signal
            signal = x[..., 0]
            if gen is None:
                mask = torch.rand(signal.shape, device=device) < args.mask_rate
            else:
                mask = torch.rand(signal.shape, generator=gen, device=device) < args.mask_rate
            adj_norm = _batch_adj_norm(batch, device, fixed_adj_norm)
            pred = model(x, adj_norm, mask)
            loss = signal_recon_loss(pred, signal, mask)
            m = {"acc": 0.0, "margin": 0.0}

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        bs = x.shape[0]
        agg["loss"] += loss.item() * bs
        agg["acc"] += m["acc"] * bs
        agg["margin"] += m["margin"] * bs
        agg["n"] += bs
        if train and (step + 1) % log_every == 0:
            n = agg["n"]
            print(f"  [step {step+1:>5}] loss={agg['loss']/n:.4f} "
                  f"acc={agg['acc']/n:.3f} ({time.time()-t0:.1f}s)", flush=True)
    n = max(agg["n"], 1)
    return {k: v / n for k, v in agg.items() if k != "n"}


def build_model(args, encoder, feature_dim, device):
    if args.pretext == "edge_mask":
        return EdgePretrainer(encoder, latent_dim=args.node_dim,
                              decoder_hidden=args.decoder_hidden).to(device)
    if args.pretext == "feature_mask":
        return FeatureMaskPretrainer(
            encoder, in_dim=feature_dim, latent_dim=args.node_dim,
            mask_channels_lo=args.graph_feat_dim,
            decoder_hidden=args.decoder_hidden).to(device)
    return EncoderPretrainer(encoder, latent_dim=args.node_dim,
                             decoder_hidden=args.decoder_hidden).to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cli.add_dblp_synthetic_flags(p)
    add_dataset_cli_flags(p)
    cli.add_window_flags(p)
    cli.add_encoder_flags(p)
    cli.add_optim_flags(p)
    cli.add_run_flags(p, out_dir="outputs/encoder")
    p.add_argument("--decoder_hidden", type=int, default=64)
    p.add_argument("--pretext",
                   choices=["edge_mask", "feature_mask", "masked_signal"],
                   default="edge_mask")
    p.add_argument("--mask_rate", type=float, default=0.15)
    p.add_argument("--neg_per_pos", type=int, default=1)
    p.add_argument("--graph_feat_dim", type=int, default=8,
                   help="Channels [0, this) are graph-derived; channels past "
                        "it are semantic and masked under feature_mask.")
    cli.add_early_stop_flags(p, patience=0, min_delta=1e-4)
    p.add_argument("--plot_losses", action="store_true")
    args = p.parse_args()

    set_deterministic(args.seed)
    out_dir = setup_output_dir(args.out_dir, args)
    device = torch.device(args.device)
    print(f"[setup] device={args.device} data={args.data_source} "
          f"encoder={args.encoder_kind} pretext={args.pretext} "
          f"D={args.node_dim} mask_rate={args.mask_rate}", flush=True)

    train_ds = build_dynamic_dataset(args, split="train")
    val_ds = build_dynamic_dataset(args, split="val")
    # METR-LA: val must use train normalisation stats.
    if hasattr(train_ds, "mean") and hasattr(val_ds, "mean"):
        val_ds.mean, val_ds.std = train_ds.mean, train_ds.std
    print(f"[data] train={len(train_ds)} val={len(val_ds)} "
          f"N={train_ds.num_nodes} F={train_ds.feature_dim}", flush=True)

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    if len(train_loader) == 0:
        raise RuntimeError(
            f"Train loader empty: {len(train_ds)} windows, "
            f"batch_size={args.batch_size}. Reduce --batch_size.")

    # Fixed adjacency (METR-LA): dataset has `.adj` and batches carry no "adj".
    fixed_adj_norm = None
    sample = train_ds[0]
    if "adj" not in sample and hasattr(train_ds, "adj"):
        fixed_adj_norm = normalize_adjacency(
            torch.from_numpy(train_ds.adj).to(device))
        if args.pretext != "masked_signal":
            raise ValueError(
                f"--pretext {args.pretext} needs per-snapshot adjacency, but "
                f"{args.data_source} is fixed-topology. Use --pretext masked_signal.")

    encoder = build_encoder(
        args.encoder_kind, in_dim=train_ds.feature_dim,
        hidden_dim=args.encoder_hidden, out_dim=args.node_dim, freeze=False,
        bidirectional=args.bidirectional).to(device)
    model = build_model(args, encoder, train_ds.feature_dim, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable={n_params:,}  pretext={args.pretext}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    if args.smoke:
        print("[smoke] 3 train steps")
        for step, batch in enumerate(train_loader):
            if step >= 3:
                break
            run_epoch(model, [batch], opt, args, device, fixed_adj_norm, train=True)
        print("[smoke] ok")
        return

    best_val = float("inf")
    since_improve = 0
    train_losses, val_losses = [], []
    for epoch in range(args.epochs):
        print(f"[epoch {epoch+1}/{args.epochs}]", flush=True)
        train_log = run_epoch(model, train_loader, opt, args, device,
                              fixed_adj_norm, train=True)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        val_log = run_epoch(model, val_loader, None, args, device,
                            fixed_adj_norm, train=False, gen=gen)
        train_losses.append(train_log["loss"])
        val_losses.append(val_log["loss"])
        print(f"  train loss={train_log['loss']:.4f} acc={train_log['acc']:.3f}  "
              f"val loss={val_log['loss']:.4f} acc={val_log['acc']:.3f}", flush=True)

        if val_log["loss"] < best_val - args.early_stop_min_delta:
            best_val = val_log["loss"]
            since_improve = 0
            print(f"[save] val loss improved to {best_val:.4f}")
            save_encoder_ckpt(
                out_dir / f"encoder_{args.data_source}_{args.encoder_kind}_{args.pretext}.ckpt", encoder, args,
                feature_dim=train_ds.feature_dim, epoch=epoch,
                val_loss=val_log["loss"], val_acc=val_log["acc"],
                pretrain_objective=args.pretext)
        else:
            since_improve += 1
            if (args.early_stop_patience > 0
                    and since_improve >= args.early_stop_patience):
                print(f"[early stop] no improvement for {since_improve} epochs; "
                      f"stopping at {epoch+1}/{args.epochs}", flush=True)
                break

    print(f"[done] best val loss = {best_val:.4f}  -> {out_dir/f'encoder_{args.data_source}_{args.encoder_kind}_{args.pretext}.ckpt'}",
          flush=True)
    
    if args.plot_losses:
        import matplotlib.pyplot as plt
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
        plt.title(f"Pretraining {args.pretext} on {args.data_source}")
        plt.savefig(out_dir / f"losses_{args.data_source}_{args.encoder_kind}_{args.pretext}.png", dpi=200)
        print(f"[plot] saved {out_dir/f'losses_{args.data_source}_{args.encoder_kind}_{args.pretext}.png'}", flush=True)


if __name__ == "__main__":
    main()
