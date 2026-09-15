"""Stage 1 of the STEP baseline: pre-train TSFormer.

    python -m src.baselines.step.pretrain_tsformer \
        --data_source metr_la --history_size 12 --num_preds 12 \
        --long_history 2016 --patch_size 12 --out_dir outputs/step_pre_s42

Writes ``tsformer.ckpt`` next to an ``args.json``; ``train_step.py`` loads it,
freezes it, and trains the graph learner + forecasting backend on top.

Objective (unchanged from the reference): patchify each node's long history
independently, drop ``--mask_ratio`` of the time patches, encode what is left,
re-insert learned mask tokens carrying only the dropped patches' positions,
decode, and score masked-MAE against the true values of the dropped patches
only. The loss is computed in the ORIGINAL signal units with ``null_val=0.0``,
so METR-LA's missing loop-detector readings -- and any left-padding at the very
start of the timeline -- are excluded rather than fitted.

Note the mask ratio is 0.75 here against STD-MAE's 0.25: STEP masks along one
axis and reconstructs from the remaining quarter, which is the harder pretext
its paper argues for. That difference is the method, not a knob to equalise.

Model selection follows every other trainer in this repo: best-val-loss weights
are restored and ``--early_stop_patience`` non-improving epochs stop the run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.baselines.common import BestValTracker
from src.baselines.long_history import build_long_history_datasets
from src.baselines.masked_pretrain import masked_mae
from src.baselines.step.step import TSFormer
from src.custom_codes.dataset_factory import add_dataset_cli_flags
from src.train import cli
from src.train.engine import set_deterministic


def add_tsformer_flags(p) -> None:
    """The ``tsformer_args`` block of the reference configs, as CLI flags."""
    p.add_argument("--long_history", type=int, default=2016,
                   help="Length of the TSFormer input window, in steps. Must "
                        "be a multiple of --patch_size. Reference: 288*7 = "
                        "2016 (one week) on METR-LA. Far smaller on England "
                        "COVID, whose whole timeline is 61 daily snapshots.")
    p.add_argument("--patch_size", type=int, default=12,
                   help="Steps per token. Reference: 12 (one hour) on METR-LA.")
    p.add_argument("--embed_dim", type=int, default=96)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--mlp_ratio", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mask_ratio", type=float, default=0.75,
                   help="Reference: 0.75 (STD-MAE's is 0.25 -- that difference "
                        "is the method).")
    p.add_argument("--encoder_depth", type=int, default=4)
    p.add_argument("--decoder_depth", type=int, default=1)


def tsformer_args_from(args, mode: str) -> dict:
    """The kwargs both stages must agree on. Saved into the checkpoint so the
    forecasting stage rebuilds the identical module instead of re-reading flags
    that may have drifted."""
    return {
        "patch_size": args.patch_size,
        "in_channel": 1,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "num_token": args.long_history // args.patch_size,
        "mask_ratio": args.mask_ratio,
        "encoder_depth": args.encoder_depth,
        "decoder_depth": args.decoder_depth,
        "mode": mode,
    }


def run_epoch(model, loader, mean, std, device, train: bool, opt=None,
              clip: float = 5.0) -> float:
    model.train(train)
    total, n = 0.0, 0
    for batch in loader:
        long_hist = batch["long_history"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            recon, label = model(long_hist)
            # BasicTS scores the loss AFTER re-scaling, which is what makes
            # null_val=0.0 mean "missing reading" rather than "mean speed".
            loss = masked_mae(recon * std + mean, label * std + mean, 0.0)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        bs = long_hist.shape[0]
        total += float(loss.detach()) * bs
        n += bs
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_cli_flags(p)
    cli.add_window_flags(p, history_size=12, num_preds=12)
    cli.add_run_flags(p, out_dir="outputs/step_pretrain")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    add_tsformer_flags(p)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--milestones", type=int, nargs="*", default=[50],
                   help="MultiStepLR milestones (reference: [50], gamma 0.5).")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--clip_grad", type=float, default=5.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--window_stride", type=int, default=1,
                   help="Pre-train on every k-th window. METR-LA's stride-1 "
                        "windows overlap in 2015 of 2016 steps, so k>1 buys a "
                        "large speed-up for very little information. Affects "
                        "PRE-TRAINING ONLY; the forecasting stage and every "
                        "reported metric still use all windows.")
    args = p.parse_args()

    if args.long_history % args.patch_size != 0:
        raise SystemExit(f"--long_history {args.long_history} must be a "
                         f"multiple of --patch_size {args.patch_size}.")
    if args.long_history < args.patch_size:
        raise SystemExit("--long_history must be at least one patch long.")

    set_deterministic(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args_tsformer.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    device = torch.device(args.device)
    print(f"[setup] TSFormer pre-training  data={args.data_source} "
          f"long_history={args.long_history} patch={args.patch_size} "
          f"num_token={args.long_history // args.patch_size} "
          f"mask_ratio={args.mask_ratio} embed_dim={args.embed_dim}", flush=True)

    ds, _ = build_long_history_datasets(args, args.long_history,
                                        splits=("train", "val"))
    mean, std = ds["train"].mean, ds["train"].std

    def loader(view, shuffle):
        data = view
        if args.window_stride > 1:
            data = Subset(view, list(range(0, len(view), args.window_stride)))
        return DataLoader(data, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    train_loader = loader(ds["train"], True)
    val_loader = loader(ds["val"], False)
    if len(val_loader.dataset) == 0:
        raise SystemExit("the val split has no windows; model selection needs "
                         "at least one.")

    model = TSFormer(**tsformer_args_from(args, "pre-train")).to(device)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"[model] TSFormer trainable={n_params:,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8,
                           betas=(0.9, 0.95), weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones,
                                                 gamma=args.gamma)
    tracker = BestValTracker(args.early_stop_patience, args.early_stop_min_delta)

    n_epochs = 1 if args.smoke else args.epochs
    for epoch in range(n_epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, mean, std, device,
                               train=True, opt=opt, clip=args.clip_grad)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, mean, std, device, train=False)
        sched.step()
        improved, stop = tracker.update(val_loss, model)
        print(f"[epoch {epoch+1}/{n_epochs}] mae={train_loss:.4f} "
              f"val_mae={val_loss:.4f}{'  (best)' if improved else ''} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if stop:
            print(f"  early stopping: no val improvement for "
                  f"{args.early_stop_patience} epochs", flush=True)
            break

    if tracker.restore(model):
        print(f"[best] restored TSFormer from best val epoch "
              f"(val_mae={tracker.best:.4f})", flush=True)

    ckpt_path = out_dir / "tsformer.ckpt"
    torch.save({
        "model": model.state_dict(),
        # The forecasting stage rebuilds from THIS, not from its own flags, so
        # a mismatched --patch_size / --embed_dim cannot silently produce a
        # differently-shaped encoder than the one that was pre-trained.
        "tsformer_args": tsformer_args_from(args, "forecasting"),
        "long_history": args.long_history,
        "val_loss": tracker.best,
        "args": vars(args),
    }, ckpt_path)
    print(f"[done] best val masked-MAE = {tracker.best:.4f}  -> {ckpt_path}",
          flush=True)


if __name__ == "__main__":
    main()
