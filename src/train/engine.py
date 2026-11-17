"""Shared training-engine utilities for the DELOS entry scripts.

Collects the determinism / dataloader / LR-schedule / checkpoint boilerplate
that was copy-pasted across pretrain_encoder_*, train_cjepa_*, and the
trajectory/hetero trainers. The unified entry points
(`pretrain_encoder.py`, `train_delos.py`) import from here so a single edit
changes the run harness everywhere.
"""
from __future__ import annotations

import json
import math
import os
import random as _random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.world_models.delos_encoder import build_encoder


def set_deterministic(seed: int) -> None:
    """Make the run reproducible across machines / GPU types.

    Sets Python/NumPy/torch (CPU + all CUDA) RNGs, cuDNN deterministic
    kernels, and CUBLAS workspace config. Without this, same-seed
    pretraining can land in different basins, which makes controlled
    ablations impossible to interpret.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int,
                drop_last: bool = False) -> DataLoader:
    """Standard DataLoader. ``drop_last`` defaults False so scarce-snapshot
    splits (DBLP yearly) never produce empty epochs."""
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=drop_last,
    )


def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int,
                     min_ratio: float = 0.05) -> float:
    """Linear warmup then cosine decay to ``min_ratio`` of the base LR."""
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def setup_output_dir(out_dir: str, args) -> Path:
    """Create the output dir and dump args.json (the standard run-record)."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    return path


def save_encoder_ckpt(path, encoder, args, feature_dim: int, **extra) -> None:
    """Save encoder weights + the metadata needed to rebuild it downstream.

    The schema matches the previous pretrain scripts so existing
    `--encoder_ckpt` consumers (train_cjepa_*, diagnose_*, ablations) keep
    loading unchanged.
    """
    payload = {
        "encoder": encoder.state_dict(),
        "encoder_kind": args.encoder_kind,
        "node_dim": args.node_dim,
        "encoder_hidden": args.encoder_hidden,
        "bidirectional": getattr(args, "bidirectional", True),
        "feature_dim": feature_dim,
        "args": vars(args),
    }
    payload.update(extra)
    torch.save(payload, path)


def check_encoder_ckpt_batch_size(ckpt: dict, args, ckpt_path: str) -> None:
    """Refuse to train on a pretrained encoder produced at a different batch size.

    ``save_encoder_ckpt`` stores the pretraining ``args``; a run that consumes
    the encoder (DELOS with ``--encoder_ckpt``) must use the same
    ``--batch_size`` so the two stages sit on one optimisation budget and the
    table compares objectives, not batch sizes. A checkpoint whose trainer had
    no ``--batch_size`` flag (the baseline SSL encoders sample nodes per
    snapshot instead) cannot be verified and is let through with a notice.
    """
    saved_args = ckpt.get("args", {}) or {}
    saved = saved_args.get("batch_size")
    if saved is None:
        print(f"[encoder] {ckpt_path}: no batch_size recorded in the pretraining "
              f"args; cannot verify it matches --batch_size {args.batch_size}.",
              flush=True)
        return
    if int(saved) != int(args.batch_size):
        raise SystemExit(
            f"--encoder_ckpt {ckpt_path!r} was pretrained with batch_size={saved} "
            f"but this run uses --batch_size {args.batch_size}. Pretrained "
            f"encoders must be consumed at the batch size they were trained "
            f"with: pass --batch_size {saved}, or re-run the encoder "
            f"pretraining with --batch_size {args.batch_size}.")
    print(f"[encoder] batch_size {saved} matches the pretraining run.", flush=True)


def load_encoder_from_ckpt(ckpt_path: str, feature_dim: int, device,
                           freeze: bool = True):
    """Rebuild a frozen encoder from a pretrain checkpoint. Returns
    (encoder, meta_dict)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = build_encoder(
        ckpt.get("encoder_kind", "tgcn"),
        in_dim=ckpt.get("feature_dim", feature_dim),
        hidden_dim=ckpt["encoder_hidden"],
        out_dim=ckpt["node_dim"],
        freeze=freeze,
        bidirectional=ckpt.get("bidirectional", True),
    ).to(device)
    enc.load_state_dict(ckpt["encoder"])
    return enc, ckpt
