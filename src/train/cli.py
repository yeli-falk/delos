"""Shared argparse building blocks for the DELOS entry scripts.

Each entry point composes these with :func:`add_dataset_cli_flags` (from
``dataset_factory``) and adds only its block-specific arguments, so the common
flags (encoder, optimisation, run dirs, DBLP synthetic generators) stay in one
place.
"""
from __future__ import annotations

import torch

from src.world_models.delos_encoder import ENCODER_REGISTRY


def _bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "y", "t")



def add_dblp_synthetic_flags(p, synthetic_nodes: int = 100,
                             synthetic_years: int = 30) -> None:
    """`--data_path` plus the synthetic-DBLP/SBM generator flags.

    Kept here (rather than repeated per entry point) so the generator config a
    run records in args.json is identical across pretrain / train / baselines —
    the probe rebuilds the dataset from those saved args.
    """
    p.add_argument("--data_path", default=None,
                   help="Path to the real DBLP edges CSV (ignored when "
                        "--synthetic is set).")
    p.add_argument("--synthetic", action="store_true",
                   help="Generate a synthetic timeline instead of loading "
                        "--data_path.")
    p.add_argument("--synthetic_nodes", type=int, default=synthetic_nodes)
    p.add_argument("--synthetic_years", type=int, default=synthetic_years)
    p.add_argument("--synthetic_kind", choices=["dblp", "sbm"], default="dblp")
    p.add_argument("--synthetic_blocks", type=int, default=8,
                   help="Number of planted communities (--synthetic_kind sbm).")


def add_window_flags(p, history_size: int = 3, num_preds: int = 1) -> None:
    p.add_argument("--history_size", type=int, default=history_size)
    p.add_argument("--num_preds", type=int, default=num_preds)


def add_encoder_flags(p, default_kind: str = "tgcn", node_dim: int = 64,
                      encoder_hidden: int = 64) -> None:
    p.add_argument("--encoder_kind", choices=sorted(ENCODER_REGISTRY),
                   default=default_kind,
                   help="Node encoder architecture (registry-backed).")
    p.add_argument("--node_dim", type=int, default=node_dim,
                   help="Encoder output (latent) dim.")
    p.add_argument("--encoder_hidden", type=int, default=encoder_hidden)
    p.add_argument("--bidirectional", type=_bool, default=True,
                   help="TGCN bidirectional GRU. Set False for a temporally "
                        "causal encoder with no future-leak (assumption A4).")


def add_optim_flags(p, lr: float = 1e-3, epochs: int = 20,
                    batch_size: int = 8) -> None:
    p.add_argument("--batch_size", type=int, default=batch_size)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--weight_decay", type=float, default=1e-4)


def add_early_stop_flags(p, patience: int = 20, min_delta: float = 0.0) -> None:
    """Best-val-loss selection + early stopping, shared by every trainer.

    Defined once so DELOS and the baselines cannot drift apart on model
    selection — the JEPA selecting on val while a baseline keeps its final
    epoch would flatter the JEPA for a reason unrelated to representation
    quality. ``--early_stop_patience 0`` disables stopping but every trainer
    still restores its best-val weights.
    """
    p.add_argument("--early_stop_patience", type=int, default=patience,
                   help="Stop after this many epochs with no val improvement. "
                        "0 = never stop early (best-val weights are still "
                        "restored at the end).")
    p.add_argument("--early_stop_min_delta", type=float, default=min_delta,
                   help="Minimum val-loss drop that counts as an improvement.")


def add_run_flags(p, out_dir: str) -> None:
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default=out_dir)
    p.add_argument("--smoke", action="store_true",
                   help="Run a handful of steps and exit (shape/wiring check).")
