"""Encoder-side regression arms for the node-signal forecasting table.

Fits a forecast head on an encoder's last-history embedding and evaluates it
with the SAME persistence + inflection protocol as the DELOS predictor
(``src/train/probe.py::probe_regression_dynamic``). Works on any regression
``--data_source`` — England COVID (dynamic topology) and METR-LA (fixed) both
resolve through ``probes.graph_inputs``.

Two arms, selected by ``--train_encoder``:

  frozen (default)     encoder weights fixed, only the head is fit. This is how
                       CLDG / DDGCL / IDOL and the DELOS-encoder control are
                       scored: what does the representation already contain?
  supervised end-to-end
                       ``--train_encoder``: encoder + head optimised jointly on
                       the forecast loss. With ``--random_init`` (no
                       ``--encoder_ckpt``) this is the fully supervised
                       reference — no self-supervised pretraining at all — that
                       every SSL arm is trying to reach without labels.

Both arms share one fit loop (``probes.fit_regression_probe_encoder_only``), so
their gap is attributable to the encoder, not to a divergent training recipe.

Writes test_metrics.json in the schema DELOS writes, so the aggregator reads
every arm uniformly.

Usage:
    # frozen probe of a baseline encoder
    python scripts/regression/probe_encoder_regression.py \
        --encoder_ckpt outputs/ec_cldg_s42/encoder.ckpt \
        --data_source england_covid --history_size 8 --num_preds 3 \
        --persist_residual

    # supervised end-to-end reference (no pretraining)
    python scripts/regression/probe_encoder_regression.py --random_init \
        --data_source metr_la --history_size 12 --num_preds 12 \
        --encoder_kind gcn --encoder_hidden 128 --node_dim 128 \
        --train_encoder --persist_residual --out_dir outputs/metrla_supervised_s42

    # every encoder under a glob
    python scripts/regression/probe_encoder_regression.py \
        --encoder_pattern 'outputs/ec_*_s42' \
        --data_source england_covid --history_size 8 --num_preds 6 \
        --persist_residual
    # NB: --encoder_pattern is a plain glob (no brace {a,b} expansion).
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import sys
import types
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.world_models.delos_encoder import build_encoder
from src.world_models.probes import (
    evaluate_regression_encoder_only,
    fit_regression_probe_encoder_only,
)


def _dataset_args(saved: dict, history_size: int, num_preds: int,
                  seed: int, cli: argparse.Namespace | None = None
                  ) -> types.SimpleNamespace:
    """Args namespace for build_dynamic_dataset.

    Two modes, and the choice matters:

    * ``--data_source`` given -> the dataset is built ENTIRELY from this
      script's own CLI flags. Use this whenever the probe is one arm of a
      comparison: every arm then reads the same explicit spec.
    * omitted -> rebuilt from the checkpoint's saved args (convenient for
      one-off probes of an old checkpoint).

    The saved-args path is a silent-drift hazard and has already bitten once:
    a dataset flag added AFTER a checkpoint was written is simply absent from
    its saved args, so ``getattr(args, flag, default)`` falls back to the
    default and the probe runs on a *different dataset* than the arms it is
    being compared against — with no error, just an incomparable number.
    Hence the feature-dim assertion in run_one_encoder, and the cross-arm
    persistence check in scripts/synthetic/aggregate_synth_sweep.py.
    """
    if cli is not None and cli.data_source:
        ns = types.SimpleNamespace(**vars(cli))
    else:
        ns = types.SimpleNamespace(**saved)
        ns.data_source = saved.get("data_source") or "england_covid"
    ns.history_size = history_size
    ns.num_preds = num_preds
    ns.seed = seed
    return ns


def run_one_encoder(ckpt_path: Path | None, args, out_dir: Path | None = None):
    """Fit + evaluate one arm. ``ckpt_path=None`` means a random-init encoder."""
    # Seed the forecast head init + fit so the encoder-only baseline numbers
    # are reproducible across runs (the head is random-init; without this the
    # baseline MAEs swing by ~0.5 MAE run-to-run on this tiny dataset).
    torch.manual_seed(args.seed)
    ck = ({} if ckpt_path is None
          else torch.load(ckpt_path, map_location="cpu", weights_only=False))
    if ckpt_path is not None and "encoder" not in ck:
        raise ValueError(f"{ckpt_path} has no 'encoder' field.")
    saved = ck.get("args", {}) or {}
    ds_args = _dataset_args(saved, args.history_size, args.num_preds,
                            args.seed, cli=args)

    train_ds = build_dynamic_dataset(ds_args, "train")
    val_ds = build_dynamic_dataset(ds_args, "val")
    eval_ds = build_dynamic_dataset(ds_args, args.probe_split)
    # Cheap tripwire for "probed the wrong dataset": the encoder's input width
    # is baked into its weights, so a mismatch here means the rebuilt dataset
    # is not the one it was trained on.
    saved_fdim = ck.get("feature_dim")
    if saved_fdim is not None and saved_fdim != train_ds.feature_dim:
        raise ValueError(
            f"{ckpt_path} was trained on feature_dim={saved_fdim} but the "
            f"rebuilt dataset has {train_ds.feature_dim}. Pass the dataset "
            f"flags explicitly (--data_source ... plus its knobs).")
    for ds in (val_ds, eval_ds):
        ds.mean, ds.std = train_ds.mean, train_ds.std   # normalise consistently
    mk = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers)
    train_loader, val_loader, eval_loader = (mk(train_ds, True),
                                             mk(val_ds, False),
                                             mk(eval_ds, False))

    # Checkpoint metadata wins when present (its weights fix these dims); the
    # CLI supplies them for a random-init encoder.
    enc_kind = ck.get("encoder_kind", saved.get("encoder_kind", args.encoder_kind))
    enc_hidden = ck.get("encoder_hidden", saved.get("encoder_hidden",
                                                    args.encoder_hidden))
    node_dim = ck.get("node_dim", saved.get("node_dim", args.node_dim))
    enc_bidir = ck.get("bidirectional", saved.get("bidirectional",
                                                  args.bidirectional))
    encoder = build_encoder(enc_kind, in_dim=train_ds.feature_dim,
                            hidden_dim=enc_hidden, out_dim=node_dim,
                            freeze=not args.train_encoder,
                            bidirectional=enc_bidir).to(args.device)
    if ckpt_path is not None:
        encoder.load_state_dict(ck["encoder"])
        print(f"[encoder] loaded {enc_kind} from {ckpt_path}", flush=True)
    else:
        print(f"[encoder] random init {enc_kind} "
              f"(hidden={enc_hidden} out={node_dim})", flush=True)

    # Mirrors src/train/probe.py: the probe sees the persistence base only when
    # that base is noisy, so every arm in a sweep gets the same treatment.
    sees_base = getattr(train_ds, "target_noise_sigma", 0.0) > 0.0
    head = fit_regression_probe_encoder_only(
        encoder, train_loader, args.history_size, args.num_preds, args.device,
        epochs=args.probe_epochs, lr=args.lr, hidden_dim=args.probe_hidden,
        persist_residual=args.persist_residual, probe_sees_base=sees_base,
        train_encoder=args.train_encoder, encoder_lr=args.encoder_lr,
        # Only the end-to-end arm can overfit the encoder, so that is the arm
        # that needs val-based restoration; the frozen arm fits a tiny head and
        # keeps its historical behaviour (no val pass) so its published numbers
        # do not shift underneath the comparison.
        val_loader=val_loader if args.train_encoder else None,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta)
    metrics = evaluate_regression_encoder_only(
        encoder, head, eval_loader, args.history_size, args.num_preds,
        args.device, persist_residual=args.persist_residual,
        valid_mode=("active" if getattr(eval_ds, "signed_target", False)
                    else "positive"),
        probe_sees_base=sees_base)

    target = Path(out_dir) if out_dir else Path(ckpt_path).parent
    target.mkdir(parents=True, exist_ok=True)
    out_path = target / "test_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics, out_path


def _print_table(tag, metrics):
    print(f"[{tag}] horizon  MAE  MAE_persist  |  infl: MAE  MAE_persist  (n)")
    for h, m in sorted(metrics.items()):
        line = f"  +{h:>2}: {m['MAE']:.3f}     {m['MAE_persist']:.3f}"
        if "MAE_inflect" in m:
            line += (f"     |  {m['MAE_inflect']:.3f}     "
                     f"{m['MAE_persist_inflect']:.3f}  (n={m['n_inflect']})")
        win = "  <-- beats persistence" if m['MAE'] < m['MAE_persist'] else ""
        print(line + win)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder_ckpt", default=None,
                   help="A single bare encoder.ckpt (save_encoder_ckpt schema).")
    p.add_argument("--encoder_pattern", default=None,
                   help="Glob of directories each holding an encoder.ckpt.")
    p.add_argument("--random_init", action="store_true",
                   help="No checkpoint: build the encoder from the CLI encoder "
                        "flags. With --train_encoder this is the supervised "
                        "end-to-end reference.")
    # Full dataset CLI. Pass --data_source (plus its knobs) to build the
    # dataset from these flags instead of from the checkpoint's saved args —
    # required when this probe is one arm of a multi-arm comparison.
    add_dataset_cli_flags(p)
    p.set_defaults(data_source=None)
    p.add_argument("--out_dir", default=None,
                   help="Where to write test_metrics.json. Default: next to "
                        "the checkpoint. Needed when several arms probe the "
                        "SAME shared encoder and must not overwrite each other.")
    p.add_argument("--history_size", type=int, default=8)
    p.add_argument("--num_preds", type=int, default=3)
    p.add_argument("--probe_epochs", type=int, default=10)
    p.add_argument("--probe_hidden", type=int, default=64)
    p.add_argument("--persist_residual", action="store_true")
    p.add_argument("--probe_split", choices=["test", "val"], default="test",
                   help="Split to evaluate on. 'val' during hyperparameter "
                        "search so no configuration is ranked on test.")
    # Encoder shape, used only when there is no checkpoint to read it from.
    p.add_argument("--encoder_kind", default="gcn")
    p.add_argument("--encoder_hidden", type=int, default=128)
    p.add_argument("--node_dim", type=int, default=128)
    p.add_argument("--bidirectional", type=lambda s: str(s).lower() in
                   ("1", "true", "yes", "y", "t"), default=False)
    p.add_argument("--train_encoder", action="store_true",
                   help="Optimise the encoder jointly with the head "
                        "(supervised end-to-end arm).")
    p.add_argument("--encoder_lr", type=float, default=None,
                   help="Encoder LR for --train_encoder (default: --lr).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    targets: list[Path | None] = []
    if args.random_init:
        if not args.out_dir:
            print("--random_init needs --out_dir (there is no checkpoint "
                  "directory to write next to).", file=sys.stderr)
            sys.exit(1)
        targets.append(None)
    if args.encoder_ckpt:
        targets.append(Path(args.encoder_ckpt))
    if args.encoder_pattern:
        for d in sorted(_glob.glob(args.encoder_pattern)):
            ckpt = Path(d) / "encoder.ckpt"
            if ckpt.exists():
                targets.append(ckpt)
            else:
                print(f"[skip] {d} has no encoder.ckpt", file=sys.stderr)
    if not targets:
        print("Pass --encoder_ckpt, --encoder_pattern or --random_init.",
              file=sys.stderr)
        sys.exit(1)
    if args.out_dir and len(targets) > 1:
        print(f"--out_dir with {len(targets)} encoders would overwrite one "
              f"test_metrics.json; use it with a single encoder.",
              file=sys.stderr)
        sys.exit(1)

    failures = 0
    for ckpt in targets:
        tag = "random_init" if ckpt is None else ckpt.parent.name
        try:
            metrics, out = run_one_encoder(
                ckpt, args, out_dir=Path(args.out_dir) if args.out_dir else None)
            _print_table(tag, metrics)
            print(f"  -> {out}", flush=True)
        except Exception as e:
            failures += 1
            print(f"  [{tag}] FAILED: {type(e).__name__}: {e}", flush=True)
    # A failed arm that exits 0 shows up in the table as a missing row, which
    # reads as "not applicable" rather than "this run crashed".
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
