"""Parameter-free forecasting baselines for node-signal regression.

No training, no checkpoints. Each baseline scores the same test windows the
learned methods are evaluated on, through the SAME accounting helpers in
``src/world_models/probes.py`` (``_accum_reg`` / ``_finalize_reg``), so the
output JSON has an identical schema — MAE / RMSE / MAPE, the persistence
reference, and the inflection subset — and drops straight into the comparison
table next to DELOS, CLDG, DDGCL and IDOL.

Reusing those helpers rather than re-deriving the sums is deliberate. The
earlier England-COVID-only version computed its own MAE and had no inflection
columns, so its rows could not be placed beside the learned ones without a
reader assuming the two shared a protocol they did not.

Baselines:
  - persistence      : every future step = the last observed step. The canonical
                       no-change forecaster, and the reference every learned
                       method has to beat. Identical by construction to the
                       ``MAE_persist`` column the learned methods report, which
                       makes it a free cross-check that all arms scored the same
                       windows.
  - window_mean      : every future step = the mean over that node's history.
  - historical_mean  : every future step = that node's mean over the train
                       split (a static climatology).

Works on any ``--data_source`` whose items expose ``signal`` — England COVID and
METR-LA both do.

Usage:
    python scripts/regression/run_baselines.py --data_source england_covid \
        --history_size 8 --num_preds 3 --out_dir outputs/ec_baselines

    python scripts/regression/run_baselines.py --data_source metr_la \
        --history_size 12 --num_preds 12 --out_dir outputs/metrla_baselines
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.world_models.probes import (
    _accum_reg,
    _blank_reg_sums,
    _finalize_reg,
)

BASELINES = ("persistence", "window_mean", "historical_mean")


def _valid_mode(ds) -> str:
    """Match src/train/probe.py::_valid_mode so every arm scores the same cells."""
    return "active" if getattr(ds, "signed_target", False) else "positive"


def historical_mean(train_ds, history_size, num_preds, batch_size, num_workers):
    """Per-node mean over the train split, over OBSERVED entries only.

    Computed by streaming the train windows rather than reading a contiguous
    ``raw_signal``, because the two datasets store their timelines differently
    (England COVID keeps one, METR-LA's HF parquet layout is pre-windowed and
    keeps none). Overlapping windows re-weight timesteps unevenly, which is
    harmless for a climatology and keeps this path dataset-agnostic.
    """
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    total = count = None
    for batch in loader:
        signal = batch["signal"].float()                      # (B, T_total, N)
        active = (batch["active"] if "active" in batch
                  else signal > 0).float()
        if total is None:
            total = torch.zeros(signal.shape[-1])
            count = torch.zeros(signal.shape[-1])
        total += (signal * active).sum(dim=(0, 1))
        count += active.sum(dim=(0, 1))
    if total is None:
        raise SystemExit("train split is empty; cannot build a climatology.")
    return total / count.clamp(min=1.0)                        # (N,)


def evaluate_baselines(train_ds, test_ds, history_size, num_preds,
                       batch_size, num_workers):
    hist_mean = historical_mean(train_ds, history_size, num_preds,
                                batch_size, num_workers)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    horizons = tuple(range(num_preds))
    can_inflect = history_size >= 2
    mode = _valid_mode(test_ds)
    sums = {name: {h: _blank_reg_sums() for h in horizons} for name in BASELINES}

    for batch in loader:
        signal = batch["signal"].float()                       # (B, T_total, N)
        sig_obs = batch.get("signal_obs", batch["signal"]).float()
        active = batch["active"] if "active" in batch else None
        hist = sig_obs[:, :history_size]
        last_obs = hist[:, -1]                                 # (B, N)
        prev_obs = sig_obs[:, history_size - 2] if can_inflect else None
        # Average the history over observed entries only, so a missing sensor
        # reading (stored as 0.0) does not drag the window mean toward zero.
        if active is not None:
            am = active[:, :history_size].float()
            win_mean = (hist * am).sum(dim=1) / am.sum(dim=1).clamp(min=1.0)
        else:
            win_mean = hist.mean(dim=1)
        clim = hist_mean.unsqueeze(0).expand_as(last_obs)

        est = {"persistence": last_obs, "window_mean": win_mean,
               "historical_mean": clim}
        for h in horizons:
            valid = (active[:, history_size + h] if (active is not None
                                                     and mode == "active")
                     else None)
            true = signal[:, history_size + h]
            for name in BASELINES:
                _accum_reg(sums[name][h], est[name], true, last_obs, prev_obs,
                           can_inflect, valid=valid)

    return {name: _finalize_reg(s, report_mape=(mode == "positive"))
            for name, s in sums.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_cli_flags(p)
    p.add_argument("--history_size", type=int, default=8)
    p.add_argument("--num_preds", type=int, default=3)
    p.add_argument("--out_dir", default="outputs/regression_baselines")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(data_source="england_covid")
    args = p.parse_args()

    train_ds = build_dynamic_dataset(args, "train")
    test_ds = build_dynamic_dataset(args, "test")
    test_ds.mean, test_ds.std = train_ds.mean, train_ds.std
    print(f"[data] source={args.data_source} train_windows={len(train_ds)} "
          f"test_windows={len(test_ds)} N={train_ds.num_nodes}", flush=True)

    results = evaluate_baselines(train_ds, test_ds, args.history_size,
                                 args.num_preds, args.batch_size,
                                 args.num_workers)
    for name, metrics in results.items():
        print(f"\n[{name}] horizon  MAE  RMSE")
        for h, m in sorted(metrics.items()):
            infl = m.get("MAE_inflect")
            extra = f"  infl={infl:.3f}" if infl is not None else ""
            print(f"  +{h:>2}: {m['MAE']:.3f}  {m['RMSE']:.3f}{extra}")

    # Cross-check: `persistence` here and the `MAE_persist` column every learned
    # method writes are the same estimator. If they disagree, the arms are not
    # scoring the same windows and the table is invalid — better to see it now.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] wrote {out_dir/'baseline_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
