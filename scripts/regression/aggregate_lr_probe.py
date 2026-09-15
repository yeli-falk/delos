"""Read the LR-range probe and report where the predictor is stable.

Consumes the ``history.json`` each cell of ``slurm_lr_probe.sh`` writes and
classifies every learning rate by the SHAPE of its training curve, not by its
final loss. That distinction is the point: a diverging run and a merely
slow one can end at similar losses, but only one of them is worth searching
around.

Verdicts:
  diverged   train loss ends above its own minimum by more than --tol. With a
             frozen encoder the targets are static, so a rising TRAIN loss can
             only be the optimiser -- this LR is above the stable range.
  overfit    train loss still falling but val loss turned up early. The LR is
             usable; the run needs regularisation or fewer epochs, not a
             smaller step.
  stable     both still falling at the end. Candidates for the search grid.
  slow       stable but barely moved -- likely below the useful range.

Usage:
    python scripts/regression/aggregate_lr_probe.py --root outputs/reg_lrprobe/metr_la
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def classify(hist, tol, slow_frac):
    tr = [h["train_loss"] for h in hist]
    va = [h["val_loss"] for h in hist]
    if len(tr) < 2:
        return "too-short", {}
    tr_min, va_min = min(tr), min(va)
    rise = (tr[-1] - tr_min) / max(abs(tr_min), 1e-9)
    improved = (tr[0] - tr_min) / max(abs(tr[0]), 1e-9)
    facts = {"train_first": tr[0], "train_min": tr_min, "train_last": tr[-1],
             "val_min": va_min, "val_argmin": va.index(va_min) + 1,
             "train_rise_frac": rise, "train_improve_frac": improved}
    if rise > tol:
        return "diverged", facts
    if va.index(va_min) + 1 < len(va) and va[-1] > va_min * (1 + tol):
        return "overfit", facts
    if improved < slow_frac:
        return "slow", facts
    return "stable", facts


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--tol", type=float, default=0.02,
                   help="Relative rise in train loss that counts as divergence.")
    p.add_argument("--slow_frac", type=float, default=0.02,
                   help="Minimum relative train-loss improvement to not be 'slow'.")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist. Run slurm_lr_probe.sh first.")

    rows = []
    for d in sorted(p_ for p_ in root.iterdir() if p_.is_dir()):
        if d.name.startswith("_"):
            continue
        hp = d / "history.json"
        if not hp.exists():
            print(f"[skip] {d.name}: no history.json (unfinished or failed)")
            continue
        m = re.search(r"lr_([0-9.eE+-]+)", d.name)
        if not m:
            print(f"[skip] {d.name}: cannot parse an LR from the name")
            continue
        with open(hp) as f:
            hist = json.load(f)
        verdict, facts = classify(hist, args.tol, args.slow_frac)
        rows.append((float(m.group(1)), verdict, facts, hist))

    if not rows:
        raise SystemExit(f"No finished probe cells under {root}.")
    rows.sort(key=lambda r: r[0])

    print(f"\n{root}   ({len(rows)} learning rates)")
    print(f"{'lr':>9}  {'verdict':<9} {'train first->min->last':<28} "
          f"{'val min':>9} {'@ep':>4}")
    print("-" * 72)
    for lr, verdict, f, _ in rows:
        traj = f"{f['train_first']:.4f} -> {f['train_min']:.4f} -> {f['train_last']:.4f}"
        flag = "  <-- rising" if verdict == "diverged" else ""
        print(f"{lr:>9.1e}  {verdict:<9} {traj:<28} "
              f"{f['val_min']:>9.4f} {f['val_argmin']:>4}{flag}")

    stable = [lr for lr, v, _, _ in rows if v in ("stable", "overfit")]
    print()
    if not stable:
        print("No LR was stable. Extend the ladder downward:")
        print(f"  LRS=\"1e-6 3e-6 1e-5\" DATASET=... sbatch --array=0-2 "
              f"scripts/regression/slurm_lr_probe.sh")
        return
    hi = max(stable)
    print(f"Stable range: {min(stable):.1e} .. {hi:.1e}")
    # Centre the grid at most a factor ~2 below the highest stable LR: that is
    # the usual sweet spot, and staying under `hi` keeps every grid cell inside
    # the range this probe actually verified.
    grid = sorted({float(f"{hi/4:.1e}"), float(f"{hi/2:.1e}"), float(f"{hi:.1e}")})
    print("Suggested search grid (edit LRS in slurm_hparam_search.sh):")
    print("        for lr in " + " ".join(f"{g:.0e}" for g in grid) + "; do")
    diverged = [lr for lr, v, _, _ in rows if v == "diverged"]
    if diverged:
        print(f"Do NOT include: {', '.join(f'{d:.0e}' for d in diverged)} (diverged)")


if __name__ == "__main__":
    main()
