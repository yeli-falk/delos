"""Choose a run's checkpoint on the VALIDATION PROBE instead of the val JEPA loss.

Why: ``best.ckpt`` is whatever epoch minimised the validation JEPA loss. With an
EMA target that loss is measured against a moving target, so it is not
comparable across epochs, and nothing ties its minimum to the epoch whose
latents the downstream probe reads best. This script selects on the quantity
that is actually reported -- but on the VALIDATION split, so the test split is
still never used for selection.

Protocol (what ``SELECT_BY_PROBE=1`` in slurm_all_methods.sh runs):

  1. train with ``--snapshot_every N``            -> <run>/snapshots/ep<NNNN>/best.ckpt
  2. this script ``--stage``                      -> copies the loss-selected
     <run>/best.ckpt to <run>/snapshots/loss_best/best.ckpt so every candidate
     lives under snapshots/ and is probed by ONE call
  3. run_block_probe.py --pattern "<run>/snapshots/*" --probe_split val
                                                  -> state_probe_val.json per candidate
  4. this script ``--apply``                      -> picks the candidate with the
     best val metric of the forecast column, keeps the original as
     <run>/best_loss.ckpt, promotes the winner to <run>/best.ckpt and writes
     <run>/ckpt_selection.json
  5. the usual test probe on <run>/best.ckpt      -> <run>/state_probe.json

Ties / missing metrics: a candidate without a finite metric is ignored; if no
snapshot beats it, ``loss_best`` stays selected and nothing is replaced. The
selection is idempotent: once ckpt_selection.json exists, ``--apply`` is a no-op
unless ``--force``.

FAIRNESS NOTE. This uses validation LABELS to pick an epoch of a
self-supervised run. That is ordinary validation-based model selection, but
every method in a table must get the same treatment -- either select all of
them this way or none -- otherwise the comparison is between selection rules.

Usage:
    python scripts/classification/select_ckpt_by_probe.py --run_dir <run> --stage
    python scripts/dblp/run_block_probe.py --pattern "<run>/snapshots/*" \
        --task binary --state_label_mode state --probe_split val
    python scripts/classification/select_ckpt_by_probe.py --run_dir <run> --apply --prune
    python scripts/classification/select_ckpt_by_probe.py --run_dir <run>          # report only
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

LOSS_BEST = "loss_best"


def metric_of(path: Path, column: str, metric: str):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            blob = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    col = blob.get(column)
    if not isinstance(col, dict):
        return None
    v = col.get(metric)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def epoch_of(ckpt: Path):
    """Epoch stored in the checkpoint (1-based), without needing torch on the
    report-only path; falls back to the directory name."""
    try:
        import torch
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        return int(ck.get("epoch", -1)) + 1
    except Exception:                                   # noqa: BLE001
        name = ckpt.parent.name
        return int(name[2:]) if name.startswith("ep") and name[2:].isdigit() else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--stage", action="store_true",
                   help="copy <run>/best.ckpt to snapshots/loss_best/ so it is a candidate")
    p.add_argument("--apply", action="store_true",
                   help="promote the winner to <run>/best.ckpt (original kept as best_loss.ckpt)")
    p.add_argument("--prune", action="store_true",
                   help="with --apply: delete the snapshot .ckpt files afterwards "
                        "(their json results are kept)")
    p.add_argument("--force", action="store_true", help="redo an existing selection")
    p.add_argument("--column", default="jepa", help="probe column to select on")
    p.add_argument("--metric", default="auroc_test",
                   help="metric key ('_test' is the writer's naming; the file is VAL)")
    args = p.parse_args()

    run = Path(args.run_dir)
    snaps = run / "snapshots"
    best = run / "best.ckpt"
    sel_file = run / "ckpt_selection.json"

    if args.stage:
        # After a selection best.ckpt IS a promoted snapshot; the loss-selected
        # one is then best_loss.ckpt. Always stage the loss-selected weights.
        src = run / "best_loss.ckpt" if (run / "best_loss.ckpt").exists() else best
        if not src.exists():
            raise SystemExit(f"{src} missing: nothing to stage")
        d = snaps / LOSS_BEST
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "best.ckpt").exists() or args.force:
            shutil.copy2(src, d / "best.ckpt")
        print(f"[stage] {src.name} -> {d / 'best.ckpt'}")
        if not args.apply:
            return

    if not snaps.exists():
        raise SystemExit(f"{snaps} missing: train with --snapshot_every N first")

    cands = []
    for d in sorted(x for x in snaps.iterdir() if x.is_dir()):
        v = metric_of(d / "state_probe_val.json", args.column, args.metric)
        cands.append({"name": d.name, "value": v,
                      "has_ckpt": (d / "best.ckpt").exists()})
    scored = [c for c in cands if c["value"] is not None]
    if not scored:
        raise SystemExit(f"no candidate under {snaps} has a val probe result; run "
                         f"run_block_probe.py --pattern \"{snaps}/*\" --probe_split val")

    # Winner: best val metric; on a tie prefer loss_best (change nothing).
    winner = max(scored, key=lambda c: (c["value"], c["name"] == LOSS_BEST))
    base = next((c for c in scored if c["name"] == LOSS_BEST), None)

    print(f"\n{run}   select on VAL {args.column}.{args.metric}")
    for c in cands:
        mark = " <- selected" if c is winner else ""
        val = "   --  " if c["value"] is None else f"{c['value']:.4f}"
        print(f"  {c['name']:<12}{val}{mark}")
    if base is not None:
        print(f"  gain over loss-selected: {winner['value'] - base['value']:+.4f}")
    unscored = [c["name"] for c in cands if c["value"] is None]
    if unscored:
        print(f"  [warn] no val probe for: {', '.join(unscored)}")

    if not args.apply:
        return
    if sel_file.exists() and not args.force:
        print(f"[skip] {sel_file.name} exists (use --force to redo)")
        return
    src = snaps / winner["name"] / "best.ckpt"
    if not src.exists():
        raise SystemExit(f"{src} missing (pruned already?) -- cannot promote")

    if not (run / "best_loss.ckpt").exists() and best.exists():
        shutil.copy2(best, run / "best_loss.ckpt")
    if winner["name"] != LOSS_BEST:
        shutil.copy2(src, best)
        # The test probe of the previous best.ckpt no longer describes best.ckpt.
        for stale in ("state_probe.json", "state_probe_val.json"):
            f = run / stale
            if f.exists():
                f.rename(run / f"{f.stem}.loss_selected.json")
    record = {"selected": winner["name"], "selected_epoch": epoch_of(src),
              "column": args.column, "metric": args.metric, "split": "val",
              "selected_value": winner["value"],
              "loss_best_value": None if base is None else base["value"],
              "candidates": {c["name"]: c["value"] for c in cands}}
    with open(sel_file, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[apply] best.ckpt <- snapshots/{winner['name']}  "
          f"(epoch {record['selected_epoch']}); original kept as best_loss.ckpt")

    if args.prune:
        n = 0
        for d in snaps.iterdir():
            ck = d / "best.ckpt"
            if ck.exists():
                ck.unlink(); n += 1
        print(f"[prune] removed {n} snapshot checkpoints (json results kept)")


if __name__ == "__main__":
    main()
