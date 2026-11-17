"""Check that runs meant to be compared were launched with the same settings.

Every trainer dumps its argparse namespace to ``<run_dir>/args.json``
(``src/train/engine.setup_output_dir``). This script walks a results tree, reads
those files and reports, for each key you ask about (default: ``batch_size``),
whether the runs agree:

  1. WITHIN a results root  -- e.g. every cell of ``outputs/cls_hparam/mooc_variants``
     or every arm of ``outputs/cls_mooc``. A mismatch here means cells of one
     sweep / rows of one table are not comparable.
  2. ACROSS roots of the same dataset and task family -- e.g. the
     hyperparameter search ran at batch 8 but the final table at batch 4, so
     the tuned flags were selected under a different optimisation regime than
     the one that is reported. The family is the prefix of the top-level folder
     (``cls_mooc`` and ``cls_hparam/...`` -> ``cls``; ``lp_*`` -> ``lp``), so
     classification runs are never compared against link-prediction runs.

Runs are split by KIND, because only like is compared with like:
  jepa      DyG-JEPA trainer            (args has ``mask_scheme``)
  pretrain  shared-encoder pretraining  (args has ``pretext``)
  finetune  supervised fine-tuning      (args has ``label_fraction``)
  baseline  everything else (CLDG / DDGCL / IDOL / MaskDGNN / ...)
Baselines that sample node batches report ``node_batch_size`` instead of
``batch_size``; a key a run does not have is shown as ``-`` and never counted
as a disagreement.

Nested ``args.json`` files (e.g. ``<run>/collapse_check/args.json`` written by
the analysis step) are skipped: they describe an evaluation, not a training run.

Usage:
    python scripts/check_run_consistency.py                       # batch_size, ./outputs
    python scripts/check_run_consistency.py --root outputs/cls_hparam
    python scripts/check_run_consistency.py --keys batch_size lr epochs
    python scripts/check_run_consistency.py --preset comparability   # everything that
        # must match for two runs to describe the same experiment
    python scripts/check_run_consistency.py --only mooc --kinds jepa --list

Exit status is 1 when any disagreement is found (handy in a submit script).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PRESETS = {
    "batch": ["batch_size"],
    # Settings that change WHAT is being measured; two runs that differ in any
    # of these cannot sit in the same table or share tuned hyperparameters.
    "comparability": [
        "batch_size", "history_size", "num_preds",
        "encoder_kind", "encoder_hidden", "node_dim",
        "with_edge_features", "jodie_num_bins",
        "jodie_top_k_users", "jodie_top_k_items",
        "cldg_num_bins", "cldg_binning", "cldg_top_k", "cldg_file_features",
        "tgbn_weight_features",
        "chase_top_k", "chase_node_features", "chase_feat_norm", "chase_feat_pca",
        "no_attention_mask", "attn_impl",
        "early_stop_patience", "epochs",
    ],
}
MISSING = "-"

# The JODIE top-K knob was removed (always full size since 2026-09-15), so newer
# args.json files have no ``jodie_top_k_*`` key at all. Absent must therefore
# read as "full", not as "unknown": otherwise a stale 3000-user subset run next
# to a current run is never reported. Older runs spelled full size explicitly.
JODIE_FULL = {
    "jodie_mooc": {"jodie_top_k_users": 7047, "jodie_top_k_items": 97},
    "jodie_wikipedia": {"jodie_top_k_users": 8227, "jodie_top_k_items": 1000},
    "jodie_reddit": {"jodie_top_k_users": 10000, "jodie_top_k_items": 984},
}


def normalise(a: dict) -> dict:
    full = JODIE_FULL.get(a.get("data_source"))
    if full:
        for k, n in full.items():
            v = a.get(k)
            a[k] = "full" if v is None or v >= n else v
    return a


def run_kind(a: dict) -> str:
    if "label_fraction" in a:
        return "finetune"
    if "mask_scheme" in a:
        return "jepa"
    if "pretext" in a:
        return "pretrain"
    return "baseline"


def dataset_of(a: dict) -> str:
    ds = a.get("data_source") or ("metr_la" if "metr_la_synthetic" in a else "?")
    if a.get("synthetic"):
        ds += "(synthetic)"
    return str(ds)


def family_of(run: Path, root: Path) -> str:
    """Task family = prefix of the first folder under the scan root."""
    try:
        first = run.relative_to(root).parts[0]
    except (ValueError, IndexError):
        return "?"
    return first.split("_", 1)[0]


def find_runs(root: Path):
    """Yield (run_dir, args) for every top-level args.json under root."""
    for f in sorted(root.rglob("args.json")):
        run = f.parent
        # skip args.json nested inside another run (analysis sub-dirs)
        if any((p / "args.json").exists() for p in run.parents
               if root in p.parents or p == root):
            continue
        try:
            with open(f) as fh:
                a = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[warn] unreadable {f}: {e}", file=sys.stderr)
            continue
        if isinstance(a, dict):
            yield run, a


def fmt_val(v) -> str:
    return MISSING if v is None else str(v)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs", help="results tree to scan")
    p.add_argument("--keys", nargs="+", default=None,
                   help="args.json keys to check (default: batch_size)")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="named key set; 'comparability' = everything that must match")
    p.add_argument("--kinds", nargs="+", default=None,
                   choices=["jepa", "pretrain", "finetune", "baseline"],
                   help="restrict to these run kinds")
    p.add_argument("--only", default=None,
                   help="substring filter on the run path or dataset (e.g. 'mooc')")
    p.add_argument("--list", action="store_true", help="also print every run")
    args = p.parse_args()

    keys = args.keys or PRESETS[args.preset or "batch"]
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist")

    runs = []
    for run, a in find_runs(root):
        kind, ds = run_kind(a), dataset_of(a)
        if args.kinds and kind not in args.kinds:
            continue
        if args.only and args.only not in str(run) and args.only not in ds:
            continue
        runs.append((run, run.parent, kind, f"{family_of(run, root)}:{ds}", normalise(a)))
    if not runs:
        raise SystemExit("no runs found (no args.json under the root / filters)")
    print(f"scanned {len(runs)} runs under {root}   keys: {', '.join(keys)}")

    if args.list:
        print("\n--- runs")
        for run, _, kind, ds, a in runs:
            vals = "  ".join(f"{k}={fmt_val(a.get(k))}" for k in keys)
            print(f"  {kind:<9}{ds:<26}{vals}   {run}")

    n_bad = 0
    for key in keys:
        print(f"\n================ {key} ================")
        # ---- 1) within each results root ---------------------------------------
        by_root = defaultdict(lambda: defaultdict(list))     # (root, kind) -> val -> [runs]
        for run, rroot, kind, ds, a in runs:
            if key in a and a[key] is not None:
                by_root[(rroot, kind)][a[key]].append(run)
        bad_roots = {k: v for k, v in by_root.items() if len(v) > 1}
        print(f"[1] within a results root: "
              f"{len(by_root) - len(bad_roots)}/{len(by_root)} (root, kind) groups agree")
        for (rroot, kind), vals in sorted(bad_roots.items(), key=lambda x: str(x[0])):
            n_bad += 1
            print(f"  MISMATCH  {rroot}  [{kind}]")
            for v, rs in sorted(vals.items(), key=lambda x: -len(x[1])):
                names = ", ".join(r.name for r in rs[:6]) + (" ..." if len(rs) > 6 else "")
                print(f"      {key}={v!s:<10} x{len(rs):<4} {names}")

        # ---- 2) across roots of the same dataset -------------------------------
        by_ds = defaultdict(lambda: defaultdict(set))         # (dataset, kind) -> val -> {roots}
        for run, rroot, kind, ds, a in runs:
            if key in a and a[key] is not None:
                by_ds[(ds, kind)][a[key]].add(rroot)
        bad_ds = {k: v for k, v in by_ds.items() if len(v) > 1}
        print(f"[2] across roots of one task family + dataset: "
              f"{len(by_ds) - len(bad_ds)}/{len(by_ds)} (dataset, kind) groups agree")
        for (ds, kind), vals in sorted(bad_ds.items()):
            n_bad += 1
            print(f"  DIFFERS   {ds}  [{kind}]")
            for v, roots in sorted(vals.items(), key=lambda x: str(x[0])):
                print(f"      {key}={v!s:<10} {', '.join(sorted(str(r) for r in roots))}")

    print()
    if n_bad:
        print(f"{n_bad} disagreement(s). [1] breaks a sweep/table internally; "
              f"[2] means tuned flags were chosen under different settings than the "
              f"runs that report them (not always wrong, but it must be deliberate).")
        sys.exit(1)
    print("all checked groups agree.")


if __name__ == "__main__":
    main()
