"""Rank the node-signal forecasting hyperparameter-search cells.

Scans a sweep root written by ``scripts/regression/slurm_hparam_search.sh``.
Every cell there probed the VALIDATION split, so ranking on its numbers does
not touch test.

Cells are ranked by validation forecast MAE, averaged over the reported
horizons, and shown against that cell's own persistence reference — a
configuration that does not beat persistence has not learned to forecast, no
matter how it places in the ranking.

Usage:
    python scripts/regression/aggregate_hparam.py \
        --root outputs/reg_hparam/metr_la_predictor
    python scripts/regression/aggregate_hparam.py \
        --root outputs/reg_hparam/metr_la_variants --by_arm
    # emit the winner as flags to paste into slurm_all_methods.sh
    python scripts/regression/aggregate_hparam.py --root ... --emit_flags
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path


def cell_score(metrics: dict, horizons: list[int] | None):
    """(mean MAE, mean persistence MAE) over the scored horizons."""
    keys = sorted(metrics, key=lambda k: int(k))
    if horizons is not None:
        keys = [k for k in keys if int(k) in horizons]
    mae = [metrics[k]["MAE"] for k in keys if "MAE" in metrics[k]]
    per = [metrics[k]["MAE_persist"] for k in keys if "MAE_persist" in metrics[k]]
    if not mae:
        return None, None
    return st.mean(mae), (st.mean(per) if per else float("nan"))


def collect(root: Path, horizons):
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.startswith("_"):          # _shared_encoder_s*
            continue
        mpath = d / "test_metrics.json"
        if not mpath.exists():
            print(f"[skip] {d.name}: no metrics (unfinished or failed)")
            continue
        with open(mpath) as f:
            metrics = json.load(f)
        mae, per = cell_score(metrics, horizons)
        if mae is None:
            continue
        hp = ""
        hp_file = d / "hp_args.txt"
        if hp_file.exists():
            hp = hp_file.read_text().strip()
        arm = d.name.split("_", 1)[0]
        rows.append({"cell": d.name, "arm": arm, "val_MAE": mae,
                     "val_MAE_persist": per, "gain_vs_persist": per - mae,
                     "hp": hp})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--horizons", type=int, nargs="+", default=None,
                   help="Restrict the ranking to these 0-indexed steps.")
    p.add_argument("--by_arm", action="store_true",
                   help="Report the winner within each variant arm (use after "
                        "STAGE=variants).")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--csv", default=None)
    p.add_argument("--flag_prefix", default=None,
                   help="Env-var prefix for --emit_flags. Default: inferred "
                        "from the sweep root (VARIANT_HP for the JEPA variants, "
                        "BASELINE_HP for the baseline sweep).")
    p.add_argument("--emit_flags", action="store_true",
                   help="Print the winner(s) as shell assignments ready to "
                        "paste into slurm_all_methods.sh.")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist.")
    rows = collect(root, args.horizons)
    if not rows:
        raise SystemExit(f"No finished cells under {root}.")
    rows.sort(key=lambda r: r["val_MAE"])

    print(f"\n{len(rows)} cells under {root}  (ranked on VAL, lower is better)")
    print(f"{'rank':<5}{'cell':<52}{'val_MAE':>10}{'persist':>10}{'gain':>9}")
    print("-" * 86)
    for i, r in enumerate(rows[:args.top], 1):
        print(f"{i:<5}{r['cell'][:50]:<52}{r['val_MAE']:>10.4f}"
              f"{r['val_MAE_persist']:>10.4f}{r['gain_vs_persist']:>+9.4f}")

    beaten = [r for r in rows if r["gain_vs_persist"] > 0]
    print(f"\n{len(beaten)}/{len(rows)} cells beat persistence on val.")
    if not beaten:
        print("  None did. Tune the recipe (--persist_residual, --residual_head,")
        print("  --change_weight) before reading anything into this ranking.")

    winners = {}
    if args.by_arm:
        print(f"\n{'arm':<12}{'best cell':<52}{'val_MAE':>10}{'gain':>9}")
        print("-" * 81)
        for arm in sorted({r["arm"] for r in rows}):
            best = min((r for r in rows if r["arm"] == arm),
                       key=lambda r: r["val_MAE"])
            winners[arm] = best
            print(f"{arm:<12}{best['cell'][:50]:<52}{best['val_MAE']:>10.4f}"
                  f"{best['gain_vs_persist']:>+9.4f}")
    else:
        winners["best"] = rows[0]

    if args.emit_flags:
        prefix = args.flag_prefix or (
            "BASELINE_HP" if root.name.endswith("_baselines") else "VARIANT_HP")
        print("\n# --- paste into scripts/regression/slurm_all_methods.sh ---")
        for arm, r in winners.items():
            # hp_args.txt line looks like: arm=ema flags=... base=...
            flags = r["hp"].split("flags=", 1)[-1].split("base=")[0].strip()
            if args.by_arm:
                print(f'{prefix}_{arm}="{flags}"')
            else:
                print(f'BASE_HP="{flags}"')

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[saved] {args.csv}")


if __name__ == "__main__":
    main()
