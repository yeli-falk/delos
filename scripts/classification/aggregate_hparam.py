"""Rank the node-classification hyperparameter-search cells.

The classification twin of ``scripts/regression/aggregate_hparam.py``. Scans a
sweep root written by ``scripts/classification/slurm_hparam_search.sh``. Every
cell there probed the VALIDATION split, so ranking on its numbers does not touch
test.

Two differences from the regression version, both load-bearing:

* **Higher is better.** AUROC/AUPRC, not MAE -- the sort is descending.
* **Cells are ranked on the paired gap when one exists.** For the JEPA stages a
  cell reports both ``jepa`` (the predictor rollout) and ``encoder_last``
  (copy-last) from ONE encoder, so ``gap = jepa - encoder_last`` is a
  within-cell difference with far less variance than either column. Ranking on
  raw AUROC would mostly rank encoders; ranking on the gap ranks *predictors*,
  which is what these stages are tuning. ``--rank_by auroc`` overrides.

Baselines are encoder-only (no ``jepa`` column), so they are always ranked on
their raw ``encoder_last`` metric.

Usage:
    python scripts/classification/aggregate_hparam.py \
        --root outputs/cls_hparam/mooc_predictor --emit_flags
    python scripts/classification/aggregate_hparam.py \
        --root outputs/cls_hparam/mooc_variants --by_arm --emit_flags
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
from pathlib import Path

FORECAST_COL = "jepa"
LAST_COL = "encoder_last"
DEGREE_COL = "degree_history_baseline"


def _get(blob, column, metric):
    """A metric spelled "subset_<name>.<metric>" reads the nested subset dict
    (multiclass state probes: e.g. subset_switch.acc_test)."""
    col = blob.get(column) if isinstance(blob, dict) else None
    if "." in metric:
        sub, metric = metric.split(".", 1)
        col = col.get(sub) if isinstance(col, dict) else None
    if not isinstance(col, dict) or metric not in col:
        return None
    v = col[metric]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def infer_metric(root: Path) -> str:
    """The probe task decides which columns exist: binary writes auroc_test,
    multiclass acc_test / macro_f1_test, ranking ndcg10_test. Read the first
    finished cell and pick in that order, so a chase_music / synthetic_dyn /
    cldg_* / tgbn_* root aggregates without --metric."""
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        mpath = d / "state_probe_val.json"
        if d.name.startswith("_") or not mpath.exists():
            continue
        with open(mpath) as f:
            blob = json.load(f)
        col = blob.get(LAST_COL) or blob.get(FORECAST_COL) or {}
        for m in ("auroc_test", "acc_test", "ndcg10_test"):
            if isinstance(col, dict) and isinstance(col.get(m), (int, float)):
                return m
    return "auroc_test"


def collect(root: Path, metric: str):
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.startswith("_"):          # _shared_encoder_s*
            continue
        mpath = d / "state_probe_val.json"
        if not mpath.exists():
            print(f"[skip] {d.name}: no state_probe_val.json (unfinished or failed)")
            continue
        with open(mpath) as f:
            blob = json.load(f)
        fc = _get(blob, FORECAST_COL, metric)
        last = _get(blob, LAST_COL, metric)
        deg = _get(blob, DEGREE_COL, metric)
        if fc is None and last is None:
            # The file exists but carries no `metric` column: almost always a
            # multiclass / ranking root aggregated with the default AUROC.
            have = sorted(k for k in (blob.get(LAST_COL) or {})
                          if isinstance(blob.get(LAST_COL), dict)
                          and isinstance(blob[LAST_COL].get(k), (int, float)))
            print(f"[skip] {d.name}: state_probe_val.json has no '{metric}' "
                  f"(available: {', '.join(have) or 'none'}) -- pass --metric")
            continue
        hp = ""
        hp_file = d / "hp_args.txt"
        if hp_file.exists():
            hp = hp_file.read_text().strip()
        # Arm from hp_args.txt ("arm=<name> flags=..."): arm names may contain
        # underscores (ema_cos, lejepa_full), so splitting the dir name at the
        # first "_" would file them under the wrong arm.
        arm = d.name.split("_", 1)[0]
        if hp.startswith("arm="):
            arm = hp.split()[0][len("arm="):]
        m = re.search(r"_s(\d+)$", d.name)
        rows.append({"cell": d.name, "arm": arm,
                     "config": d.name[:m.start()] if m else d.name,
                     "seed": int(m.group(1)) if m else None,
                     "val_forecast": fc, "val_last": last,
                     "gap": (None if (fc is None or last is None) else fc - last),
                     "val_degree": deg, "hp": hp})
    return rows


def average_seeds(rows):
    """Collapse cells that differ only in the seed into one row (mean per
    column, std of the forecast and of the gap, n seeds). Single-seed sweeps
    come out unchanged apart from the ``n`` column.

    Single-seed validation noise is of the same order as the differences a
    sweep ranks, so a second seed is what makes the ranking meaningful.
    """
    groups = {}
    for r in rows:
        groups.setdefault((r["arm"], r["config"]), []).append(r)
    out = []
    for (arm, config), rs in groups.items():
        def mean(k):
            v = [r[k] for r in rs if r[k] is not None]
            return sum(v) / len(v) if v else None
        def std(k):
            v = [r[k] for r in rs if r[k] is not None]
            return st.stdev(v) if len(v) >= 2 else 0.0
        out.append({"cell": config, "arm": arm, "config": config,
                    "n": len(rs), "seeds": sorted(r["seed"] for r in rs if r["seed"] is not None),
                    "val_forecast": mean("val_forecast"), "val_last": mean("val_last"),
                    "gap": mean("gap"), "val_degree": mean("val_degree"),
                    "std_forecast": std("val_forecast"), "std_gap": std("gap"),
                    "hp": rs[0]["hp"]})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--metric", default=None,
                   help="Key inside each column (auroc_test / auprc_test / acc_test "
                        "/ ndcg10_test ...). Default: inferred from the first finished "
                        "cell -- auroc_test for a binary probe, acc_test for a "
                        "multiclass one, ndcg10_test for a ranking one. The "
                        "'_test' suffix is the writer's naming; the file is the "
                        "VAL split.")
    p.add_argument("--rank_by", choices=["auto", "gap", "auroc"], default="auto",
                   help="'gap' ranks the predictor's lift over copy-last, paired "
                        "within the cell -- meaningful ONLY when every cell "
                        "shares one encoder (STAGE=predictor). 'auroc' ranks the "
                        "raw forecast column. 'auto' (default) uses the gap when "
                        "copy-last is identical across cells and the forecast "
                        "otherwise: once the encoder trains, copy-last moves per "
                        "cell and a large gap can just mean a WORSE encoder.")
    p.add_argument("--by_arm", action="store_true")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--csv", default=None)
    p.add_argument("--flag_prefix", default=None)
    p.add_argument("--emit_flags", action="store_true")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist.")
    if args.metric is None:
        args.metric = infer_metric(root)
        print(f"[metric] {args.metric} (inferred from the cells; override with --metric)")
    rows = collect(root, args.metric)
    if not rows:
        raise SystemExit(f"No finished cells under {root}.")

    n_cells = len(rows)
    rows = average_seeds(rows)
    multi = any(r["n"] > 1 for r in rows)

    has_gap = any(r["gap"] is not None for r in rows)
    rank_by = args.rank_by
    if rank_by == "auto":
        lasts = [r["val_last"] for r in rows
                 if r["val_last"] is not None and r["val_forecast"] is not None]
        # The threshold is not ~0: even with one frozen encoder the copy-last
        # probe is refit per cell and wobbles slightly.
        shared_encoder = bool(lasts) and (max(lasts) - min(lasts) < 0.01)
        rank_by = "gap" if (has_gap and shared_encoder) else "auroc"
        if has_gap and not shared_encoder:
            print(f"[rank] copy-last differs across cells "
                  f"({min(lasts):.4f} .. {max(lasts):.4f}): the encoder is not shared, "
                  f"so ranking on the FORECAST, not the gap (override: --rank_by gap).")
    key = "gap" if (rank_by == "gap" and has_gap) else "val_forecast"
    if key == "val_forecast" and not any(r["val_forecast"] is not None for r in rows):
        key = "val_last"          # encoder-only sweep (baselines)
    rows = [r for r in rows if r[key] is not None]
    rows.sort(key=lambda r: r[key], reverse=True)          # HIGHER is better

    print(f"\n{len(rows)} configurations ({n_cells} cells) under {root}  "
          f"(ranked on VAL {key}, HIGHER is better"
          f"{', mean over seeds' if multi else ''})")
    # Width follows the longest cell name so that long names are not cut.
    cw = max(50, max(len(r["cell"]) for r in rows) + 2)
    print(f"{'rank':<5}{'cell':<{cw}}{'fcast':>9}{'±':>8}{'last':>9}{'gap':>9}{'degree':>9}{'n':>4}")
    print("-" * 103)
    def _f(v): return f"{v:.4f}" if v is not None else "   --  "
    for i, r in enumerate(rows[:args.top], 1):
        sd = f"{r['std_forecast']:.4f}" if r["n"] > 1 else "   -- "
        print(f"{i:<5}{r['cell']:<{cw}}{_f(r['val_forecast']):>9}{sd:>8}"
              f"{_f(r['val_last']):>9}{_f(r['gap']):>9}{_f(r['val_degree']):>9}{r['n']:>4}")
    if multi:
        ns = sorted({r["n"] for r in rows})
        if len(ns) > 1:
            print(f"[warn] configurations have different seed counts {ns}: "
                  f"means over fewer seeds are noisier; finish the missing cells "
                  f"before trusting the order.")
    elif len(rows) > 1:
        print("[note] one seed per configuration: differences below ~0.01-0.02 are "
              "within single-seed validation noise.")

    if has_gap:
        beaten = [r for r in rows if (r["gap"] or 0) > 0]
        print(f"\n{len(beaten)}/{len(rows)} cells have the forecast BEATING copy-last on val.")
        if not beaten:
            print("  None do. The predictor is adding nothing over the frozen")
            print("  encoder's last-bin latent; ranking these is ranking noise.")
    # A cell that cannot beat the degree-only control is not showing learned
    # dynamics, just 'this user is winding down'.
    if rows and rows[0]["val_degree"] is not None:
        top_val = rows[0]["val_forecast"] or rows[0]["val_last"]
        if top_val is not None and top_val <= rows[0]["val_degree"]:
            print(f"\n[WARN] the top cell ({top_val:.4f}) does not beat the "
                  f"degree-history control ({rows[0]['val_degree']:.4f}). "
                  f"That is an activity-trend result, not a dynamics result.")

    winners = {}
    if args.by_arm:
        aw = max(12, max(len(r["arm"]) for r in rows) + 2)
        print(f"\n{'arm':<{aw}}{'best cell':<{cw}}{'score':>10}")
        print("-" * 72)
        for arm in sorted({r["arm"] for r in rows}):
            best = max((r for r in rows if r["arm"] == arm), key=lambda r: r[key])
            winners[arm] = best
            print(f"{arm:<{aw}}{best['cell']:<{cw}}{best[key]:>10.4f}")
    else:
        winners["best"] = rows[0]

    if args.emit_flags:
        prefix = args.flag_prefix or (
            "BASELINE_HP" if root.name.endswith("_baselines") else "VARIANT_HP")
        print("\n# --- paste into scripts/classification/slurm_all_methods.sh ---")
        for arm, r in winners.items():
            flags = r["hp"].split("flags=", 1)[-1].split("base=")[0].strip()
            if args.by_arm:
                print(f'{prefix}_{arm}="{flags}"')
            else:
                print(f'BASE_HP="{flags}"')

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"\n[saved] {args.csv}")


if __name__ == "__main__":
    main()
