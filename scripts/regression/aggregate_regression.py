"""Aggregate node-signal forecasting results into a mean +/- std LaTeX table.

Reads the per-seed ``test_metrics.json`` every arm of
``scripts/regression/slurm_all_methods.sh`` writes, groups them by method,
and reports mean +/- sample std across seeds.

Layout it expects (produced by that launcher):

    outputs/reg_<dataset>/
      baselines/baseline_metrics.json        persistence / window_mean /
                                             historical_mean -- parameter-free,
                                             so identical across seeds and
                                             reported without a std
      supervised_s<seed>/test_metrics.json   supervised end-to-end
      jepa_{frozen,naive,ema,vicreg}_s<seed>/test_metrics.json
      jepa_enc_s<seed>/test_metrics.json     encoder + head, no predictor
      {grace,graphmae,bgrl,cldg,ddgcl,idol,maskdgnn,dysubc,dyg2vec,dygwm,sgjepa}_s<seed>/test_metrics.json
      step_s<seed>/test_metrics.json         STEP (TSFormer pre-training + a
                                             learned graph + its own Graph
                                             WaveNet backend)
      stdmae_s<seed>/test_metrics.json       STD-MAE (S-MAE + T-MAE pre-training
                                             + its own Graph WaveNet backend)

Horizons default to the dataset's published reporting convention:
METR-LA at +3 / +6 / +12 steps (15 / 30 / 60 min, the DCRNN protocol),
England COVID at every horizon it forecasts.

Usage:
    python scripts/regression/aggregate_regression.py --dataset metr_la \
        --seeds 42 43 44 --latex
    python scripts/regression/aggregate_regression.py --dataset england_covid \
        --seeds 42 43 44 --metrics MAE RMSE MAPE --latex
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

# Display name -> directory stem under outputs/reg_<dataset>/.
# Ordered as the table should read: parameter-free, supervised, contrastive SSL,
# then the DyG-JEPA arms.
METHODS = [
    ("Persistence",             "baseline:persistence"),
    ("Window mean",             "baseline:window_mean"),
    ("Historical mean",         "baseline:historical_mean"),
    ("Supervised (end-to-end)", "supervised"),
    ("GRACE (static)",          "grace"),
    ("GraphMAE (static)",       "graphmae"),
    ("BGRL (static)",           "bgrl"),
    ("CLDG",                    "cldg"),
    ("DDGCL",                   "ddgcl"),
    ("IDOL",                    "idol"),
    ("MaskDGNN",                "maskdgnn"),
    ("DySubC",                  "dysubc"),
    ("DyG2Vec",                 "dyg2vec"),
    ("DyG-WM (SSL)",            "dygwm"),
    ("SG-JEPA",                 "sgjepa"),
    ("STEP",                    "step"),
    ("STD-MAE",                 "stdmae"),
    ("DyG-JEPA (enc. only)",    "jepa_enc"),
    ("DyG-JEPA (frozen)",       "jepa_frozen"),
    ("DyG-JEPA (naive)",        "jepa_naive"),
    ("DyG-JEPA (EMA)",          "jepa_ema"),
    ("DyG-JEPA (VICReg)",       "jepa_vicreg"),
]

# Horizons reported per dataset, as 0-indexed forecast steps.
DEFAULT_HORIZONS = {
    "metr_la": [2, 5, 11],          # +3 / +6 / +12 steps = 15 / 30 / 60 min
    "england_covid": [0, 1, 2],
}
# Human-facing horizon labels.
HORIZON_LABEL = {
    "metr_la": {2: "15 min", 5: "30 min", 11: "60 min"},
    "england_covid": {0: "+1 d", 1: "+2 d", 2: "+3 d"},
}


def _load(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect(root: Path, stem: str, seeds: list[int]) -> tuple[list[dict], list[int]]:
    """Per-seed metric dicts for one method, plus the seeds that were missing."""
    if stem.startswith("baseline:"):
        blob = _load(root / "baselines" / "baseline_metrics.json")
        name = stem.split(":", 1)[1]
        if not blob or name not in blob:
            return [], list(seeds)
        # Parameter-free: one deterministic result, no seed variance.
        return [blob[name]], []
    runs, missing = [], []
    for s in seeds:
        m = _load(root / f"{stem}_s{s}" / "test_metrics.json")
        (runs if m else missing).append(m if m else s)
    return runs, missing


def stat(runs: list[dict], horizon: int, metric: str):
    """(mean, std, n) of ``metric`` at ``horizon`` across seeds."""
    vals = []
    for r in runs:
        # test_metrics.json keys are horizons; JSON turns int keys into strings.
        entry = r.get(str(horizon), r.get(horizon))
        if isinstance(entry, dict) and metric in entry:
            v = entry[metric]
            if v is not None and v == v:            # drop NaN (empty subset)
                vals.append(float(v))
    if not vals:
        return None
    return (st.mean(vals),
            st.stdev(vals) if len(vals) >= 2 else 0.0,
            len(vals))


def persistence_check(root: Path, seeds: list[int], horizons: list[int]) -> None:
    """Warn if the arms disagree on the persistence reference.

    ``MAE_persist`` is a parameter-free estimator of the same test windows, so
    every arm must report the same value. A disagreement means the arms did not
    score identical windows -- different history length, different split, a
    stale checkpoint -- and the table is comparing different experiments. This
    has to be loud: the resulting numbers look perfectly plausible.
    """
    seen: dict[int, dict[str, float]] = {h: {} for h in horizons}
    for label, stem in METHODS:
        if stem.startswith("baseline:"):
            continue
        runs, _ = collect(root, stem, seeds)
        for h in horizons:
            s = stat(runs, h, "MAE_persist")
            if s:
                seen[h][label] = s[0]
    for h in horizons:
        vals = seen[h]
        if len(vals) < 2:
            continue
        lo, hi = min(vals.values()), max(vals.values())
        if hi - lo > 1e-3 * max(abs(hi), 1.0):
            print(f"[WARN] horizon {h}: arms disagree on the persistence "
                  f"reference (MAE_persist {lo:.4f}..{hi:.4f}). The arms are "
                  f"NOT scoring the same windows; do not merge these rows.")
            for k, v in sorted(vals.items(), key=lambda kv: kv[1]):
                print(f"         {k:<24} {v:.4f}")


def fmt(cell, prec: int) -> str:
    if cell is None:
        return "--"
    mu, sd, n = cell
    return f"{mu:.{prec}f} ± {sd:.{prec}f}" if n >= 2 else f"{mu:.{prec}f}"


def latex_cell(cell, prec: int, best: bool) -> str:
    if cell is None:
        return "--"
    mu, sd, n = cell
    body = (f"{mu:.{prec}f} \\pm {sd:.{prec}f}" if n >= 2 else f"{mu:.{prec}f}")
    return f"$\\mathbf{{{body}}}$" if best else f"${body}$"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   choices=["metr_la", "england_covid"])
    p.add_argument("--root", default=None,
                   help="Results root (default: outputs/reg_<dataset>).")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--horizons", type=int, nargs="+", default=None,
                   help="0-indexed forecast steps (default: this dataset's "
                        "published reporting convention).")
    p.add_argument("--metrics", nargs="+", default=["MAE", "RMSE"],
                   help="Any of MAE RMSE MAPE MAE_inflect.")
    p.add_argument("--precision", type=int, default=3)
    p.add_argument("--latex", action="store_true",
                   help="Also emit a booktabs table body.")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    root = Path(args.root or f"outputs/reg_{args.dataset}")
    if not root.exists():
        raise SystemExit(f"{root} does not exist. Run "
                         f"scripts/regression/slurm_all_methods.sh first.")
    horizons = args.horizons or DEFAULT_HORIZONS[args.dataset]
    labels = HORIZON_LABEL.get(args.dataset, {})

    rows, blob = [], {}
    for label, stem in METHODS:
        runs, missing = collect(root, stem, args.seeds)
        if not runs:
            print(f"[skip] {label}: no results under {root}/{stem}_s*")
            continue
        if missing:
            print(f"[warn] {label}: missing seeds {missing} "
                  f"(mean over {len(runs)} of {len(args.seeds)})")
        cells = {(h, m): stat(runs, h, m) for h in horizons for m in args.metrics}
        rows.append((label, cells, len(runs)))
        blob[label] = {
            f"h{h}_{m}": (None if cells[(h, m)] is None else
                          {"mean": cells[(h, m)][0], "std": cells[(h, m)][1],
                           "n": cells[(h, m)][2]})
            for h in horizons for m in args.metrics}

    if not rows:
        raise SystemExit(f"No results found under {root}.")

    print()
    persistence_check(root, args.seeds, horizons)

    # --- plain-text view ----------------------------------------------------
    head = f"{'method':<26}"
    for h in horizons:
        for m in args.metrics:
            head += f"{labels.get(h, f'+{h+1}') + ' ' + m:>26}"
    print(f"\n{args.dataset}  (seeds {args.seeds}, lower is better)")
    print(head)
    print("-" * len(head))
    for label, cells, n in rows:
        line = f"{label:<26}"
        for h in horizons:
            for m in args.metrics:
                line += f"{fmt(cells[(h, m)], args.precision):>26}"
        print(line + f"  n={n}")

    # --- LaTeX --------------------------------------------------------------
    if args.latex:
        # Bold the best (lowest) mean per column, over the LEARNED methods only:
        # bolding a parameter-free baseline reads as "our method lost" when the
        # point of the row is to be the reference line.
        learned = [r for r in rows if not r[0].endswith("mean")
                   and r[0] != "Persistence"]
        best = {}
        for h in horizons:
            for m in args.metrics:
                vals = [(c[(h, m)][0], lab) for lab, c, _ in learned
                        if c[(h, m)] is not None]
                if vals:
                    best[(h, m)] = min(vals)[1]

        ncols = len(horizons) * len(args.metrics)
        print("\n% --- paste into your results table -----------------------")
        print("\\begin{tabular}{l" + "r" * ncols + "}")
        print("\\toprule")
        hdr = "Method"
        for h in horizons:
            hdr += (f" & \\multicolumn{{{len(args.metrics)}}}{{c}}"
                    f"{{{labels.get(h, f'+{h+1}')}}}")
        print(hdr + " \\\\")
        start = 2
        cmids = ""
        for _ in horizons:
            cmids += f"\\cmidrule(lr){{{start}-{start+len(args.metrics)-1}}}"
            start += len(args.metrics)
        print(cmids)
        print(" " + " & ".join([""] + args.metrics * len(horizons)) + " \\\\")
        print("\\midrule")
        for i, (label, cells, _) in enumerate(rows):
            # Rule off the parameter-free block from the learned methods.
            if i and rows[i - 1][0] == "Historical mean":
                print("\\midrule")
            cs = [latex_cell(cells[(h, m)], args.precision,
                             best.get((h, m)) == label)
                  for h in horizons for m in args.metrics]
            print(f"{label} & " + " & ".join(cs) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print(f"% mean $\\pm$ std over {len(args.seeds)} seeds "
              f"({', '.join(map(str, args.seeds))}); lower is better.")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(blob, f, indent=2)
        print(f"\n[saved] {args.json_out}")


if __name__ == "__main__":
    main()
