"""Aggregate dynamic link-prediction results into a mean +/- std LaTeX table.

Third of the three task aggregators, alongside
``scripts/regression/aggregate_regression.py`` and
``scripts/classification/aggregate_classification.py``.

Two writers produce results, and they share a schema but not a filename:

  DyG-JEPA arms   train_dyg_jepa.py --downstream_task link_pred
                  -> <run>/test_metrics.json
  baselines and   analyze_jepa.py --link_sources ...
  controls        -> <run>/analysis.json, under ["link_pred"][source]

Both are ``{"all": {horizon: {MRR, Hits@1, Hits@3, Hits@10, ...}}, "new": {...}}``,
so this reads either and reports them in one table.

The two edge subsets answer different questions and BOTH belong in the paper:

  all   every positive edge in the horizon. Dominated by edges that already
        exist in the history window, which a model can score well by echoing
        the current graph.
  new   only edges absent from every earlier snapshot of the window. Formation,
        not persistence -- the link-prediction analogue of the forecasting
        inflection subset, and the harder number.

Metrics are averaged over horizons (one number per metric), matching how the
existing scripts/dblp/aggregate_seeds.py reports them.

Usage:
    python scripts/linkpred/aggregate_linkpred.py --dataset mooc --seeds 42 43 44 --latex
    python scripts/linkpred/aggregate_linkpred.py --dataset reddit --seeds 42 43 44 \
        --subset new --metrics MRR Hits@10
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

# Display name -> (directory stem, source key inside analysis.json or None).
# None means "read test_metrics.json directly" (the DyG-JEPA arms).
METHODS = [
    ("Random",              ("controls", "random")),
    ("Supervised (end-to-end)", ("controls", "supervised")),
    # MaskDGNN's own encoder + edge head (the paper's protocol), written by
    # its trainer as link_e2e[_val].json; the frozen-probe row is below.
    ("MaskDGNN (end-to-end)", ("maskdgnn", "e2e")),
    ("CLDG",                ("cldg", "encoder")),
    ("DDGCL",               ("ddgcl", "encoder")),
    ("IDOL",                ("idol", "encoder")),
    ("MaskDGNN",            ("maskdgnn", "encoder")),
    ("DyG-JEPA (frozen)",   ("jepa_frozen", None)),
    ("DyG-JEPA (naive)",    ("jepa_naive", None)),
    ("DyG-JEPA (EMA)",      ("jepa_ema", None)),
    ("DyG-JEPA (VICReg)",   ("jepa_vicreg", None)),
]
# Four ranking metrics; AUC is computed by link_ranking_metrics and can be
# added back with --metrics, but the paper table does not carry it.
DEFAULT_METRICS = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
# Header labels for the paper table. The JSON keys stay `Hits@k`; only the
# printed column names are shortened, matching the template.
METRIC_LABEL = {"Hits@1": "H@1", "Hits@3": "H@3", "Hits@10": "H@10"}
# Display names for the dataset column of the combined table.
DATASET_LABEL = {"mooc": "MOOC", "reddit": "Reddit",
                 "wikipedia": "Wikipedia", "dblp": "DBLP",
                 "reddit_body": "Reddit-Body", "reddit_title": "Reddit-Title"}


def _load(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def read_run(root: Path, stem: str, source, seed: int, split: str):
    """One run's {"all":..., "new":...} block, from either writer."""
    suffix = "" if split == "test" else "_val"
    d = root / f"{stem}_s{seed}"
    if source is None:
        return _load(d / "test_metrics.json")
    if source == "e2e":
        return _load(d / f"link_e2e{suffix}.json")
    blob = _load(d / f"analysis{suffix}.json")
    if not blob:
        return None
    return (blob.get("link_pred") or {}).get(source) or None


def metric_over_horizons(block, subset: str, metric: str):
    """Mean of ``metric`` across horizons within one run, or None."""
    sub = (block or {}).get(subset)
    if not isinstance(sub, dict):
        return None
    vals = []
    for _, m in sub.items():
        if isinstance(m, dict) and metric in m:
            v = m[metric]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vals.append(float(v))
    return st.mean(vals) if vals else None


def agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return (st.mean(vals), st.stdev(vals) if len(vals) >= 2 else 0.0, len(vals))


def fmt(cell, prec):
    if cell is None:
        return "--"
    mu, sd, n = cell
    return f"{mu:.{prec}f} ± {sd:.{prec}f}" if n >= 2 else f"{mu:.{prec}f}"


def latex_cell(cell, prec, best=False, sd_prec=None):
    """One table cell: ``$0.254_{0.01}$``, or ``$\\mathbf{...}$`` when best.

    ``prec`` sets the mean's decimals, ``sd_prec`` the subscript's (default 2).
    The subscript is not decoration: at 3 seeds several between-method gaps in
    these tables are smaller than the spread, and a bare mean would present
    those as wins. Note a std below 0.005 renders as ``0.00`` at 2 decimals.
    """
    if cell is None:
        return "--"
    if sd_prec is None:
        sd_prec = 2
    mu, sd, n = cell
    body = f"{mu:.{prec}f}_{{{sd:.{sd_prec}f}}}" if n >= 2 else f"{mu:.{prec}f}"
    return f"$\\mathbf{{{body}}}$" if best else f"${body}$"


def combined_latex(per_ds, metrics, subsets, prec, sd_prec=2):
    """The paper table: datasets stacked with \\multirow, both subsets across.

    Bolding is per column WITHIN a dataset, and excludes Random -- a floor is
    not a competitor, and bolding it would read as "every method lost".
    """
    ncol = len(metrics) * len(subsets)
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Link-prediction performance across datasets and methods. The "
          r"\emph{all} setting scores every positive edge at the target snapshot; "
          r"the \emph{new} setting restricts to edges absent at every prior "
          r"history step in the window. Best per column within a dataset in "
          r"\textbf{bold}.}")
    print(r"\label{tab:results_combined}")
    print(r"\footnotesize")
    print(r"\setlength{\tabcolsep}{5pt}")
    print(r"\renewcommand{\arraystretch}{1.05}")
    print(r"% wrap in \resizebox{\textwidth}{!}{ ... } if it overflows \textwidth")
    print(r"\begin{tabular}{l l " + " ".join("c" * len(metrics) for _ in subsets) + "}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Method}")
    print("  & " + " & ".join(
        rf"\multicolumn{{{len(metrics)}}}{{c}}{{{sub}}}" for sub in subsets) + r" \\")
    start = 3
    print(" ".join(rf"\cmidrule(lr){{{start + i*len(metrics)}-"
                   rf"{start + (i+1)*len(metrics) - 1}}}" for i in range(len(subsets))))
    print(" & & " + " & ".join(METRIC_LABEL.get(m, m)
                               for _ in subsets for m in metrics) + r" \\")
    print(r"\midrule")
    for di, (ds, rows) in enumerate(per_ds.items()):
        if di:
            print(r"\midrule")
        learned = [r for r in rows if r[0] != "Random"]
        best = {}
        for sub in subsets:
            for m in metrics:
                v = [(c[(sub, m)][0], lab) for lab, c, _ in learned if c[(sub, m)]]
                if v:
                    best[(sub, m)] = max(v)[1]
        label = DATASET_LABEL.get(ds, ds)
        print(rf"\multirow{{{len(rows)}}}{{*}}{{{label}}}")
        for lab, cells, _ in rows:
            cs = [latex_cell(cells[(sub, m)], prec,
                             best.get((sub, m)) == lab, sd_prec)
                  for sub in subsets for m in metrics]
            print(f"  & {lab} & " + " & ".join(cs) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\vspace{2pt}")
    print(r"\end{table*}")


def collect_dataset(root, seeds, subsets, metrics, split, exclude=()):
    """(label, cells, n) rows for one dataset."""
    rows = []
    for label, (stem, source) in METHODS:
        if any(label.lower().startswith(x.lower()) for x in exclude):
            continue
        runs, missing = [], []
        for s in seeds:
            b = read_run(root, stem, source, s, split)
            (runs if b else missing).append(b if b else s)
        if not runs:
            print(f"[skip] {root.name}/{label}: no results")
            continue
        if missing:
            print(f"[warn] {root.name}/{label}: missing seeds {missing} "
                  f"(mean over {len(runs)} of {len(seeds)})")
        cells = {(sub, m): agg([metric_over_horizons(r, sub, m) for r in runs])
                 for sub in subsets for m in metrics}
        rows.append((label, cells, len(runs)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=None,
                   help="Single dataset. Use --datasets for the combined table.")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Two or more datasets -> ONE combined table with a "
                        "\\multirow dataset column, both subsets side by side, "
                        "and best-per-column bolded WITHIN each dataset. This is "
                        "the paper layout; --dataset gives the per-dataset view.")
    p.add_argument("--root", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--subset", choices=["all", "new", "both"], default="both")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--precision", type=int, default=3,
                   help="Decimals for the MEAN.")
    p.add_argument("--std_precision", type=int, default=2,
                   help="Decimals for the std subscript. At 2, a spread below "
                        "0.005 renders as 0.00; raise it where the seed spread "
                        "matters at that scale.")
    p.add_argument("--latex", action="store_true")
    p.add_argument("--json_out", default=None)
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Method labels to omit, matched case-insensitively on "
                        "a prefix. e.g. --exclude Random Supervised gives the "
                        "7 learned rows only.")
    args = p.parse_args()

    subsets_pre = ["all", "new"] if args.subset == "both" else [args.subset]
    if args.datasets:
        per_ds = {}
        for ds in args.datasets:
            r = Path(f"outputs/lp_{ds}")
            if not r.exists():
                print(f"[skip] {r} does not exist")
                continue
            rows = collect_dataset(r, args.seeds, subsets_pre, args.metrics,
                                   args.split, args.exclude)
            if rows:
                per_ds[ds] = rows
        if not per_ds:
            raise SystemExit("No results found for any requested dataset.")
        combined_latex(per_ds, args.metrics, subsets_pre, args.precision,
                       args.std_precision)
        return
    if not args.dataset:
        raise SystemExit("Pass --dataset <name> or --datasets <a> <b> ...")

    root = Path(args.root or f"outputs/lp_{args.dataset}")
    if not root.exists():
        raise SystemExit(f"{root} does not exist. Run "
                         f"scripts/linkpred/slurm_all_methods.sh first.")
    subsets = ["all", "new"] if args.subset == "both" else [args.subset]

    rows, blob_out = [], {}
    for label, (stem, source) in METHODS:
        runs, missing = [], []
        for s in args.seeds:
            b = read_run(root, stem, source, s, args.split)
            (runs if b else missing).append(b if b else s)
        if not runs:
            print(f"[skip] {label}: no results under {root}/{stem}_s*")
            continue
        if missing:
            print(f"[warn] {label}: missing seeds {missing} "
                  f"(mean over {len(runs)} of {len(args.seeds)})")
        cells = {(sub, m): agg([metric_over_horizons(r, sub, m) for r in runs])
                 for sub in subsets for m in args.metrics}
        rows.append((label, cells, len(runs)))
        blob_out[label] = {
            f"{sub}_{m}": (None if cells[(sub, m)] is None else
                           {"mean": cells[(sub, m)][0], "std": cells[(sub, m)][1],
                            "n": cells[(sub, m)][2]})
            for sub in subsets for m in args.metrics}

    if not rows:
        raise SystemExit(f"No results found under {root}.")

    for sub in subsets:
        print(f"\n{args.dataset}  subset={sub}  (seeds {args.seeds}, "
              f"split={args.split}, HIGHER is better)")
        hdr = f"  {'method':<26}" + "".join(f"{m:>20}" for m in args.metrics)
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for label, cells, n in rows:
            line = f"  {label:<26}"
            for m in args.metrics:
                line += f"{fmt(cells[(sub, m)], args.precision):>20}"
            print(line + f"  n={n}")
    print("\n  all = every positive edge; new = edges absent from the history")
    print("  window (formation, not persistence -- the harder subset).")

    if args.latex:
        # Bold the best LEARNED method: Random is a floor, not a competitor.
        learned = [r for r in rows if r[0] != "Random"]
        for sub in subsets:
            best = {}
            for m in args.metrics:
                v = [(c[(sub, m)][0], lab) for lab, c, _ in learned
                     if c[(sub, m)]]
                if v:
                    best[m] = max(v)[1]
            print(f"\n% ===== subset: {sub} =====")
            print("\\begin{tabular}{l" + "r" * len(args.metrics) + "}")
            print("\\toprule")
            print("Method & " + " & ".join(args.metrics) + " \\\\")
            print("\\midrule")
            for i, (label, cells, _) in enumerate(rows):
                if i and rows[i - 1][0] == "IDOL":
                    print("\\midrule")
                if i and rows[i - 1][0] == "Supervised (end-to-end)":
                    print("\\midrule")
                cs = [latex_cell(cells[(sub, m)], args.precision,
                                 best.get(m) == label, args.std_precision)
                      for m in args.metrics]
                print(f"{label} & " + " & ".join(cs) + " \\\\")
            print("\\bottomrule")
            print("\\end{tabular}")
            print(f"% mean $\\pm$ std over {len(args.seeds)} seeds "
                  f"({', '.join(map(str, args.seeds))}); higher is better; "
                  f"subset `{sub}'.")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(blob_out, f, indent=2)
        print(f"\n[saved] {args.json_out}")


if __name__ == "__main__":
    main()
