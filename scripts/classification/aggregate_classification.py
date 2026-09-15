"""Aggregate dynamic node-classification results into a mean +/- std LaTeX table.

The classification twin of ``scripts/regression/aggregate_regression.py``. Reads
the per-seed ``state_probe.json`` every arm of
``scripts/classification/slurm_all_methods.sh`` writes.

Layout it expects:

    outputs/cls_<dataset>/
      jepa_{frozen,naive,ema,vicreg}_s<seed>/state_probe.json
      {cldg,ddgcl,idol,maskdgnn}_s<seed>/state_probe.json

Every state_probe.json carries several COLUMNS, not just one number:

  jepa                     the JEPA predictor's rolled-out latent  (the forecast)
  encoder_last             the frozen encoder at the last history bin (copy-last)
  encoder_oracle           an encoder that SAW the target bin       (upper bound)
  degree_history_baseline  degree at each history bin  (activity-trend control)

Baselines are encoder-only, so they have no ``jepa`` column: their
``encoder_last`` IS their prediction.

Two things are reported, because they answer different questions:

  AUROC / AUPRC   the headline per-method numbers, comparable to the literature.
                  ~3% positive rate here, so accuracy is meaningless and AUPRC
                  is the harder of the two.
  gap             ``jepa - encoder_last``, PAIRED within a seed. This is the
                  claim: a forecast that beats copy-last. Pairing matters --
                  both columns come from ONE encoder within a seed, so the
                  difference has far less variance than either column alone, and
                  it can resolve differences the raw columns cannot.

Higher is better for every metric here (unlike the regression table).

Usage:
    python scripts/classification/aggregate_classification.py \
        --dataset mooc --seeds 42 43 44 --latex
    python scripts/classification/aggregate_classification.py \
        --dataset mooc --seeds 42 43 44 --metrics auroc_test auprc_test acc_test
    # synthetic_dyn multiclass state: accuracy overall and on the switch rows
    python scripts/classification/aggregate_classification.py \
        --dataset synthetic_dyn --seeds 42 43 44 \
        --metrics acc_test subset_switch.acc_test subset_stay.acc_test
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

# Display name -> directory stem under outputs/cls_<dataset>/.
# Contrastive baselines first, then the DyG-JEPA arms, as the table should read.
METHODS = [
    ("CLDG",                "cldg"),
    ("DDGCL",               "ddgcl"),
    ("IDOL",                "idol"),
    ("MaskDGNN",            "maskdgnn"),
    ("DyG-JEPA (frozen)",   "jepa_frozen"),
    ("DyG-JEPA (naive)",    "jepa_naive"),
    ("DyG-JEPA (EMA)",      "jepa_ema"),
    ("DyG-JEPA (VICReg)",   "jepa_vicreg"),
    ("DyG-JEPA (input-space)", "jepa_input"),
    # JEPA-recipe variants (opt-in flags in train_dyg_jepa.py; see README).
    ("DyG-JEPA (frozen + ctx-mask)",  "jepa_ctx"),
    ("DyG-JEPA (frozen + target-LN)", "jepa_tnorm"),
    ("DyG-JEPA (EMA cosine)",         "jepa_ema_cos"),
    ("DyG-JEPA (SIGReg)",             "jepa_sigreg"),
    ("DyG-JEPA (LeJEPA)",             "jepa_lejepa"),
    ("DyG-JEPA (I-JEPA recipe)",      "jepa_ijepa"),
    ("DyG-JEPA (LeJEPA + recipe)",    "jepa_lejepa_full"),
]

FORECAST_COL = "jepa"
LAST_COL = "encoder_last"
ORACLE_COL = "encoder_oracle"
DEGREE_COL = "degree_history_baseline"

PERSIST_COL = "label_persistence"     # multiclass state only: copy the label

METRIC_LABEL = {"auroc_test": "AUROC", "auprc_test": "AUPRC",
                "acc_test": "Acc", "macro_f1_test": "macro-F1",
                "n_pos_test": "n_pos",
                # subset metrics (multiclass state): "<subset>.<metric>"
                "subset_switch.acc_test": "Acc(switch)",
                "subset_stay.acc_test": "Acc(stay)",
                "subset_switch.macro_f1_test": "F1(switch)",
                "change_f1": "changeF1", "change_precision": "changeP",
                "change_recall": "changeR",
                "latent_r2_test": "R2(z)", "delta_r2_test": "R2(dz)"}


def _load(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _get(blob, column, metric):
    """One metric from one column, or None for missing / NaN. A metric spelled
    "subset_<name>.<metric>" reads the nested subset dict (multiclass state)."""
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


PROBE_STEM = "state_probe"


def collect(root: Path, stem: str, seeds: list[int], split: str):
    """Per-seed state_probe blobs for one method, plus the missing seeds."""
    suffix = "" if split == "test" else "_val"
    runs, missing = [], []
    for s in seeds:
        blob = _load(root / f"{stem}_s{s}" / f"{PROBE_STEM}{suffix}.json")
        if blob:
            runs.append(blob)
        else:
            missing.append(s)
    return runs, missing


def agg(values):
    """(mean, std, n) over the non-None values."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return (st.mean(vals), st.stdev(vals) if len(vals) >= 2 else 0.0, len(vals))


def paired_gap(runs, metric):
    """mean/std of (forecast - copy-last), differenced WITHIN each seed.

    Never mean(forecast) - mean(last): pairing is the whole point. Both columns
    share one encoder per seed, so the per-seed difference cancels the encoder
    variance that dominates either column on its own.
    """
    diffs = []
    for r in runs:
        f, l = _get(r, FORECAST_COL, metric), _get(r, LAST_COL, metric)
        if f is not None and l is not None:
            diffs.append(f - l)
    return agg(diffs)


def fmt(cell, prec, signed=False):
    if cell is None:
        return "--"
    mu, sd, n = cell
    s = f"{mu:+.{prec}f}" if signed else f"{mu:.{prec}f}"
    return f"{s} ± {sd:.{prec}f}" if n >= 2 else s


def latex_cell(cell, prec, best=False, signed=False):
    if cell is None:
        return "--"
    mu, sd, n = cell
    m = f"{mu:+.{prec}f}" if signed else f"{mu:.{prec}f}"
    body = f"{m} \\pm {sd:.{prec}f}" if n >= 2 else m
    return f"$\\mathbf{{{body}}}$" if best else f"${body}$"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   choices=["mooc", "reddit", "wikipedia", "synthetic_dyn"])
    p.add_argument("--root", default=None,
                   help="Results root (default: outputs/cls_<dataset>).")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--metrics", nargs="+", default=["auroc_test", "auprc_test"])
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--precision", type=int, default=3)
    p.add_argument("--latex", action="store_true")
    p.add_argument("--json_out", default=None)
    p.add_argument("--no_gap", action="store_true",
                   help="Omit the paired forecast-minus-copy-last column.")
    p.add_argument("--probe_stem", default="state_probe",
                   help="Result file stem per arm: state_probe (default) or switch_probe "
                        "(binary switch-event probe derived from a state checkpoint).")
    args = p.parse_args()
    global PROBE_STEM
    PROBE_STEM = args.probe_stem

    root = Path(args.root or f"outputs/cls_{args.dataset}")
    if not root.exists():
        raise SystemExit(f"{root} does not exist. Run "
                         f"scripts/classification/slurm_all_methods.sh first.")

    rows, blob_out = [], {}
    for label, stem in METHODS:
        runs, missing = collect(root, stem, args.seeds, args.split)
        if not runs:
            print(f"[skip] {label}: no results under {root}/{stem}_s*")
            continue
        if missing:
            print(f"[warn] {label}: missing seeds {missing} "
                  f"(mean over {len(runs)} of {len(args.seeds)})")
        # Baselines are encoder-only: copy-last IS their prediction, so their
        # headline column is encoder_last and they have no paired gap.
        has_fcast = any(_get(r, FORECAST_COL, args.metrics[0]) is not None
                        for r in runs)
        col = FORECAST_COL if has_fcast else LAST_COL
        cells = {m: agg([_get(r, col, m) for r in runs]) for m in args.metrics}
        gaps = ({m: paired_gap(runs, m) for m in args.metrics}
                if has_fcast else {m: None for m in args.metrics})
        rows.append((label, cells, gaps, has_fcast, len(runs), runs))
        blob_out[label] = {
            "column": col, "n_seeds": len(runs),
            **{m: (None if cells[m] is None else
                   {"mean": cells[m][0], "std": cells[m][1]})
               for m in args.metrics},
            **{f"gap_{m}": (None if gaps[m] is None else
                            {"mean": gaps[m][0], "std": gaps[m][1]})
               for m in args.metrics}}

    if not rows:
        raise SystemExit(f"No results found under {root}.")

    # --- the shared reference columns, from the DyG-JEPA runs ----------------
    # encoder_oracle / degree_history_baseline are matched controls present in
    # every run; report them once rather than per method.
    ref_runs = next((r for _, _, _, hf, _, r in rows if hf), None)
    if ref_runs:
        print(f"\nshared controls (from the DyG-JEPA runs, {len(ref_runs)} seeds)")
        for cname, clabel in ((PERSIST_COL, "label_persistence (floor)"),
                              (ORACLE_COL, "encoder_oracle (upper bound)"),
                              (LAST_COL, "encoder_last (copy-last)"),
                              (DEGREE_COL, "degree_history (trend control)")):
            cells = [fmt(agg([_get(r, cname, m) for r in ref_runs]),
                         args.precision) for m in args.metrics]
            if any(c != "--" for c in cells):
                print(f"  {clabel:<32}" +
                      "".join(f"{METRIC_LABEL.get(m, m)} {c:<18}"
                              for m, c in zip(args.metrics, cells)))

    # --- text table ----------------------------------------------------------
    show_gap = not args.no_gap
    head = f"\n{args.dataset}  (seeds {args.seeds}, split={args.split}, HIGHER is better)"
    print(head)
    hdr = f"  {'method':<22}"
    for m in args.metrics:
        hdr += f"{METRIC_LABEL.get(m, m):>18}"
        if show_gap:
            hdr += f"{'gap(' + METRIC_LABEL.get(m, m) + ')':>20}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, cells, gaps, has_fcast, n, _ in rows:
        line = f"  {label:<22}"
        for m in args.metrics:
            line += f"{fmt(cells[m], args.precision):>18}"
            if show_gap:
                line += f"{fmt(gaps[m], args.precision, signed=True):>20}"
        print(line + f"  n={n}")
    print("\n  gap = forecast - copy-last, differenced WITHIN each seed.")
    print("  Baselines are encoder-only, so copy-last IS their prediction (no gap).")

    # --- LaTeX ---------------------------------------------------------------
    if args.latex:
        best = {}
        for m in args.metrics:
            vals = [(c[m][0], lab) for lab, c, _, _, _, _ in rows if c[m]]
            if vals:
                best[m] = max(vals)[1]          # higher is better
            gv = [(g[m][0], lab) for lab, _, g, _, _, _ in rows if g[m]]
            if gv:
                best[f"gap_{m}"] = max(gv)[1]
        ncol = len(args.metrics) * (2 if show_gap else 1)
        print("\n% --- paste into your results table -----------------------")
        print("\\begin{tabular}{l" + "r" * ncol + "}")
        print("\\toprule")
        hdr = "Method"
        for m in args.metrics:
            lbl = METRIC_LABEL.get(m, m)
            hdr += (f" & \\multicolumn{{2}}{{c}}{{{lbl}}}" if show_gap
                    else f" & {lbl}")
        print(hdr + " \\\\")
        if show_gap:
            start, cm = 2, ""
            for _ in args.metrics:
                cm += f"\\cmidrule(lr){{{start}-{start+1}}}"
                start += 2
            print(cm)
            print(" " + " & ".join([""] + ["value", "gap"] * len(args.metrics)) + " \\\\")
        print("\\midrule")
        for i, (label, cells, gaps, has_fcast, _, _) in enumerate(rows):
            if i and rows[i - 1][0] == "IDOL":
                print("\\midrule")
            cs = []
            for m in args.metrics:
                cs.append(latex_cell(cells[m], args.precision,
                                     best.get(m) == label))
                if show_gap:
                    cs.append(latex_cell(gaps[m], args.precision,
                                         best.get(f"gap_{m}") == label,
                                         signed=True))
            print(f"{label} & " + " & ".join(cs) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print(f"% mean $\\pm$ std over {len(args.seeds)} seeds "
              f"({', '.join(map(str, args.seeds))}); higher is better. "
              f"gap = forecast $-$ copy-last, paired within seed.")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(blob_out, f, indent=2)
        print(f"\n[saved] {args.json_out}")


if __name__ == "__main__":
    main()
