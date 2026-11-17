"""Ablation table for DELOS on the MOOC node-state task -> LaTeX.

Companion of ``scripts/classification/slurm_ablation_mooc.sh``. Reads one
``state_probe.json`` per (arm, seed), where an arm's results live either in
``$ROOT/abl_<arm>_s<seed>/`` (``abl_ema_<arm>_s<seed>/`` for ``--base ema``)
or -- for the arms that coincide with a main-table run -- in that run's own
directory (``jepa_ema_cos_s<seed>`` etc.).

``--base`` picks which main-table row is the full model: ``ema_cos`` (cosine
EMA schedule, default) or ``ema`` (constant EMA). The EMA-schedule row flips
with it: ``const_ema`` is a removal only under ``ema_cos``, ``cos_ema`` an
addition only under ``ema``.

Every row is the ``jepa`` column (the predictor's forecast) of the probe. Two
paired differences are reported next to the raw numbers, both differenced
WITHIN a seed so the encoder variance cancels:

  gap  = forecast - copy-last (``encoder_last``) of the SAME run
  Δ    = arm - full model, same seed

Rows are: full model, then each removed component. By default the table shows
value and Δ only; ``--gap`` adds the copy-last gap column and ``--floors`` adds
the copy-last / degree-trend / oracle rows. Higher is better for every metric.

Usage:
    python scripts/classification/aggregate_ablation.py --dataset mooc --latex
    python scripts/classification/aggregate_ablation.py --dataset mooc \
        --seeds 42 43 44 --arms full no_node_mask no_pretrain --latex --json_out abl.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_classification import (  # noqa: E402
    DEGREE_COL, FORECAST_COL, LAST_COL, METRIC_LABEL, ORACLE_COL,
    _get, agg, fmt, latex_cell, paired_gap,
)

# arm -> (row label, legacy main-table stem or None); `full` is patched per --base
ARMS = {
    "full":          ("DELOS (full: cosine-EMA target)", "jepa_ema_cos"),
    "const_ema":     ("w/o cosine anneal (constant EMA)",    "jepa_ema"),
    "cos_ema":       ("+ cosine anneal (cosine EMA)",        "jepa_ema_cos"),
    "no_target_net": ("w/o EMA target (shared encoder)",     "jepa_naive"),
    "frozen_enc":    ("w/o encoder training (frozen)",       "jepa_frozen"),
    "no_node_mask":  ("w/o node masking",                    None),
    "input_space":   ("w/o latent target (input-space loss)", None),
    "no_change_w":   ("w/o change weighting",                None),
    "no_residual":   ("w/o residual head",                   None),
    "no_pretrain":   ("w/o encoder pretraining",             None),
    "mlp_pred":      ("w/o graph predictor (per-node MLP)",  None),
    "gru_pred":      ("w/o graph predictor (per-node GRU)",  None),
    "no_attn_mask":  ("w/o structural attention mask",       None),
    "ctx_mask":      ("+ context-input masking",             None),
}
# base -> (full-row label, full-row legacy stem, abl dir prefix, schedule arm it owns)
BASES = {
    "ema_cos": ("DELOS (full: cosine-EMA target)",  "jepa_ema_cos", "abl",     "const_ema"),
    "ema":     ("DELOS (full: constant-EMA target)", "jepa_ema",     "abl_ema", "cos_ema"),
}
FLOORS = ((LAST_COL, "copy-last (encoder at last history step)"),
          (DEGREE_COL, "degree-trend control"),
          (ORACLE_COL, "oracle encoder (sees the target step)"))


def load_arm(root: Path, arm: str, seeds, split: str, prefix: str = "abl"):
    """{seed: blob} for one arm, preferring <prefix>_<arm> over the legacy stem."""
    _, legacy = ARMS[arm]
    fname = "state_probe.json" if split == "test" else "state_probe_val.json"
    runs = {}
    for s in seeds:
        cands = [root / f"{prefix}_{arm}_s{s}" / fname]
        if legacy:
            cands.append(root / f"{legacy}_s{s}" / fname)
        for c in cands:
            if c.exists():
                with open(c) as f:
                    runs[s] = json.load(f)
                break
    return runs


def paired_delta(runs, full_runs, metric):
    """mean/std of (arm - full) over the seeds both have."""
    diffs = []
    for s, r in runs.items():
        if s not in full_runs:
            continue
        a, b = _get(r, FORECAST_COL, metric), _get(full_runs[s], FORECAST_COL, metric)
        if a is not None and b is not None:
            diffs.append(a - b)
    return agg(diffs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mooc")
    p.add_argument("--root", default=None, help="default: outputs/cls_<dataset>")
    p.add_argument("--base", choices=list(BASES), default="ema_cos",
                   help="which main-table row is the full model (default ema_cos)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--arms", nargs="+", default=list(ARMS),
                   help="arms to include, in table order (unknown names rejected)")
    p.add_argument("--metrics", nargs="+", default=["auroc_test", "auprc_test"])
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--precision", type=int, default=3)
    p.add_argument("--latex", action="store_true")
    p.add_argument("--gap", action="store_true",
                   help="add the forecast - copy-last column (off by default)")
    p.add_argument("--floors", action="store_true",
                   help="add the copy-last / degree-trend / oracle rows (off by default)")
    p.add_argument("--no_delta", action="store_true", help="omit the arm - full column")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    full_label, full_stem, prefix, sched_arm = BASES[args.base]
    ARMS["full"] = (full_label, full_stem)
    other_sched = {"const_ema": "cos_ema", "cos_ema": "const_ema"}[sched_arm]
    bad = [a for a in args.arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; known: {list(ARMS)}")
    if other_sched in args.arms:
        raise SystemExit(f"arm {other_sched!r} is not defined under --base {args.base} "
                         f"(it would coincide with `full`); use {sched_arm!r}")
    root = Path(args.root or f"outputs/cls_{args.dataset}")
    if not root.exists():
        raise SystemExit(f"{root} does not exist; run slurm_ablation_mooc.sh first")

    full_runs = load_arm(root, "full", args.seeds, args.split, prefix)
    if not full_runs:
        print("[warn] no `full` runs found: Δ column will be empty")

    rows, blob_out = [], {}
    for arm in args.arms:
        runs = load_arm(root, arm, args.seeds, args.split, prefix)
        label = ARMS[arm][0]
        if not runs:
            print(f"[skip] {arm}: no state_probe under {root}/{prefix}_{arm}_s* "
                  f"{'or ' + ARMS[arm][1] + '_s*' if ARMS[arm][1] else ''}")
            continue
        missing = [s for s in args.seeds if s not in runs]
        if missing:
            print(f"[warn] {arm}: missing seeds {missing} (n={len(runs)})")
        cells = {m: agg([_get(r, FORECAST_COL, m) for r in runs.values()]) for m in args.metrics}
        gaps = {m: paired_gap(list(runs.values()), m) for m in args.metrics}
        deltas = {m: (None if arm == "full" else paired_delta(runs, full_runs, m))
                  for m in args.metrics}
        rows.append((arm, label, cells, gaps, deltas, len(runs)))
        blob_out[arm] = {"label": label, "n_seeds": len(runs), "seeds": sorted(runs),
                         **{m: cells[m] and {"mean": cells[m][0], "std": cells[m][1]} for m in args.metrics},
                         **{f"gap_{m}": gaps[m] and {"mean": gaps[m][0], "std": gaps[m][1]} for m in args.metrics},
                         **{f"delta_{m}": deltas[m] and {"mean": deltas[m][0], "std": deltas[m][1]} for m in args.metrics}}
    if not rows:
        raise SystemExit("no ablation results found")

    # label-free floors, measured inside the full runs (or the first arm found)
    ref = full_runs or load_arm(root, rows[0][0], args.seeds, args.split, prefix)
    floors = []
    for col, flabel in (FLOORS if args.floors else ()):
        cells = {m: agg([_get(r, col, m) for r in ref.values()]) for m in args.metrics}
        if any(cells[m] is not None for m in args.metrics):
            floors.append((flabel, cells))
            blob_out[f"floor_{col}"] = {m: cells[m] and {"mean": cells[m][0], "std": cells[m][1]} for m in args.metrics}

    show_gap, show_delta = args.gap, not args.no_delta
    prec = args.precision

    # --- text ------------------------------------------------------------------
    print(f"\n{args.dataset} ablation  (base={args.base}, seeds {args.seeds}, split={args.split}, HIGHER is better)")
    hdr = f"  {'arm':<44}"
    for m in args.metrics:
        lbl = METRIC_LABEL.get(m, m)
        hdr += f"{lbl:>18}"
        if show_gap:
            hdr += f"{'gap':>16}"
        if show_delta:
            hdr += f"{'Δ vs full':>16}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for arm, label, cells, gaps, deltas, n in rows:
        line = f"  {label:<44}"
        for m in args.metrics:
            line += f"{fmt(cells[m], prec):>18}"
            if show_gap:
                line += f"{fmt(gaps[m], prec, signed=True):>16}"
            if show_delta:
                line += f"{fmt(deltas[m], prec, signed=True):>16}"
        print(line + f"  n={n}")
    for flabel, cells in floors:
        line = f"  {flabel:<44}"
        for m in args.metrics:
            line += f"{fmt(cells[m], prec):>18}"
            line += " " * (16 * (show_gap + show_delta))
        print(line)
    notes = []
    if show_gap:
        notes.append("gap = forecast - copy-last of the same run")
    if show_delta:
        notes.append("Δ = arm - full model")
    if notes:
        print("\n  " + "; ".join(notes) + "; paired within seed.")

    # --- LaTeX -----------------------------------------------------------------
    if args.latex:
        best = {m: max((c[m][0], a) for a, _, c, _, _, _ in rows if c[m])[1]
                for m in args.metrics if any(c[m] for _, _, c, _, _, _ in rows)}
        per_metric = 1 + show_gap + show_delta
        print("\n% --- ablation table (paste) ------------------------------------")
        print("\\begin{tabular}{l" + "r" * (per_metric * len(args.metrics)) + "}")
        print("\\toprule")
        if per_metric > 1:
            print("Variant" + "".join(f" & \\multicolumn{{{per_metric}}}{{c}}{{{METRIC_LABEL.get(m, m)}}}"
                                       for m in args.metrics) + " \\\\")
            cm, start = "", 2
            for _ in args.metrics:
                cm += f"\\cmidrule(lr){{{start}-{start + per_metric - 1}}}"; start += per_metric
            print(cm)
            sub = ["value"] + (["gap"] if show_gap else []) + (["$\\Delta$"] if show_delta else [])
            print(" & " + " & ".join(sub * len(args.metrics)) + " \\\\")
        else:
            print("Variant" + "".join(f" & {METRIC_LABEL.get(m, m)}" for m in args.metrics) + " \\\\")
        print("\\midrule")
        for i, (arm, label, cells, gaps, deltas, n) in enumerate(rows):
            if i == 1:
                print("\\midrule")
            cs = []
            for m in args.metrics:
                cs.append(latex_cell(cells[m], prec, best.get(m) == arm))
                if show_gap:
                    cs.append(latex_cell(gaps[m], prec, signed=True))
                if show_delta:
                    cs.append("--" if deltas[m] is None else latex_cell(deltas[m], prec, signed=True))
            note = "" if n == len(args.seeds) else f" ($n{{=}}{n}$)"
            print(f"{label}{note} & " + " & ".join(cs) + " \\\\")
        if floors:
            print("\\midrule")
            for flabel, cells in floors:
                cs = []
                for m in args.metrics:
                    cs.append(latex_cell(cells[m], prec))
                    cs += ["--"] * (show_gap + show_delta)
                print(f"\\textit{{{flabel}}} & " + " & ".join(cs) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        tail = f"% mean $\\pm$ std over seeds {', '.join(map(str, args.seeds))}; higher is better."
        if show_gap:
            tail += " gap = forecast $-$ copy-last of the same run, paired within seed."
        if show_delta:
            tail += " $\\Delta$ = variant $-$ full model, paired within seed."
        if floors:
            tail += " Floor rows are measured inside the full-model runs."
        print(tail)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(blob_out, f, indent=2)
        print(f"\n[saved] {args.json_out}")


if __name__ == "__main__":
    main()
