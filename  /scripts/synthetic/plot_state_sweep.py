"""Plot the synthetic node-state sweeps: where does the JEPA forecast beat the
contrastive encoders (CLDG / DDGCL / IDOL), and by how much, along each axis?

Reads the per-seed ``state_probe.json`` files that ``sweep_state.sh`` produces
(``<root>/<AXIS>_<value>/<arm>_s<seed>/state_probe.json``) -- not the
aggregator's summary, so the matched controls (copy-last, oracle encoder,
label persistence) are available too.

Two figures per run:

  <out>/state_sweep_<metric>.pdf  accuracy on SWITCH rows vs axis value, one
                                  panel per axis. This is the anticipation
                                  measure: label persistence is 0 there by
                                  construction, so every point above 0 is
                                  something the representation knew before the
                                  state changed. Baselines are solid colored
                                  lines, JEPA arms are blue, the frozen
                                  encoder's own copy-last is a grey dashed line
                                  and the oracle encoder (saw the target step)
                                  a grey dotted ceiling.
  <out>/state_sweep_margin_<metric>.pdf
                                  JEPA(frozen) minus each baseline on switch
                                  rows, with the combined std, so the reader
                                  sees directly where the margin clears noise.

and a text/LaTeX table of the same margins (``--latex``). The comparison to
the contrastive baselines is PAIRED WITHIN SEED: the seed is the generated
graph, so every arm of a seed sees the same data and the per-seed difference
removes the graph-realisation variance that dominates the raw bands. The table
also reports how many seeds JEPA wins.

Usage (from the repo root, after the sweeps finished):
    python scripts/synthetic/plot_state_sweep.py \
        --axes SYN_MOMENTUM SYN_ANCHOR SYN_NODE_NOISE NPRED SYN_DIFFUSION \
        --seeds 42 43 44 --out figures/synth_state --latex
    # a single axis, custom root:
    python scripts/synthetic/plot_state_sweep.py --axes NPRED --root outputs/synth_state
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

# --- what is plotted --------------------------------------------------------
# (arm directory stem, legend label, column inside state_probe.json)
JEPA_ARMS = [("jepa_frozen", "DyG-JEPA (frozen enc.)", "jepa"),
             ("jepa_ema", "DyG-JEPA (EMA enc.)", "jepa"),
             ("jepa_input", "same predictor, input-space target", "jepa")]
BASELINES = [("cldg", "CLDG", "encoder_last"),
             ("ddgcl", "DDGCL", "encoder_last"),
             ("idol", "IDOL", "encoder_last")]
# matched controls, read from the frozen-JEPA run (same encoder as its forecast)
CONTROLS = [("encoder_last", "copy-last (same encoder)"),
            ("encoder_oracle", "oracle encoder (saw target)")]

# Fixed categorical order (validated: adjacent CVD dE >= 8, normal dE >= 15 in
# light mode). JEPA is slot 1 (blue); the two JEPA arms share the hue and differ
# by line style + marker, the baselines take the next slots in order.
COLORS = {"jepa_frozen": "#2a78d6", "jepa_ema": "#2a78d6", "jepa_input": "#2a78d6",
          "cldg": "#eb6834", "ddgcl": "#1baf7a", "idol": "#4a3aa7"}
STYLE = {"jepa_frozen": dict(ls="-", marker="o"),
         "jepa_ema": dict(ls="--", marker="s"),
         "jepa_input": dict(ls=":", marker="x"),
         "cldg": dict(ls="-", marker="^"),
         "ddgcl": dict(ls="-", marker="D"),
         "idol": dict(ls="-", marker="v")}
GREY = "#8a8985"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"

AXIS_LABEL = {
    "SYN_MOMENTUM": "momentum ρ  (how much history matters)",
    "SYN_ANCHOR": "anchor spread σ_a  (community separation)",
    "SYN_NODE_NOISE": "observation noise ν",
    "SYN_OU_SIGMA": "latent noise σ",
    "SYN_THETA": "mean reversion θ",
    "SYN_DIFFUSION": "feature diffusion α  (graph shapes features)",
    "SYN_BETA": "edge sharpness β",
    "SYN_PERSISTENCE": "edge persistence η",
    "SYN_DENSITY": "edge density κ",
    "SYN_K": "communities K",
    "SYN_DLAT": "latent dim d",
    "NPRED": "forecast horizon (steps)",
    "HIST": "history length (steps)",
}


def _get(blob: dict, column: str, metric: str):
    col = blob.get(column)
    if "." in metric:
        sub, metric = metric.split(".", 1)
        col = col.get(sub) if isinstance(col, dict) else None
    if not isinstance(col, dict) or metric not in col:
        return None
    v = col[metric]
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _values_on_disk(root: Path, axis: str) -> list[str]:
    pat = re.compile(rf"^{re.escape(axis)}_(.+)$")
    vals = [m.group(1) for d in root.iterdir() if d.is_dir() and (m := pat.match(d.name))]
    return sorted(vals, key=lambda s: float(s) if re.match(r"^-?\d+(\.\d+)?$", s) else s)


PROBE_FILE = "state_probe.json"


def load_axis(root: Path, axis: str, values: list[str], seeds: list[int], metric: str):
    """-> {value: {series_key: (mean, std, n)}} for every arm and control."""
    out = {}
    for v in values:
        cell = {}
        per_seed = {}
        for stem, _, column in JEPA_ARMS + BASELINES:
            xs = {}
            for s in seeds:
                p = root / f"{axis}_{v}" / f"{stem}_s{s}" / PROBE_FILE
                if p.exists():
                    x = _get(json.load(open(p)), column, metric)
                    if x is not None:
                        xs[s] = x
            if xs:
                vals = list(xs.values())
                cell[stem] = (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals))
                per_seed[stem] = xs
        cell["_per_seed"] = per_seed
        for column, _ in CONTROLS:
            xs = []
            for s in seeds:
                p = root / f"{axis}_{v}" / f"jepa_frozen_s{s}" / PROBE_FILE
                if p.exists():
                    x = _get(json.load(open(p)), column, metric)
                    if x is not None:
                        xs.append(x)
            if xs:
                cell[column] = (float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, len(xs))
        # paired gap vs copy-last, per JEPA arm
        for stem, _, column in JEPA_ARMS:
            ds = []
            for s in seeds:
                p = root / f"{axis}_{v}" / f"{stem}_s{s}" / PROBE_FILE
                if p.exists():
                    b = json.load(open(p))
                    a, l = _get(b, column, metric), _get(b, "encoder_last", metric)
                    if a is not None and l is not None:
                        ds.append(a - l)
            if ds:
                cell[f"gap:{stem}"] = (float(np.mean(ds)), float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0, len(ds))
        out[v] = cell
    return out


def _xpos(values: list[str]):
    """Numeric x when every value parses, else categorical positions."""
    try:
        return [float(v) for v in values], False
    except ValueError:
        return list(range(len(values))), True


def _style_axis(ax, xlabel, ylabel, values, x, categorical):
    ax.set_xlabel(xlabel, color=TEXT2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT2, fontsize=9)
    if categorical:
        ax.set_xticks(x)
        ax.set_xticklabels(values)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=TEXT2, labelsize=8)


def plot_grid(data: dict, axes: list[str], metrics: list[str], labels: dict, out: Path,
              controls: str = "oracle"):
    """Rows = metrics, columns = axes. Methods only, plus the oracle ceiling
    (controls='oracle'), or also the frozen encoder's copy-last
    (controls='all'), or neither (controls='none')."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrow, ncol = len(metrics), len(axes)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    handles = {}
    for r, metric in enumerate(metrics):
        for c, axis in enumerate(axes):
            ax = axs[r][c]
            cells = data[metric][axis]
            values = list(cells.keys())
            x, categorical = _xpos(values)
            shown = {"none": [], "oracle": ["encoder_oracle"],
                     "all": ["encoder_last", "encoder_oracle"]}[controls]
            for column, label in CONTROLS:
                if column not in shown:
                    continue
                y = [cells[v].get(column, (np.nan,))[0] for v in values]
                (h,) = ax.plot(x, y, ls="--" if column == "encoder_last" else ":", lw=1.6,
                               color=GREY, zorder=1)
                handles[label] = h
            for stem, label, _ in BASELINES + JEPA_ARMS:
                if not any(stem in cells[v] for v in values):
                    continue
                y = np.array([cells[v].get(stem, (np.nan, np.nan))[0] for v in values])
                e = np.array([cells[v].get(stem, (np.nan, np.nan))[1] for v in values])
                st = STYLE[stem]
                ax.fill_between(x, y - e, y + e, color=COLORS[stem], alpha=0.12, lw=0, zorder=2)
                (h,) = ax.plot(x, y, color=COLORS[stem], lw=2, ls=st["ls"], marker=st["marker"],
                               ms=6, mec="white", mew=1.2, zorder=3)
                handles[label] = h
            _style_axis(ax, AXIS_LABEL.get(axis, axis) if r == nrow - 1 else "",
                        labels[metric] if c == 0 else "", values, x, categorical)
            if r < nrow - 1:
                ax.tick_params(labelbottom=False)
    order = [l for _, l, _ in JEPA_ARMS] + [l for _, l, _ in BASELINES] + [l for _, l in CONTROLS]
    fig.legend([handles[l] for l in order if l in handles], [l for l in order if l in handles],
               loc="lower center", ncol=min(4, len(handles)), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Node-state forecast on the synthetic generator (band = ±1 std over seeds)",
                 fontsize=10, color=TEXT)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def margins(data: dict, axes: list[str]):
    """{axis: {value: {key: (mean_diff, std_diff, wins, n)}}} for JEPA(frozen)
    minus each baseline and minus its own copy-last.

    Every difference is PAIRED WITHIN A SEED. The seed is the generated graph
    (and the probe's train/test rows), so all arms of one seed see the same
    data; differencing inside the seed removes the graph-realisation variance
    that dominates the raw bands. sqrt(sd_a^2 + sd_b^2) would treat two arms
    on the same graph as independent and overstate the noise several-fold.
    ``wins`` = number of seeds where JEPA is strictly ahead."""
    out = {}
    for axis in axes:
        out[axis] = {}
        for v, cells in data[axis].items():
            row = {}
            ps = cells.get("_per_seed", {})
            j = ps.get("jepa_frozen", {})
            for stem, _, _ in BASELINES:
                b = ps.get(stem, {})
                common = sorted(set(j) & set(b))
                if common:
                    d = np.array([j[s] - b[s] for s in common])
                    row[stem] = (float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                                 int((d > 0).sum()), len(d))
            if "gap:jepa_frozen" in cells:
                m, sd, n = cells["gap:jepa_frozen"]
                row["copy_last"] = (m, sd, None, n)
            out[axis][v] = row
    return out


def plot_margin(mg: dict, axes: list[str], metrics: list[str], labels: dict, out: Path,
                band: str = "std", controls: str = "oracle"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrow, ncol = len(metrics), len(axes)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
    handles = {}
    series = [(stem, f"vs {label}", COLORS[stem], "-", STYLE[stem]["marker"]) for stem, label, _ in BASELINES]
    if controls == "all":
        series = [("copy_last", "vs copy-last (same encoder)", GREY, "--", "o")] + series
    for r, metric in enumerate(metrics):
        for c, axis in enumerate(axes):
            ax = axs[r][c]
            cellmg = mg[metric][axis]
            values = list(cellmg.keys())
            x, categorical = _xpos(values)
            ax.axhline(0, color="#c3c2b7", lw=1, zorder=1)
            for key, label, color, ls, marker in series:
                if not any(key in cellmg[v] for v in values):
                    continue
                y = np.array([cellmg[v].get(key, (np.nan, np.nan, None, 0))[0] for v in values])
                e = np.array([cellmg[v].get(key, (np.nan, np.nan, None, 0))[1] for v in values])
                if band == "se":
                    nseeds = np.array([cellmg[v].get(key, (np.nan, np.nan, None, 0))[3] for v in values])
                    e = e / np.sqrt(np.maximum(nseeds, 1))
                ax.fill_between(x, y - e, y + e, color=color, alpha=0.12, lw=0, zorder=2)
                (h,) = ax.plot(x, y, color=color, lw=2, ls=ls, marker=marker, ms=6,
                               mec="white", mew=1.2, zorder=3)
                handles[label] = h
            _style_axis(ax, AXIS_LABEL.get(axis, axis) if r == nrow - 1 else "",
                        f"JEPA (frozen) − method,\n{labels[metric]}" if c == 0 else "",
                        values, x, categorical)
            if r < nrow - 1:
                ax.tick_params(labelbottom=False)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=min(4, len(handles)),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    btxt = "±1 std over seeds" if band == "std" else "±1 standard error of the mean"
    fig.suptitle(f"Margin of the JEPA forecast over each method "
                 f"(paired within seed; band = {btxt}; above 0 = JEPA wins)", fontsize=10, color=TEXT)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_table(mg: dict, data: dict, axes: list[str], latex: bool, metric_label: str = "",
                controls: str = "oracle"):
    cols = (["copy_last"] if controls == "all" else []) + [s for s, _, _ in BASELINES]
    names = {"copy_last": "copy-last", "cldg": "CLDG", "ddgcl": "DDGCL", "idol": "IDOL"}
    print(f"\n[{metric_label}] JEPA (frozen) minus method, PAIRED within seed (same generated graph). "
          "* = margin exceeds 1 std, ** = 2 std; k/n = seeds where JEPA is ahead. '--' = arm not run.")
    hdr = f"{'axis':<16}{'value':>8}{'JEPA':>10}" + "".join(f"{names[c]:>24}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    rows_tex = []
    for axis in axes:
        for v, row in mg[axis].items():
            j = data[axis][v].get("jepa_frozen")
            jtxt = "--" if j is None else f"{j[0]:.3f}"
            line = f"{axis:<16}{v:>8}{jtxt:>10}"
            tex = [axis.replace("_", r"\_"), v, jtxt]
            for c in cols:
                if c not in row:
                    line += f"{'--':>20}"; tex.append("--"); continue
                m, s, wins, n = row[c]
                star = "**" if abs(m) > 2 * s > 0 else ("*" if abs(m) > s > 0 else "")
                w = "" if wins is None else f" {wins}/{n}"
                line += f"{f'{m:+.3f} ± {s:.3f}{star}{w}':>24}"
                tex.append(f"${m:+.3f} \\pm {s:.3f}" + ("^{**}" if star == "**" else "^{*}" if star else "") + "$")
            print(line)
            rows_tex.append(" & ".join(tex) + r" \\")
    if latex:
        print("\n% --- LaTeX ---")
        print(r"\begin{tabular}{llr" + "r" * len(cols) + "}")
        print(r"\toprule")
        print("axis & value & JEPA & " + " & ".join(f"vs {names[c]}" for c in cols) + r" \\")
        print(r"\midrule")
        print("\n".join(rows_tex))
        print(r"\bottomrule")
        print(r"\end{tabular}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--axes", nargs="+", required=True)
    p.add_argument("--root", default="outputs/synth_state",
                   help="Sweep root; each axis lives under <root>/<AXIS>/<AXIS>_<value>/. "
                        "The binary switch-event sweeps live under outputs/synth_switch; "
                        "plot them with --metrics auroc_test auprc_test.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--metrics", nargs="+", default=["subset_switch.acc_test", "acc_test"],
                   help="One figure row per metric. subset_switch.acc_test = accuracy on the rows "
                        "whose state changes; acc_test = overall; subset_stay.acc_test = stay rows.")
    p.add_argument("--controls", choices=["none", "oracle", "all"], default="oracle",
                   help="Grey reference lines: 'oracle' = the encoder that saw the target step "
                        "(ceiling); 'all' = also the frozen encoder's copy-last; 'none'.")
    p.add_argument("--out", default="figures/synth_state")
    p.add_argument("--band", choices=["std", "se"], default="std",
                   help="Band on the margin figure: 'std' = spread of the per-seed "
                        "margins; 'se' = standard error of their mean, the band that "
                        "answers 'is the mean margin above zero'. The title states which.")
    p.add_argument("--probe_file", default="state_probe.json",
                   help="Per-arm result file to read. switch_probe.json = the binary "
                        "switch-event probe derived from a multiclass-state checkpoint "
                        "(run_block_probe.py --task binary on a community-target run).")
    p.add_argument("--latex", action="store_true")
    args = p.parse_args()
    global PROBE_FILE
    PROBE_FILE = args.probe_file

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = {"subset_switch.acc_test": "accuracy on switch rows",
              "subset_stay.acc_test": "accuracy on stay rows",
              "acc_test": "overall accuracy",
              "macro_f1_test": "macro-F1",
              "change_f1": "change-detection F1",
              "change_precision": "change-detection precision",
              "change_recall": "change-detection recall",
              "latent_r2_test": "R² of true latent z(target)",
              "delta_r2_test": "R² of latent change Δz",
              "auroc_test": "switch-event AUROC",
              "auprc_test": "switch-event AUPRC",
              "subset_onset.auroc_test": "switch-event AUROC (onset rows)"}
    data, mg = {}, {}
    axes = None
    for metric in args.metrics:
        data[metric], mg[metric] = {}, {}
        for axis in args.axes:
            aroot = root / axis if (root / axis).exists() else root
            values = _values_on_disk(aroot, axis)
            if not values:
                if metric == args.metrics[0]:
                    print(f"[skip] no {axis}_* directories under {aroot}")
                continue
            data[metric][axis] = load_axis(aroot, axis, values, args.seeds, metric)
            if metric == args.metrics[0]:
                for v in values:
                    for stem, _, _ in JEPA_ARMS + BASELINES:
                        cell = data[metric][axis][v]
                        if stem in cell and cell[stem][2] < len(args.seeds):
                            print(f"[warn] {axis}={v} {stem}: {cell[stem][2]}/{len(args.seeds)} seeds")
        mg[metric] = margins(data[metric], list(data[metric].keys()))
        axes = list(data[metric].keys())
    if not axes:
        raise SystemExit("nothing to plot")
    tag = "_".join(m.replace(".", "_").replace("_acc_test", "").replace("acc_test", "overall")
                   .replace("_test", "") for m in args.metrics)
    plot_grid(data, axes, args.metrics, {m: labels.get(m, m) for m in args.metrics},
              out / f"state_sweep_{tag}.pdf", controls=args.controls)
    mtag = f"{tag}_se" if args.band == "se" else tag
    plot_margin(mg, axes, args.metrics, {m: labels.get(m, m) for m in args.metrics},
                out / f"state_sweep_margin_{mtag}.pdf", band=args.band, controls=args.controls)
    for metric in args.metrics:
        print_table(mg[metric], data[metric], axes, args.latex, labels.get(metric, metric), args.controls)
    print(f"\n[saved] {out}/state_sweep_{tag}.pdf  {out}/state_sweep_margin_{mtag}.pdf  (+ .png)")


if __name__ == "__main__":
    main()
