"""Horizon showcase figure + table for the synthetic node-state task.

Reads the per-seed ``state_probe.json`` that ``horizon_showcase.sh`` produces
(``<root>/<arm>_s<seed>/state_probe.json``, one run per method and seed,
trained and probed with a long forecast window). Every column carries a
``subset_h<k>`` block fitted per horizon (run_block_probe.py::_fit_per_horizon),
so a single run gives the whole curve h = 1..NPRED.

What is plotted, per horizon:

  DELOS arms      the rolled-out latent (`jepa` column)
  baselines          their embedding at the last observed bin, with a head
                     trained for that horizon (`encoder_last`) -- direct
                     multi-step forecasting, the only option without a
                     dynamics model
  copy-last          the JEPA run's OWN `encoder_last`: the same (EMA-trained)
                     encoder as its rollout, so the vertical distance between
                     the blue line and the grey dashed one is the predictor's
                     contribution
  label persistence  the label at the last observed bin still holds (overall
                     panel only; it is 0 on switch rows by construction)

Outputs (<out>/):
  horizon_curves.pdf|png   one panel per metric, methods vs steps ahead
  horizon_margin.pdf|png   DELOS minus each baseline, PAIRED WITHIN
                           SEED (the seed is the generated graph, so every arm
                           of a seed sees the same data), band = std of the
                           per-seed differences; and minus its own copy-last
  horizon_table.tex|txt    mean +/- std per method at the table horizons,
                           switch-row and overall accuracy, plus seeds-won
                           counts for DELOS vs each baseline

Usage:
    python scripts/synthetic/plot_horizon.py --root outputs/synth_horizon/NPRED_12 \
        --seeds 42 43 44 45 46 --table_horizons 1 3 6 9 12 --out figures/synth_horizon --latex
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

# (arm directory stem, legend label, column inside state_probe.json)
# The DELOS row is the EMA arm (encoder trained jointly with an EMA target);
# --ref_arm picks which JEPA run supplies the matched controls and the paired
# margins. jepa_frozen is kept as an optional extra row, off by default.
JEPA_ARMS = [("jepa_ema", "DELOS rollout", "jepa")]
OPTIONAL_JEPA = [("jepa_frozen", "DELOS rollout (frozen enc.)", "jepa")]
BASELINES = [("cldg", "CLDG", "encoder_last"),
             ("ddgcl", "DDGCL", "encoder_last"),
             ("idol", "IDOL", "encoder_last"),
             ("maskdgnn", "MaskDGNN", "encoder_last"),
             ("dysubc", "DySubC", "encoder_last"),
             ("dyg2vec", "DyG2Vec", "encoder_last"),
             ("dygwm", "DyG-WM (SSL)", "encoder_last"),
             ("sgjepa", "SG-JEPA", "encoder_last"),
             # static SSL controls (one snapshot per step); drawn only if their
             # runs exist or they are named in --baselines
             ("graphmae", "GraphMAE (static)", "encoder_last"),
             ("bgrl", "BGRL (static)", "encoder_last"),
             ("grace", "GRACE (static)", "encoder_last")]
# Matched controls, read from the reference JEPA run (same encoder as its rollout).
CONTROLS = [("encoder_last", "copy-last + direct head (same encoder)"),
            ("encoder_diff", "copy-last + Δlatent (same encoder)"),
            ("label_persistence", "label persistence"),
            ("encoder_oracle", "oracle encoder (saw target)")]

# Fixed categorical order, validated (dataviz palette, light surface: adjacent
# CVD dE >= 8, normal >= 15). JEPA is slot 1 (blue; an optional second JEPA
# arm shares the hue and differs by line style + marker); the six temporal
# baselines take slots 2-7 in order; DyG-WM / SG-JEPA take brown / cyan.
COLORS = {"jepa_ema": "#2a78d6", "jepa_frozen": "#2a78d6",
          "cldg": "#eb6834", "ddgcl": "#1baf7a", "idol": "#4a3aa7",
          "maskdgnn": "#eda100", "dysubc": "#e87ba4", "dyg2vec": "#008300",
          "dygwm": "#8d6e63", "sgjepa": "#00acc1",
          # the static block shares slot 8 (red) and differs by marker/line style
          "graphmae": "#e34948", "bgrl": "#e34948", "grace": "#e34948"}
STYLE = {"jepa_ema": dict(ls="-", marker="o", lw=2.4),
         "jepa_frozen": dict(ls="--", marker="s", lw=2.0),
         "cldg": dict(ls="-", marker="^", lw=1.6),
         "ddgcl": dict(ls="-", marker="D", lw=1.6),
         "idol": dict(ls="-", marker="v", lw=1.6),
         "maskdgnn": dict(ls="-", marker="P", lw=1.6),
         "dysubc": dict(ls="-", marker="X", lw=1.6),
         "dyg2vec": dict(ls="-", marker="<", lw=1.6),
         "dygwm": dict(ls="-", marker=">", lw=1.6),
         "sgjepa": dict(ls="-", marker="h", lw=1.6),
         "graphmae": dict(ls=":", marker="s", lw=1.4),
         "bgrl": dict(ls=":", marker="^", lw=1.4),
         "grace": dict(ls=":", marker="o", lw=1.4)}
REF_ARM = "jepa_ema"          # overridden by --ref_arm
# Layout, set from the CLI in main(). The figure is drawn at its destination
# size so that the point sizes are the ones seen next to the body text and
# nothing is scaled at \includegraphics time.
WIDTH_IN = 3.33      # --fig_width: one column of a two-column template
HEIGHT_IN = None     # --fig_height: default = panel width * 0.95
FONT_PT = 7.0        # --font_pt
SCALE = 1.0          # line widths / marker sizes, relative to a 4.4in panel
#GREY = "#8a8985"
#TEXT = "#0b0b0b"
#TEXT2 = "#52514e"

METRIC_LABEL = {
    "switch_acc_test": "accuracy on switch rows",
    "acc_test": "overall accuracy",
    "stay_acc_test": "accuracy on STAY rows",
    "macro_f1_test": "macro-F1",
    "vel_r2_test": "R² to the hidden velocity\nat the last observed step",
    "delta_r2_test": "R² to the true latent CHANGE Δz",
    "latent_r2_test": "R² to the true latent z",
    "change_f1": "change-alarm F1",
}
# Panel title (the subset) and y-axis label (the quantity) when --panel_titles
# is on: "(a) switch rows" over an "accuracy" axis reads better in a column
# than one rotated "accuracy on switch rows".
PANEL_TITLE = {"switch_acc_test": "switch rows", "acc_test": "overall",
               "stay_acc_test": "stay rows", "macro_f1_test": "macro-F1",
               "change_f1": "change alarm", "vel_r2_test": "hidden velocity",
               "delta_r2_test": r"latent change $\Delta z$",
               "latent_r2_test": "latent $z$"}
PANEL_UNIT = {"switch_acc_test": "accuracy", "acc_test": "accuracy",
              "stay_acc_test": "accuracy", "macro_f1_test": "macro-F1",
              "change_f1": "F1", "vel_r2_test": "$R^2$",
              "delta_r2_test": "$R^2$", "latent_r2_test": "$R^2$"}
PANEL_TITLES = True        # --no_panel_titles

# label_persistence only carries these per horizon
PERSIST_METRICS = {"acc_test", "switch_acc_test", "stay_acc_test"}


def _load_runs(root: Path, stem: str, seeds: list[int], probe_file: str) -> dict[int, dict]:
    runs = {}
    for s in seeds:
        p = root / f"{stem}_s{s}" / probe_file
        if p.exists():
            runs[s] = json.load(open(p))
    return runs


def _horizons(blob: dict, column: str) -> list[int]:
    col = blob.get(column, {})
    ks = [int(m.group(1)) for k in col if (m := re.match(r"^subset_h(\d+)$", k))]
    return sorted(ks)


def _val(blob: dict, column: str, k: int, metric: str):
    v = blob.get(column, {}).get(f"subset_h{k}", {}).get(metric)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _agg(xs: list[float], band: str):
    if not xs:
        return (np.nan, np.nan, 0)
    sd = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    if band == "se":
        sd /= math.sqrt(len(xs))
    return (float(np.mean(xs)), sd, len(xs))


def collect(root: Path, seeds: list[int], metrics: list[str], band: str, probe_file: str):
    """-> horizons, {metric: {series: {k: (mean, band, n)}}}, per-seed raw
    {metric: {series: {k: {seed: value}}}}"""
    runs = {stem: _load_runs(root, stem, seeds, probe_file) for stem, _, _ in JEPA_ARMS + BASELINES}
    ref = runs.get(REF_ARM)
    if not ref:
        raise SystemExit(f"no {REF_ARM}_s<seed>/{probe_file} under {root} (the reference "
                         "JEPA run supplies the matched controls; pass --ref_arm)")
    horizons = sorted(set().union(*[_horizons(b, "encoder_last") for b in ref.values()]))
    if not horizons:
        raise SystemExit("no per-horizon blocks (subset_h<k>) -- was the run probed with "
                         "num_preds > 1 and the per-horizon probe (2026-09-24+)?")
    data, raw = {}, {}
    for m in metrics:
        data[m], raw[m] = {}, {}
        for stem, _, column in JEPA_ARMS + BASELINES:
            if not runs[stem]:
                continue
            per = {k: {s: v for s, b in runs[stem].items() if (v := _val(b, column, k, m)) is not None}
                   for k in horizons}
            if any(per.values()):
                raw[m][stem] = per
                data[m][stem] = {k: _agg(list(per[k].values()), band) for k in horizons}
        for column, _ in CONTROLS:
            if column == "label_persistence" and m not in PERSIST_METRICS:
                continue
            per = {k: {s: v for s, b in ref.items()
                       if (v := _val(b, column, k, m)) is not None} for k in horizons}
            if any(per.values()):
                raw[m][column] = per
                data[m][column] = {k: _agg(list(per[k].values()), band) for k in horizons}
    return horizons, data, raw


def _short(metric: str) -> str:
    """First line of the metric's full label, for metrics with no short name."""
    return METRIC_LABEL.get(metric, metric).split("\n")[0]


def paper_style(plt, n_panels: int):
    """Set the rcParams for a figure drawn at its FINAL size and return figsize.

    Also sets the module-level ``SCALE``: the per-series line widths and marker
    sizes in ``STYLE`` were chosen for a 4.4in panel, so they are multiplied by
    the panel's share of that width wherever they are used.
    """
    global SCALE
    w = WIDTH_IN
    panel_w = w / max(n_panels, 1)
    h = HEIGHT_IN if HEIGHT_IN else panel_w * 0.95
    SCALE = max(panel_w / 4.4, 0.40)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "xtick.labelsize": FONT_PT - 1,
        "ytick.labelsize": FONT_PT - 1,
        "legend.fontsize": FONT_PT - 1,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        # Tight handles and gaps decide whether a 6-entry legend fits one row
        # at column width.
        "legend.handlelength": 1.2, "legend.handletextpad": 0.3,
        "legend.columnspacing": 0.7, "legend.borderaxespad": 0.0,
        "axes.titlepad": 3.0,
    })
    return (w, h)


def _style_axis(ax, xlabel, ylabel, horizons):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([k + 1 for k in horizons])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    #for sp in ("left", "bottom"):
        #ax.spines[sp].set_color("#d9d8d2")
    #ax.grid(axis="y", color="#e8e7e2", lw=0.8)
    ax.set_axisbelow(True)


def _add_legend(fig, handles, keys, cols: int = 0, top: float = 1.0):
    """Legend in ONE row directly under the axes.

    ``cols`` = 0 decides the column count from how much text there is: one row
    while the entries plausibly fit the figure width, two otherwise (the margin
    figure's labels are long). Pass ``--legend_cols`` to force it.

    The reserved strip at the bottom is sized to the number of legend ROWS, so
    the gap between the axes and the legend stays small instead of being a
    fixed 16% of the figure height; ``bbox_inches="tight"`` crops the rest at
    save time.
    """
    n = max(1, len(keys))
    arts = [handles[l] for l in keys]
    # Start from one row and add a row only when the legend is measured to
    # overflow the figure (estimating the width from the label text is unreliable).
    ncol, leg = (cols if cols > 0 else n), None
    while True:
        if leg is not None:
            leg.remove()
        leg = fig.legend(arts, keys, loc="lower center", ncol=ncol,
                         frameon=False, bbox_to_anchor=(0.5, 0.0))
        if cols > 0 or ncol <= 1:
            break
        try:
            w_in = leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
        except Exception:                      # no renderer: keep one row
            break
        if w_in <= fig.get_figwidth():
            break
        ncol = max(1, math.ceil(n / (math.ceil(n / ncol) + 1)))
    rows = math.ceil(n / ncol)
    fig.tight_layout(rect=(0, 0.06 * rows + 0.02, 1, top))


def plot_curves(horizons, data, metrics, out: Path, controls: str, band: str,
                legend_cols: int = 0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([k + 1 for k in horizons])
    fig, axs = plt.subplots(1, len(metrics), figsize=paper_style(plt, len(metrics)),
                            squeeze=False)
    handles = {}
    shown = {"none": [], "copylast": ["encoder_last", "encoder_diff", "label_persistence"],
             "all": ["encoder_last", "encoder_diff", "label_persistence", "encoder_oracle"]}[controls]
    for c, m in enumerate(metrics):
        ax = axs[0][c]
        cells = data[m]
        for column, label in CONTROLS:
            if column not in shown or column not in cells:
                continue
            y = np.array([cells[column][k][0] for k in horizons])
            ls = {"encoder_last": "--", "encoder_diff": ":", "label_persistence": "-.",
                  "encoder_oracle": (0, (1, 1))}[column]
            (h,) = ax.plot(x, y, ls=ls, lw=(1.6 if column != "encoder_diff" else 2.0) * SCALE,
                           marker="x" if column == "encoder_diff" else None,
                           ms=5 * SCALE, zorder=1)
            handles[label] = h
        for stem, label, _ in BASELINES + JEPA_ARMS:       # JEPA drawn last = on top
            if stem not in cells:
                continue
            y = np.array([cells[stem][k][0] for k in horizons])
            e = np.array([cells[stem][k][1] for k in horizons])
            st = STYLE[stem]
            ax.fill_between(x, y - e, y + e, color=COLORS[stem], alpha=0.10, lw=0, zorder=2)
            (h,) = ax.plot(x, y, color=COLORS[stem], lw=st["lw"] * SCALE, ls=st["ls"],
                           marker=st["marker"], ms=6 * SCALE, mec="white", mew=1.0 * SCALE,
                           zorder=4 if stem.startswith("jepa") else 3)
            handles[label] = h
        if PANEL_TITLES:
            ax.set_title(f"({chr(97 + c)}) {PANEL_TITLE.get(m, _short(m))}")
        _style_axis(ax, "steps ahead (h)",
                    PANEL_UNIT.get(m, _short(m)) if PANEL_TITLES else METRIC_LABEL.get(m, m),
                    horizons)
    order = ([l for _, l, _ in JEPA_ARMS] + [l for _, l, _ in BASELINES]
             + [l for _, l in CONTROLS])
    keys = [l for l in order if l in handles]
    #fig.suptitle("Node-state forecast vs horizon on the synthetic generator "
                 #f"(band = ±1 {'std' if band == 'std' else 'se'} over seeds)", fontsize=10, color=TEXT)
    _add_legend(fig, handles, keys, legend_cols, top=1.0)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def margins(horizons, raw, metrics, band: str):
    """Paired within seed: JEPA(REF_ARM) - other, per horizon.
    -> {metric: {other: {k: (mean, band, n, wins)}}}"""
    out = {}
    for m in metrics:
        out[m] = {}
        ref = raw[m].get(REF_ARM)
        if not ref:
            continue
        for other in [s for s, _, _ in BASELINES] + [s for s, _, _ in JEPA_ARMS if s != REF_ARM] + ["encoder_last"]:
            if other not in raw[m]:
                continue
            per = {}
            for k in horizons:
                d = [ref[k][s] - raw[m][other][k][s] for s in ref[k] if s in raw[m][other][k]]
                mean, sd, n = _agg(d, band)
                per[k] = (mean, sd, n, int(sum(v > 0 for v in d)))
            out[m][other] = per
    return out


def plot_margin(horizons, mg, metrics, out: Path, band: str, legend_cols: int = 0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([k + 1 for k in horizons])
    fig, axs = plt.subplots(1, len(metrics), figsize=paper_style(plt, len(metrics)),
                            squeeze=False)
    handles = {}
    label_of = {s: l for s, l, _ in BASELINES + JEPA_ARMS}
    label_of["encoder_last"] = "own copy-last + direct head (rollout gain)"
    for c, m in enumerate(metrics):
        ax = axs[0][c]
        ax.axhline(0, lw=1, zorder=1)
        for other, per in mg[m].items():
            y = np.array([per[k][0] for k in horizons])
            e = np.array([per[k][1] for k in horizons])
            if other == "encoder_last":
                (h,) = ax.plot(x, y, ls="--", lw=1.8 * SCALE, marker="o", ms=5 * SCALE,
                               mec="white", mew=1.0 * SCALE, zorder=3)
                ax.fill_between(x, y - e, y + e,  alpha=0.10, lw=0, zorder=2)
            else:
                st = STYLE[other]
                ax.fill_between(x, y - e, y + e, color=COLORS[other], alpha=0.10, lw=0, zorder=2)
                (h,) = ax.plot(x, y, color=COLORS[other], lw=st["lw"] * SCALE, ls=st["ls"],
                               marker=st["marker"], ms=6 * SCALE, mec="white",
                               mew=1.0 * SCALE, zorder=3)
            handles[label_of[other]] = h
        if PANEL_TITLES:
            ax.set_title(f"({chr(97 + c)}) {PANEL_TITLE.get(m, _short(m))}")
            _style_axis(ax, "steps ahead (h)", "rollout − method", horizons)
        else:
            _style_axis(ax, "steps ahead (h)",
                        "DELOS rollout − method\n" + METRIC_LABEL.get(m, m).split("\n")[0],
                        horizons)
    keys = list(handles)
    _add_legend(fig, handles, keys, legend_cols, top=1.0)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fmt(cell, prec=3, signed=False, best=False, latex=False):
    if cell is None or np.isnan(cell[0]):
        return "--"
    s = f"{cell[0]:{'+' if signed else ''}.{prec}f} ± {cell[1]:.{prec}f}"
    if latex:
        s = s.replace("±", r"$\pm$")
        if best:
            s = r"\textbf{" + s + "}"
    return s


def write_table(horizons, data, mg, table_metrics, table_h, out_dir: Path, latex: bool, prec: int):
    """One block per metric: rows = floor / copy-last / baselines / JEPA arms,
    columns = the table horizons; then the paired-margin block with seeds won."""
    ks = [h - 1 for h in table_h if h - 1 in horizons]
    lines_txt, lines_tex = [], []
    for m in table_metrics:
        cells = data[m]
        rows = [(l, c) for c, l in CONTROLS if c in cells and c != "encoder_oracle"]
        rows += [(l, s) for s, l, _ in BASELINES + JEPA_ARMS if s in cells]
        # bold: best method (not control) per horizon
        best = {}
        for k in ks:
            cand = [(cells[s][k][0], s) for s, _, _ in BASELINES + JEPA_ARMS
                    if s in cells and not np.isnan(cells[s][k][0])]
            best[k] = max(cand)[1] if cand else None
        title = METRIC_LABEL.get(m, m).split("\n")[0]
        hdr = f"{'':40}" + "".join(f"{'h=' + str(k + 1):>18}" for k in ks)
        lines_txt += ["", f"=== {title} ===", hdr]
        lines_tex += [r"\begin{tabular}{l" + "r" * len(ks) + "}", r"\toprule",
                      f"{title} & " + " & ".join(f"$h={k + 1}$" for k in ks) + r" \\", r"\midrule"]
        for label, key in rows:
            cs = [cells[key].get(k) for k in ks]
            lines_txt.append(f"{label:40}" + "".join(f"{_fmt(c, prec):>18}" for c in cs))
            lines_tex.append(label + " & " + " & ".join(
                _fmt(c, prec, best=(best.get(k) == key), latex=True) for c, k in zip(cs, ks)) + r" \\")
        lines_tex += [r"\bottomrule", r"\end{tabular}", ""]
        # margins
        if mg.get(m):
            lines_txt += ["", f"--- DELOS rollout minus method, paired within seed  [{title}]  (mean ± std, seeds won)"]
            lines_tex += [r"\begin{tabular}{l" + "r" * len(ks) + "}", r"\toprule",
                          f"rollout $-$ method ({title}) & " + " & ".join(f"$h={k + 1}$" for k in ks) + r" \\",
                          r"\midrule"]
            label_of = {s: l for s, l, _ in BASELINES + JEPA_ARMS}
            label_of["encoder_last"] = "own copy-last + direct head"
            for other, per in mg[m].items():
                cs = [per.get(k) for k in ks]
                txt = "".join(f"{(_fmt(c, prec, signed=True) + f' ({c[3]}/{c[2]})') if c else '--':>24}"
                              for c in cs)
                lines_txt.append(f"{label_of[other]:40}" + txt)
                lines_tex.append(label_of[other] + " & " + " & ".join(
                    (_fmt(c, prec, signed=True, latex=True) + f" ({c[3]}/{c[2]})") if c else "--"
                    for c in cs) + r" \\")
            lines_tex += [r"\bottomrule", r"\end{tabular}", ""]
    txt = "\n".join(lines_txt)
    print(txt)
    (out_dir / "horizon_table.txt").write_text(txt + "\n")
    if latex:
        (out_dir / "horizon_table.tex").write_text("\n".join(lines_tex))
        print(f"\n[latex] {out_dir / 'horizon_table.tex'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs/synth_horizon/NPRED_12")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--metrics", nargs="+", default=["switch_acc_test", "acc_test"],
                   help="per-horizon keys inside subset_h<k>: switch_acc_test acc_test "
                        "stay_acc_test macro_f1_test vel_r2_test delta_r2_test latent_r2_test "
                        "change_f1. vel_r2_test needs a probe from 2026-09-24 or later; "
                        "delta_r2_test rises with h through mean reversion (position-readable) "
                        "and is not an anticipation measure at long horizons.")
    p.add_argument("--table_metrics", nargs="+", default=["switch_acc_test", "acc_test"])
    p.add_argument("--table_horizons", type=int, nargs="+", default=[1, 3, 6, 9, 12],
                   help="steps ahead (1-indexed) shown in the table")
    p.add_argument("--plot_horizons", type=int, nargs="+", default=None,
               help="steps ahead (1-indexed) to draw in the figures; default: all available")
    p.add_argument("--h_max", type=int, default=None,
               help="draw only h <= h_max (ignored if --plot_horizons is given)")
    p.add_argument("--controls", choices=["none", "copylast", "all"], default="copylast",
                   help="grey reference lines: the reference JEPA run's own copy-last + the "
                        "persistence floor (default), also the oracle encoder (all), or none")
    p.add_argument("--ref_arm", default="jepa_ema",
                   help="JEPA run that supplies the matched controls and paired margins")
    p.add_argument("--with_frozen", action="store_true",
                   help="also draw the jepa_frozen arm if its runs exist")
    p.add_argument("--baselines", nargs="*", default=None,
                   help="baseline stems to draw, in this order (default: every one whose "
                        "runs exist): cldg ddgcl idol maskdgnn dysubc dyg2vec dygwm sgjepa graphmae bgrl grace")
    p.add_argument("--main_label", default=None,
                   help="legend/table name for a tagged --ref_arm (e.g. 'DELOS')")
    p.add_argument("--jepa_arms", nargs="*", default=[],
                   help="extra JEPA run stems to draw as rows, e.g. predictor variants "
                        "written with JEPA_TAG (jepa_ema_ff jepa_ema_cw4); label = stem suffix")
    p.add_argument("--fig_width", type=float, default=3.33,
                   help="figure width in INCHES, matched to where the paper "
                        "includes it: 3.33 = one column of a two-column "
                        "template, ~7.0 = full width. Draw at the final size "
                        "and include it without scaling, or the fonts shrink "
                        "by whatever \\includegraphics scales by.")
    p.add_argument("--fig_height", type=float, default=None,
                   help="figure height in inches (default: panel width * 0.95)")
    p.add_argument("--font_pt", type=float, default=7.0,
                   help="base font size in points; ticks and legend are one less")
    p.add_argument("--no_panel_titles", dest="panel_titles", action="store_false",
                   help="draw the metric as a long rotated y-axis label instead "
                        "of an '(a) switch rows' panel title over a short one")
    p.set_defaults(panel_titles=True)
    p.add_argument("--legend_cols", type=int, default=0,
                   help="legend columns (0 = auto: one row while the labels fit, else two)")
    p.add_argument("--band", choices=["std", "se"], default="std")
    p.add_argument("--probe_file", default="state_probe.json")
    p.add_argument("--out", default="figures/synth_horizon")
    p.add_argument("--prec", type=int, default=3)
    p.add_argument("--latex", action="store_true")
    a = p.parse_args()
    global REF_ARM, WIDTH_IN, HEIGHT_IN, FONT_PT, PANEL_TITLES
    REF_ARM = a.ref_arm
    WIDTH_IN, HEIGHT_IN, FONT_PT = a.fig_width, a.fig_height, a.font_pt
    PANEL_TITLES = a.panel_titles
    if a.baselines is not None:
        known = {st: (st, l, c) for st, l, c in BASELINES}
        bad = [b for b in a.baselines if b not in known]
        if bad:
            raise SystemExit(f"unknown baseline(s) {bad}; choose from {list(known)}")
        BASELINES[:] = [known[b] for b in a.baselines]
    if a.with_frozen or a.ref_arm == "jepa_frozen":
        JEPA_ARMS[:] = OPTIONAL_JEPA + JEPA_ARMS if a.ref_arm == "jepa_frozen" else JEPA_ARMS + OPTIONAL_JEPA
    # Extra JEPA rows (predictor variants): same hue, cycled line styles. The
    # reference arm is always drawn (and drawn solid, as the main line), even
    # when it is a tagged variant not listed in --jepa_arms.
    extra_styles = [dict(ls="--", marker="s", lw=2.0), dict(ls=":", marker="^", lw=2.0),
                    dict(ls="-.", marker="D", lw=2.0), dict(ls="--", marker="v", lw=2.0)]
    if not any(REF_ARM == st for st, _, _ in JEPA_ARMS):
        suffix = REF_ARM.split("jepa_", 1)[-1]
        # A tagged reference arm REPLACES the default jepa_ema row (list
        # jepa_ema in --jepa_arms to keep it as a comparison line). With
        # --main_label the reference is drawn under that plain name, for the
        # paper figure.
        JEPA_ARMS[:] = [(st, l, c) for st, l, c in JEPA_ARMS if st != "jepa_ema" or "jepa_ema" in a.jepa_arms]
        JEPA_ARMS.insert(0, (REF_ARM, a.main_label or f"DELOS rollout ({suffix})", "jepa"))
        COLORS[REF_ARM] = COLORS["jepa_ema"]
        STYLE[REF_ARM] = dict(ls="-", marker="o", lw=2.4)
        if "jepa_ema" in a.jepa_arms:
            JEPA_ARMS[:] = [x for x in JEPA_ARMS if x[0] != "jepa_ema"] + [("jepa_ema", "DELOS rollout (ema)", "jepa")]
            STYLE["jepa_ema"] = dict(ls="--", marker="s", lw=2.0)
    for i, stem in enumerate(a.jepa_arms):
        if any(stem == st for st, _, _ in JEPA_ARMS):
            continue
        suffix = stem.split("jepa_", 1)[-1]
        JEPA_ARMS.append((stem, f"DELOS rollout ({suffix})", "jepa"))
        COLORS[stem] = COLORS["jepa_ema"]
        STYLE[stem] = extra_styles[i % len(extra_styles)]

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = list(dict.fromkeys(a.metrics + a.table_metrics))
    horizons, data, raw = collect(root, a.seeds, metrics, a.band, a.probe_file)
    mg = margins(horizons, raw, metrics, a.band)          # full set: table still sees every h

    if a.plot_horizons:
        plot_h = [h - 1 for h in a.plot_horizons if h - 1 in horizons]
    elif a.h_max:
        plot_h = [k for k in horizons if k + 1 <= a.h_max]
    else:
        plot_h = horizons
    if not plot_h:
        raise SystemExit(f"no requested horizon available; have h = {[k + 1 for k in horizons]}")

    present = [m for m in a.metrics if data.get(m)]
    plot_curves(plot_h, data, present, out / "horizon_curves.pdf", a.controls, a.band, a.legend_cols)
    plot_margin(plot_h, mg, [m for m in present if mg.get(m)], out / "horizon_margin.pdf",
                a.band, a.legend_cols)
    write_table(horizons, data, mg, [m for m in a.table_metrics if data.get(m)],
                a.table_horizons, out, a.latex, a.prec)
    print(f"[figures] {out / 'horizon_curves.pdf'}  {out / 'horizon_margin.pdf'}")


if __name__ == "__main__":
    main()
