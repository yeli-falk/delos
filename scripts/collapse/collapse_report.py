"""Tables and figures for the representation-collapse study.

Reads what scripts/collapse/collapse_metrics.py wrote under a results root
(<stem>_s<seed>/collapse.json, collapse_dynamics.json, the reference rows in
_collapse/) plus each run's state_probe.json and history.json, and writes to
--out:

  collapse_table.txt / .tex   one row per method, mean +/- std over seeds:
                              RankMe (pooled and per snapshot), spread,
                              pairwise cosine, within-node-over-time variance
                              share, forecast-latent RankMe, forecast error vs
                              copy-last, downstream probe score
  fig_spectrum.pdf            singular-value spectra (log scale): DELOS and
                              its controls | DELOS vs the SSL baselines
  fig_dynamics.pdf            val loss, RankMe and spread over training for
                              the arms that kept snapshots
  fig_temporal.pdf            within-node-over-time variance share per method
                              | forecast error relative to copy-last
  fig_rank_vs_probe.pdf       RankMe of the probed latent vs the probe score,
                              one dot per (method, seed); final model + SSL
                              baselines by default (--scatter all for every run)
  fig_spectrum_probe.pdf      the baselines spectrum panel and that scatter
                              side by side, one shared legend
  collapse_summary.json       every aggregated number

Roles:
  --main        the DELOS arm the paper reports (blue). Default: the first
                of jepa_ema_cos / jepa_ema / jepa_frozen that has results.
  --collapsed   the positive control(s) expected to collapse (dark grey).
                Default: every jepa_naive* stem found.
The view columns are documented in collapse_metrics.py. Seeds are whatever is
on disk unless --seeds is given; every cell prints its own n.

Usage:
    python scripts/collapse/collapse_report.py --root outputs/collapse/mooc \
        --out figures/collapse/mooc --latex
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.classification.aggregate_classification import METHODS  # noqa: E402
from scripts.synthetic import plot_horizon as PH  # noqa: E402
from scripts.synthetic.plot_horizon import COLORS, STYLE  # noqa: E402

RUN_RE = re.compile(r"^(?P<stem>.+)_s(?P<seed>\d+)$")
LABEL = {stem: name for name, stem in METHODS}
ORDER = [stem for _, stem in METHODS]
REFS = [("ref_input", "Raw node features"),
        ("ref_random", "Random-init encoder"),
        ("ref_pretrain", "Edge-mask pretrain (init)")]
BLUE = COLORS["jepa_ema"]
# color=None means "take the next colour from matplotlib's default cycle"
# (resolved once in _resolve_colors): the controls and the collapsed arms were
# drawn in grey before, and text / spines / the grid used a charcoal palette.
CTRL = None
# Other DELOS arms share the JEPA hue and differ by line style + marker
# (fixed order, as in plot_horizon.py).
JEPA_ALT = [("--", "s"), ("-.", "^"), (":", "D"), ((0, (5, 1)), "v"),
            ((0, (3, 1, 1, 1)), "P"), ((0, (1, 1)), "X"), ((0, (7, 2)), "<")]
PROBE_METRIC = {"binary": ("auroc_test", "AUROC"), "multiclass": ("acc_test", "Acc"),
                "ranking": ("ndcg10_test", "NDCG@10")}
# Lettered titles on the combined paper figure only (the standalone diagnostic
# figures carry none; their captions name the panels). --no_panel_titles.
PANEL_TITLES = True
COMBINED_TITLES = ("(a) spectrum", "(b) rank vs probe")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load(root: Path, seeds, suffix: str):
    """-> {stem: {seed: {"c": collapse, "dyn": [...], "hist": [...], "probe": {...}}}}"""
    runs = {}
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith("_")]
    if (root / "_collapse").is_dir():
        dirs += sorted((root / "_collapse").iterdir())
    for d in dirs:
        m = RUN_RE.match(d.name)
        if not m or (seeds and m["seed"] not in seeds):
            continue
        c = _json(d / f"collapse{suffix}.json")
        if c is None:
            continue
        runs.setdefault(m["stem"], {})[int(m["seed"])] = {
            "c": c, "dyn": _json(d / f"collapse_dynamics{suffix}.json"),
            "hist": _json(d / "history.json"), "probe": _json(d / f"state_probe{suffix}.json")}
    return runs


FAMILY = "DELOS"        # name of the model family in every label (--family_label)
MAIN_LABEL = None       # label of the main arm alone (--main_label; default = FAMILY)


def label_of(stem: str, collapsed, main=None) -> str:
    if stem in dict(REFS):
        return dict(REFS)[stem]
    if stem == main:
        lab = MAIN_LABEL or FAMILY
    elif stem in LABEL:
        lab = LABEL[stem]
    else:
        base = max((s for s in LABEL if stem.startswith(s + "_")), key=len, default=None)
        lab = f"{LABEL[base]} [{stem[len(base) + 1:]}]" if base else stem
    lab = lab.replace("DELOS", FAMILY)
    return lab + (" [control]" if stem in collapsed else "")


def ordered(runs, collapsed):
    """refs, baselines, DELOS arms (METHODS order; tagged variants after
    their base arm)."""
    def key(stem):
        base = max((s for s in ORDER if stem == s or stem.startswith(s + "_")), key=len, default=None)
        return (ORDER.index(base) if base else len(ORDER), stem != base, stem)
    refs = [s for s, _ in REFS if s in runs]
    rest = sorted((s for s in runs if s not in refs), key=key)
    return refs + [s for s in rest if not s.startswith("jepa_")] + [s for s in rest if s.startswith("jepa_")]


def probe_score(blob):
    """(value, metric label, probed view) from a state_probe.json."""
    if not blob or blob.get("task") not in PROBE_METRIC:
        return None, None, None
    metric, name = PROBE_METRIC[blob["task"]]
    col = "jepa" if isinstance(blob.get("jepa"), dict) else "encoder_last"
    v = blob.get(col, {}).get(metric)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None, name, None
    return float(v), name, ("pred" if col == "jepa" else "last")


def agg(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    return (float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, len(xs))


def get(c, path):
    for k in path.split("."):
        c = c.get(k) if isinstance(c, dict) else None
    return c


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------
# (key, txt header, LaTeX header, path in collapse.json or callable, precision, in LaTeX table)
def columns(probe_name):
    last_h = lambda c: max((k for k in c.get("forecast", {}) if k.startswith("h")),  # noqa: E731
                           key=lambda k: int(k[1:]), default=None)
    return [
        ("rankme", "RankMe", r"RankMe $\uparrow$", "enc.rankme", 1, True),
        ("dim", "D", "$D$", "enc.dim", 0, False),
        ("snap_rankme", "RankMe/snap", r"RankMe / snapshot $\uparrow$", "enc.snap_rankme", 1, True),
        ("eff_rank_cov", "effrank(cov)", "eff. rank (cov)", "enc.eff_rank_cov", 1, False),
        ("spread", "spread", r"Spread $\uparrow$", "enc.spread", 3, True),
        ("cos", "cos", r"Pair cos. $\downarrow$", "enc.cos", 3, True),
        ("dead", "dead dims", "dead dims", "enc.dead_frac", 3, False),
        ("within", "time var.", r"Time var. share $\uparrow$", "temporal.within_frac", 3, True),
        ("step", "step ratio", "step ratio", "temporal.step_ratio", 3, False),
        ("pred_rankme", "RankMe(fcst)", r"Forecast RankMe $\uparrow$", "pred.rankme", 1, True),
        ("pred_spread", "spread(fcst)", "forecast spread", "pred.spread", 3, False),
        ("ratio_h1", "err/copy h1", r"Err. / copy-last ($h{=}1$) $\downarrow$", "forecast.h0.ratio", 3, True),
        ("ratio_hl", "err/copy hmax", "err / copy-last (h max)",
         lambda c: get(c, f"forecast.{last_h(c)}.ratio") if last_h(c) else None, 3, False),
        ("mag_h1", "disp.mag h1", "disp. mag.", "forecast.h0.disp_mag", 3, False),
        ("final_rankme", "RankMe@last ep", r"RankMe, last epoch $\uparrow$", "__final__", 1, True),
        ("probe", probe_name or "probe", (probe_name or "Probe") + r" $\uparrow$", "__probe__", 3, True),
    ]


def build_table(runs, order, probe_name):
    cols = columns(probe_name)
    table = {}
    for stem in order:
        row = {}
        for key, _, _, path, _, _ in cols:
            vals = []
            for r in runs[stem].values():
                if path == "__probe__":
                    vals.append(probe_score(r["probe"])[0])
                elif path == "__final__":
                    vals.append(r["dyn"][-1]["enc"]["rankme"] if r["dyn"] else None)
                elif callable(path):
                    vals.append(path(r["c"]))
                else:
                    vals.append(get(r["c"], path))
            row[key] = agg(vals)
        table[stem] = row
    return cols, table


def _cell(c, prec, latex=False):
    if c is None:
        return "--"
    if c[2] == 1 or prec == 0:
        return f"{c[0]:.{prec}f}"
    return (f"${c[0]:.{prec}f} \\pm {c[1]:.{prec}f}$" if latex else f"{c[0]:.{prec}f}±{c[1]:.{prec}f}")


def write_table(cols, table, order, names, out: Path, latex: bool):
    cols = [c for c in cols if any(table[s][c[0]] is not None for s in order)]
    w = max(len(v) for v in names.values()) + 2
    lines = [f"{'method':<{w}}" + "".join(f"{h:>16}" for _, h, *_ in cols) + f"{'n':>4}"]
    for s in order:
        n = max((c[2] for c in table[s].values() if c), default=0)
        lines.append(f"{names[s]:<{w}}" + "".join(
            f"{_cell(table[s][k], p):>16}" for k, _, _, _, p, _ in cols) + f"{n:>4}")
    txt = "\n".join(lines)
    (out / "collapse_table.txt").write_text(txt + "\n")
    print(txt)
    if latex:
        tc = [c for c in cols if c[5]]
        tex = [r"% RankMe / spread / cosine / time-variance: encoder latents of active nodes, held-out windows.",
               r"% [control] rows are the arms trained WITHOUT the anti-collapse mechanism.",
               r"\begin{tabular}{l" + "r" * len(tc) + "}", r"\toprule",
               "Method & " + " & ".join(h for _, _, h, *_ in tc) + r" \\", r"\midrule"]
        prev_group = None
        for s in order:
            group = 0 if s.startswith("ref_") else (2 if s.startswith("jepa_") else 1)
            if prev_group is not None and group != prev_group:
                tex.append(r"\midrule")
            prev_group = group
            tex.append(names[s].replace("_", r"\_") + " & " + " & ".join(
                _cell(table[s][k], p, latex=True) for k, _, _, _, p, _ in tc) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}", ""]
        (out / "collapse_table.tex").write_text("\n".join(tex))
        print(f"[saved] {out / 'collapse_table.tex'}")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def style_of(stem, main, collapsed, jepa_others):
    if stem == main:
        return dict(color=BLUE, ls="-", marker="o", lw=2.4)
    if stem in collapsed:
        i = sorted(collapsed).index(stem)
        return dict(color=CTRL, ls=["--", ":", "-."][i % 3], marker=["X", "P", "*"][i % 3], lw=2.0)
    if stem == "ref_random":
        return dict(color=None, ls=":", marker=None, lw=1.6)
    if stem == "ref_pretrain":
        return dict(color=None, ls="-.", marker=None, lw=1.6)
    if stem == "ref_input":
        return dict(color=None, ls="-", marker=None, lw=1.2)
    if stem in STYLE and not stem.startswith("jepa_"):
        return dict(color=COLORS[stem], **STYLE[stem])
    if stem.startswith("jepa_"):
        ls, mk = JEPA_ALT[jepa_others.index(stem) % len(JEPA_ALT)] if stem in jepa_others else ("--", "s")
        return dict(color=BLUE, ls=ls, marker=mk, lw=1.6)
    return dict(color=None, ls="-", marker="o", lw=1.4)


def _resolve_colors(plt, sty, order):
    """Give every series whose style carries ``color=None`` one colour from
    matplotlib's default cycle, skipping the hues the project palette already
    uses. Resolved ONCE, so a series' shaded band and its line always match
    (``fill_between`` and ``plot`` would otherwise each draw the next colour)."""
    from matplotlib.colors import to_rgb
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    taken = [to_rgb(st["color"]) for st in sty.values() if st["color"]]
    # skip cycle hues that are perceptually close to a palette hue: the default
    # cycle's blue sits next to the main arm's, which would read as the same arm
    far = lambda c: all(sum((a - b) ** 2 for a, b in zip(to_rgb(c), t)) > 0.0625 for t in taken)
    free = [c for c in cycle if far(c)] or list(cycle)
    i = 0
    for s in order:
        if sty[s]["color"] is None:
            sty[s]["color"] = free[i % len(free)]
            i += 1


def _axis(ax, xlabel=None, ylabel=None):
    # sizes come from the rcParams paper_style sets, as in plot_horizon;
    # hard-coding them here made these labels 9pt next to the horizon
    # figure's 7pt in the same paper
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_axisbelow(True)


def _save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"[saved] {path}")


def _mean_spectrum(runs_of_stem, view):
    specs = [np.asarray(r["c"][view]["spectrum"]) for r in runs_of_stem.values()
             if view in r["c"] and "spectrum" in r["c"][view]]
    if not specs or len({len(s) for s in specs}) != 1:
        return None
    # geometric mean over seeds: the axis is logarithmic
    return np.exp(np.mean([np.log(np.clip(s, 1e-8, None)) for s in specs], axis=0))


def _spectrum_series(runs, order, names, main, collapsed):
    """(controls panel, baselines panel) series lists: (stem, view, label, style overrides)."""
    controls = [(main, "enc", names[main] + " (encoder)", {}),
                (main, "pred", names[main] + " (forecast)", dict(ls="--", marker=None, lw=1.8))]
    controls += [(s, "enc", names[s], {}) for s in order if s in collapsed]
    controls += [(s, "enc", names[s], {}) for s in ("ref_pretrain", "ref_random") if s in runs]
    baselines = [(main, "enc", names[main], {})] + [
        (s, "enc", names[s], {}) for s in order
        if not s.startswith(("jepa_", "ref_")) and s not in collapsed]
    return controls, baselines[:8]


def _draw_spectrum(ax, runs, series, sty, main, title, ylabel=True, legend=True):
    for stem, view, lab, over in series:
        spec = _mean_spectrum(runs[stem], view)
        if spec is None:
            continue
        st = {**sty[stem], **over}
        x = np.arange(1, len(spec) + 1)
        ax.plot(x, spec, color=st["color"], ls=st["ls"], lw=st["lw"] * PH.SCALE,
                marker=st["marker"], ms=4.5 * PH.SCALE, markevery=max(1, len(x) // 8),
                markeredgecolor="white", markeredgewidth=0.8 * PH.SCALE, label=lab,
                zorder=3 if stem == main else 2)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.5)
    # no panel titles: the figure caption names the panels (``title`` is kept
    # in the signature because the callers describe the panel they build)
    _axis(ax, "singular value index", r"$\sigma_i / \sigma_1$" if ylabel else None)
    if legend:
        ax.legend(frameon=False, loc="lower left")


def fig_spectrum(plt, runs, order, sty, names, main, collapsed, out: Path, panels_mode="baselines"):
    controls, baselines = _spectrum_series(runs, order, names, main, collapsed)
    panels = []
    if panels_mode in ("both", "controls"):
        panels.append((f"{FAMILY} and its controls", controls))
    if panels_mode in ("both", "baselines") and len(baselines) > 1:
        panels.append((f"{FAMILY} vs SSL baselines", baselines))
    if not panels:
        print("[spectrum] nothing to draw -> skipped")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=PH.paper_style(plt, len(panels)),
                             sharey=True, squeeze=False)
    for i, (ax, (title, series)) in enumerate(zip(axes[0], panels)):
        _draw_spectrum(ax, runs, series, sty, main, title, ylabel=(i == 0))
    fig.tight_layout()
    _save(fig, out / "fig_spectrum.pdf")
    plt.close(fig)


def _curve(seed_series, min_n):
    """{seed: {epoch: value}} -> (epochs, mean, std) over the epochs at least
    `min_n` seeds reached (runs early-stop at different epochs)."""
    epochs = sorted(set().union(*[set(s) for s in seed_series.values()])) if seed_series else []
    xs, mu, sd = [], [], []
    for e in epochs:
        v = [s[e] for s in seed_series.values() if e in s and s[e] is not None]
        if len(v) >= min_n:
            xs.append(e); mu.append(np.mean(v)); sd.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
    return np.array(xs), np.array(mu), np.array(sd)


def fig_dynamics(plt, runs, arms, sty, names, out: Path):
    rows = [("val. JEPA loss", "loss", True),
            ("RankMe, encoder latents", "enc.rankme", False),
            ("RankMe, forecast latents", "pred.rankme", False),
            ("spread, encoder latents", "enc.spread", False)]
    w, h = PH.paper_style(plt, 1)
    fig, axes = plt.subplots(len(rows), 1, figsize=(w, h * len(rows)), sharex=True)
    handles = {}
    for ax, (ylabel, path, logy) in zip(axes, rows):
        for stem in arms:
            series = {}
            for seed, r in runs[stem].items():
                if not r["dyn"]:        # a run without snapshots contributes no curve
                    continue
                if path == "loss":
                    series[seed] = {h["epoch"]: h["val_loss"] for h in (r["hist"] or [])}
                else:
                    series[seed] = {d["epoch"]: get(d, path) for d in r["dyn"]}
                    # epoch 0 = the pretrained encoder the arm starts from
                    ref = runs.get("ref_pretrain", {}).get(seed)
                    if ref and path.startswith("enc.") and get(r["c"], "_meta.flags.encoder_ckpt"):
                        series[seed][0] = get(ref["c"], path)
            x, mu, sd = _curve(series, min_n=max(1, math.ceil(len(series) / 2)))
            if not len(x):
                continue
            st = sty[stem]
            lo = np.maximum(mu - sd, mu * 0.1) if logy else mu - sd   # log axis: no band below 0
            ax.fill_between(x, lo, mu + sd, color=st["color"], alpha=0.10, lw=0, zorder=2)
            (h,) = ax.plot(x, mu, color=st["color"], ls=st["ls"], lw=st["lw"] * PH.SCALE,
                           marker=st["marker"] if path != "loss" else None, ms=4.5 * PH.SCALE,
                           markeredgecolor="white", markeredgewidth=0.8 * PH.SCALE, zorder=3)
            handles[names[stem]] = h
        if logy:
            ax.set_yscale("log")
        _axis(ax, None, ylabel)
    axes[-1].set_xlabel("epoch")
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=min(3, len(handles)),
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.035 * math.ceil(len(handles) / 3)))
    _save(fig, out / "fig_dynamics.pdf")
    plt.close(fig)


def _dot_panel(ax, stems, cells, names, main, collapsed, xlabel, vline=None, sty=None):
    y = np.arange(len(stems))[::-1]
    for yi, s in zip(y, stems):
        c = cells[s]
        # the per-series colour resolved in _resolve_colors (BLUE for the main
        # arm); the controls were drawn grey before.
        color = sty[s]["color"] if sty else (BLUE if s == main else None)
        ax.errorbar(c[0], yi, xerr=c[1], fmt="o", color=color,
                    ms=(7 if s == main else 5.5) * PH.SCALE, elinewidth=1.4 * PH.SCALE,
                    capsize=0, markeredgecolor="white", markeredgewidth=0.8 * PH.SCALE, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([names[s] for s in stems])
    _axis(ax, xlabel)
    if vline is not None:
        ax.axvline(vline[0], lw=1, ls="--", zorder=1)
        ax.text(vline[0], len(stems) - 0.4, " " + vline[1], va="bottom")
    ax.set_ylim(-0.6, len(stems) - 0.2)


def fig_temporal(plt, runs, order, table, sty, names, main, collapsed, out: Path):
    a_stems = [s for s in order if table[s]["within"] is not None]
    j_stems = [s for s in order if any("forecast" in r["c"] for r in runs[s].values())]
    horizons = sorted({int(k[1:]) for s in j_stems for r in runs[s].values()
                       for k in r["c"].get("forecast", {}) if k.startswith("h")})
    n = max(len(a_stems), 3)
    w, h = PH.paper_style(plt, 2 if j_stems else 1)
    fig, axes = plt.subplots(1, 2 if j_stems else 1, figsize=(w, max(h, 0.11 * n + 0.6)),
                             squeeze=False)
    _dot_panel(axes[0][0], a_stems, {s: table[s]["within"] for s in a_stems}, names, main, collapsed,
               "within-node share of latent variance", sty=sty)
    if j_stems:
        ax = axes[0][1]
        if len(horizons) > 1:
            for s in j_stems:
                pts = [agg([get(r["c"], f"forecast.h{k}.ratio") for r in runs[s].values()]) for k in horizons]
                x = np.array([k + 1 for k, p in zip(horizons, pts) if p])
                mu = np.array([p[0] for p in pts if p]); sd = np.array([p[1] for p in pts if p])
                st = sty[s]
                ax.fill_between(x, mu - sd, mu + sd, color=st["color"], alpha=0.10, lw=0, zorder=2)
                ax.plot(x, mu, color=st["color"], ls=st["ls"], lw=st["lw"] * PH.SCALE,
                        marker=st["marker"], ms=5 * PH.SCALE, markeredgecolor="white",
                        markeredgewidth=0.8 * PH.SCALE, label=names[s], zorder=3)
            ax.axhline(1.0, lw=1, ls="--", zorder=1)
            ax.text(horizons[-1] + 1, 1.0, "copy-last ", va="top", ha="right")
            ax.set_xticks([k + 1 for k in horizons])
            _axis(ax, "steps ahead", "forecast error / copy-last")
            ax.legend(frameon=False)
        else:
            _dot_panel(ax, j_stems, {s: table[s]["ratio_h1"] for s in j_stems}, names, main, collapsed,
                       "forecast error / copy-last", vline=(1.0, "copy-last"), sty=sty)
    fig.tight_layout()
    _save(fig, out / "fig_temporal.pdf")
    plt.close(fig)


def _scatter_points(runs, stems):
    pts = {}
    for s in stems:
        for r in runs[s].values():
            v, _, view = probe_score(r["probe"])
            if v is not None and view in r["c"]:
                pts.setdefault(s, []).append((r["c"][view]["rankme"], v))
    return pts


def _draw_scatter(ax, pts, sty, names, main, probe_name, legend=True):
    for s, xy in pts.items():
        st = sty[s]
        xs, ys = zip(*xy)
        ax.scatter(xs, ys, s=(64 if s == main else 40) * PH.SCALE ** 2, color=st["color"],
                   marker=st["marker"] or "o", edgecolor="white", linewidth=0.8 * PH.SCALE,
                   label=names[s], zorder=3 if s == main else 2)
    allx = np.array([p[0] for v in pts.values() for p in v]); ally = np.array([p[1] for v in pts.values() for p in v])
    rx, ry = allx.argsort().argsort(), ally.argsort().argsort()
    rho = float(np.corrcoef(rx, ry)[0, 1]) if len(allx) > 2 and rx.std() > 0 and ry.std() > 0 else float("nan")
    _axis(ax, "RankMe", f"probe {probe_name}")
    if legend:
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return rho


def fig_scatter(plt, runs, order, sty, names, main, collapsed, probe_name, out: Path,
                arms="main", legend_cols=0):
    """Standalone rank-vs-probe scatter and the combined spectrum | scatter
    figure. arms='main' keeps the final model and the SSL baselines (the other
    DELOS arms and the controls are dropped); 'all' keeps every run."""
    if arms == "main":
        stems = [s for s in order if s == main or (not s.startswith(("jepa_", "ref_")) and s not in collapsed)]
    else:
        stems = order
    pts = _scatter_points(runs, stems)
    if sum(len(v) for v in pts.values()) < 3:
        print("[scatter] fewer than 3 runs with both a probe score and collapse metrics -> skipped")
        return None
    fig, ax = plt.subplots(figsize=PH.paper_style(plt, 1))
    rho = _draw_scatter(ax, pts, sty, names, main, probe_name)
    fig.tight_layout()
    _save(fig, out / "fig_rank_vs_probe.pdf")
    plt.close(fig)

    # spectrum | scatter side by side, one legend (same series, same styles)
    _, baselines = _spectrum_series(runs, order, names, main, collapsed)
    if len(baselines) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PH.paper_style(plt, 2))
        _draw_spectrum(ax1, runs, baselines, sty, main, f"{FAMILY} vs SSL baselines",
                       legend=False)
        _draw_scatter(ax2, pts, sty, names, main, probe_name, legend=False)
        # both panels draw the same series, so one legend under the figure --
        # an in-axes legend covers the curves at column width
        if PANEL_TITLES:
            ax1.set_title(COMBINED_TITLES[0])
            ax2.set_title(COMBINED_TITLES[1])
        h, l = ax1.get_legend_handles_labels()
        PH._add_legend(fig, dict(zip(l, h)), l, legend_cols, top=1.0)
        _save(fig, out / "fig_spectrum_probe.pdf")
        plt.close(fig)
    return rho


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--out", default=None, help="output directory (default: <root>/collapse_report)")
    p.add_argument("--seeds", nargs="*", default=None)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--main", default=None, help="stem of the reported DELOS arm")
    p.add_argument("--collapsed", nargs="*", default=None, help="stems of the collapse controls")
    p.add_argument("--stems", nargs="*", default=None, help="only these stems (references are always kept)")
    p.add_argument("--dyn_arms", nargs="*", default=None,
                   help="arms drawn in fig_dynamics (default: main, controls, jepa_frozen)")
    p.add_argument("--no_panel_titles", dest="panel_titles", action="store_false",
                   help="drop the '(a) spectrum' / '(b) rank vs probe' titles on "
                        "the combined figure")
    p.set_defaults(panel_titles=True)
    p.add_argument("--legend_cols", type=int, default=0,
                   help="legend columns on the combined figure (0 = as many as fit one row)")
    p.add_argument("--fig_width", type=float, default=3.33,
                   help="figure width in INCHES, matched to where the paper "
                        "includes it: 3.33 = one column of a two-column "
                        "template, ~7.0 = full width. Drawing at the final "
                        "size keeps the fonts at their stated point size; "
                        "include the PDF without scaling.")
    p.add_argument("--fig_height", type=float, default=None,
                   help="figure height in inches (default: panel width * 0.95)")
    p.add_argument("--font_pt", type=float, default=7.0,
                   help="base font size in points; ticks and legend are one less")
    p.add_argument("--latex", action="store_true")
    p.add_argument("--family_label", default="DELOS",
                   help="model-family name used in every DELOS label (default DELOS)")
    p.add_argument("--main_label", default=None,
                   help="label of the main arm on its own (default: the family name)")
    p.add_argument("--scatter", choices=["main", "all"], default="main",
                   help="fig_rank_vs_probe: the final model + SSL baselines (default) or every run")
    p.add_argument("--spectrum", choices=["baselines", "controls", "both"], default="baselines",
                   help="panels of fig_spectrum: vs the SSL baselines (default), the controls, or both")
    a = p.parse_args()
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "collapse_report"
    out.mkdir(parents=True, exist_ok=True)

    runs = load(root, a.seeds, "_val" if a.split == "val" else "")
    if a.stems:
        runs = {s: r for s, r in runs.items() if s in a.stems or s.startswith("ref_")}
    if not runs:
        raise SystemExit(f"no collapse.json under {root} -- run scripts/collapse/collapse_metrics.py first")
    main_arm = a.main or next((s for s in ("jepa_ema_cos", "jepa_ema", "jepa_frozen") if s in runs), None)
    if main_arm not in runs:
        raise SystemExit(f"main arm {main_arm!r} has no results under {root}; pass --main <stem> "
                         f"(found: {sorted(runs)})")
    collapsed = set(a.collapsed) if a.collapsed is not None else {
        s for s in runs if s == "jepa_naive" or s.startswith("jepa_naive_")}
    collapsed &= set(runs)
    global FAMILY, MAIN_LABEL
    FAMILY, MAIN_LABEL = a.family_label, a.main_label
    order = ordered(runs, collapsed)
    names = {s: label_of(s, collapsed, main_arm) for s in order}
    jepa_others = [s for s in order if s.startswith("jepa_") and s != main_arm and s not in collapsed]
    sty = {s: style_of(s, main_arm, collapsed, jepa_others) for s in order}
    probe_name = next((n for s in order for r in runs[s].values()
                       if (n := probe_score(r["probe"])[1])), None)
    print(f"[report] root={root} main={main_arm} controls={sorted(collapsed) or 'none'} "
          f"methods={len(order)} probe={probe_name}\n")

    cols, table = build_table(runs, order, probe_name)
    write_table(cols, table, order, names, out, a.latex)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    global PANEL_TITLES
    PANEL_TITLES = a.panel_titles
    PH.WIDTH_IN, PH.HEIGHT_IN, PH.FONT_PT = a.fig_width, a.fig_height, a.font_pt
    PH.paper_style(plt, 1)
    _resolve_colors(plt, sty, order)
    fig_spectrum(plt, runs, order, sty, names, main_arm, collapsed, out, a.spectrum)
    with_dyn = [s for s in order if any(r["dyn"] for r in runs[s].values())]
    dyn_arms = [s for s in (a.dyn_arms or [main_arm, *sorted(collapsed), "jepa_frozen"]) if s in with_dyn]
    dyn_arms = list(dict.fromkeys(dyn_arms))
    if dyn_arms:
        fig_dynamics(plt, runs, dyn_arms, sty, names, out)
    else:
        print("[dynamics] no run has collapse_dynamics.json (trained without --snapshot_every?) -> skipped")
    fig_temporal(plt, runs, order, table, sty, names, main_arm, collapsed, out)
    rho = fig_scatter(plt, runs, order, sty, names, main_arm, collapsed, probe_name, out,
                      a.scatter, a.legend_cols)

    summary = {"root": str(root), "main": main_arm, "controls": sorted(collapsed),
               "probe_metric": probe_name, "spearman_rank_vs_probe": rho,
               "rows": {s: {"label": names[s], **{k: (dict(mean=c[0], std=c[1], n=c[2]) if c else None)
                                                  for k, c in table[s].items()}} for s in order}}
    with open(out / "collapse_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {out / 'collapse_summary.json'}")


if __name__ == "__main__":
    main()
