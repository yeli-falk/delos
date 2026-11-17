"""Aggregate missing-history sweeps across seeds into one table.

Reads the ``missing_history.json`` files written by missing_history_covid.py /
missing_history_mooc.py, groups method directories by base name (stripping the
trailing ``_s<seed>``), and reports mean +/- std per method per hidden-fraction,
plus the degradation (metric at the largest fraction vs at fraction 0). The
robustness claim is a SMALL degradation: the method still predicts hidden nodes
from graph context.

Usage:
    python scripts/analysis/dump_missing_history.py --task covid
    python scripts/analysis/dump_missing_history.py --task mooc
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def _clean(xs):
    """Drop None/NaN so an empty or positive-starved subset doesn't crash agg."""
    return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def _base(name):
    return re.sub(r"_s\d+(_cjepa)?$", r"\1", name)


def _load(d):
    p = Path(d) / "missing_history.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _fnum(x):
    x = _clean(x)
    if not x:
        return "nan"
    return f"{mean(x):.3f}±{stdev(x):.3f}" if len(x) > 1 else f"{x[0]:.3f}"


def covid(pattern):
    groups = defaultdict(list)
    for d in sorted(_glob.glob(pattern)):
        m = _load(d)
        if m:
            groups[_base(Path(d).name)].append(m)
    if not groups:
        print(f"[dump] no missing_history.json under {pattern!r}"); return
    fracs = sorted({float(f) for ms in groups.values() for m in ms for f in m},
                   key=float)

    def avg_over_horizons(mj, frac, key):
        blk = mj.get(str(frac)) or mj.get(f"{frac}")
        if not blk:
            return None
        vals = [v[key] for v in blk.values()]
        return sum(vals) / len(vals)

    # Relational-persistence reference (method-independent): take any group.
    anym = next(iter(groups.values()))[0]
    print("forecast MAE on hidden nodes (avg over horizons, mean±std over seeds)")
    print(f"{'frac':>6}: " + "  ".join(f"{f:>13.2f}" for f in fracs))
    rp = [avg_over_horizons(anym, f, "MAE_relpersist") for f in fracs]
    print(f"{'rel-pers':>8} " + "  ".join(f"{v:>13.3f}" if v else f"{'-':>13}" for v in rp))
    print("-" * (8 + 15 * len(fracs)))
    order = ["ec_dygjepa", "ec_dygjepa_encoder", "ec_cldg", "ec_ddgcl", "ec_idol"]
    for b in sorted(groups, key=lambda x: (order.index(x) if x in order else 99, x)):
        ms = groups[b]
        cells = []
        for f in fracs:
            vs = [avg_over_horizons(m, f, "MAE_model") for m in ms]
            vs = [v for v in vs if v is not None]
            cells.append(_fnum(vs) if vs else "-")
        # degradation = MAE(max frac) - MAE(frac 0)
        v0 = _clean([avg_over_horizons(m, fracs[0], "MAE_model") for m in ms])
        vN = _clean([avg_over_horizons(m, fracs[-1], "MAE_model") for m in ms])
        deg = (mean(vN) - mean(v0)) if v0 and vN else float("nan")
        print(f"{b:<20} " + "  ".join(f"{c:>13}" for c in cells) + f"   Δ={deg:+.3f}")
    print("\nΔ = MAE(max hidden) − MAE(0 hidden); SMALLER = more robust to missing "
          "history.\nBeat rel-pers to show you use the graph beyond a 1-hop average.")


def mooc(pattern):
    groups = defaultdict(list)
    for d in sorted(_glob.glob(pattern)):
        m = _load(d)
        if m:
            groups[_base(Path(d).name)].append(m)
    if not groups:
        print(f"[dump] no missing_history.json under {pattern!r}"); return
    fracs = sorted({float(f) for ms in groups.values() for m in ms for f in m},
                   key=float)
    for b, ms in sorted(groups.items()):
        print(f"\n[{b}]  state AUROC on hidden nodes (mean±std over {len(ms)} seeds)")
        print(f"{'frac':>6}: " + "  ".join(f"{f:>13.2f}" for f in fracs))
        for col, key in (("enc-last", "enc_last_auroc"), ("jepa", "jepa_auroc")):
            cells = []
            for f in fracs:
                vs = [m[str(f)][key] for m in ms if str(f) in m]
                cells.append(_fnum(vs) if vs else "-")
            v0 = _clean([m[str(fracs[0])][key] for m in ms if str(fracs[0]) in m])
            vN = _clean([m[str(fracs[-1])][key] for m in ms if str(fracs[-1]) in m])
            deg = (mean(v0) - mean(vN)) if v0 and vN else float("nan")
            print(f"{col:<8} " + "  ".join(f"{c:>13}" for c in cells) + f"   drop={deg:+.3f}")
        print("drop = AUROC(0 hidden) − AUROC(max hidden); SMALLER drop = more "
              "robust. Claim: jepa drop < enc-last drop.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["covid", "mooc"], required=True)
    p.add_argument("--pattern", default=None)
    args = p.parse_args()
    if args.task == "covid":
        covid(args.pattern or "outputs/ec_*")
    else:
        mooc(args.pattern or "outputs/jodie_mooc_*_cjepa")


if __name__ == "__main__":
    main()
