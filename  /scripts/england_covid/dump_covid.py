"""Aggregate England COVID forecast metrics across seeds into one table.

Reads every ``test_metrics.json`` written by the all-methods run
(scripts/england_covid/slurm_all_methods.sh) — DyG-JEPA (full + encoder-only
control) and the CLDG/DDGCL/IDOL baselines — groups directories that share a
base name (stripping the trailing ``_s<seed>``), and reports mean ± std per
method, per horizon, for both the overall forecast and the inflection subset
(the turning points where persistence is provably wrong).

Persistence is parameter-free and identical across methods/seeds, so it's
printed once as the reference line; each method's ``gap`` column is
mean(MAE - MAE_persist) — negative means it beats persistence.

Usage:
    python scripts/england_covid/dump_covid.py                 # default glob
    python scripts/england_covid/dump_covid.py --pattern 'outputs/ec_*_s*'
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def _base(name: str) -> str:
    """Strip a trailing _s<seed> to group a method's seeds together."""
    return re.sub(r"_s\d+$", "", name)


def _load(d: Path):
    p = d / "test_metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _agg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    mu = mean(vals)
    sd = stdev(vals) if len(vals) > 1 else 0.0
    return mu, sd, len(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="outputs/ec_*_s*",
                   help="Glob of per-seed method directories.")
    # Preferred display order; anything else is appended alphabetically.
    p.add_argument("--order", nargs="*", default=[
        "ec_dygjepa", "ec_dygjepa_encoder", "ec_cldg", "ec_ddgcl", "ec_idol"])
    args = p.parse_args()

    groups = defaultdict(list)                       # base -> [metrics dict]
    for d in sorted(_glob.glob(args.pattern)):
        m = _load(Path(d))
        if m is not None:
            groups[_base(Path(d).name)].append(m)
    if not groups:
        print(f"[dump_covid] no test_metrics.json under {args.pattern!r}")
        return

    # Horizons present anywhere (as sorted ints).
    horizons = sorted({int(h) for ms in groups.values() for m in ms for h in m},
                      key=int)

    # Persistence reference (identical across methods/seeds — take the first).
    any_ms = next(iter(groups.values()))[0]
    print(f"{'horizon':<9}{'persist_MAE':>12}{'persist_infl':>13}")
    for h in horizons:
        e = any_ms.get(str(h), {})
        print(f"+{h:<8}{e.get('MAE_persist', float('nan')):>12.3f}"
              f"{e.get('MAE_persist_inflect', float('nan')):>13.3f}")
    print()

    # Per method, per horizon: MAE (mean±std), gap vs persist, inflection MAE + gap.
    def _order_key(b):
        return (args.order.index(b) if b in args.order else len(args.order), b)

    hdr = f"{'method':<20}{'seeds':>6}  " + "".join(
        f"h{h}:MAE          gap     infl_gap    " for h in horizons)
    print(hdr)
    print("-" * len(hdr))
    for b in sorted(groups, key=_order_key):
        ms = groups[b]
        cells = ""
        for h in horizons:
            hs = str(h)
            mae = _agg([m[hs]["MAE"] for m in ms if hs in m])
            gap = _agg([m[hs]["MAE"] - m[hs]["MAE_persist"]
                        for m in ms if hs in m and "MAE_persist" in m[hs]])
            igap = _agg([m[hs]["MAE_inflect"] - m[hs]["MAE_persist_inflect"]
                         for m in ms if hs in m and "MAE_inflect" in m[hs]])
            if mae is None:
                cells += f"{'-':>12}  {'-':>6}  {'-':>8}    "
                continue
            g = f"{gap[0]:+.3f}" if gap else "-"
            ig = f"{igap[0]:+.3f}" if igap else "-"
            cells += f"{mae[0]:>6.3f}±{mae[1]:.3f}  {g:>7}  {ig:>8}    "
        n_seeds = len(ms)
        print(f"{b:<20}{n_seeds:>6}  {cells}")
    print()
    print("MAE = mean±std across seeds. gap = mean(MAE - MAE_persist); "
          "infl_gap = same on the inflection subset. NEGATIVE = beats "
          "persistence (the win, esp. infl_gap at h>=1).")


if __name__ == "__main__":
    main()
