"""Merge the MOOC state-prediction results for every method into one table.

Reads the ``state_probe.json`` each method writes and prints a mean±std row per
method across seeds. Every method is probed by the SAME deterministic probe on
the SAME rows, so the columns are directly comparable:

  oracle   encoder that SAW the target bin        (upper bound)
  last     encoder at the last history bin        (copy-last / persistence)
  fcast    the forecast column -- DyG-JEPA's predictor latent, JODIE's temporal
           projection. Absent for CLDG / DDGCL / IDOL: they are encoder-only
           SSL methods with no forecast operator, so `last` IS their prediction.
  gap      fcast - last, PAIRED within a seed. This is the claim: a forecast
           that beats persistence. Pairing matters because both columns share
           one encoder within a seed, so the difference has far less variance
           than either column alone.
  deghist  degree-at-each-history-bin probe = the activity-trend control. A
           method must clear this to be showing learned dynamics rather than
           "this user is winding down".

Usage:
    python scripts/dblp/aggregate_mooc_state.py                     # defaults
    python scripts/dblp/aggregate_mooc_state.py --seeds 42 43 44
    python scripts/dblp/aggregate_mooc_state.py --split val         # HP-search view
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

# method -> (directory template, does it have a forecast column?)
# {tag} = the shared-config TAG, {seed} = seed.
LAYOUT = {
    "DyG-JEPA": ("outputs/{tag}_s{seed}_cjepa",       True),
    "JODIE":    ("outputs/jodie_{dataset}_s{seed}",   True),
    "CLDG":     ("outputs/cldg_{tag}_s{seed}",        False),
    "DDGCL":    ("outputs/ddgcl_{tag}_s{seed}",       False),
    "IDOL":     ("outputs/idol_{tag}_s{seed}",        False),
}

# --layout variants: the DyG-JEPA encoder-training arms from
# slurm_mooc_jepa_variants.sh. Same columns, same probe, same rows — the only
# difference between arms is how the encoder was trained.
VARIANT_LAYOUT = {
    "frozen": ("outputs/mooc_jepa_variants/frozen_final_s{seed}", True),
    "ema":    ("outputs/mooc_jepa_variants/ema_final_s{seed}",    True),
    "naive":  ("outputs/mooc_jepa_variants/naive_final_s{seed}",  True),
    "vicreg": ("outputs/mooc_jepa_variants/vicreg_final_s{seed}", True),
}


def _get(d, *keys):
    """Nested lookup returning None for a missing key or a nan (empty subset)."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    if isinstance(d, float) and math.isnan(d):
        return None
    return d


def _cell(vals, places: int = 4) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.{places}f}±{s:.{places}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="jodie_mooc_gcn_u3000i97",
                   help="TAG from _dataset_config.sh, WITHOUT the _s<seed> "
                        "suffix (the script appends it).")
    p.add_argument("--dataset", default="mooc", help="JODIE dataset short name.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--layout", choices=["methods", "variants"], default="methods",
                   help="'methods' compares DyG-JEPA against the baselines; "
                        "'variants' compares the DyG-JEPA encoder-training arms "
                        "(frozen / ema / naive / vicreg).")
    p.add_argument("--split", choices=["test", "val"], default="test",
                   help="'test' reads state_probe.json (the reported number); "
                        "'val' reads state_probe_val.json (HP-search view).")
    args = p.parse_args()

    layout = VARIANT_LAYOUT if args.layout == "variants" else LAYOUT
    fname = "state_probe.json" if args.split == "test" else "state_probe_val.json"
    rows, missing = {}, []
    for method, (tmpl, _) in layout.items():
        for seed in args.seeds:
            path = Path(tmpl.format(tag=args.tag, dataset=args.dataset,
                                    seed=seed)) / fname
            if not path.exists():
                missing.append(f"{method} s{seed}: {path}")
                continue
            rows.setdefault(method, []).append(json.loads(path.read_text()))

    what = ("DyG-JEPA encoder-training variants" if args.layout == "variants"
            else "method comparison")
    print(f"\nMOOC state prediction ({what}) — {args.split} split, "
          f"seeds={args.seeds}")

    def table(metric: str, title: str, extra_cols: bool):
        hdr = (f"{'method':<10}{'fcast':>16}{'last':>16}{'gap (paired)':>16}"
               f"{'deghist':>16}" + (f"{'oracle':>16}" if extra_cols else "")
               + f"{'seeds':>7}")
        print(f"\n--- {title} ---")
        print(hdr)
        print("-" * len(hdr))
        for method, (_, has_fcast) in layout.items():
            rs = rows.get(method, [])
            if not rs:
                print(f"{method:<10}{'(no runs found)':>16}")
                continue
            last = [_get(r, "encoder_last", metric) for r in rs]
            fc = [_get(r, "jepa", metric) for r in rs] if has_fcast else []
            gap = ([a - b for a, b in zip(fc, last)
                    if a is not None and b is not None] if has_fcast else [])
            dh = [_get(r, "degree_history_baseline", metric) for r in rs]
            line = (f"{method:<10}{_cell(fc):>16}{_cell(last):>16}"
                    f"{_cell(gap):>16}{_cell(dh):>16}")
            if extra_cols:
                line += _cell([_get(r, "encoder_oracle", metric) for r in rs]).rjust(16)
            print(line + f"{len(rs):>7}")

    table("auroc_test", "AUROC (chance = 0.5)", extra_cols=True)

    # AP's floor is the positive rate, not 0.5 — without it the numbers look
    # tiny when they may be several times chance.
    npos = [r.get("n_pos_test") for r in next(iter(rows.values()))]
    nrow = [r.get("n_valid_rows") for r in next(iter(rows.values()))]
    floor = (npos[0] / nrow[0]) if npos and nrow and npos[0] and nrow[0] else None
    ftxt = f"random floor = {floor:.4f}" if floor else "random floor = n/a"
    table("auprc_test", f"Average precision ({ftxt})", extra_cols=False)
    if floor:
        print(f"\nAP is the metric that matters at {floor:.1%} positives; "
              f"divide by {floor:.4f} for lift over chance.")

    # Row counts must be identical across methods — if they are not, the runs
    # used different subsets and the table is not a like-for-like comparison.
    print()
    sig = {}
    for method, rs in rows.items():
        for r in rs:
            sig.setdefault((r.get("n_valid_rows"), r.get("n_pos_test")),
                           []).append(method)
    if len(sig) == 1:
        (nv, npos), _ = next(iter(sig.items()))
        print(f"[check] all methods probed identical rows "
              f"(n_valid={nv}, n_pos_test={npos}) — comparable.")
    else:
        print("[WARN] methods were probed on DIFFERENT row counts — they did "
              "not see the same dataset, so this table is not comparable:")
        for (nv, npos), ms in sig.items():
            print(f"        n_valid={nv} n_pos_test={npos}: {sorted(set(ms))}")

    if missing:
        print(f"\n[missing] {len(missing)} run(s) not found:")
        for m in missing[:12]:
            print(f"   {m}")
        if len(missing) > 12:
            print(f"   ... and {len(missing)-12} more")

    print("\ngap = fcast - last, paired within seed. >0 = the forecast beats "
          "persistence.\nCLDG/DDGCL/IDOL have no forecast operator, so their "
          "`last` column IS their prediction\nand they are compared against "
          "DyG-JEPA's and JODIE's `fcast`.")


if __name__ == "__main__":
    main()
