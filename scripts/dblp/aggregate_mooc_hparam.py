"""Rank the MOOC state-prediction hyperparameter search, per method.

Reads ``outputs/mooc_hp/<method>_<slug>_s<seed>/state_probe_val.json`` — the
VAL-split probe written by slurm_mooc_hparam.sh — and prints each method's
cells best-first, plus the winning flag string to paste into the final run.

The ranking metric is deliberately per-method:
  * DyG-JEPA / JODIE have a forecast column, so they rank on the PAIRED gap
    (forecast AUROC - persistence AUROC). Ranking them on raw forecast AUROC
    would reward a strong encoder rather than a useful predictor, which is not
    the claim being made.
  * CLDG / DDGCL / IDOL are encoder-only, so they rank on persistence AUROC —
    the column that IS their prediction.

Note on field names: ``auroc_test`` inside state_probe_val.json is the probe's
held-out portion of the VAL windows, not the test split. The test split is
untouched by the search.

Usage:
    python scripts/dblp/aggregate_mooc_hparam.py
    python scripts/dblp/aggregate_mooc_hparam.py --top 5
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

# Env var each method's winning flags go into for the final 3-seed run.
ENV_VAR = {"jepa": "JEPA_HP", "cldg": "CLDG_HP", "ddgcl": "DDGCL_HP",
           "idol": "IDOL_HP", "jodie": "JODIE_HP",
           # DyG-JEPA variants (slurm_mooc_jepa_variants.sh)
           "frozen": "VARIANT_HP_frozen", "ema": "VARIANT_HP_ema",
           "naive": "VARIANT_HP_naive", "vicreg": "VARIANT_HP_vicreg"}
# Method order for display; anything unlisted is appended alphabetically.
ORDER = ["jepa", "cldg", "ddgcl", "idol", "jodie",
         "frozen", "ema", "naive", "vicreg"]
DIR_RE = re.compile(r"^(?P<method>[a-z]+)_(?P<slug>.+)_s(?P<seed>\d+)$")


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    if isinstance(d, float) and math.isnan(d):
        return None
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs/mooc_hp")
    p.add_argument("--top", type=int, default=8,
                   help="Cells to show per method (0 = all).")
    p.add_argument("--rank", choices=["auto", "gap", "fcast"], default="auto",
                   help="'gap' = fcast - last. Only valid when `last` is a FIXED "
                        "reference (frozen encoder): if the encoder is trainable, "
                        "each cell has its own `last` and maximising the gap "
                        "rewards cells whose encoder DEGRADED. 'fcast' ranks on "
                        "absolute forecast quality. 'auto' (default) picks gap "
                        "when `last` is near-constant within the arm, else fcast.")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist — has the search run yet?")

    by_method: dict[str, list] = {}
    for f in sorted(root.glob("*/state_probe_val.json")):
        m = DIR_RE.match(f.parent.name)
        if not m:
            continue
        method = m.group("method")
        r = json.loads(f.read_text())
        hp_file = f.parent / "hp_args.txt"
        hp = hp_file.read_text().strip() if hp_file.exists() else ""
        last = _get(r, "encoder_last", "auroc_test")
        fc = _get(r, "jepa", "auroc_test")
        # A method has a forecast operator iff its probe emitted a `jepa`
        # column. Hardcoding the set means every new variant is silently
        # mis-ranked as encoder-only.
        if fc is not None and last is not None:
            score = fc - last
        elif fc is not None:
            score = None
        else:
            score = last
        by_method.setdefault(method, []).append({
            "slug": m.group("slug"), "hp": hp, "seed": int(m.group("seed")),
            "score": score, "fcast": fc, "last": last,
            "deghist": _get(r, "degree_history_baseline", "auroc_test"),
            "n_pos": r.get("n_pos_test"), "n_rows": r.get("n_valid_rows"),
        })

    if not by_method:
        raise SystemExit(f"no state_probe_val.json found under {root}")

    methods = ([m for m in ORDER if m in by_method]
               + sorted(m for m in by_method if m not in ORDER))
    for method in methods:
        cells = by_method[method]
        has_fc = any(c["fcast"] is not None for c in cells)
        metric = ("gap = fcast - last" if has_fc
                  else "last_auroc (encoder-only: persistence IS the prediction)")
        # Decide the ranking statistic per arm.
        lasts = [c["last"] for c in cells if c["last"] is not None]
        spread = (max(lasts) - min(lasts)) if len(lasts) > 1 else 0.0
        if args.rank == "auto":
            use_gap = has_fc and spread <= 0.01
        else:
            use_gap = (args.rank == "gap") and has_fc
        if has_fc:
            for c in cells:
                c["score"] = (
                    (c["fcast"] - c["last"]) if (use_gap and c["fcast"] is not None
                                                 and c["last"] is not None)
                    else c["fcast"])
            metric = ("gap = fcast - last" if use_gap
                      else f"fcast (absolute; `last` spreads {spread:.4f} across "
                           f"cells, so it is not a fixed reference)")
        cells.sort(key=lambda c: (c["score"] is None, -(c["score"] or 0)))
        keep = cells if args.top == 0 else cells[: args.top]
        print(f"\n=== {method.upper()}  ({len(cells)} cells, ranked by {metric}) ===")
        print(f"{'rank':>4}  {'score':>9}  {'fcast':>8}  {'last':>8}  "
              f"{'deghist':>8}  config")
        print("-" * 96)
        for i, c in enumerate(keep, 1):
            f = lambda v: f"{v:.4f}" if v is not None else "   -  "
            print(f"{i:>4}  {f(c['score']):>9}  {f(c['fcast']):>8}  "
                  f"{f(c['last']):>8}  {f(c['deghist']):>8}  "
                  f"{c['hp'] or c['slug']}")
        best = cells[0]
        if best["score"] is None:
            print("  -> best: n/a (no scoreable cell)")
        else:
            env = ENV_VAR.get(method, f"{method.upper()}_HP")
            if best["hp"]:
                print(f'  -> best (score={best["score"]:.4f}):  '
                      f'{env}="{best["hp"]}"')
            else:
                # `slug` is a display label ("base"), NOT a flag string —
                # pasting it would reach argparse as an unrecognized positional
                # and kill the run. An arm with no flags wants the var UNSET.
                print(f'  -> best (score={best["score"]:.4f}):  no tunable '
                      f'flags for this arm — leave {env} UNSET')

    # Every cell must have been probed on the same rows.
    sig = {(c["n_rows"], c["n_pos"]) for cs in by_method.values() for c in cs}
    print()
    if len(sig) == 1:
        nr, npos = next(iter(sig))
        print(f"[check] every cell probed identical val rows "
              f"(n_valid={nr}, n_pos={npos}).")
        if npos:
            se = math.sqrt(0.6 * 0.4 / npos)     # AUROC SE, AUC ~ 0.6
            print(f"[noise] with only {npos} positives the AUROC standard error "
                  f"is ~{se:.3f}. Treat any two cells within ~{se:.3f} as tied — "
                  f"ranking below that is sampling noise, not signal.")
    else:
        print(f"[WARN] cells were probed on {len(sig)} different row counts "
              f"{sorted(sig)} — configs did not share a dataset.")
    print("\nSelection used the VAL split only. Re-run the winners across "
          "3 seeds and report the TEST split\n(state_probe.json) via "
          "aggregate_mooc_state.py.")


if __name__ == "__main__":
    main()
