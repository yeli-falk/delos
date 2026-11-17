"""Paired per-seed difference between each two-level arm and the flat DELOS
run of the SAME seed (scripts/synthetic/hier_showcase.sh --aggregate).

A hint arm starts from exactly the flat run's initialisation and RNG stream,
so the per-seed difference removes the seed-to-seed spread that dominates the
unpaired table. Reported per horizon for the `jepa` (rolled-out forecast)
column: mean difference, its standard deviation over seeds, and how many
seeds are positive.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def _load(root: Path, stem: str, seed: int):
    f = root / f"{stem}_s{seed}" / "state_probe.json"
    if not f.exists():
        return None
    with open(f) as fh:
        return json.load(fh)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--tags", nargs="+", required=True,
                   help="JEPA_TAGs of the two-level arms, e.g. h3last_fc")
    p.add_argument("--base", default="jepa_ema", help="flat arm stem")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6, 9, 12])
    p.add_argument("--column", default="jepa")
    a = p.parse_args()
    root = Path(a.root)
    for metric, title in (("switch_acc_test", "SWITCH-row accuracy"),
                          ("acc_test", "overall accuracy")):
        print(f"\n{title}: arm - flat, column '{a.column}'   mean (sd) [seeds > 0 / n]")
        print(f"{'arm':<18}" + "".join(f"{'h=' + str(h):>24}" for h in a.horizons))
        for tag in a.tags:
            cells = []
            for h in a.horizons:
                d = []
                for s in a.seeds:
                    flat, arm = _load(root, a.base, s), _load(root, f"{a.base}_{tag}", s)
                    try:
                        d.append(arm[a.column][f"subset_h{h - 1}"][metric]
                                 - flat[a.column][f"subset_h{h - 1}"][metric])
                    except (TypeError, KeyError):
                        continue
                if not d:
                    cells.append(f"{'-':>24}")
                    continue
                d = np.asarray(d)
                sd = d.std(ddof=1) if d.size > 1 else float("nan")
                cells.append(f"{d.mean():+.4f} ({sd:.4f}) [{int((d > 0).sum())}/{d.size}]".rjust(24))
            print(f"{tag:<18}" + "".join(cells))


if __name__ == "__main__":
    main()
