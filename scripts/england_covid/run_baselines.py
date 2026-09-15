"""Deprecated location — delegates to scripts/regression/run_baselines.py.

The parameter-free baselines were generalised to every node-signal regression
dataset (England COVID *and* METR-LA) and now report through the same
accounting helpers as the learned arms, so their JSON carries the persistence
reference and the inflection subset too. Keeping a second copy here is exactly
the duplication that makes two result tables silently describe different
experiments, so this file forwards to the one implementation and exists only so
existing command lines keep working.

Update your scripts to call:
    python scripts/regression/run_baselines.py --data_source england_covid ...
"""
import runpy
import sys
from pathlib import Path

_NEW = Path(__file__).resolve().parents[1] / "regression" / "run_baselines.py"

if __name__ == "__main__":
    print(f"[deprecated] scripts/england_covid/run_baselines.py -> "
          f"{_NEW.relative_to(Path(__file__).resolve().parents[2])}",
          file=sys.stderr)
    runpy.run_path(str(_NEW), run_name="__main__")
