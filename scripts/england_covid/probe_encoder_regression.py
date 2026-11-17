"""Deprecated location — delegates to scripts/regression/probe_encoder_regression.py.

The probe covers every node-signal regression dataset. This file only forwards
to that single implementation so that existing command lines keep working.

Update your scripts to call:
    python scripts/regression/probe_encoder_regression.py ...
"""
import runpy
import sys
from pathlib import Path

_NEW = Path(__file__).resolve().parents[1] / "regression" / "probe_encoder_regression.py"

if __name__ == "__main__":
    print(f"[deprecated] scripts/england_covid/probe_encoder_regression.py -> "
          f"{_NEW.relative_to(Path(__file__).resolve().parents[2])}",
          file=sys.stderr)
    runpy.run_path(str(_NEW), run_name="__main__")
