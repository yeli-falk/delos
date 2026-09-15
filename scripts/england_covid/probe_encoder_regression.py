"""Deprecated location — delegates to scripts/regression/probe_encoder_regression.py.

This probe was generalised to every node-signal regression dataset (England
COVID *and* METR-LA) and gained the supervised end-to-end arm. Keeping a second
copy here is exactly the duplication that makes two result tables silently
describe different experiments, so this file now forwards to the one
implementation and exists only so existing command lines keep working.

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
