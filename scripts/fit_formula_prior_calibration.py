#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET = SCRIPT_DIR / "analyze_formula_prior_generalization.py"


if __name__ == "__main__":
    if not TARGET.exists():
        raise FileNotFoundError(f"Missing calibration backend script: {TARGET}")
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")
