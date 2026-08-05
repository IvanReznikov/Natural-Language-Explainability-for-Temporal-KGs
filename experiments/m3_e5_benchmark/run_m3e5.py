#!/usr/bin/env python3
"""Compatibility wrapper for the canonical `run_m3_e5.py` entrypoint."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_m3_e5.py")), run_name="__main__")
