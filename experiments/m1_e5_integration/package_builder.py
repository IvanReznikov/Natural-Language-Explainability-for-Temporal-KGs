#!/usr/bin/env python3
"""
Utility to build the temporal-nlg package artifacts for M1.

- Ensures pyproject.toml exists
- Runs `python -m build`
- Drops artifacts into ./dist
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def run_build() -> int:
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        print("pyproject.toml not found; cannot build package.")
        return 1

    print("Building package via python -m build ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "build"], cwd=ROOT, check=True
        )
    except subprocess.CalledProcessError as exc:
        print(f"Build failed with exit code {exc.returncode}")
        return exc.returncode

    dist = ROOT / "dist"
    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))

    if wheels or sdists:
        print("Artifacts:")
        for p in wheels + sdists:
            print(f" - {p}")
    else:
        print("Build completed but no artifacts were produced.")

    return 0


def main() -> int:
    return run_build()


if __name__ == "__main__":
    raise SystemExit(main())
