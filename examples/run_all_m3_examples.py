#!/usr/bin/env python3
"""Run all Milestone 3 example scripts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "examples" / "milestone3"


def discover_scripts() -> list[Path]:
    scripts = []
    for path in sorted(TARGET_DIR.glob("*.py")):
        if path.name.startswith("run_all_"):
            continue
        scripts.append(path)
    return scripts


def run_scripts(scripts: list[Path], list_only: bool = False, fail_fast: bool = False) -> int:
    if not scripts:
        print("No example scripts found.")
        return 0

    if list_only:
        for script in scripts:
            print(script.relative_to(ROOT))
        return 0

    failed = []
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{ROOT}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(ROOT)
    )

    for script in scripts:
        rel = script.relative_to(ROOT)
        print(f"\n=== Running {rel} ===")
        result = subprocess.run([sys.executable, str(script)], cwd=str(ROOT), env=env, timeout=300)
        if result.returncode != 0:
            failed.append((rel, result.returncode))
            if fail_fast:
                break

    print("\n=== Summary ===")
    print(f"Total: {len(scripts)}")
    print(f"Failed: {len(failed)}")
    if failed:
        for rel, code in failed:
            print(f"- {rel} (exit {code})")
        return 1
    print("All scripts passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Milestone 3 examples")
    parser.add_argument("--list", action="store_true", help="Only list scripts")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at first failure")
    args = parser.parse_args()

    scripts = discover_scripts()
    return run_scripts(scripts, list_only=args.list, fail_fast=args.fail_fast)


if __name__ == "__main__":
    raise SystemExit(main())
