#!/usr/bin/env python3
"""Run Milestone 1 experiment scripts with sensible defaults and skips."""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "experiments"


def latest_result_dir(parent: Path, report_name: str) -> Path | None:
    if not parent.exists():
        return None

    matches = [path.parent for path in parent.glob(f"*/{report_name}")]
    if not matches:
        return None

    return max(matches, key=lambda path: path.stat().st_mtime)


def discover_scripts() -> list[Path]:
    scripts = []
    for exp_dir in sorted(TARGET_DIR.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("m1_"):
            continue
        for path in sorted(exp_dir.rglob("*.py")):
            if path.name.startswith("run_all_"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if 'if __name__ == "__main__":' in text or "if __name__ == '__main__':" in text:
                scripts.append(path)
    return scripts


def script_config(script: Path) -> dict:
    rel = script.relative_to(ROOT).as_posix()
    cfg = {"args": [], "skip": None, "timeout": 900}
    latest_hybrid_dir = latest_result_dir(ROOT / "output" / "m1_e3_hybrid", "m1_e3_report.json")
    latest_accuracy_dir = latest_result_dir(ROOT / "output" / "m1_e4_accuracy", "m1_e4_report.json")

    if rel.endswith("m1_e1_templates/m1_e1_evaluation_debug.py"):
        cfg["args"] = ["output/m1_e1_templates/20260521_105530"]
    elif rel.endswith("m1_e2_llm_nlg/lora_inference.py"):
        cfg["args"] = [
            "--instruction",
            "Render a short temporal explanation for a single fact.",
            "--input-text",
            "The event happened on 2024-05-01.",
            "--base-model",
            "microsoft/phi-4-mini-instruct",
            "--adapter-path",
            "models/temporal_nlg_lora",
            "--max-new-tokens",
            "32",
            "--no-4bit",
        ]
        cfg["timeout"] = 1800
    elif rel.endswith("m1_e2_llm_nlg/m1_e2_evaluation_debug.py"):
        cfg["args"] = ["output/m1_e2_llm_nlg/74aad412bee64201a15074219b8f6232"]
    elif rel.endswith("m1_e3_hybrid/m1_e3_evaluation_debug.py"):
        if latest_hybrid_dir is None:
            cfg["skip"] = "requires a completed hybrid output directory"
        else:
            cfg["args"] = [str(latest_hybrid_dir.relative_to(ROOT))]
    elif rel.endswith("m1_e3_hybrid/run_eval.py"):
        cfg["args"] = ["--examples", "20", "--report-only"]
        cfg["timeout"] = 1800
    elif rel.endswith("m1_e4_accuracy/m1_e4_evaluation_debug.py"):
        if latest_accuracy_dir is None:
            cfg["skip"] = "requires a completed accuracy output directory"
        else:
            cfg["args"] = [str(latest_accuracy_dir.relative_to(ROOT))]
    elif rel.endswith("m1_e4_accuracy/run_eval.py"):
        cfg["args"] = ["--report-only", "output/m1_e2_llm_nlg/74aad412bee64201a15074219b8f6232"]
        cfg["timeout"] = 1800
    elif rel.endswith("m1_e5_integration/human_eval_aggregate.py"):
        cfg["args"] = ["docs/human_eval_template.csv"]
    elif rel.endswith("m1_e5_integration/package_builder.py"):
        cfg["timeout"] = 1800

    return cfg


def run_scripts(scripts: list[Path], list_only: bool = False, fail_fast: bool = False) -> int:
    if not scripts:
        print("No runnable experiment scripts found.")
        return 0

    if list_only:
        for script in scripts:
            print(script.relative_to(ROOT))
        return 0

    failed = []
    skipped = []
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{ROOT}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(ROOT)
    )

    for script in scripts:
        rel = script.relative_to(ROOT)
        cfg = script_config(script)
        if cfg["skip"]:
            print(f"\n=== Skipping {rel} ===")
            print(f"SKIPPED: {cfg['skip']}")
            skipped.append((rel, cfg["skip"]))
            continue
        print(f"\n=== Running {rel} ===")
        try:
            result = subprocess.run(
                [sys.executable, str(script), *cfg["args"]],
                cwd=str(ROOT),
                env=env,
                timeout=cfg["timeout"],
            )
        except subprocess.TimeoutExpired:
            failed.append((rel, "timeout"))
            if fail_fast:
                break
            continue

        if result.returncode != 0:
            failed.append((rel, result.returncode))
            if fail_fast:
                break

    print("\n=== Summary ===")
    print(f"Total: {len(scripts)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed: {len(failed)}")
    if skipped:
        for rel, reason in skipped:
            print(f"- SKIPPED {rel}: {reason}")
    if failed:
        for rel, code in failed:
            print(f"- {rel} (exit {code})")
        return 1
    print("All runnable scripts passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Milestone 1 experiments")
    parser.add_argument("--list", action="store_true", help="Only list scripts")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at first failure")
    args = parser.parse_args()

    scripts = discover_scripts()
    return run_scripts(scripts, list_only=args.list, fail_fast=args.fail_fast)


if __name__ == "__main__":
    raise SystemExit(main())
