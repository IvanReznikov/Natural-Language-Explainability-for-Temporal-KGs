#!/usr/bin/env python3
"""Run Milestone 3 experiment scripts with sensible defaults and skips."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "experiments"
DATASET = "data/jsonls/temporal_graph.jsonl"
M3_E5_EVAL_SET = ROOT / "data" / "jsonls" / "temporal_evaluation_set_v2.jsonl"

ORDER = {
    "experiments/m3_e2_fidelity/run_fidelity_eval.py": 10,
    "experiments/m3_e3_human_eval/run_comprehension.py": 20,
    "experiments/m3_e3_human_eval/run_utility.py": 30,
    "experiments/m3_e3_human_eval/run_cognitive_load.py": 40,
    "experiments/m3_e4_efficiency_consistency/run_efficiency.py": 50,
    "experiments/m3_e4_efficiency_consistency/run_consistency.py": 60,
    "experiments/m3_e4_efficiency_consistency/run_coherence.py": 70,
    "experiments/m3_e4_efficiency_consistency/run_granularity.py": 80,
    "experiments/m3_e4_efficiency_consistency/run_generate_predictions.py": 90,
    "experiments/m3_e5_benchmark/run_m3_e5.py": 100,
}


def is_runnable_script(path: Path) -> bool:
    if path.name.startswith("run_all_"):
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return 'if __name__ == "__main__":' in text or "if __name__ == '__main__':" in text


def discover_scripts() -> list[Path]:
    scripts = []
    for exp_dir in sorted(TARGET_DIR.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("m3_"):
            continue
        for path in sorted(exp_dir.rglob("*.py")):
            if is_runnable_script(path):
                scripts.append(path)
    return sorted(
        scripts,
        key=lambda path: (ORDER.get(path.relative_to(ROOT).as_posix(), 999), path.as_posix()),
    )


def latest_output_file(pattern: str) -> Path | None:
    candidates = list((ROOT / "output").glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def has_cuda_gpu() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


HAS_CUDA_GPU = has_cuda_gpu()


def script_config(script: Path) -> dict:
    rel = script.relative_to(ROOT).as_posix()
    cfg = {"args": [], "skip": None, "timeout": 900}
    latest_eff_scenarios = latest_output_file("m3_e4a_efficiency/m3_e4a_scenarios.jsonl")

    if rel.endswith("m3_e2_fidelity/run_fidelity_eval.py"):
        cfg["args"] = ["--dataset", DATASET, "--output-dir", "output/m3_e2_fidelity_smoke"]
    elif rel.endswith("m3_e3_human_eval/run_comprehension.py"):
        cfg["args"] = [
            "export",
            "--dataset",
            DATASET,
            "--output-dir",
            "output/m3_e3a_comprehension",
        ]
    elif rel.endswith("m3_e3_human_eval/run_utility.py"):
        cfg["args"] = ["export", "--dataset", DATASET, "--output-dir", "output/m3_e3b_utility"]
    elif rel.endswith("m3_e3_human_eval/run_cognitive_load.py"):
        cfg["args"] = [
            "export",
            "--dataset",
            DATASET,
            "--output-dir",
            "output/m3_e3c_cognitive_load",
        ]
    elif rel.endswith("m3_e4_efficiency_consistency/run_efficiency.py"):
        cfg["args"] = ["export", "--dataset", DATASET, "--output-dir", "output/m3_e4a_efficiency"]
    elif rel.endswith("m3_e4_efficiency_consistency/run_consistency.py"):
        cfg["args"] = ["export", "--dataset", DATASET, "--output-dir", "output/m3_e4b_consistency"]
    elif rel.endswith("m3_e4_efficiency_consistency/run_coherence.py"):
        cfg["args"] = ["export", "--dataset", DATASET, "--output-dir", "output/m3_e4c_coherence"]
    elif rel.endswith("m3_e4_efficiency_consistency/run_granularity.py"):
        cfg["args"] = ["export", "--dataset", DATASET, "--output-dir", "output/m3_e4d_granularity"]
    elif rel.endswith("m3_e4_efficiency_consistency/run_generate_predictions.py"):
        if latest_eff_scenarios is None:
            cfg["skip"] = "requires m3_e4a_efficiency/m3_e4a_scenarios.jsonl"
        else:
            cfg["args"] = [
                "efficiency",
                "--scenarios",
                str(latest_eff_scenarios.relative_to(ROOT)),
                "--dataset",
                DATASET,
                "--methods",
                "template,baseline",
                "--output",
                "output/m3_e4a_efficiency/m3_e4a_predictions.jsonl",
                "--runs-out",
                "output/m3_e4a_efficiency/m3_e4a_runs.jsonl",
            ]
            cfg["timeout"] = 1800
    elif rel.startswith("experiments/m3_e5_benchmark/"):
        if rel.endswith("run_m3_e5_lcel_openai_models.py") or rel.endswith(
            "run_lcel_gpt_models.py"
        ):
            cfg["skip"] = "OpenAI runs are disabled by orchestrator policy"
        elif not M3_E5_EVAL_SET.exists():
            cfg["skip"] = "data/jsonls/temporal_evaluation_set_v2.jsonl is missing"
        elif not HAS_CUDA_GPU:
            cfg["skip"] = "local qwen M3-E5 requires CUDA GPU; none detected"
        else:
            cfg["args"] = [
                "--run-all",
                "--eval-set",
                str(M3_E5_EVAL_SET.relative_to(ROOT)),
                "--graph-dir",
                "data/jsonls/temporal_graph_output_v3",
                "--output-dir",
                "output/m3_e5_results",
            ]
            cfg["timeout"] = 10800

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
    parser = argparse.ArgumentParser(description="Run all Milestone 3 experiments")
    parser.add_argument("--list", action="store_true", help="Only list scripts")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at first failure")
    args = parser.parse_args()

    scripts = discover_scripts()
    return run_scripts(scripts, list_only=args.list, fail_fast=args.fail_fast)


if __name__ == "__main__":
    raise SystemExit(main())
