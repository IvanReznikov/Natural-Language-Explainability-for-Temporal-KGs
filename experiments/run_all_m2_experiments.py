#!/usr/bin/env python3
"""Run Milestone 2 experiment scripts with sensible defaults and skips."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "experiments"
GOLD_E3 = "experiments/m2_e3_parse/data/temporal_queries_gold.jsonl"

ORDER = {
    "experiments/m2_e2_intent/run_intent_classifier.py": 10,
    "experiments/m2_e3_construct/run_construct.py": 20,
    "experiments/m2_e3_optimize/run_optimize.py": 30,
    "experiments/m2_e3_parse/run_parse.py": 40,
    "experiments/m2_e4_taxonomy/split_data.py": 50,
    "experiments/m2_e4_taxonomy/report_splits.py": 60,
    "experiments/m2_e4_taxonomy/train_taxonomy_default.py": 70,
    "experiments/m2_e4_taxonomy/predict_taxonomy.py": 80,
    "experiments/m2_e5_trace_meta_query/generate_trace_corpus.py": 90,
    "experiments/m2_e5_trace_meta_query/meta_query_cli.py": 100,
    "experiments/m2_e5_trace_meta_query/trace_benchmark.py": 110,
    "experiments/m2_e5_trace_meta_query/trace_integration_demo.py": 120,
    "experiments/m2_e6_query_store_triggers/generate_query_corpus.py": 130,
    "experiments/m2_e6_query_store_triggers/change_detection_demo.py": 140,
    "experiments/m2_e6_query_store_triggers/e2e_chain_demo.py": 150,
    "experiments/m2_e6_query_store_triggers/query_store_benchmark.py": 160,
    "experiments/m2_e6_query_store_triggers/trigger_demo.py": 170,
    "experiments/m2_e7_harness/generate_e2e_queries.py": 180,
    "experiments/m2_e7_harness/generate_traces.py": 190,
    "experiments/m2_e7_harness/run_e2e.py": 200,
}


def is_runnable_script(path: Path) -> bool:
    if path.name.startswith("run_all_"):
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "if __name__ == \"__main__\":" in text or "if __name__ == '__main__':" in text


def discover_scripts() -> list[Path]:
    scripts = []
    for exp_dir in sorted(TARGET_DIR.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("m2_"):
            continue
        for path in sorted(exp_dir.rglob("*.py")):
            if is_runnable_script(path):
                scripts.append(path)
    return sorted(scripts, key=lambda path: (ORDER.get(path.relative_to(ROOT).as_posix(), 999), path.as_posix()))


def latest_output_file(pattern: str) -> Path | None:
    candidates = list((ROOT / "output").glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def script_config(script: Path) -> dict:
    rel = script.relative_to(ROOT).as_posix()
    cfg = {"args": [], "skip": None, "timeout": 900}
    latest_construct = latest_output_file("m2_e3_construct/*/outputs.jsonl")
    latest_optimize = latest_output_file("m2_e3_optimize/*/optimized.jsonl")
    latest_parse = latest_output_file("m2_e3_eval/*/preds.jsonl")

    if rel.endswith("m2_e2_intent/run_intent_classifier.py"):
        cfg["args"] = [
            "--dataset",
            "experiments/m2_e2_intent/data/annotated_queries.jsonl",
            "--output-dir",
            "output/m2_e2_intent",
        ]
    elif rel.endswith("m2_e3_construct/run_construct.py"):
        cfg["args"] = [
            "--data",
            "experiments/m2_e3_parse/data/temporal_queries_gold.jsonl",
            "--output-dir",
            "output/m2_e3_construct",
            "--use-gold",
        ]
    elif rel.endswith("m2_e3_construct/eval_construct.py"):
        if latest_construct is None:
            cfg["skip"] = "requires a construct outputs.jsonl artifact"
        else:
            cfg["args"] = [
                "--gold",
                GOLD_E3,
                "--pred",
                str(latest_construct.relative_to(ROOT)),
            ]
    elif rel.endswith("m2_e3_optimize/run_optimize.py"):
        cfg["args"] = [
            "--data",
            "experiments/m2_e3_parse/data/temporal_queries_gold.jsonl",
            "--output-dir",
            "output/m2_e3_optimize",
        ]
    elif rel.endswith("m2_e3_optimize/eval_optimize.py"):
        if latest_optimize is None:
            cfg["skip"] = "requires an optimize optimized.jsonl artifact"
        else:
            cfg["args"] = [
                "--gold",
                GOLD_E3,
                "--pred",
                str(latest_optimize.relative_to(ROOT)),
            ]
    elif rel.endswith("m2_e3_parse/run_parse.py"):
        cfg["args"] = [
            "--data",
            "experiments/m2_e3_parse/data/temporal_queries_gold.jsonl",
            "--output-dir",
            "output/m2_e3_eval",
        ]
    elif rel.endswith("m2_e3_parse/eval_parse.py"):
        if latest_parse is None:
            cfg["skip"] = "requires a parse preds.jsonl artifact"
        else:
            cfg["args"] = [
                "--gold",
                GOLD_E3,
                "--pred",
                str(latest_parse.relative_to(ROOT)),
            ]
    elif rel.endswith("m2_e4_taxonomy/char_cnn_run.py"):
        cfg["skip"] = "experimental training script without stable automation defaults"
    elif rel.endswith("m2_e4_taxonomy/predict_taxonomy.py"):
        model_path = ROOT / "output" / "m2_e4_taxonomy" / "e4a_taxonomy" / "taxonomy_model.joblib"
        if not model_path.exists():
            cfg["skip"] = "taxonomy model has not been trained yet"
        else:
            cfg["args"] = ["--text", "Revenue grew 12 percent year over year."]
    elif rel.endswith("m2_e5_trace_meta_query/generate_trace_corpus.py"):
        cfg["args"] = ["--count", "50", "--output", "output/m2_e5_trace_meta_query/small_traces.jsonl"]
    elif rel.endswith("m2_e5_trace_meta_query/meta_query_cli.py"):
        cfg["args"] = ["output/m2_e5_trace_meta_query/small_traces.jsonl", "list-rules"]
    elif rel.endswith("m2_e7_harness/run_e2e.py"):
        cfg["timeout"] = 1200

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
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(ROOT)

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
    parser = argparse.ArgumentParser(description="Run all Milestone 2 experiments")
    parser.add_argument("--list", action="store_true", help="Only list scripts")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at first failure")
    args = parser.parse_args()

    scripts = discover_scripts()
    return run_scripts(scripts, list_only=args.list, fail_fast=args.fail_fast)


if __name__ == "__main__":
    raise SystemExit(main())
