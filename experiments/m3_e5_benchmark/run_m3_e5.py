#!/usr/bin/env python3
"""M3-E5: End-to-End QA System Benchmark runner.

Usage examples:

  # Run a single configuration
  python experiments/m3_e5_benchmark/run_m3_e5.py \\
      --llm-id llm_9b --mode graph_large_emb --emb-id emb_4b \\
      --eval-set data/jsonls/temporal_evaluation_set_v2.jsonl \\
      --graph-dir data/jsonls/temporal_graph_output_v3 \\
      --output-dir output/m3_e5_results

  # Run all 36 configurations (skips completed ones)
  python experiments/m3_e5_benchmark/run_m3_e5.py --run-all \\
      --eval-set data/jsonls/temporal_evaluation_set_v2.jsonl \\
      --graph-dir data/jsonls/temporal_graph_output_v3 \\
      --output-dir output/m3_e5_results

  # Aggregate completed runs into MATRIX.json
  python experiments/m3_e5_benchmark/run_m3_e5.py --aggregate \\
      --output-dir output/m3_e5_results

  # List all run configs and their status
  python experiments/m3_e5_benchmark/run_m3_e5.py --list \\
      --output-dir output/m3_e5_results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

LLM_IDS = ["llm_0.8b", "llm_2b", "llm_4b", "llm_9b"]
EMB_IDS = ["emb_0.6b", "emb_4b"]
MODES_NO_EMB = ["pure_llm"]
MODES_WITH_EMB = ["rag_small_emb", "rag_large_emb", "graph_small_emb", "graph_large_emb"]

# Canonical embedding used by each retrieval mode
MODE_EMB_HINT = {
    "rag_small_emb": "emb_0.6b",
    "rag_large_emb": "emb_4b",
    "graph_small_emb": "emb_0.6b",
    "graph_large_emb": "emb_4b",
}


def all_run_configs() -> List[Dict]:
    """Return all 36 canonical run configurations."""
    configs = []
    for llm in LLM_IDS:
        # pure_llm — no embedding
        configs.append({"llm_id": llm, "mode": "pure_llm", "emb_id": None})
        # retrieval modes — both embedding sizes
        for mode in MODES_WITH_EMB:
            for emb in EMB_IDS:
                configs.append({"llm_id": llm, "mode": mode, "emb_id": emb})
    return configs


def run_id_for(llm_id: str, mode: str, emb_id: Optional[str]) -> str:
    if emb_id:
        return f"{llm_id}__{mode}__{emb_id}"
    return f"{llm_id}__{mode}"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_eval_set(path: Path) -> List[Dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_summary(run_dir: Path) -> Optional[Dict]:
    p = run_dir / "summary.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------------------
# Scoring utilities (shared with the notebook pipeline)
# ---------------------------------------------------------------------------

def score_exact(prediction: str, gold: str) -> float:
    return 1.0 if prediction.strip().lower() == gold.strip().lower() else 0.0


def score_contains(prediction: str, gold: str) -> float:
    return 1.0 if gold.strip().lower() in prediction.strip().lower() else 0.0


def compute_metrics(predictions: List[Dict]) -> Dict:
    """Compute aggregate metrics from a predictions list."""
    if not predictions:
        return {}

    exact_scores = [p.get("scores", {}).get("exact", 0.0) for p in predictions]
    contains_scores = [p.get("scores", {}).get("contains", 0.0) for p in predictions]
    latencies = [p.get("latency_sec", 0.0) for p in predictions if p.get("latency_sec") is not None]

    by_difficulty: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "exact": [], "contains": []})
    by_domain: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "exact": [], "contains": []})

    for p, e, c in zip(predictions, exact_scores, contains_scores):
        diff = p.get("difficulty", "unknown")
        domain = p.get("domain", "unknown")
        by_difficulty[diff]["n"] += 1
        by_difficulty[diff]["exact"].append(e)
        by_difficulty[diff]["contains"].append(c)
        by_domain[domain]["n"] += 1
        by_domain[domain]["exact"].append(e)
        by_domain[domain]["contains"].append(c)

    def agg(d: Dict) -> Dict:
        return {k: {"n": v["n"], "exact": _mean(v["exact"]), "contains": _mean(v["contains"])} for k, v in d.items()}

    return {
        "n": len(predictions),
        "exact": _mean(exact_scores),
        "contains": _mean(contains_scores),
        "latency_sec_mean": _mean(latencies) if latencies else 0.0,
        "by_difficulty": agg(by_difficulty),
        "by_domain": agg(by_domain),
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Pipeline stub — replace with real pipeline imports
# ---------------------------------------------------------------------------

def run_pipeline(
    question: Dict,
    llm_id: str,
    mode: str,
    emb_id: Optional[str],
    graph_dir: Optional[Path],
) -> Tuple[str, float, Optional[str], Optional[float]]:
    """Run the QA pipeline for a single question.

    Returns:
        (prediction, latency_sec, graph_answer_text, graph_confidence)

    This function is a stub. Replace the body with real pipeline calls that:
    - Load or reference the LLM/embedding services
    - Perform retrieval (RAG or graph-augmented) based on `mode`
    - Return the raw prediction string

    The stub returns empty predictions so the harness can be run without
    services to validate the matrix/scoring/output logic.
    """
    t0 = time.perf_counter()

    # ---------- REPLACE BELOW WITH REAL PIPELINE ----------
    try:
        from services.llm_server import query_llm  # type: ignore
        from services.embeddings_server import embed  # type: ignore  # noqa: F401
    except ImportError:
        query_llm = None  # type: ignore

    prediction = ""
    graph_text = None
    graph_conf = None

    if query_llm is None:
        # Stub: services not available
        prediction = ""
    elif mode == "pure_llm":
        prediction = query_llm(question["question"])
    elif mode.startswith("rag_"):
        prediction = query_llm(question["question"])
    elif mode.startswith("graph_"):
        prediction = query_llm(question["question"])
    # ---------- REPLACE ABOVE WITH REAL PIPELINE ----------

    latency = time.perf_counter() - t0
    return prediction, latency, graph_text, graph_conf


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------

def run_single(
    cfg: Dict,
    eval_set: List[Dict],
    output_dir: Path,
    graph_dir: Optional[Path],
    resume: bool = True,
) -> Dict:
    """Execute one run configuration, write results, return summary dict."""
    llm_id = cfg["llm_id"]
    mode = cfg["mode"]
    emb_id = cfg.get("emb_id")
    rid = run_id_for(llm_id, mode, emb_id)
    run_dir = output_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip if already completed
    existing = load_summary(run_dir)
    if resume and existing and existing.get("status") == "completed":
        print(f"[SKIP] {rid} — already completed")
        return existing

    print(f"[RUN ] {rid}  ({len(eval_set)} questions)")
    predictions = []
    debug_log = []

    for i, item in enumerate(eval_set):
        question_text = item.get("question", item.get("query", ""))
        gold = item.get("gold_answer", item.get("answer", ""))

        pred, latency, graph_text, graph_conf = run_pipeline(
            item, llm_id, mode, emb_id, graph_dir
        )

        row = {
            "idx": i + 1,
            "question": question_text,
            "gold_answer": gold,
            "difficulty": item.get("difficulty", "unknown"),
            "domain": item.get("domain", item.get("category_type", "unknown")),
            "system": rid,
            "prediction": pred,
            "scores": {
                "exact": score_exact(pred, gold),
                "contains": score_contains(pred, gold),
            },
            "latency_sec": latency,
            "error": None,
        }
        if graph_text is not None:
            row["graph_answer_text"] = graph_text
        if graph_conf is not None:
            row["graph_confidence"] = graph_conf

        predictions.append(row)
        debug_log.append({"idx": i + 1, "question": question_text, "prediction": pred, "gold": gold})

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(eval_set)} done …")

    write_jsonl(run_dir / "predictions.jsonl", predictions)
    write_jsonl(run_dir / "debug_log.jsonl", debug_log)

    metrics = compute_metrics(predictions)
    summary = {
        "run_id": rid,
        "llm_id": llm_id,
        "emb_id": emb_id,
        "mode": mode,
        "dataset": str(eval_set[0].get("_source", "temporal_evaluation_set_v2.jsonl")),
        "n_questions": len(eval_set),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "metrics": metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  exact={metrics['exact']:.3f}  contains={metrics['contains']:.3f}")
    return summary


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(output_dir: Path) -> None:
    """Collect all completed run summaries into MATRIX.json."""
    matrix = {}
    for run_dir in sorted(output_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        s = load_summary(run_dir)
        if s:
            matrix[s["run_id"]] = s
    out = output_dir / "MATRIX.json"
    out.write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}  ({len(matrix)} runs)")
    # Print summary table
    print(f"\n{'Run ID':<45} {'exact':>6} {'contains':>8} {'n':>5}  status")
    print("-" * 70)
    for rid, s in sorted(matrix.items()):
        m = s.get("metrics", {})
        status = s.get("status", "?")
        exact = m.get("exact")
        contains = m.get("contains")
        n = m.get("n") or 0
        exact_s = f"{exact:6.3f}" if exact is not None else "   N/A"
        contains_s = f"{contains:8.3f}" if contains is not None else "     N/A"
        print(f"{rid:<45} {exact_s} {contains_s} {n:>5}  {status}")


# ---------------------------------------------------------------------------
# List / status
# ---------------------------------------------------------------------------

def list_configs(output_dir: Path) -> None:
    configs = all_run_configs()
    print(f"\n{'Run ID':<45} {'status':<12}")
    print("-" * 60)
    for cfg in configs:
        rid = run_id_for(cfg["llm_id"], cfg["mode"], cfg.get("emb_id"))
        run_dir = output_dir / rid
        s = load_summary(run_dir)
        status = s.get("status", "not_started") if s else "not_started"
        print(f"{rid:<45} {status:<12}")
    print(f"\nTotal: {len(configs)} configurations")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="M3-E5 QA benchmark runner", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--eval-set", default="data/jsonls/temporal_evaluation_set_v2.jsonl")
    p.add_argument("--graph-dir", default="data/jsonls/temporal_graph_output_v3")
    p.add_argument("--output-dir", default="output/m3_e5_results")

    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--run-all", action="store_true", help="Run all 36 configurations")
    mode_group.add_argument("--aggregate", action="store_true", help="Aggregate completed runs into MATRIX.json")
    mode_group.add_argument("--list", action="store_true", help="List all configs and status")

    p.add_argument("--llm-id", choices=LLM_IDS, help="LLM model ID for single run")
    p.add_argument("--mode", choices=MODES_NO_EMB + MODES_WITH_EMB, help="Pipeline mode for single run")
    p.add_argument("--emb-id", choices=EMB_IDS, help="Embedding model ID (required for non-pure_llm modes)")
    p.add_argument("--no-resume", action="store_true", help="Re-run even if already completed")
    return p


def main() -> int:
    args = build_parser().parse_args()

    root = Path(__file__).parent.parent.parent
    eval_path = root / args.eval_set
    graph_path = root / args.graph_dir if args.graph_dir else None
    output_path = root / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_configs(output_path)
        return 0

    if args.aggregate:
        aggregate(output_path)
        return 0

    eval_set = load_eval_set(eval_path)
    print(f"Eval set: {eval_path} ({len(eval_set)} questions)")

    if args.run_all:
        configs = all_run_configs()
        print(f"Running {len(configs)} configurations …\n")
        for cfg in configs:
            try:
                run_single(cfg, eval_set, output_path, graph_path, resume=not args.no_resume)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR {run_id_for(**cfg)}: {exc}", file=sys.stderr)
        aggregate(output_path)
        return 0

    # Single run
    if not args.llm_id or not args.mode:
        print("ERROR: --llm-id and --mode required for a single run (or use --run-all)", file=sys.stderr)
        return 1

    if args.mode != "pure_llm" and not args.emb_id:
        print(f"ERROR: --emb-id required for mode '{args.mode}'", file=sys.stderr)
        return 1

    cfg = {"llm_id": args.llm_id, "mode": args.mode, "emb_id": args.emb_id}
    run_single(cfg, eval_set, output_path, graph_path, resume=not args.no_resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())


