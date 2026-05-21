#!/usr/bin/env python3
"""Run all Milestone 2 performance benchmarks.

Benchmarks:
- Intent classifier inference throughput (TF-IDF + LR)
- TMS trace recorder overhead (trace_bench replicated)
- Parser hybrid inference throughput
"""
from __future__ import annotations
import importlib.util
import json
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "benchmarks" / "milestone2"
ROOT = Path(__file__).parent.parent.parent


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_module_from_path(module_name: str, path: Path):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bench_intent_classifier_inference() -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    dataset_path = ROOT / "experiments" / "m2_e2_intent" / "data" / "annotated_queries.jsonl"
    if not dataset_path.exists():
        return {"benchmark": "intent_classifier_inference", "skipped": True, "reason": "intent dataset not found"}

    rows = _load_jsonl(dataset_path)
    queries = [str(row.get("query") or "").strip() for row in rows if row.get("query")]
    intents = [row.get("intents") or [] for row in rows if row.get("query")]
    if not queries or not intents:
        return {"benchmark": "intent_classifier_inference", "skipped": True, "reason": "empty dataset"}

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(queries)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(intents)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=400))
    clf.fit(X, y)

    # Reuse transformed vectors for pure inference throughput timing.
    N = min(max(X.shape[0] * 20, 500), 5000)
    t0 = time.perf_counter()
    for _ in range(N):
        _ = clf.predict_proba(X)
    elapsed = time.perf_counter() - t0
    return {
        "benchmark": "intent_classifier_inference",
        "labels": list(mlb.classes_),
        "n": N,
        "total_sec": elapsed,
        "per_item_ms": elapsed / N * 1000,
    }


def bench_tms_trace() -> dict:
    from temporal_nlg.tms.trace import TraceRecorder

    recorder = TraceRecorder()
    N = 500
    t0 = time.perf_counter()
    for i in range(N):
        with recorder.session(f"q{i}", meta={"idx": i}) as trace:
            recorder.record_rule_firing(
                trace,
                rule_id=f"rule_{i % 3}",
                rule_name=f"rule_{i % 3}",
                inputs=[{"fact_id": f"f{i}"}],
                conclusion={"fact_id": f"g{i}"},
                confidence=1.0,
                latency_ms=0.1,
            )
    elapsed = time.perf_counter() - t0
    return {"benchmark": "tms_trace_recorder", "n": N, "total_sec": elapsed, "per_item_ms": elapsed / N * 1000}


def bench_parser_hybrid_inference() -> dict:
    parser_script = ROOT / "experiments" / "m2_e3_parse" / "run_parse.py"
    data_path = ROOT / "experiments" / "m2_e3_parse" / "data" / "temporal_queries_gold.jsonl"

    if not parser_script.exists():
        return {"benchmark": "parser_hybrid_inference", "skipped": True, "reason": "run_parse.py not found"}
    if not data_path.exists():
        return {"benchmark": "parser_hybrid_inference", "skipped": True, "reason": "gold parse dataset not found"}

    module = _load_module_from_path("m2_e3_run_parse", parser_script)
    parse_row_rules = getattr(module, "parse_row_rules", None)
    if parse_row_rules is None:
        return {"benchmark": "parser_hybrid_inference", "skipped": True, "reason": "parse_row_rules missing"}

    rows = _load_jsonl(data_path)
    if not rows:
        return {"benchmark": "parser_hybrid_inference", "skipped": True, "reason": "empty parser dataset"}

    N = min(max(len(rows) * 10, 500), 5000)
    t0 = time.perf_counter()
    for i in range(N):
        _ = parse_row_rules(rows[i % len(rows)])
    elapsed = time.perf_counter() - t0
    return {
        "benchmark": "parser_hybrid_inference",
        "n": N,
        "total_sec": elapsed,
        "per_item_ms": elapsed / N * 1000,
    }


def main() -> None:
    from datetime import datetime, timezone

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for fn in [bench_intent_classifier_inference, bench_tms_trace, bench_parser_hybrid_inference]:
        try:
            r = fn()
            if r.get("skipped"):
                print(f"  {r['benchmark']:35s}  SKIPPED ({r.get('reason', '')})")
            else:
                print(f"  {r['benchmark']:35s}  {r['per_item_ms']:.3f} ms/item")
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            print(f"  {fn.__name__}: ERROR — {exc}")

    out = OUTPUT_DIR / f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
