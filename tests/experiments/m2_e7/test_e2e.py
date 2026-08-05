import json
import runpy
import subprocess
import sys
from pathlib import Path

RUN_E2E_PATH = Path("experiments/m2_e7_harness/run_e2e.py")


def load_run_e2e_module():
    return runpy.run_path(RUN_E2E_PATH)


def test_build_results_from_traces_extracts_intent():
    mod = load_run_e2e_module()
    build_results_from_traces = mod["build_results_from_traces"]

    traces = [
        {
            "query_id": "q1",
            "rule_traces": [
                {"conclusion": {"fact_id": "intent", "value": "medical"}},
            ],
        },
        {
            "query_id": "q2",
            "rule_traces": [],
        },
    ]

    results = build_results_from_traces(traces)

    assert results[0]["intent"] == "medical"
    assert results[1]["intent"] is None
    assert results[0]["rule_traces"]


def test_eval_results_counts_failures():
    mod = load_run_e2e_module()
    eval_results = mod["eval_results"]

    results = [
        {"query_id": "q1", "intent": "medical", "rule_traces": []},
        {"query_id": "q2", "intent": "science", "rule_traces": []},
    ]
    expected = {
        "q1": {"intent": "medical", "required_facts": [], "max_latency_ms": None},
        "q3": {"intent": "science", "required_facts": [], "max_latency_ms": None},
    }

    ok, total, failures = eval_results(results, expected)

    assert ok == 1
    assert total == 2
    assert len(failures) == 1
    assert failures[0]["query_id"] == "q2"
    assert "missing expected entry" in failures[0]["reason"]


def test_eval_results_checks_facts_and_latency():
    mod = load_run_e2e_module()
    eval_results = mod["eval_results"]

    results = [
        {
            "query_id": "q10",
            "intent": "medical",
            "rule_traces": [
                {"conclusion": {"fact_id": "diagnosis", "value": "flu"}, "latency_ms": 5.0},
            ],
        },
        {
            "query_id": "q11",
            "intent": "science",
            "rule_traces": [
                {
                    "conclusion": {"fact_id": "experiment", "value": "double_slit"},
                    "latency_ms": 15.0,
                },
            ],
        },
    ]

    expected = {
        "q10": {"intent": "medical", "required_facts": ["diagnosis"], "max_latency_ms": 10.0},
        "q11": {"intent": "science", "required_facts": ["experiment"], "max_latency_ms": 10.0},
    }

    ok, total, failures = eval_results(results, expected)

    assert ok == 1
    assert total == 2
    assert len(failures) == 1
    assert failures[0]["query_id"] == "q11"
    assert "latency_ms" in failures[0]["reason"]


def test_run_e2e_cli_outputs_report(tmp_path):
    mod = load_run_e2e_module()

    queries_path = tmp_path / "queries.jsonl"
    traces_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "results.jsonl"
    report_path = tmp_path / "report.json"

    queries = [
        {"query_id": "q10", "text": "medical q", "intent": "medical"},
        {"query_id": "q11", "text": "science q", "intent": "science"},
    ]
    traces = [
        {
            "query_id": "q10",
            "rule_traces": [
                {"conclusion": {"fact_id": "intent", "value": "medical"}},
            ],
        },
        {
            "query_id": "q11",
            "rule_traces": [
                {"conclusion": {"fact_id": "intent", "value": "science"}},
            ],
        },
    ]

    with queries_path.open("w", encoding="utf-8") as f:
        for row in queries:
            f.write(json.dumps(row) + "\n")

    with traces_path.open("w", encoding="utf-8") as f:
        for row in traces:
            f.write(json.dumps(row) + "\n")

    cmd = [
        sys.executable,
        str(RUN_E2E_PATH),
        "--queries",
        str(queries_path),
        "--trace",
        str(traces_path),
        "--output",
        str(output_path),
        "--report",
        str(report_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert proc.returncode == 0

    report = json.loads(report_path.read_text())
    assert report["ok"] == 2
    assert report["fail"] == 0
    assert output_path.exists()
