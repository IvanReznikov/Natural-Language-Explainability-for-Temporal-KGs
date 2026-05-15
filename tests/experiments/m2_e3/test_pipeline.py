from pathlib import Path

from experiments.m2_e3_construct import run_construct
from experiments.m2_e3_optimize import run_optimize
from experiments.m2_e3_parse import run_parse


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(run_construct.json.dumps(row) + "\n")


def test_pipeline_rule_only_smoke(tmp_path):
    gold_rows = [
        {
            "id": "q1",
            "text": "Did the storm cause flooding?",
            "frame": {"cause": "storm", "effect": "flooding"},
            "canonical_query": "CAUSAL(cause='storm', effect='flooding')",
        },
        {
            "id": "q2",
            "text": "Compare GDP between 2019 and 2021",
            "frame": {"metric": "gdp", "a": "2019", "b": "2021"},
            "canonical_query": "COMPARE(metric='gdp', a='2019', b='2021')",
        },
        {
            "id": "q3",
            "text": "Total sales in 2023 Q4 in Europe",
            "frame": {"metric": "sales", "period": "2023-Q4", "region": "Europe"},
            "canonical_query": "AGG(metric='sales', period='2023-Q4', region='Europe')",
        },
    ]

    gold_path = tmp_path / "gold.jsonl"
    write_jsonl(gold_path, gold_rows)

    preds = [run_parse.parse_row_rules({"id": row["id"], "text": row["text"]}) for row in gold_rows]
    preds_path = tmp_path / "preds.jsonl"
    run_parse.save_jsonl(preds_path, preds)

    outputs = []
    for pred in preds:
        templated = run_construct.build_template(pred.get("frame", {}))
        outputs.append({"id": pred["id"], "canonical_query": templated})
    constructed_path = tmp_path / "constructed.jsonl"
    run_construct.save_jsonl(constructed_path, outputs)

    optimized = []
    for out in outputs:
        optimized_query = run_optimize.rewrite(out["canonical_query"])
        optimized.append(
            {
                "id": out["id"],
                "canonical_query": out["canonical_query"],
                "optimized_query": optimized_query,
                "cost_before": run_optimize.mock_cost(out["canonical_query"]),
                "cost_after": run_optimize.mock_cost(optimized_query),
            }
        )
    optimized_path = tmp_path / "optimized.jsonl"
    run_optimize.save_jsonl(optimized_path, optimized)

    assert gold_path.exists()
    assert preds_path.exists()
    assert constructed_path.exists()
    assert optimized_path.exists()
    assert len(preds) == len(outputs) == len(optimized) == len(gold_rows)
    assert all(record.get("canonical_query") for record in outputs)
    assert all(record.get("optimized_query", "").startswith("OPTIMIZED[") for record in optimized)
