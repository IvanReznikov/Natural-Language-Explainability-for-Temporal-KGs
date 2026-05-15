import argparse
import json
import uuid
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def rewrite(query: str) -> str:
    # Stub optimizer: add a hint marker
    return f"OPTIMIZED[{query}]"


def mock_cost(query: str) -> float:
    # Deterministic mock cost: shorter strings are cheaper
    return max(1.0, len(query) / 25.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Gold file with canonical_query")
    parser.add_argument("--pred", type=Path, default=None, help="Optional constructed outputs to optimize")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    gold_rows = {row["id"]: row for row in load_jsonl(args.data)}
    pred_rows = {row["id"]: row for row in load_jsonl(args.pred)} if args.pred else {}

    run_id = uuid.uuid4().hex
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    improved = 0
    total = 0

    for qid, gold in gold_rows.items():
        canonical = gold.get("canonical_query")
        if args.pred and qid in pred_rows:
            canonical = pred_rows[qid].get("canonical_query", canonical)
        if canonical is None:
            continue

        optimized = rewrite(canonical)
        before = gold.get("cost_before", mock_cost(canonical))
        after = mock_cost(optimized)
        total += 1
        if after < before:
            improved += 1

        outputs.append(
            {
                "id": qid,
                "canonical_query": canonical,
                "optimized_query": optimized,
                "cost_before": before,
                "cost_after": after,
            }
        )

    save_jsonl(run_dir / "optimized.jsonl", outputs)

    metrics = {
        "run_id": run_id,
        "examples": total,
        "improvement_rate": improved / total if total else 0.0,
        "notes": "Stub optimizer wraps queries; replace with real rewrites and real cost model."
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved optimized queries to {run_dir}")


if __name__ == "__main__":
    main()
