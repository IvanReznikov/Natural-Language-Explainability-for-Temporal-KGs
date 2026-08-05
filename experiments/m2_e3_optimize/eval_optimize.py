import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True, help="Gold file with cost_before")
    parser.add_argument(
        "--pred", type=Path, required=True, help="Optimized outputs with cost_after"
    )
    args = parser.parse_args()

    gold_rows = {row["id"]: row for row in load_jsonl(args.gold)}
    pred_rows = {row["id"]: row for row in load_jsonl(args.pred)}

    improved = 0
    total = 0

    for qid, gold in gold_rows.items():
        if qid not in pred_rows:
            continue
        total += 1
        before = gold.get("cost_before", 0.0)
        after = pred_rows[qid].get("cost_after", before)
        if after < before:
            improved += 1

    metrics = {
        "examples": total,
        "improvement_rate": improved / total if total else 0.0,
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
