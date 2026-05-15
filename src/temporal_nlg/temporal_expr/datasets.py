"""Dataset helpers for temporal expression evaluation (M2-E1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class TemporalDatasetExample:
    """Single example containing text and gold annotations."""

    text: str
    gold_spans: List[dict]
    gold_normalized: List[str]
    reference_time: Optional[str] = None


def load_jsonl_temporal_dataset(path: str | Path) -> List[TemporalDatasetExample]:
    """Load a JSONL dataset with required fields: text, gold_spans, gold_normalized."""

    dataset: List[TemporalDatasetExample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            dataset.append(
                TemporalDatasetExample(
                    text=obj["text"],
                    gold_spans=obj["gold_spans"],
                    gold_normalized=obj["gold_normalized"],
                    reference_time=obj.get("reference_time"),
                )
            )
    return dataset
