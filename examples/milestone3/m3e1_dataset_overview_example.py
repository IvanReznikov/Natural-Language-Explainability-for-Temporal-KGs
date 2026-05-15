#!/usr/bin/env python3
"""Milestone 3 E1 dataset overview using temporal_nlg graph index."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg import TemporalGraphIndex


def _graph_dir(root: Path) -> Path:
    candidates = [
        root / "data" / "jsonls" / "temporal_graph_output_v3",
        root / "data" / "jsonls" / "temporal_graph_output",
    ]
    for candidate in candidates:
        if (candidate / "nodes.jsonl").exists() and (candidate / "edges.jsonl").exists():
            return candidate
    raise FileNotFoundError("No graph artifact directory found under data/jsonls.")


def main() -> None:
    graph_dir = _graph_dir(ROOT)
    index = TemporalGraphIndex(graph_dir)

    node_count = len(index.node_label_by_uid)
    edge_count = len(index.edge_by_uid)
    year_count = len(index.all_years)

    category_counts = Counter(index.node_category_by_uid.values())

    print("M3-E1 Dataset Overview")
    print("graph_dir:", graph_dir)
    print("nodes:", node_count)
    print("edges:", edge_count)
    print("unique_years:", year_count)
    print("top_categories:")
    for category, count in category_counts.most_common(10):
        print(f"  - {category}: {count}")


if __name__ == "__main__":
    main()
