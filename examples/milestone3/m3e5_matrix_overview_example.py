#!/usr/bin/env python3
"""Milestone 3 E5 matrix overview using temporal_nlg graph package primitives."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg import TemporalGraphIndex


def _graph_dir(root: Path) -> Path:
    for path in (
        root / "data" / "jsonls" / "temporal_graph_output_v3",
        root / "data" / "jsonls" / "temporal_graph_output",
    ):
        if (path / "nodes.jsonl").exists() and (path / "edges.jsonl").exists():
            return path
    raise FileNotFoundError("No graph output directory found for TemporalGraphIndex.")


def _top_runs(matrix_rows: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    return sorted(matrix_rows, key=lambda row: float(row.get("exact", 0.0)), reverse=True)[:top_k]


def main() -> None:
    matrix_path = ROOT / "output" / "m3_e5_results" / "MATRIX.json"
    if not matrix_path.exists():
        raise FileNotFoundError(f"MATRIX.json not found: {matrix_path}")

    graph_index = TemporalGraphIndex(_graph_dir(ROOT))
    matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix_rows: List[Dict[str, Any]] = list(matrix_payload.get("rows", []))

    print("M3-E5 Matrix Overview")
    print("matrix_rows:", len(matrix_rows))
    print("graph_nodes:", len(graph_index.node_label_by_uid))
    print("graph_edges:", len(graph_index.edge_by_uid))

    print("\nTop runs by exact:")
    for row in _top_runs(matrix_rows, top_k=5):
        run_id = row.get("run_id") or row.get("id") or "unknown"
        exact = float(row.get("exact", 0.0))
        mode = row.get("mode", "n/a")
        llm_id = row.get("llm_id", "n/a")
        print(f"  - {run_id}: exact={exact:.3f}, llm={llm_id}, mode={mode}")


if __name__ == "__main__":
    main()
