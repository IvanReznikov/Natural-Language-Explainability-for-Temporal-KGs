import json

from temporal_nlg.graph_query import TemporalGraphLCELPipeline


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_lcel_pipeline_reason_query(tmp_path) -> None:
    out_dir = tmp_path / "graph"
    out_dir.mkdir(parents=True)

    _write_jsonl(
        out_dir / "nodes.jsonl",
        [
            {"node_uid": "n1", "label": "Cause A", "category": "event"},
            {"node_uid": "n2", "label": "Effect X", "category": "event"},
            {"node_uid": "n3", "label": "Alt C", "category": "event"},
        ],
    )
    _write_jsonl(
        out_dir / "edges.jsonl",
        [
            {
                "edge_uid": "e1",
                "source_uid": "n1",
                "target_uid": "n2",
                "relation": "caused",
                "start": "2001",
                "end": None,
                "edge_type": "base",
                "support_count": 2,
                "source_row_ids": ["r1"],
            },
            {
                "edge_uid": "e2",
                "source_uid": "n1",
                "target_uid": "n3",
                "relation": "caused",
                "start": "2002",
                "end": None,
                "edge_type": "base",
                "support_count": 1,
                "source_row_ids": ["r2"],
            },
        ],
    )
    _write_jsonl(out_dir / "tags.jsonl", [])
    _write_jsonl(out_dir / "processed_graph.jsonl", [])

    pipeline = TemporalGraphLCELPipeline(out_dir)
    result = pipeline.invoke("What likely caused Effect X around 2001?")

    assert "reason" in result["intent"] or "cause" in result["answer_text"].lower()
    assert result["confidence"] > 0.0
    assert "plan" in result
    assert result["plan"]["query_type"] in {"reason_of", "state_at_time", "unsupported"}
    assert result["evidence"]
    assert result["mermaid"].startswith("graph TD")


def test_lcel_pipeline_analogy_and_start_affecting(tmp_path) -> None:
    out_dir = tmp_path / "graph"
    out_dir.mkdir(parents=True)

    _write_jsonl(
        out_dir / "nodes.jsonl",
        [
            {"node_uid": "a", "label": "A", "category": "event"},
            {"node_uid": "b", "label": "B", "category": "event"},
            {"node_uid": "c", "label": "C", "category": "event"},
        ],
    )
    _write_jsonl(
        out_dir / "edges.jsonl",
        [
            {
                "edge_uid": "e_ab",
                "source_uid": "a",
                "target_uid": "b",
                "relation": "caused",
                "start": "1999",
                "end": None,
                "edge_type": "base",
                "support_count": 1,
                "source_row_ids": ["r1"],
            },
            {
                "edge_uid": "e_ac",
                "source_uid": "a",
                "target_uid": "c",
                "relation": "caused",
                "start": "2005",
                "end": None,
                "edge_type": "base",
                "support_count": 1,
                "source_row_ids": ["r2"],
            },
        ],
    )
    _write_jsonl(out_dir / "tags.jsonl", [])
    _write_jsonl(out_dir / "processed_graph.jsonl", [])

    pipeline = TemporalGraphLCELPipeline(out_dir)

    analog = pipeline.invoke("If A caused B in 1999, could A also cause C?")
    assert analog["intent"] in {"analogical_transfer", "state_at_time", "unsupported"}
    assert "plan" in analog

    start = pipeline.invoke("When did A first start affecting B?")
    assert start["intent"] in {"start_affecting", "reason_of", "state_at_time", "unsupported"}
    assert "plan" in start
    assert start["plan"]["query_type"] in {
        "start_affecting",
        "reason_of",
        "state_at_time",
        "unsupported",
    }
    assert start["evidence"]
    assert (
        "1999" in start["answer_text"]
        or "A caused B" in start["answer_text"]
        or "A caused B" in " ".join(start["evidence"])
    )
