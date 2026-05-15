import json
import importlib

_module_name = "".join(["temporal", "_", "graph", "_", "processing", ".process_temporal_graph"])
_graph_module = importlib.import_module(_module_name)
_parse_temporal = _graph_module._parse_temporal
_remove_lines_by_number_desc = _graph_module._remove_lines_by_number_desc
_sample_line_numbers = _graph_module._sample_line_numbers
process_temporal_graph = _graph_module.process_temporal_graph


def test_sample_line_numbers_spread() -> None:
    lines = _sample_line_numbers(total_rows=100, sample_size=7)
    assert len(lines) == 7
    assert 1 in lines
    assert 100 in lines
    assert 50 in lines
    assert lines == sorted(lines)


def test_parse_temporal_accepts_year_month() -> None:
    assert _parse_temporal("1975-01") == "1975-01-01"
    assert _parse_temporal("1975") == "1975"
    assert _parse_temporal("1975-01-22") == "1975-01-22"


def test_parse_temporal_extended_formats() -> None:
    assert _parse_temporal("late 2010s") == "2018-01-01"
    assert _parse_temporal("2022-Q4") == "2022-10-01"
    assert _parse_temporal("2025-Q2") == "2025-04-01"
    assert _parse_temporal("2026-H2") == "2026-07-01"
    assert _parse_temporal("recent years") == "PRESENT"
    assert _parse_temporal("present") == "PRESENT"
    assert _parse_temporal("19th century") == "1801-01-01"
    assert _parse_temporal("20th century") == "1901-01-01"
    assert _parse_temporal("18050") == "18050"
    assert _parse_temporal("-252000000") == "-252000000"
    assert _parse_temporal("-1479-01-01") == "-1479-01-01"
    assert _parse_temporal("690-10-16") == "690-10-16"
    assert _parse_temporal("-0539") == "-539"
    assert _parse_temporal("1993-02-26T12:18:00") == "1993-02-26"


def test_remove_lines_descending(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    target = tmp_path / "target.jsonl"
    source.write_text("a\n" "b\n" "c\n" "d\n", encoding="utf-8")

    removed = _remove_lines_by_number_desc(source, target, [2, 4])
    assert removed == 2
    assert target.read_text(encoding="utf-8") == "a\n" "c\n"


def test_process_temporal_graph_outputs(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"

    rows = [
        {
            "id": "r1",
            "query": "q1",
            "domain": "historical",
            "intent": "causal",
            "time_scope": "causal",
            "gold_facts": [{"fact_id": "f1", "subject": "A", "relation": "caused", "object": "B"}],
            "graph_nodes": [
                {"node_id": "n1", "label": "A", "category": "event"},
                {"node_id": "n2", "label": "B", "category": "event"},
            ],
            "graph_edges": [
                {
                    "edge_id": "e1",
                    "source": "n1",
                    "target": "n2",
                    "relation": "caused",
                    "start": "1913",
                    "end": None,
                    "weight": 1.0,
                }
            ],
        },
        {
            "id": "r2",
            "query": "q2",
            "gold_facts": [],
            "graph_nodes": [{"node_id": "n1", "label": "X", "category": "event"}],
            "graph_edges": [
                {
                    "source": "n1",
                    "target": "n_missing",
                    "relation": "linked_to",
                    "start": "2020-01-01",
                    "end": "2010-01-01",
                    "weight": -5,
                }
            ],
        },
    ]

    with input_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    report = process_temporal_graph(input_jsonl=input_path, output_dir=output_dir, sample_size=2)

    assert report["rows_total"] == 2
    assert report["rows_sampled"] == 2
    assert report["cleaned_rows"] == 2
    assert report["kg_nodes"] >= 2
    assert report["kg_edges"] >= 1
    assert report["issues_total"] >= 1
    assert "edge_target_not_found" in report["issues_by_code"]

    cleaned_path = output_dir / "temporal_graph_cleaned.jsonl"
    issues_path = output_dir / "quality_issues.jsonl"
    nodes_path = output_dir / "temporal_kg_nodes.jsonl"
    edges_path = output_dir / "temporal_kg_edges.jsonl"
    report_path = output_dir / "quality_report.json"

    assert cleaned_path.exists()
    assert issues_path.exists()
    assert nodes_path.exists()
    assert edges_path.exists()
    assert report_path.exists()
