#!/usr/bin/env python3
"""
Milestone 1: Graph Indexing Example

Builds a lightweight index of rendered temporal facts keyed by entity for
quick lookup in downstream workflows.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg.core.templates import TemplateRenderer, TemplateType, TemporalFact


def _print_header(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def build_index() -> None:
    """Create and query a simple in-memory index."""
    renderer = TemplateRenderer()

    facts = [
        TemporalFact(TemplateType.POINT_IN_TIME, event="founding", entity="Kyoto", date="794"),
        TemporalFact(TemplateType.INTERVAL, entity="Heian period", event="historical era", start_date="794", end_date="1185", context="Japan"),
        TemporalFact(TemplateType.OVERLAP, events=["Renaissance", "Age of Discovery"], time_period="1300-1700", context="Europe"),
    ]

    index = {fact.content.get("entity") or f"fact-{idx}": renderer.render(fact) for idx, fact in enumerate(facts, 1)}

    _print_header("Index Contents")
    for key, text in index.items():
        print(f"{key:20} -> {text}")

    targets = ["Kyoto", "Heian period"]
    _print_header("Lookup Results")
    for target in targets:
        print(f"{target:20} -> {index.get(target, 'not found')}")


def main() -> None:
    _print_header("Milestone 1 - Graph Indexing Example")
    build_index()


if __name__ == "__main__":
    main()
