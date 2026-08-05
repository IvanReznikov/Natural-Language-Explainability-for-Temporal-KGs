#!/usr/bin/env python3
"""
Milestone 1: Delta Encoding Example

Simulates capturing deltas between a baseline timeline and new events using
built-in template rendering. Outputs human-friendly sentences for the changes.
"""

import sys
from pathlib import Path

# Ensure src/ is on the path for local execution
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg.core.templates import TemplateRenderer, TemplateType, TemporalFact
from temporal_nlg.data.loaders import generate_examples


def _print_header(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def delta_encoding_demo() -> None:
    """Render only the new facts (the delta) between two snapshots."""
    renderer = TemplateRenderer()

    baseline = [
        TemporalFact(
            TemplateType.POINT_IN_TIME, event="was born", entity="Ada Lovelace", date="1815-12-10"
        ),
        TemporalFact(
            TemplateType.INTERVAL,
            entity="First Industrial Revolution",
            event="industrialization phase",
            start_date="1760",
            end_date="1840",
        ),
    ]

    updates = baseline + [
        TemporalFact(
            TemplateType.SEQUENCE,
            entity="Apollo Program",
            events=["concept", "landing"],
            timestamps=["1961", "1969"],
            context="NASA",
        ),
        TemporalFact(
            TemplateType.CAUSALITY,
            cause="steam engine adoption",
            effect="mass production boom",
            context="Industrial impact",
        ),
    ]

    existing_entities = {fact.content.get("entity") for fact in baseline}
    delta = [fact for fact in updates if fact.content.get("entity") not in existing_entities]

    _print_header("Baseline Snapshot")
    for fact in baseline:
        print(f"- {renderer.render(fact)}")

    _print_header("Incoming Snapshot")
    for fact in updates:
        print(f"- {renderer.render(fact)}")

    _print_header("Delta (New Facts Only)")
    for fact in delta:
        print(f"- {renderer.render(fact)}")


def batch_delta_generation() -> None:
    """Generate a small batch of deltas from synthetic data."""
    renderer = TemplateRenderer()
    generated = generate_examples(TemplateType.POINT_IN_TIME, n=3)

    _print_header("Synthetic Delta Batch")
    for idx, fact in enumerate(generated, 1):
        print(f"{idx}. {renderer.render(fact)}")


def main() -> None:
    _print_header("Milestone 1 - Delta Encoding Example")
    delta_encoding_demo()
    batch_delta_generation()


if __name__ == "__main__":
    main()
