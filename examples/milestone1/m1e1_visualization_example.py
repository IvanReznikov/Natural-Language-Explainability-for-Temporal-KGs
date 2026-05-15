#!/usr/bin/env python3
"""
Milestone 1: Visualization Example

Creates text-based timelines for interval and overlap facts to illustrate
how rendered explanations map to simple visual markers.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg.core.templates import TemplateRenderer, TemplateType, TemporalFact


def _print_header(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def _timeline(label: str, start: str, end: str) -> None:
    span = f"{start} --- {end}" if end else start
    print(f"{label:18} | {span}")


def visualize_timelines() -> None:
    renderer = TemplateRenderer()

    facts = [
        TemporalFact(TemplateType.INTERVAL, entity="World War II", event="global conflict", start_date="1939", end_date="1945", context="global conflict"),
        TemporalFact(TemplateType.OVERLAP, events=["Renaissance", "Age of Discovery", "Scientific Revolution"], time_period="1300-1700", context="Europe"),
    ]

    _print_header("Rendered Explanations")
    for fact in facts:
        print(f"- {renderer.render(fact)}")

    _print_header("ASCII Timelines")
    _timeline("World War II", "1939", "1945")
    _timeline("Renaissance", "1300", "1600")
    _timeline("Age of Discovery", "1400", "1700")
    _timeline("Scientific Revolution", "1543", "1687")


def main() -> None:
    _print_header("Milestone 1 - Visualization Example")
    visualize_timelines()


if __name__ == "__main__":
    main()
