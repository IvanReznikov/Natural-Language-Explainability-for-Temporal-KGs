#!/usr/bin/env python3
"""
Milestone 1: Changelog Example

Turns milestone updates into narrative changelog entries using sequence and
causality templates.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg.core.templates import TemplateRenderer, TemplateType, TemporalFact


def _print_header(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def changelog_entries() -> None:
    """Render a few structured changelog entries."""
    renderer = TemplateRenderer()

    entries = [
        TemporalFact(
            TemplateType.SEQUENCE,
            entity="Temporal NLG",
            events=["baseline templates", "LLM hybrid", "evaluation tooling"],
            timestamps=["v0.1.0", "v0.2.0", "v0.3.0"],
            context="feature rollout"
        ),
        TemporalFact(
            TemplateType.CAUSALITY,
            cause="coverage reaching 80%",
            effect="confidence to ship milestone 1",
            context="quality gate"
        ),
    ]

    _print_header("Rendered Changelog")
    for entry in entries:
        print(f"- {renderer.render(entry)}")


def main() -> None:
    _print_header("Milestone 1 - Changelog Example")
    changelog_entries()


if __name__ == "__main__":
    main()
