#!/usr/bin/env python3
"""
Milestone 1: Hybrid Generation Example

Demonstrates routing between template, polished, and LLM strategies using
HybridGenerator with an offline-safe stub LLM.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from temporal_nlg.core.templates import TemplateRenderer, TemplateType, TemporalFact
from temporal_nlg.models.hybrid_generator import HybridGenerator, GenerationResult


class _StubLLMGenerator:
    """Offline-safe stand-in that avoids real LLM calls."""

    def generate(self, fact):
        return f"[stub-llm] Generated text for {getattr(fact, 'event', 'fact')}"


def _print_header(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def demo_hybrid_strategies() -> dict:
    renderer = TemplateRenderer()
    generator = HybridGenerator(enable_caching=True, template_renderer=renderer)
    generator.llm_generator = _StubLLMGenerator()

    fact_template = TemporalFact(
        TemplateType.POINT_IN_TIME,
        event="launch",
        entity="Mission Orion",
        date="2025-03-21",
    )
    fact_polish = TemporalFact(
        TemplateType.SEQUENCE,
        events=["design", "integration", "launch"],
        timestamps=["T-12m", "T-3m", "T0"],
        context="Extended context for polish"
    )
    fact_llm = TemporalFact(
        TemplateType.CAUSALITY,
        cause="Solar flare",
        effect="Communications blackout",
        temporal_relation="triggered",
        context="A complex causal chain with multiple signals and overrides"
    )

    results = {
        "template": generator.generate(fact_template, force_strategy="template"),
        "polish": generator.generate(fact_polish, force_strategy="polish"),
        "llm": generator.generate(fact_llm, force_strategy="llm"),
    }
    return results


def main() -> None:
    _print_header("Milestone 1 - Hybrid Generation Example")
    results = demo_hybrid_strategies()

    for name, result in results.items():
        assert isinstance(result, GenerationResult)
        print(f"[{name.upper()}] {result.text} (strategy={result.strategy}, conf={result.confidence:.2f})")


if __name__ == "__main__":
    main()
