"""Lightweight TIMEX3 tagger baseline for M2-E1.

The implementation favors determinism and extensibility over breadth. It
covers common absolute dates (month name + day + year, ISO date) and a
handful of relative phrases needed to stand up the normalization pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .schemas import Span, TemporalExpression, TemporalExpressionType


@dataclass
class TaggerConfig:
    """Configuration hooks for the tagger."""

    enable_relative: bool = True


class TemporalTagger:
    """Rule-centric temporal expression tagger.

    This is intentionally simple: it establishes interfaces and a baseline
    needed to exercise normalization and evaluation for M2-E1.
    """

    _MONTH_PATTERN = (
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    )
    _WEEKDAY_PATTERN = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"

    def __init__(self, config: TaggerConfig | None = None):
        self.config = config or TaggerConfig()
        self._absolute_patterns = [
            re.compile(rf"\b{self._MONTH_PATTERN}\s+\d{{1,2}},\s*\d{{4}}", re.IGNORECASE),
            re.compile(rf"\b{self._MONTH_PATTERN}\s+\d{{1,2}}-\d{{1,2}},\s*\d{{4}}", re.IGNORECASE),
            re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
            re.compile(r"\b\d{1,2}:\d{2}\b"),
            re.compile(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", re.IGNORECASE),
            re.compile(r"\b\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}\b"),
        ]
        self._relative_patterns = [
            re.compile(r"\b(today|yesterday|tomorrow)\b", re.IGNORECASE),
            re.compile(r"\b(last|next)\s+(week|month|year)\b", re.IGNORECASE),
            re.compile(rf"\b(last|next)\s+{self._WEEKDAY_PATTERN}\b", re.IGNORECASE),
            re.compile(
                r"\b(today|tomorrow|yesterday)\s+at\s+\d{1,2}(:\d{2})?\s*(am|pm)?\b", re.IGNORECASE
            ),
            re.compile(rf"\bevery\s+{self._WEEKDAY_PATTERN}\b", re.IGNORECASE),
            re.compile(r"\b(every day|daily|weekly|monthly|yearly)\b", re.IGNORECASE),
            re.compile(r"\b(twice a (day|week|month|year))\b", re.IGNORECASE),
            re.compile(r"\baround\s+(noon|midnight)\b", re.IGNORECASE),
        ]
        self._duration_patterns = [
            re.compile(
                r"\b(?P<num1>\d+(?:\.\d+)?)\s*(?P<unit1>years?|yrs?|y|months?|mo|weeks?|w|days?|d|hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)"
                r"(?:\s*(?:and\s+)?)"
                r"(?P<num2>\d+(?:\.\d+)?)\s*(?P<unit2>hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>days?|weeks?|months?|years?|hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b",
                re.IGNORECASE,
            ),
        ]

    def tag(self, text: str) -> List[TemporalExpression]:
        """Identify temporal expressions in text."""

        expressions: List[TemporalExpression] = []
        expressions.extend(self._tag_absolute(text))
        if self.config.enable_relative:
            expressions.extend(self._tag_relative(text))
        expressions.extend(self._tag_durations(text))

        # Deduplicate overlapping spans, keeping longest span when overlaps occur
        unique: List[TemporalExpression] = []
        for expr in expressions:
            replaced = False
            for idx, existing in enumerate(unique):
                if _spans_overlap(expr.span, existing.span):
                    new_len = expr.span.end - expr.span.start
                    old_len = existing.span.end - existing.span.start
                    if new_len > old_len:
                        unique[idx] = expr
                    replaced = True
                    break
            if not replaced:
                unique.append(expr)

        return sorted(unique, key=lambda e: e.span.start)

    def _tag_absolute(self, text: str) -> List[TemporalExpression]:
        matches: List[TemporalExpression] = []
        for pattern in self._absolute_patterns:
            for m in pattern.finditer(text):
                span = Span(start=m.start(), end=m.end(), text=m.group(0))
                expr_type = (
                    TemporalExpressionType.TIME
                    if ":" in m.group(0)
                    else TemporalExpressionType.DATE
                )
                matches.append(
                    TemporalExpression(
                        text=m.group(0),
                        span=span,
                        expr_type=expr_type,
                    )
                )
        return matches

    def _tag_relative(self, text: str) -> List[TemporalExpression]:
        matches: List[TemporalExpression] = []
        for pattern in self._relative_patterns:
            for m in pattern.finditer(text):
                span = Span(start=m.start(), end=m.end(), text=m.group(0))
                raw = m.group(0)
                lowered = raw.lower()
                is_set = (
                    lowered.startswith("every")
                    or "twice a" in lowered
                    or lowered in {"daily", "weekly", "monthly", "yearly", "every day"}
                )
                expr_type = TemporalExpressionType.SET if is_set else TemporalExpressionType.DATE
                metadata = {"relative": True}
                if expr_type == TemporalExpressionType.SET:
                    metadata["recurring"] = True
                matches.append(
                    TemporalExpression(
                        text=raw,
                        span=span,
                        expr_type=expr_type,
                        metadata=metadata,
                    )
                )
        return matches

    def _tag_durations(self, text: str) -> List[TemporalExpression]:
        matches: List[TemporalExpression] = []
        for pattern in self._duration_patterns:
            for m in pattern.finditer(text):
                span = Span(start=m.start(), end=m.end(), text=m.group(0))
                matches.append(
                    TemporalExpression(
                        text=m.group(0),
                        span=span,
                        expr_type=TemporalExpressionType.DURATION,
                    )
                )
        return matches


def _spans_overlap(a: Span, b: Span) -> bool:
    return a.start < b.end and b.start < a.end
