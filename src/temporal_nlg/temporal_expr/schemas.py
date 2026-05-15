"""Lightweight data structures for temporal expression processing (M2-E1).

These schemas keep tagging, normalization, and context handling decoupled
while remaining simple enough for baseline implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class TemporalExpressionType(Enum):
    """TIMEX3-aligned expression categories."""

    DATE = "DATE"
    TIME = "TIME"
    DURATION = "DURATION"
    SET = "SET"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Span:
    """Character span of an expression in the source text."""

    start: int
    end: int
    text: str


@dataclass
class TemporalExpression:
    """Tagged temporal expression prior to normalization."""

    text: str
    span: Span
    expr_type: TemporalExpressionType
    value: Optional[str] = None
    granularity: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class NormalizedTemporal:
    """Normalized representation following ISO 8601 style conventions."""

    expression: TemporalExpression
    normalized: Optional[str]
    expr_type: TemporalExpressionType
    granularity: Optional[str]
    alternatives: List[Dict[str, object]] = field(default_factory=list)
    reference_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class DocumentContext:
    """Context useful for resolving relative or ambiguous expressions."""

    reference_time: Optional[datetime] = None
    prior_expressions: List[NormalizedTemporal] = field(default_factory=list)
    domain_anchor: Optional[str] = None
