"""Context-aware wrapper for temporal normalization."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from .normalizer import TemporalNormalizer
from .schemas import DocumentContext, NormalizedTemporal, TemporalExpression


class ContextAwareResolver:
    """Resolve temporal expressions using document/dialog context."""

    def __init__(self, normalizer: Optional[TemporalNormalizer] = None):
        self.normalizer = normalizer or TemporalNormalizer()

    def resolve_all(
        self, expressions: List[TemporalExpression], context: Optional[DocumentContext] = None
    ) -> List[NormalizedTemporal]:
        resolved: List[NormalizedTemporal] = []
        active_context = context or DocumentContext(reference_time=datetime.utcnow())
        for expr in expressions:
            normalized = self.normalizer.normalize(expr, context=active_context)
            active_context.prior_expressions.append(normalized)
            resolved.append(normalized)
        return resolved
