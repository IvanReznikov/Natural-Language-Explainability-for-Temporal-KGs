"""Temporal expression tagging and normalization (Milestone 2, E1)."""

from .context_resolver import ContextAwareResolver
from .normalizer import TemporalNormalizer
from .schemas import (
    DocumentContext,
    NormalizedTemporal,
    Span,
    TemporalExpression,
    TemporalExpressionType,
)
from .tagger import TaggerConfig, TemporalTagger
from .datasets import TemporalDatasetExample, load_jsonl_temporal_dataset
from .evaluation import evaluate_dataset

__all__ = [
    "ContextAwareResolver",
    "TemporalDatasetExample",
    "DocumentContext",
    "evaluate_dataset",
    "NormalizedTemporal",
    "Span",
    "TaggerConfig",
    "TemporalExpression",
    "TemporalExpressionType",
    "TemporalTagger",
    "TemporalNormalizer",
    "load_jsonl_temporal_dataset",
]
