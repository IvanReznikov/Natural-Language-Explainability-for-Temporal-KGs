"""Temporal NLG framework packaged for Milestone 1."""

from temporal_nlg.core.templates import TemplateRenderer, TemplateType, TemporalFact
from temporal_nlg.models import (
    GenerationResult,
    HybridGenerator,
    LLMGenerator,
    QwenEmbeddingModel,
    QwenLocalGenerator,
)
from temporal_nlg.evaluation import AccuracyEvaluator, calculate_flesch_score
from temporal_nlg.temporal_expr import (
    ContextAwareResolver,
    DocumentContext,
    NormalizedTemporal,
    Span,
    TaggerConfig,
    TemporalExpression,
    TemporalExpressionType,
    TemporalNormalizer,
    TemporalTagger,
)
from temporal_nlg.graph_query import (
    GraphAnswer,
    GraphRetriever,
    TemporalGraphIndex,
    TemporalGraphLCELPipeline,
    answer_to_mermaid,
)

__all__ = [
    "AccuracyEvaluator",
    "GenerationResult",
    "HybridGenerator",
    "LLMGenerator",
    "QwenLocalGenerator",
    "QwenEmbeddingModel",
    "ContextAwareResolver",
    "DocumentContext",
    "NormalizedTemporal",
    "Span",
    "TaggerConfig",
    "TemporalExpression",
    "TemporalExpressionType",
    "TemporalNormalizer",
    "TemporalTagger",
    "GraphAnswer",
    "GraphRetriever",
    "TemporalGraphIndex",
    "TemporalGraphLCELPipeline",
    "answer_to_mermaid",
    "TemplateRenderer",
    "TemplateType",
    "TemporalFact",
    "calculate_flesch_score",
]

__version__ = "0.1.0"
