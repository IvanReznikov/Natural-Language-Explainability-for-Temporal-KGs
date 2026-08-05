"""
Temporal NLG Models

Production-ready generators for temporal fact verbalization.
"""

from .llm_generator import LLMGenerator
from .hybrid_generator import HybridGenerator, GenerationResult
from .qwen_generator import QwenEmbeddingModel, QwenLocalGenerator

__all__ = [
    "LLMGenerator",
    "HybridGenerator",
    "GenerationResult",
    "QwenLocalGenerator",
    "QwenEmbeddingModel",
]
