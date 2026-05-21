#!/usr/bin/env python3
"""
Hybrid Template-LLM Generator

Intelligently routes between template rendering and LLM generation
for optimal quality/speed tradeoff. Based on M1-E3 experiments.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from langchain_openai import ChatOpenAI

from ..core.templates import TemporalFact, TemplateRenderer
from .llm_generator import LLMGenerator


@dataclass
class GenerationResult:
    """Result from hybrid generation."""
    text: str
    strategy: str  # "template", "polished", or "llm"
    confidence: float
    flesch_score: Optional[float] = None
    template_id: Optional[str] = None


class HybridGenerator:
    """
    Hybrid generator combining templates and LLM.
    
    Routes facts to the best generation strategy:
    - Simple facts → Template (fast, accurate)
    - Medium complexity → Template + LLM polish
    - Complex facts → Pure LLM
    
    Example:
        >>> from temporal_nlg.models import HybridGenerator
        >>> generator = HybridGenerator()
        >>> fact = TemporalFact(...)
        >>> result = generator.generate(fact)
        >>> print(result.text, result.strategy)
    """
    
    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        polish_threshold: float = 0.7,
        enable_caching: bool = True,
        llm_polisher: ChatOpenAI = None,
        llm_generator: LLMGenerator = None,
        template_renderer: TemplateRenderer = None,
    ):
        """
        Initialize hybrid generator.
        
        Args:
            model: LLM model name
            polish_threshold: Quality threshold for using polish (0-1)
            enable_caching: Cache template renders
        """
        self.model = model
        self.polish_threshold = polish_threshold
        self.enable_caching = enable_caching
        
        self.template_renderer = template_renderer or TemplateRenderer()
        self.llm_generator = llm_generator or LLMGenerator(model=model)
        if llm_polisher is not None:
            self.llm_polisher = llm_polisher
        else:
            try:
                self.llm_polisher = ChatOpenAI(model=model, temperature=0.3, max_tokens=40)
            except Exception:
                class _StubLLM:
                    def invoke(self, payload):
                        return type("Resp", (), {"content": "stub polish"})()
                self.llm_polisher = _StubLLM()
        
        # Caches
        self._template_cache: Dict[str, str] = {}
    
    def generate(
        self,
        fact: TemporalFact,
        force_strategy: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate explanation using best strategy.
        
        Args:
            fact: TemporalFact to verbalize
            force_strategy: Force specific strategy ("template", "polish", "llm")
            
        Returns:
            GenerationResult with text and metadata
        """
        # Get cache key
        cache_key = self._get_cache_key(fact)
        
        # Decide strategy
        if force_strategy:
            strategy = force_strategy
        else:
            strategy = self._route_fact(fact)
        
        # Execute strategy
        if strategy == "template":
            return self._generate_template(fact, cache_key)
        elif strategy == "polish":
            return self._generate_polished(fact, cache_key)
        else:  # llm
            return self._generate_llm(fact)
    
    def _route_fact(self, fact: TemporalFact) -> str:
        """
        Route fact to best strategy based on complexity.
        
        Returns:
            "template", "polish", or "llm"
        """
        # Count complexity indicators
        complexity_score = 0
        
        # Multiple dates
        start = getattr(fact, 'start_date', None)
        end = getattr(fact, 'end_date', None)
        if isinstance(start, str) and isinstance(end, str) and start and end:
            complexity_score += 1
        
        # Long context
        context = getattr(fact, 'context', None)
        if isinstance(context, str) and len(context) > 50:
            complexity_score += 1
        
        # Multiple events/entities
        events = getattr(fact, 'events', None)
        if isinstance(events, list) and len(events) > 2:
            complexity_score += 2
        
        # Route based on complexity
        if complexity_score == 0:
            return "template"
        elif complexity_score <= 2:
            return "polish"
        else:
            return "llm"
    
    def _generate_template(
        self,
        fact: TemporalFact,
        cache_key: str
    ) -> GenerationResult:
        """Generate using template only."""
        # Check cache
        if self.enable_caching and cache_key in self._template_cache:
            text = self._template_cache[cache_key]
        else:
            text = self.template_renderer.render(fact)
            if self.enable_caching:
                self._template_cache[cache_key] = text
        
        return GenerationResult(
            text=text,
            strategy="template",
            confidence=0.9,
            template_id=self.template_renderer.last_template_id
        )
    
    def _generate_polished(
        self,
        fact: TemporalFact,
        cache_key: str
    ) -> GenerationResult:
        """Generate using template + LLM polish."""
        # Get template render
        template_result = self._generate_template(fact, cache_key)
        
        # Polish with LLM
        polish_prompt = f"""Improve readability while preserving ALL facts:

Original: {template_result.text}

Output ONLY the improved version (one sentence)."""
        
        try:
            response = self.llm_polisher.invoke(polish_prompt)
            polished_text = getattr(response, "content", str(response)).strip()
            
            return GenerationResult(
                text=polished_text,
                strategy="polished",
                confidence=0.85,
                template_id=template_result.template_id
            )
        except Exception:
            # Fallback to template
            return template_result
    
    def _generate_llm(self, fact: TemporalFact) -> GenerationResult:
        """Generate using LLM only."""
        text = self.llm_generator.generate(fact)
        
        return GenerationResult(
            text=text,
            strategy="llm",
            confidence=0.7
        )
    
    def _get_cache_key(self, fact: TemporalFact) -> str:
        """Generate cache key for fact."""
        return str(hash(str(fact)))
    
    def batch_generate(
        self,
        facts: List[TemporalFact],
        show_progress: bool = False
    ) -> List[GenerationResult]:
        """
        Generate explanations for multiple facts.
        
        Args:
            facts: List of TemporalFact instances
            show_progress: Show progress bar
            
        Returns:
            List of GenerationResult objects
        """
        results = []
        
        iterator = facts
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(facts, desc="Generating")
            except ImportError:
                pass
        
        for fact in iterator:
            try:
                result = self.generate(fact)
                results.append(result)
            except Exception as e:
                results.append(GenerationResult(
                    text=f"[ERROR: {str(e)}]",
                    strategy="error",
                    confidence=0.0
                ))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            "cache_size": len(self._template_cache),
            "model": self.model,
            "polish_threshold": self.polish_threshold
        }
