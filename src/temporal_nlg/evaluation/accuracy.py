#!/usr/bin/env python3
"""
Accuracy Evaluation Module

Measures factual accuracy of generated temporal explanations.
Based on M1-E4 experiments.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class AccuracyMetrics:
    """Accuracy evaluation results."""
    date_preservation: float  # 0-1
    entity_preservation: float  # 0-1
    relation_preservation: float  # 0-1
    hallucination_detected: bool
    overall_accuracy: float  # 0-1


class AccuracyEvaluator:
    """
    Evaluate factual accuracy of generated text.
    
    Checks preservation of:
    - Temporal information (dates, periods)
    - Entities (people, places, organizations)
    - Relations (causality, sequences)
    
    Example:
        >>> from temporal_nlg.evaluation import AccuracyEvaluator
        >>> evaluator = AccuracyEvaluator()
        >>> fact = {...}
        >>> output = "Einstein was born in 1879"
        >>> metrics = evaluator.evaluate(fact, output)
        >>> print(metrics.overall_accuracy)
    """
    
    def __init__(self):
        """Initialize accuracy evaluator."""
        self.temporal_words = [
            'century', 'decade', 'period', 'era', 'age', 'epoch',
            'before', 'after', 'during', 'while', 'when', 'until'
        ]
    
    def evaluate(
        self,
        fact: Any,
        generated_text: str
    ) -> AccuracyMetrics:
        """
        Evaluate accuracy of generated text against fact.
        
        Args:
            fact: Source temporal fact (TemporalFact or dict)
            generated_text: Generated explanation text
            
        Returns:
            AccuracyMetrics with detailed scores
        """
        date_score = self._score_date_preservation(fact, generated_text)
        entity_score = self._score_entity_preservation(fact, generated_text)
        relation_score = self._score_relation_preservation(fact, generated_text)
        hallucination = self._detect_hallucination(fact, generated_text)
        
        # Overall accuracy (weighted average)
        overall = (
            0.4 * date_score +
            0.3 * entity_score +
            0.3 * relation_score
        )
        
        # Penalize if hallucination detected
        if hallucination:
            overall *= 0.5
        
        return AccuracyMetrics(
            date_preservation=date_score,
            entity_preservation=entity_score,
            relation_preservation=relation_score,
            hallucination_detected=hallucination,
            overall_accuracy=overall
        )
    
    def _score_date_preservation(self, fact: Any, text: str) -> float:
        """Score date/temporal preservation (0-1)."""
        try:
            fact_str = str(fact).lower()
            text_lower = text.lower()
            
            # Extract years from fact
            fact_years = set(re.findall(r'\b(1\d{3}|2\d{3})\b', fact_str))
            if not fact_years:
                return 1.0  # No dates to preserve
            
            # Check preservation in text
            found_years = {year for year in fact_years if year in text_lower}
            preservation_rate = len(found_years) / len(fact_years)
            
            # Bonus for temporal context words
            if any(word in text_lower for word in self.temporal_words):
                preservation_rate = min(1.0, preservation_rate + 0.1)
            
            return preservation_rate
            
        except Exception:
            return 0.5  # Unknown
    
    def _score_entity_preservation(self, fact: Any, text: str) -> float:
        """Score entity preservation (0-1)."""
        try:
            # Extract entity from fact
            entity = None
            if hasattr(fact, 'entity'):
                entity = fact.entity
            elif hasattr(fact, 'event'):
                entity = fact.event
            elif isinstance(fact, dict):
                entity = fact.get('entity') or fact.get('event')
            
            if not entity:
                return 1.0  # No entity to preserve
            
            # Check if entity appears in text
            entity_lower = str(entity).lower()
            text_lower = text.lower()
            
            # Exact match
            if entity_lower in text_lower:
                return 1.0
            
            # Partial match (first/last word)
            entity_words = entity_lower.split()
            if len(entity_words) > 1:
                if entity_words[0] in text_lower or entity_words[-1] in text_lower:
                    return 0.7
            
            return 0.0
            
        except Exception:
            return 0.5
    
    def _score_relation_preservation(self, fact: Any, text: str) -> float:
        """Score relational structure preservation (0-1)."""
        try:
            # Identify fact type
            fact_type = None
            if hasattr(fact, 'fact_type'):
                fact_type = str(fact.fact_type).lower()
            elif isinstance(fact, dict):
                fact_type = str(fact.get('type', '')).lower()
            
            if not fact_type:
                return 0.5
            
            text_lower = text.lower()
            
            # Check for type-specific markers
            if 'causal' in fact_type:
                markers = ['because', 'caused', 'led to', 'resulted in', 'due to']
                return 1.0 if any(m in text_lower for m in markers) else 0.3
            
            elif 'sequence' in fact_type:
                markers = ['then', 'next', 'after', 'before', 'followed by']
                return 1.0 if any(m in text_lower for m in markers) else 0.3
            
            elif 'interval' in fact_type:
                markers = ['from', 'to', 'between', 'until', 'during']
                return 1.0 if any(m in text_lower for m in markers) else 0.3
            
            elif 'overlap' in fact_type:
                markers = ['while', 'during', 'simultaneously', 'at the same time']
                return 1.0 if any(m in text_lower for m in markers) else 0.3
            
            return 0.7  # Unknown type
            
        except Exception:
            return 0.5
    
    def _detect_hallucination(self, fact: Any, text: str) -> bool:
        """
        Detect potential hallucinations.
        
        Returns:
            True if hallucination detected
        """
        try:
            # Extract years from text
            text_years = set(re.findall(r'\b(1\d{3}|2\d{3})\b', text))
            fact_str = str(fact)
            fact_years = set(re.findall(r'\b(1\d{3}|2\d{3})\b', fact_str))
            
            # Check for extra years not in fact
            extra_years = text_years - fact_years
            if extra_years:
                return True
            
            # Check for suspiciously long output (>50 words)
            if len(text.split()) > 50:
                return True
            
            return False
            
        except Exception:
            return False
    
    def batch_evaluate(
        self,
        facts: List[Any],
        texts: List[str]
    ) -> List[AccuracyMetrics]:
        """
        Evaluate multiple fact-text pairs.
        
        Args:
            facts: List of source facts
            texts: List of generated texts
            
        Returns:
            List of AccuracyMetrics
        """
        if len(facts) != len(texts):
            raise ValueError("Facts and texts must have same length")
        
        return [
            self.evaluate(fact, text)
            for fact, text in zip(facts, texts)
        ]
    
    def aggregate_metrics(
        self,
        metrics: List[AccuracyMetrics]
    ) -> Dict[str, float]:
        """
        Aggregate metrics across multiple evaluations.
        
        Args:
            metrics: List of AccuracyMetrics
            
        Returns:
            Dictionary of aggregated scores
        """
        if not metrics:
            return {}
        
        return {
            "mean_date_preservation": sum(m.date_preservation for m in metrics) / len(metrics),
            "mean_entity_preservation": sum(m.entity_preservation for m in metrics) / len(metrics),
            "mean_relation_preservation": sum(m.relation_preservation for m in metrics) / len(metrics),
            "hallucination_rate": sum(m.hallucination_detected for m in metrics) / len(metrics),
            "mean_overall_accuracy": sum(m.overall_accuracy for m in metrics) / len(metrics),
        }
