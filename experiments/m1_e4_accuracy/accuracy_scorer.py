#!/usr/bin/env python3
"""
M1-E4: Accuracy Scoring Module
Scores preservation of dates, entities, and relations
"""

from typing import Dict, List, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re


class AccuracyScorer:
    """Score accuracy of generated outputs"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def score_date_preservation(self, fact: Any, output: str) -> float:
        """
        Score how well dates from fact are preserved in output
        Returns: 0-1 score
        """
        try:
            # Extract date from fact
            fact_str = str(fact).lower()
            output_str = output.lower()

            # Look for year patterns in fact
            year_patterns = re.findall(r"\b(1\d{3}|2\d{3})\b", fact_str)
            if not year_patterns:
                # No dates in fact, trivially preserved
                return 1.0

            # Check if at least one year appears in output
            found_years = []
            for year in set(year_patterns):
                if year in output_str:
                    found_years.append(year)

            # Score based on fraction of years preserved
            preservation_rate = len(found_years) / len(set(year_patterns)) if year_patterns else 1.0

            # Boost score if key temporal words preserved
            temporal_words = ["century", "decade", "period", "era", "age", "epoch"]
            if any(word in output_str for word in temporal_words):
                preservation_rate = min(1.0, preservation_rate + 0.2)

            return min(1.0, preservation_rate)

        except Exception:
            return 0.5  # Unknown = neutral score

    def score_entity_preservation(
        self, fact_entities: Dict[str, List[str]], output_entities: Dict[str, List[str]]
    ) -> float:
        """
        Score how well entities from fact are preserved in output
        Uses precision + recall for F1-like metric
        """
        try:
            total_score = 0.0
            entity_types = set(fact_entities.keys()) | set(output_entities.keys())

            if not entity_types:
                return 1.0  # No entities, trivially preserved

            for entity_type in entity_types:
                fact_ents = set(fact_entities.get(entity_type, []))
                output_ents = set(output_entities.get(entity_type, []))

                if not fact_ents:
                    # No entities of this type in fact
                    continue

                # Precision: fraction of output entities that match fact entities
                if output_ents:
                    matches = len(fact_ents & output_ents)
                    precision = matches / len(output_ents)
                else:
                    precision = 0.0

                # Recall: fraction of fact entities found in output
                if fact_ents:
                    matches = len(fact_ents & output_ents)
                    recall = matches / len(fact_ents)
                else:
                    recall = 1.0

                # F1 score for this type
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

                total_score += f1

            # Average across entity types
            avg_score = total_score / len(entity_types) if entity_types else 1.0
            return min(1.0, avg_score)

        except Exception:
            return 0.5

    def score_relation_preservation(
        self, fact: Any, output_entities: Dict[str, List[str]]
    ) -> float:
        """
        Score how well semantic relations are preserved
        For temporal facts: sequence, causality, overlap preservation
        """
        try:
            fact_str = str(fact)
            output_str = str(output_entities)

            # Extract relation keywords from fact
            temporal_relations = {
                "causality": ["caused", "because", "resulted", "led to", "due to"],
                "sequence": ["then", "after", "before", "followed", "preceding"],
                "overlap": ["during", "while", "same time", "concurrent", "simultaneous"],
                "interval": ["from", "to", "until", "lasted", "spanned"],
            }

            # Detect relation type from fact
            fact_lower = fact_str.lower()
            detected_relations = []
            for rel_type, keywords in temporal_relations.items():
                if any(kw in fact_lower for kw in keywords):
                    detected_relations.append(rel_type)

            if not detected_relations:
                return 1.0  # No specific relations

            # Check if output preserves these relations
            output_lower = str(output_entities).lower()
            preserved = 0
            for rel_type in detected_relations:
                keywords = temporal_relations[rel_type]
                if any(kw in output_lower for kw in keywords):
                    preserved += 1

            preservation_rate = preserved / len(detected_relations) if detected_relations else 1.0
            return min(1.0, preservation_rate)

        except Exception:
            return 0.5
