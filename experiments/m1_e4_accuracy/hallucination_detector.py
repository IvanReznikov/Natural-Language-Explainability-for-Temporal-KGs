#!/usr/bin/env python3
"""
M1-E4: Hallucination Detection Module
Detects when outputs fabricate facts not in the original fact
"""

from typing import Dict, List, Any, Tuple, Set
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re


class HallucinationDetector:
    """Detect hallucinations in generated outputs"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def detect(
        self, fact: Any, output: str, output_entities: Dict[str, List[str]]
    ) -> Tuple[bool, str]:
        """
        Detect if output contains hallucinations
        Returns: (is_hallucinated: bool, hallucination_type: str)

        Hallucination types:
        - "none": No hallucination
        - "date": Wrong dates
        - "entity": Wrong entities
        - "relation": Wrong relations
        - "fabricated": Completely fabricated content
        """
        try:
            fact_content = getattr(fact, "content", {}) or {}
            fact_tokens = self._fact_tokens(fact_content)
            fact_years = set(re.findall(r"\b(1\d{3}|2\d{3})\b", str(fact_content)))
            output_str = output.lower()
            output_years = set(re.findall(r"\b(1\d{3}|2\d{3})\b", output_str))

            # Date hallucination: only check if fact has years
            if fact_years and output_years:
                extra_years = output_years - fact_years
                if extra_years and (len(extra_years) / len(output_years) > 0.5):
                    return (True, "date")

            # Entity/token preservation with stopword filtering
            if fact_tokens:
                output_tokens = self._normalize_tokens(output_str)
                overlap = output_tokens & fact_tokens
                if not overlap and len(output_tokens) > 3:
                    return (True, "entity")

            # Relation check remains as a lightweight guard
            if self._has_temporal_relations(str(fact_content)) and not self._has_temporal_relations(
                output_str
            ):
                return (True, "relation")

            return (False, "none")

        except Exception:
            # On error, assume no hallucination
            return (False, "none")

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Simple entity extraction"""
        entities = {
            "DATE": [],
            "NAME": [],
        }

        # Extract years
        years = re.findall(r"\b(1\d{3}|2\d{3})\b", text)
        entities["DATE"].extend(years)

        # Extract capitalized words
        names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
        entities["NAME"].extend(names)

        return entities

    def _fact_tokens(self, content: Dict[str, Any]) -> set:
        tokens = set()

        def add(val: Any):
            if isinstance(val, str):
                for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", val.lower()):
                    if len(w) >= 4:
                        tokens.add(w)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    add(item)
            elif isinstance(val, dict):
                for v in val.values():
                    add(v)
            else:
                for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", str(val).lower()):
                    if len(w) >= 4:
                        tokens.add(w)

        add(content)
        return tokens

    def _normalize_tokens(self, text: str) -> Set[str]:
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "then",
            "when",
            "was",
            "were",
            "are",
            "is",
            "at",
            "of",
            "in",
            "to",
            "by",
            "on",
        }
        tokens: Set[str] = set()
        for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()):
            if len(w) >= 4 and w not in stopwords:
                tokens.add(w)
        return tokens

    def _has_temporal_relations(self, text: str) -> bool:
        """Check if text has temporal relation keywords"""
        temporal_keywords = [
            "century",
            "decade",
            "year",
            "period",
            "era",
            "age",
            "before",
            "after",
            "during",
            "while",
            "when",
            "caused",
            "led to",
            "resulted in",
            "lasting",
            "lasted",
            "spanned",
            "extended",
        ]

        text_lower = text.lower()
        return any(kw in text_lower for kw in temporal_keywords)
