#!/usr/bin/env python3
"""
M1-E4: Fact Extractor Module
Extracts entities and relations from facts and generated text
"""

from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import re

class FactExtractor:
    """Extract entities and key tokens from temporal facts"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def extract_from_fact(self, fact: Any) -> Dict[str, List[str]]:
        """Deterministic extraction from structured fact content"""
        try:
            content = getattr(fact, "content", {}) or {}
            tokens = self._flatten_fact_tokens(content)
            years = re.findall(r'\b(1\d{3}|2\d{3})\b', str(content))
            return {
                "FACT": tokens,
                "DATE": years,
            }
        except Exception:
            return {"FACT": [], "DATE": []}

    def extract_from_text(self, text: str, context: str = "text") -> Dict[str, List[str]]:
        """Lightweight pattern extraction for outputs"""
        entities = {
            "FACT": [],
            "DATE": [],
        }

        try:
            text_lower = text.lower()
            # Years
            years = re.findall(r'\b(1\d{3}|2\d{3})\b', text)
            if years:
                entities["DATE"].extend(years)

            # Tokens: keep words >=4 chars to reduce stopword noise
            words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text_lower)
            for w in words:
                if len(w) >= 4:
                    entities["FACT"].append(w)

            # Deduplicate
            for k in entities:
                seen = set()
                dedup = []
                for ent in entities[k]:
                    if ent not in seen:
                        seen.add(ent)
                        dedup.append(ent)
                entities[k] = dedup

            return entities

        except Exception:
            return entities

    def _flatten_fact_tokens(self, content: Dict[str, Any]) -> List[str]:
        tokens: List[str] = []

        def add_from_value(val: Any):
            if isinstance(val, str):
                for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", val.lower()):
                    if len(w) >= 4:
                        tokens.append(w)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    add_from_value(item)
            elif isinstance(val, dict):
                for v in val.values():
                    add_from_value(v)
            else:
                s = str(val)
                for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", s.lower()):
                    if len(w) >= 4:
                        tokens.append(w)

        for v in content.values():
            add_from_value(v)

        # Deduplicate while preserving order
        seen = set()
        dedup = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                dedup.append(t)
        return dedup