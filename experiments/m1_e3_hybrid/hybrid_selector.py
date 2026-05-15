#!/usr/bin/env python3
"""
M1-E3: HYBRID SELECTOR
======================

LLM-based intelligent routing to select best template or LLM polish strategy.
"""

from typing import Dict, List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import json

class HybridSelector:
    """Routes temporal facts to best generation strategy"""
    
    def __init__(self, model: str = "gpt-4.1-nano"):
        self.llm = ChatOpenAI(model=model)
        
        self.selector_prompt = ChatPromptTemplate.from_template("""
You are a temporal NLG routing specialist. Analyze this fact and recommend generation strategy.

FACT TYPE: {fact_type}
FACT DETAILS:
- Event/Entity: {event}
- Dates: {dates}
- Context: {context}

TEMPLATE OPTIONS:
{templates}

RECOMMENDATION RULES:
1. Simple facts (1-2 fields) → Use TEMPLATE (faster, more accurate)
2. Complex facts (3+ fields, unusual patterns) → Use TEMPLATE+POLISH
3. Very complex facts (multiple events, unclear causality) → Use LLM

Output JSON:
{{
  "strategy": "template" | "polish" | "llm",
  "top_3_templates": ["template_id_1", "template_id_2", "template_id_3"],
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}
""")
    
    def route_fact(self, fact: Dict, available_templates: List[str]) -> Dict:
        """Route fact to best generation strategy"""
        
        chain = self.selector_prompt | self.llm
        response = chain.invoke({
            "fact_type": fact.get("type", "unknown"),
            "event": fact.get("event", ""),
            "dates": fact.get("dates", ""),
            "context": fact.get("context", ""),
            "templates": "\n".join(f"- {t}" for t in available_templates[:10])
        })
        
        try:
            result = json.loads(response.content)
            return result
        except:
            # Fallback to simple template strategy
            return {
                "strategy": "template",
                "top_3_templates": available_templates[:3],
                "confidence": 0.5,
                "reasoning": "JSON parsing failed, using default template"
            }