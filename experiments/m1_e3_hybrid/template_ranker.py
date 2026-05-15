#!/usr/bin/env python3
"""
M1-E3: TEMPLATE RANKER
======================

Scores how well each template fits a given temporal fact.
Uses LLM to evaluate template-fact compatibility.
"""

from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import json


class TemplateRanker:
    """Scores template suitability for temporal facts"""
    
    def __init__(self, model: str = "gpt-4.1-nano"):
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        
        self.ranking_prompt = ChatPromptTemplate.from_template("""
Evaluate how well each template matches this temporal fact.
Rate ONLY the template fit, not the content quality.

FACT:
Type: {fact_type}
Content: {fact_content}

TEMPLATES TO RANK:
{templates}

RANKING CRITERIA (Rate 1-10):
1. Grammatical fit (does template match fact structure?)
2. Field coverage (all required fields present?)
3. Temporal accuracy (dates/intervals preserved?)
4. Semantic alignment (template expresses this type?)

Output JSON:
{{
  "rankings": [
    {{"template_id": "...", "score": 8, "reason": "..."}},
    ...
  ]
}}
""")
    
    def rank_templates(
        self,
        fact_type: str,
        fact_content: str,
        templates: List[Dict]
    ) -> Dict[str, float]:
        """
        Rank templates for this fact
        
        Returns: Dict mapping template_id -> score (0-10)
        """
        
        template_str = "\n".join(
            f"- {t['id']}: {t['template'][:60]}..."
            for t in templates[:15]
        )
        
        try:
            chain = self.ranking_prompt | self.llm
            response = chain.invoke({
                "fact_type": fact_type,
                "fact_content": fact_content[:200],
                "templates": template_str
            })
            
            result = json.loads(response.content)
            rankings = {}
            for item in result.get("rankings", []):
                rankings[item["template_id"]] = item["score"]
            
            return rankings
        
        except Exception as e:
            # Fallback: equal scoring
            return {
                t["id"]: 5.0 for t in templates
            }