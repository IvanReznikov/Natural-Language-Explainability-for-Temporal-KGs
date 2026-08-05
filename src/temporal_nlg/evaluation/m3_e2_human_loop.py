"""M3-E2: Human-in-the-loop / LLM-in-the-loop scoring.

This module fills the M3-E2 metrics that require human or expert judgement.

It supports two workflows:
- Human workflow: export tasks as JSONL, collect annotations, then merge back.
- LLM workflow: use a LangChain ChatOpenAI model to produce proxy judgements.

Important: LLM scoring is *not* a substitute for real user studies / experts.
It is provided as a scalable proxy and should be reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore


class HumanLoopScores(BaseModel):
    """Judgement-only metrics (0..1)."""

    ambiguity_resolution: Optional[float] = Field(
        default=None, description="Did the explanation resolve ambiguity? 0..1"
    )
    causal_link_accuracy: Optional[float] = Field(
        default=None, description="Are causal links correct? 0..1"
    )
    confidence_calibration: Optional[float] = Field(
        default=None, description="Is certainty appropriately calibrated? 0..1"
    )
    narrative_consistency: Optional[float] = Field(
        default=None, description="Is the narrative consistent throughout? 0..1"
    )

    notes: Optional[str] = Field(default=None, description="Short justification")


@dataclass
class M3E2HumanLoopLLMScorer:
    """LangChain-based proxy scorer for human/expert metrics."""

    model: str = "gpt-4.1-nano"
    temperature: float = 0.0
    max_tokens: int = 200

    def __post_init__(self) -> None:
        self._prompt = None
        self._chain = None

        if ChatPromptTemplate is None or ChatOpenAI is None:
            return

        self._prompt = ChatPromptTemplate.from_template("""
You are evaluating a generated explanation against the provided gold context.
Return scores in [0,1] for the judgement metrics. If a metric is not applicable,
return null.

Time-scope bucket: {bucket}
User query: {query}

Gold context (facts, abbreviated):
{gold_context}

Generated explanation:
{prediction}

Scoring guidance:
- ambiguity_resolution: Does it resolve ambiguous references/time when needed?
- causal_link_accuracy: For causal bucket, are claimed causes supported by gold?
- confidence_calibration: Is the certainty level appropriate (not overconfident)?
- narrative_consistency: For sequence bucket, is the story internally consistent?

Output JSON only.
""".strip())

        try:
            llm = ChatOpenAI(
                model=self.model, temperature=self.temperature, max_tokens=self.max_tokens
            )
            self._chain = self._prompt | llm.with_structured_output(HumanLoopScores)
        except Exception:
            # Keep scorer usable without network/API keys.
            self._chain = None

    def score(self, record: Dict[str, Any], prediction_text: str, bucket: str) -> Dict[str, Any]:
        """Return a dict of judgement metrics; may be empty on failure."""

        if not prediction_text.strip():
            return {}

        # If chain isn't available, return empty (caller will keep nulls).
        if self._chain is None:
            return {}

        query = str(record.get("query") or record.get("question") or "")
        gold_facts = record.get("gold_facts") or []
        # Keep context small.
        gold_context = "\n".join(str(f) for f in gold_facts[:12])

        payload = {
            "bucket": bucket,
            "query": query,
            "gold_context": gold_context,
            "prediction": prediction_text,
        }

        try:
            result = self._chain.invoke(payload)
            if isinstance(result, HumanLoopScores):
                return result.model_dump()
            if hasattr(result, "model_dump"):
                return result.model_dump()
            if isinstance(result, dict):
                return result
            return {}
        except Exception:
            return {}


def coerce_unit_score(value: Any) -> Optional[float]:
    """Coerce a value into a 0..1 float.

    Accepts:
    - floats/ints already in [0,1]
    - 1..5 Likert ratings (mapped to 0..1)
    - strings containing numeric values
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return 1.0 if value else 0.0

    # Disambiguation rule:
    # - ints 1..5 are treated as Likert ratings
    # - floats 0..1 are treated as unit scores
    if isinstance(value, int) and not isinstance(value, bool):
        v_int = int(value)
        if 1 <= v_int <= 5:
            return (float(v_int) - 1.0) / 4.0
        return None

    if isinstance(value, float):
        if 0.0 <= value <= 1.0:
            return float(value)
        if 1.0 <= value <= 5.0:
            return (float(value) - 1.0) / 4.0
        return None

    try:
        s = str(value).strip()

        # If the string is an integer 1..5, treat it as Likert.
        if s.isdigit():
            v_int = int(s)
            if 1 <= v_int <= 5:
                return (float(v_int) - 1.0) / 4.0

        v = float(s)
        if 0.0 <= v <= 1.0:
            return v
        if 1.0 <= v <= 5.0:
            return (v - 1.0) / 4.0
    except Exception:
        return None

    return None
