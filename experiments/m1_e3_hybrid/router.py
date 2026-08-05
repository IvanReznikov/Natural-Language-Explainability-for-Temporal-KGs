"""
M1-E3: Router - Strategic decision making for template vs LLM polish

Decides whether to use template rendering alone or add LLM polishing.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class RoutingDecision:
    """Decision from router about how to handle this fact."""

    use_template: bool
    use_polish: bool
    confidence: float
    reasoning: str
    estimated_quality: float  # 0-1


class Router:
    """Routes facts to best rendering strategy."""

    def __init__(self, threshold_flesch_polish: float = 60.0):
        """
        Args:
            threshold_flesch_polish: Flesch score threshold for polishing
        """
        self.threshold_flesch_polish = threshold_flesch_polish

    def decide(
        self,
        fact: Dict[str, Any],
        template_quality: float,
        selector_confidence: float,
    ) -> RoutingDecision:
        """
        Decide routing strategy.

        Args:
            fact: Temporal fact
            template_quality: Quality score 0-1 of template match
            selector_confidence: LLM confidence 0-1 in selector

        Returns:
            RoutingDecision with strategy
        """
        # Always use template (we have it)
        use_template = True

        # Polish if:
        # 1. Template quality is borderline (0.4-0.7)
        # 2. AND selector confidence is high (>0.7)
        use_polish = 0.4 <= template_quality <= 0.7 and selector_confidence > 0.7

        # Confidence in decision
        confidence = min(template_quality, selector_confidence)

        # Reasoning
        if use_polish:
            reasoning = f"Template quality {template_quality:.1f} + high confidence {selector_confidence:.1f} → polish"
        else:
            reasoning = f"Template sufficient (quality: {template_quality:.1f})"

        # Estimate final quality
        if use_polish:
            estimated_quality = min(0.95, template_quality + 0.2)
        else:
            estimated_quality = template_quality

        return RoutingDecision(
            use_template=use_template,
            use_polish=use_polish,
            confidence=confidence,
            reasoning=reasoning,
            estimated_quality=estimated_quality,
        )


if __name__ == "__main__":
    router = Router()

    test_fact = {"subject": "Test", "date": "2023-01-01"}
    decision = router.decide(test_fact, template_quality=0.65, selector_confidence=0.8)

    print(f"Use template: {decision.use_template}")
    print(f"Use polish: {decision.use_polish}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Reasoning: {decision.reasoning}")
