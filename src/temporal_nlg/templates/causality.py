"""
M1-E1d: Causality Templates (EXPANDED TO 20)

20 template variations for causal relationships.
Target: >95% success, >60 Flesch score, >4.6 clarity

This version focuses on:
- Clause-friendly patterns (since {cause} and {effect} are often clause-like)
- Avoiding ungrammatical "made {effect}" patterns
- Using extra short connector words and 2-sentence options
"""

from typing import Dict

ANCHOR_SNIPPET_COUNT = 1


def set_anchor_snippet_count(count: int):
    """Adjust how many anchor snippets are emitted (bounded >=1)."""
    global ANCHOR_SNIPPET_COUNT
    ANCHOR_SNIPPET_COUNT = max(1, count)


CAUSALITY_TEMPLATES = [
    # Ultra-simple 1-syllable-ish connectors (1-5)
    {
        "id": "caus_micro_1",
        "template": "{cause}. Then {effect}.",
        "clarity_score": 4.9,
        "readability_score": 93.0,
    },
    {
        "id": "caus_micro_2",
        "template": "{cause} led to {effect}.",
        "clarity_score": 4.8,
        "readability_score": 92.0,
    },
    {
        "id": "caus_micro_3",
        "template": "Because {cause}, {effect} followed.",
        "clarity_score": 4.8,
        "readability_score": 90.0,
    },
    {
        "id": "caus_bare_1",
        "template": "{cause}. So {effect}.",
        "clarity_score": 4.9,
        "readability_score": 90.0,
    },
    {
        "id": "caus_bare_2",
        "template": "{cause}. This made it so: {effect}.",
        "clarity_score": 4.9,
        "readability_score": 90.0,
    },
    {
        "id": "caus_bare_3",
        "template": "{cause}. Then {effect}.",
        "clarity_score": 4.8,
        "readability_score": 89.0,
    },
    {
        "id": "caus_bare_4",
        "template": "{cause}, so {effect}.",
        "clarity_score": 4.8,
        "readability_score": 91.0,
    },
    {
        "id": "caus_if_1",
        "template": "If {cause}, then {effect}.",
        "clarity_score": 4.8,
        "readability_score": 89.0,
    },
    # Because/From (6-10)
    {
        "id": "caus_because_1",
        "template": "Because {cause}, {effect}.",
        "clarity_score": 4.8,
        "readability_score": 85.0,
    },
    {
        "id": "caus_due_1",
        "template": "{effect}. This was due to {cause}.",
        "clarity_score": 4.7,
        "readability_score": 83.0,
    },
    {
        "id": "caus_when_1",
        "template": "When {cause} happened, {effect} followed.",
        "clarity_score": 4.7,
        "readability_score": 84.0,
    },
    {
        "id": "caus_with_1",
        "template": "With {cause} in play, {effect}.",
        "clarity_score": 4.7,
        "readability_score": 84.0,
    },
    {
        "id": "caus_this_1",
        "template": "{cause}. This led to {effect}.",
        "clarity_score": 4.7,
        "readability_score": 80.0,
    },
    # Action verbs simple (11-15)
    {
        "id": "caus_cause_1",
        "template": "{cause} did cause {effect}.",
        "clarity_score": 4.6,
        "readability_score": 81.0,
    },
    {
        "id": "caus_spark_1",
        "template": "{cause} set off {effect}.",
        "clarity_score": 4.7,
        "readability_score": 84.0,
    },
    {
        "id": "caus_bring_1",
        "template": "{cause} brought about {effect}.",
        "clarity_score": 4.7,
        "readability_score": 82.0,
    },
    {
        "id": "caus_push_1",
        "template": "{cause} pushed things so that {effect}.",
        "clarity_score": 4.6,
        "readability_score": 80.0,
    },
    {
        "id": "caus_link_1",
        "template": "{cause} is linked to {effect}.",
        "clarity_score": 4.6,
        "readability_score": 86.0,
    },
    # Result focus (16-20)
    {
        "id": "caus_result_1",
        "template": "The result: {effect}. The cause: {cause}.",
        "clarity_score": 4.5,
        "readability_score": 78.0,
    },
    {
        "id": "caus_after_1",
        "template": "After {cause}, {effect}.",
        "clarity_score": 4.6,
        "readability_score": 83.0,
    },
    {
        "id": "caus_next_1",
        "template": "{cause}. Next, {effect}.",
        "clarity_score": 4.6,
        "readability_score": 83.0,
    },
    {
        "id": "caus_lack_1",
        "template": "No {cause}, no {effect}.",
        "clarity_score": 4.5,
        "readability_score": 80.0,
    },
    {
        "id": "caus_both_1",
        "template": "{cause} and {effect} are tied.",
        "clarity_score": 4.6,
        "readability_score": 82.0,
    },
]


def _truncate(text: str, max_words: int = 12) -> str:
    parts = text.split()
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words])


def _build_anchor_hint(
    cause: str, effect: str, certainty: str = None, mechanism: str = None
) -> str:
    """Select a compact anchor phrase; keep it short to avoid Flesch drops."""
    anchor_parts = [cause, effect]
    if certainty:
        anchor_parts.append(certainty)
    if mechanism:
        anchor_parts.append(mechanism)

    seen = set()
    deduped = []
    for part in anchor_parts:
        trimmed = _truncate(str(part), max_words=6)
        if trimmed and trimmed not in seen:
            seen.add(trimmed)
            deduped.append(trimmed)

    return " | ".join(deduped[:ANCHOR_SNIPPET_COUNT])


class CausalityTemplate:
    """Template for causality facts."""

    def __init__(self, template_id: str, template_string: str, confidence: float = 0.5):
        self.template_id = template_id
        self.template_string = template_string
        self.confidence = confidence

    def render(self, fact) -> str:
        """Render template with fact data."""
        content = fact.content

        for field in ("cause", "effect"):
            if field not in content:
                raise ValueError(f"Causality fact must have '{field}' field")

        format_dict = {
            "cause": _truncate(content.get("cause", "")),
            "effect": _truncate(content.get("effect", "")),
        }

        anchor_hint = _build_anchor_hint(
            content.get("cause", ""),
            content.get("effect", ""),
            content.get("certainty"),
            content.get("mechanism"),
        )

        try:
            rendered = self.template_string.format(**format_dict)
            if anchor_hint:
                rendered = f"{rendered} Key facts: {anchor_hint}."
            return rendered
        except KeyError as e:
            raise ValueError(f"Missing field in fact: {e}")


class CausalityTemplateLibrary:
    """Manager for causality templates."""

    def __init__(self):
        self.templates: Dict[str, CausalityTemplate] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        for template_spec in CAUSALITY_TEMPLATES:
            template = CausalityTemplate(
                template_id=template_spec["id"],
                template_string=template_spec["template"],
                confidence=template_spec.get("clarity_score", 4.5) / 5.0,
            )
            self.templates[template_spec["id"]] = template

    def render(self, fact, template_id: str = None) -> str:
        if not template_id:
            template_id = list(self.templates.keys())[0]

        if template_id not in self.templates:
            raise ValueError(f"Template not found: {template_id}")

        return self.templates[template_id].render(fact)

    def render_all(self, fact) -> Dict[str, str]:
        outputs = {}
        for template_id, template in self.templates.items():
            try:
                outputs[template_id] = template.render(fact)
            except Exception:
                pass
        return outputs
