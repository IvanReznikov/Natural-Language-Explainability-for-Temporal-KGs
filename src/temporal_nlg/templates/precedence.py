"""Precedence templates for before/after relations."""

from typing import Dict

PRECEDENCE_TEMPLATES = [
    {"id": "prec_before_1", "template": "{first} happened before {second}.", "clarity_score": 4.7},
    {"id": "prec_after_1", "template": "{second} followed {first}.", "clarity_score": 4.6},
    {"id": "prec_order_1", "template": "First {first}, then {second}.", "clarity_score": 4.8},
    {
        "id": "prec_when_1",
        "template": "When {first} finished, {second} began.",
        "clarity_score": 4.5,
    },
    {
        "id": "prec_cause_hint_1",
        "template": "{first} set the stage for {second}.",
        "clarity_score": 4.5,
    },
]


class PrecedenceTemplate:
    def __init__(self, template_id: str, template_string: str, confidence: float = 0.5):
        self.template_id = template_id
        self.template_string = template_string
        self.confidence = confidence

    def render(self, fact) -> str:
        content = fact.content
        if "first" not in content or "second" not in content:
            raise ValueError("Precedence fact must have 'first' and 'second'")
        format_dict = {
            "first": content.get("first", ""),
            "second": content.get("second", ""),
        }
        return self.template_string.format(**format_dict)


class PrecedenceTemplateLibrary:
    def __init__(self):
        self.templates: Dict[str, PrecedenceTemplate] = {}
        self._load_templates()

    def _load_templates(self):
        for spec in PRECEDENCE_TEMPLATES:
            tpl = PrecedenceTemplate(
                template_id=spec["id"],
                template_string=spec["template"],
                confidence=spec.get("clarity_score", 4.5) / 5.0,
            )
            self.templates[spec["id"]] = tpl

    def render(self, fact, template_id: str = None) -> str:
        if not template_id:
            template_id = list(self.templates.keys())[0]
        if template_id not in self.templates:
            raise ValueError(f"Template not found: {template_id}")
        return self.templates[template_id].render(fact)

    def render_all(self, fact) -> Dict[str, str]:
        outputs = {}
        for tid, tpl in self.templates.items():
            try:
                outputs[tid] = tpl.render(fact)
            except Exception:
                pass
        return outputs
