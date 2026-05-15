"""Containment templates for within/contains relations."""
from typing import Dict

CONTAINMENT_TEMPLATES = [
    {"id": "cont_within_1", "template": "{inner} occurred within {outer}.", "clarity_score": 4.6},
    {"id": "cont_part_1", "template": "{inner} was part of {outer}.", "clarity_score": 4.7},
    {"id": "cont_during_1", "template": "During {outer}, {inner} took place.", "clarity_score": 4.6},
    {"id": "cont_span_1", "template": "{outer} spans the period that includes {inner}.", "clarity_score": 4.4},
    {"id": "cont_window_1", "template": "In the window of {outer}, we saw {inner}.", "clarity_score": 4.5},
]


class ContainmentTemplate:
    def __init__(self, template_id: str, template_string: str, confidence: float = 0.5):
        self.template_id = template_id
        self.template_string = template_string
        self.confidence = confidence

    def render(self, fact) -> str:
        content = fact.content
        if "inner" not in content or "outer" not in content:
            raise ValueError("Containment fact must have 'inner' and 'outer'")
        fmt = {"inner": content.get("inner", ""), "outer": content.get("outer", "")}
        return self.template_string.format(**fmt)


class ContainmentTemplateLibrary:
    def __init__(self):
        self.templates: Dict[str, ContainmentTemplate] = {}
        self._load_templates()

    def _load_templates(self):
        for spec in CONTAINMENT_TEMPLATES:
            tpl = ContainmentTemplate(
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
