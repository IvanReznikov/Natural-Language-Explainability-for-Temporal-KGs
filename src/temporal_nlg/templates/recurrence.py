"""Recurrence templates for repeating events."""
from typing import Dict

RECURRENCE_TEMPLATES = [
    {"id": "rec_basic_1", "template": "{event} happens every {frequency}.", "clarity_score": 4.7},
    {"id": "rec_basic_2", "template": "Each {frequency}, {event} occurs.", "clarity_score": 4.6},
    {"id": "rec_span_1", "template": "From {start_date} to {end_date}, {event} repeats every {frequency}.", "clarity_score": 4.5},
    {"id": "rec_until_1", "template": "{event} kept recurring every {frequency} until {end_date}.", "clarity_score": 4.4},
    {"id": "rec_count_1", "template": "{event} occurred {count} times over {frequency}.", "clarity_score": 4.4},
]


class RecurrenceTemplate:
    def __init__(self, template_id: str, template_string: str, confidence: float = 0.5):
        self.template_id = template_id
        self.template_string = template_string
        self.confidence = confidence

    def render(self, fact) -> str:
        content = fact.content
        if "event" not in content or "frequency" not in content:
            raise ValueError("Recurrence fact must have 'event' and 'frequency'")
        fmt = {
            "event": content.get("event", ""),
            "frequency": content.get("frequency", ""),
            "start_date": content.get("start_date", ""),
            "end_date": content.get("end_date", ""),
            "count": content.get("count", ""),
        }
        return self.template_string.format(**fmt)


class RecurrenceTemplateLibrary:
    def __init__(self):
        self.templates: Dict[str, RecurrenceTemplate] = {}
        self._load_templates()

    def _load_templates(self):
        for spec in RECURRENCE_TEMPLATES:
            tpl = RecurrenceTemplate(
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
