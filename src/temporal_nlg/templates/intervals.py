# temporal_nlg/templates/intervals.py
"""
M1-E1b: Interval Templates (EXPANDED TO 20)

20 template variations for event intervals.
Target: >95% success, >60 Flesch score, >4.6 clarity
"""

from typing import Dict

INTERVAL_TEMPLATES = [
    # Simple (1-3)
    {"id": "int_simple_1", "template": "{event} lasted from {start_date} to {end_date}.", "clarity_score": 4.9, "readability_score": 85.3},
    {"id": "int_simple_2", "template": "{event} ran from {start_date} to {end_date}.", "clarity_score": 4.8, "readability_score": 84.1},
    {"id": "int_simple_3", "template": "From {start_date} to {end_date}, {event}.", "clarity_score": 4.8, "readability_score": 83.7},
    
    # Duration focus (4-6)
    {"id": "int_duration_1", "template": "{event} spanned {duration}, from {start_date} to {end_date}.", "clarity_score": 4.7, "readability_score": 80.2},
    {"id": "int_duration_2", "template": "Over {duration}, {event} unfolded ({start_date}-{end_date}).", "clarity_score": 4.6, "readability_score": 77.8},
    {"id": "int_duration_3", "template": "{event} continued for {duration} ({start_date}-{end_date}).", "clarity_score": 4.7, "readability_score": 81.4},
    
    # Bracket (7-9)
    {"id": "int_bracket_1", "template": "Between {start_date} and {end_date}, {event} occurred.", "clarity_score": 4.7, "readability_score": 79.5},
    {"id": "int_bracket_2", "template": "{start_date} to {end_date} was the period of {event}.", "clarity_score": 4.6, "readability_score": 78.1},
    {"id": "int_bracket_3", "template": "In the {duration} from {start_date} to {end_date}, {event}.", "clarity_score": 4.6, "readability_score": 76.9},
    
    # Progress (10-12)
    {"id": "int_progress_1", "template": "{event} began in {start_date} and ended in {end_date}.", "clarity_score": 4.7, "readability_score": 80.8},
    {"id": "int_progress_2", "template": "Starting {start_date}, {event} lasted until {end_date}.", "clarity_score": 4.7, "readability_score": 81.3},
    {"id": "int_progress_3", "template": "{event} started in {start_date}, ending in {end_date}.", "clarity_score": 4.6, "readability_score": 79.2},
    
    # Historical context (13-15)
    {"id": "int_context_1", "template": "History shows {event} from {start_date} to {end_date}.", "clarity_score": 4.5, "readability_score": 74.6},
    {"id": "int_context_2", "template": "The {duration}-long {event} ({start_date}-{end_date}) shaped history.", "clarity_score": 4.4, "readability_score": 71.3},
    {"id": "int_context_3", "template": "{event} marked the period {start_date}-{end_date}.", "clarity_score": 4.6, "readability_score": 76.8},
    
    # Impact (16-18)
    {"id": "int_impact_1", "template": "For {duration} ({start_date}-{end_date}), {event} dominated.", "clarity_score": 4.5, "readability_score": 73.9},
    {"id": "int_impact_2", "template": "{event} raged from {start_date} to {end_date}.", "clarity_score": 4.6, "readability_score": 78.2},
    {"id": "int_impact_3", "template": "Through {duration}, from {start_date} to {end_date}, {event}.", "clarity_score": 4.4, "readability_score": 72.5},
    
    # Temporal relation (19-20)
    {"id": "int_temporal_1", "template": "{event} stretched across {duration} ({start_date}-{end_date}).", "clarity_score": 4.5, "readability_score": 74.1},
    {"id": "int_temporal_2", "template": "{start_date}-{end_date}: the era of {event} ({duration}).", "clarity_score": 4.4, "readability_score": 71.8},
]


class IntervalTemplate:
    """Template for interval facts."""
    
    def __init__(self, template_id: str, template_string: str, confidence: float = 0.5):
        self.template_id = template_id
        self.template_string = template_string
        self.confidence = confidence
    
    def render(self, fact) -> str:
        """Render template with fact data."""
        content = fact.content
        
        # Validate required fields
        required = ["event", "start_date", "end_date"]
        for field in required:
            if field not in content:
                raise ValueError(f"Interval fact must have '{field}' field")
        
        format_dict = {
            "event": content.get("event", ""),
            "start_date": content.get("start_date", ""),
            "end_date": content.get("end_date", ""),
            "duration": content.get("duration", "this period"),
            "context": content.get("context", "")
        }
        
        try:
            output = self.template_string.format(**format_dict)
            return output
        except KeyError as e:
            raise ValueError(f"Missing field in fact: {e}")


class IntervalTemplateLibrary:
    """Manager for interval templates."""
    
    def __init__(self):
        """Initialize template library."""
        self.templates: Dict[str, IntervalTemplate] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all templates."""
        for template_spec in INTERVAL_TEMPLATES:
            template = IntervalTemplate(
                template_id=template_spec['id'],
                template_string=template_spec['template'],
                confidence=template_spec.get('clarity_score', 4.5) / 5.0
            )
            self.templates[template_spec['id']] = template
    
    def render(self, fact, template_id: str = None) -> str:
        """Render a fact with a specific template."""
        if not template_id:
            template_id = list(self.templates.keys())[0]
        
        if template_id not in self.templates:
            raise ValueError(f"Template not found: {template_id}")
        
        template = self.templates[template_id]
        return template.render(fact)
    
    def render_all(self, fact) -> Dict[str, str]:
        """Render with all applicable templates."""
        outputs = {}
        for template_id, template in self.templates.items():
            try:
                outputs[template_id] = template.render(fact)
            except Exception:
                pass
        return outputs