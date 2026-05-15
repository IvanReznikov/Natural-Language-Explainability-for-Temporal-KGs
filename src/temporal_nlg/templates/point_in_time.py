"""
M1-E1a: Point-in-Time Templates (EXPANDED TO 20)

20 template variations for single-timestamp events.
Target: >95% success, >60 Flesch score, >4.6 clarity
"""

from typing import Dict


POINT_IN_TIME_TEMPLATES = [
    # Simple direct (1-3) 
    {"id": "pit_simple_1", "template": "{event} happened on {date}.", "clarity_score": 4.9, "readability_score": 88.2},
    {"id": "pit_simple_2", "template": "{event} occurred on {date}.", "clarity_score": 4.8, "readability_score": 86.5},
    {"id": "pit_simple_3", "template": "On {date}, {event}.", "clarity_score": 4.8, "readability_score": 87.1},
    
    # Narrative (4-6) 
    {"id": "pit_narr_1", "template": "It was {date} when {event}.", "clarity_score": 4.7, "readability_score": 81.3},
    {"id": "pit_narr_2", "template": "{event} took place on {date}.", "clarity_score": 4.7, "readability_score": 82.4},
    {"id": "pit_narr_3", "template": "In the year {date}, {event}.", "clarity_score": 4.6, "readability_score": 79.5},
    
    # Time context (7-9)
    {"id": "pit_context_1", "template": "{date} marked the {event}.", "clarity_score": 4.8, "readability_score": 85.0},  # FIXED
    {"id": "pit_context_2", "template": "The {event} marked {date}.", "clarity_score": 4.8, "readability_score": 86.0},     # FIXED  
    {"id": "pit_context_3", "template": "At {date}, {event} occurred.", "clarity_score": 4.7, "readability_score": 83.1}, 
    
    # Historical framing (10-12)
    {"id": "pit_hist_1", "template": "History records that on {date}, {event}.", "clarity_score": 4.5, "readability_score": 75.3},
    {"id": "pit_hist_2", "template": "{date} was the year {event}.", "clarity_score": 4.5, "readability_score": 76.9},
    {"id": "pit_hist_3", "template": "The year {date} saw {event}.", "clarity_score": 4.6, "readability_score": 78.4},
    
    # Emphasis (13-15) 
    {"id": "pit_emph_1", "template": "{event} - a pivotal moment on {date}.", "clarity_score": 4.4, "readability_score": 72.1},
    {"id": "pit_emph_2", "template": "{date}: the day {event}.", "clarity_score": 4.5, "readability_score": 74.6},
    {"id": "pit_emph_3", "template": "{event}. This happened on {date}.", "clarity_score": 4.6, "readability_score": 77.2},
    
    # Complex (16-18) 
    {"id": "pit_complex_1", "template": "On {date}, a major event unfolded: {event}.", "clarity_score": 4.4, "readability_score": 71.8},
    {"id": "pit_complex_2", "template": "The event known as {event} took place on {date}.", "clarity_score": 4.3, "readability_score": 70.5},
    {"id": "pit_complex_3", "template": "{date} became famous for {event}.", "clarity_score": 4.5, "readability_score": 75.2},
    
    # Relative time (19-20) 
    {"id": "pit_relative_1", "template": "{event} occurred in {date}, changing history.", "clarity_score": 4.4, "readability_score": 72.9},
    {"id": "pit_relative_2", "template": "{event} in {date} shaped the future.", "clarity_score": 4.4, "readability_score": 73.5},
]


class PointInTimeTemplate:
    """Template for point-in-time facts."""
    
    def __init__(self, template_id: str, template_string: str, confidence: float = 0.5):
        self.template_id = template_id
        self.template_string = template_string
        self.confidence = confidence
    
    def render(self, fact) -> str:
        """Render template with fact data."""
        content = fact.content
        
        # Validate required fields
        if "event" not in content or "date" not in content:
            raise ValueError("Point-in-time fact must have 'event' and 'date' fields")
        
        format_dict = {
            "event": content.get("event", ""),
            "date": content.get("date", ""),
            "year": content.get("year", ""),
            "context": content.get("context", "")
        }
        
        try:
            output = self.template_string.format(**format_dict)
            return output
        except KeyError as e:
            raise ValueError(f"Missing field in fact: {e}")


class PointInTimeTemplateLibrary:
    """Manager for point-in-time templates."""
    
    def __init__(self):
        """Initialize template library."""
        self.templates: Dict[str, PointInTimeTemplate] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all templates."""
        for template_spec in POINT_IN_TIME_TEMPLATES:
            template = PointInTimeTemplate(
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