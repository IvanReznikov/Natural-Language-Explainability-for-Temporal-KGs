# temporal_nlg/core/templates.py
"""
Base template classes for temporal relationship verbalization.
Provides abstract template structure and rendering engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re


class TemplateType(Enum):
    """Enumeration of temporal relationship types."""
    POINT_IN_TIME = "point_in_time"
    INTERVAL = "interval"
    SEQUENCE = "sequence"
    CAUSALITY = "causality"
    OVERLAP = "overlap"


@dataclass(init=False)
class TemporalFact:
    """Represents a temporal fact to be verbalized."""
    fact_type: TemplateType
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]]

    def __init__(self, fact_type: TemplateType, content: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        self.fact_type = fact_type
        base_content = content.copy() if content else {}
        # Allow keyword fields (event, date, etc.) to populate content for convenience.
        base_content.update(kwargs)
        self.content = base_content
        self.metadata = metadata
    
    def validate(self) -> bool:
        """Validate that required fields are present."""
        required_fields = {
            TemplateType.POINT_IN_TIME: ['entity', 'event', 'date'],
            TemplateType.INTERVAL: ['entity', 'event', 'start_date', 'end_date'],
            TemplateType.SEQUENCE: ['events', 'timestamps'],
            TemplateType.CAUSALITY: ['cause', 'effect', 'temporal_relation'],
            TemplateType.OVERLAP: ['events', 'time_period']
        }
        
        required = required_fields.get(self.fact_type, [])
        return all(field in self.content for field in required)


class Template(ABC):
    """Abstract base class for temporal templates."""
    
    def __init__(self, template_id: str, template_string: str, confidence: float = 1.0):
        """
        Initialize template.
        
        Args:
            template_id: Unique identifier for template
            template_string: Template with {placeholder} variables
            confidence: Confidence score for this template (0-1)
        """
        self.template_id = template_id
        self.template_string = template_string
        self.confidence = confidence
        self._validate_placeholders()
    
    def _validate_placeholders(self) -> None:
        """Extract and validate placeholders in template."""
        self.placeholders = set(re.findall(r'\{(\w+)\}', self.template_string))
    
    @abstractmethod
    def render(self, fact: TemporalFact) -> str:
        """
        Render template with temporal fact data.
        
        Args:
            fact: TemporalFact instance with content
            
        Returns:
            Rendered natural language string
        """
        pass
    
    @abstractmethod
    def is_applicable(self, fact: TemporalFact) -> bool:
        """
        Check if template is applicable to this fact.
        
        Args:
            fact: TemporalFact to check
            
        Returns:
            True if template can render this fact
        """
        pass
    
    def _format_value(self, value: Any) -> str:
        """Format values for natural language output."""
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            if len(value) == 0:
                return ""
            elif len(value) == 1:
                return self._format_value(value[0])
            elif len(value) == 2:
                return f"{self._format_value(value[0])} and {self._format_value(value[1])}"
            else:
                formatted = ", ".join(self._format_value(v) for v in value[:-1])
                return f"{formatted}, and {self._format_value(value[-1])}"
        else:
            return str(value)


class PointInTimeTemplate(Template):
    """Template for point-in-time events."""
    
    def is_applicable(self, fact: TemporalFact) -> bool:
        """Check if fact is point-in-time type."""
        return fact.fact_type == TemplateType.POINT_IN_TIME
    
    def render(self, fact: TemporalFact) -> str:
        """Render point-in-time event."""
        if not self.is_applicable(fact):
            raise ValueError("Fact must be POINT_IN_TIME type")
        
        # Prepare context data
        context = {
            'entity': self._format_value(fact.content.get('entity', 'It')),
            'event': self._format_value(fact.content.get('event', 'happened')),
            'date': self._format_value(fact.content.get('date', '')),
            'location': self._format_value(fact.content.get('location', '')),
            'description': self._format_value(fact.content.get('description', ''))
        }
        
        try:
            return self.template_string.format(**context)
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")


class IntervalTemplate(Template):
    """Template for interval-based events."""
    
    def is_applicable(self, fact: TemporalFact) -> bool:
        """Check if fact is interval type."""
        return fact.fact_type == TemplateType.INTERVAL
    
    def render(self, fact: TemporalFact) -> str:
        """Render interval event."""
        if not self.is_applicable(fact):
            raise ValueError("Fact must be INTERVAL type")
        
        context = {
            'entity': self._format_value(fact.content.get('entity', 'It')),
            'event': self._format_value(fact.content.get('event', 'occurred')),
            'start_date': self._format_value(fact.content.get('start_date', '')),
            'end_date': self._format_value(fact.content.get('end_date', '')),
            'duration': self._format_value(fact.content.get('duration', '')),
            'frequency': self._format_value(fact.content.get('frequency', ''))
        }
        
        try:
            return self.template_string.format(**context)
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")


class SequenceTemplate(Template):
    """Template for event sequences."""
    
    def is_applicable(self, fact: TemporalFact) -> bool:
        """Check if fact is sequence type."""
        return fact.fact_type == TemplateType.SEQUENCE
    
    def render(self, fact: TemporalFact) -> str:
        """Render sequence of events."""
        if not self.is_applicable(fact):
            raise ValueError("Fact must be SEQUENCE type")
        
        # Format events with timestamps
        events = fact.content.get('events', [])
        timestamps = fact.content.get('timestamps', [])
        
        # Build event-time pairs
        event_strings = []
        for i, event in enumerate(events):
            if i < len(timestamps):
                event_strings.append(f"{event} ({timestamps[i]})")
            else:
                event_strings.append(event)
        
        context = {
            'events': ", then ".join(event_strings) if event_strings else "no events",
            'event_count': len(events),
            'time_span': self._format_value(fact.content.get('time_span', ''))
        }
        
        try:
            return self.template_string.format(**context)
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")


class CausalityTemplate(Template):
    """Template for causal relationships."""
    
    def is_applicable(self, fact: TemporalFact) -> bool:
        """Check if fact is causality type."""
        return fact.fact_type == TemplateType.CAUSALITY
    
    def render(self, fact: TemporalFact) -> str:
        """Render causal relationship."""
        if not self.is_applicable(fact):
            raise ValueError("Fact must be CAUSALITY type")
        
        context = {
            'cause': self._format_value(fact.content.get('cause', 'Event A')),
            'effect': self._format_value(fact.content.get('effect', 'Event B')),
            'temporal_relation': self._format_value(fact.content.get('temporal_relation', 'caused')),
            'mechanism': self._format_value(fact.content.get('mechanism', '')),
            'certainty': self._format_value(fact.content.get('certainty', 'certainly'))
        }
        
        try:
            return self.template_string.format(**context)
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")


class OverlapTemplate(Template):
    """Template for overlapping events."""
    
    def is_applicable(self, fact: TemporalFact) -> bool:
        """Check if fact is overlap type."""
        return fact.fact_type == TemplateType.OVERLAP
    
    def render(self, fact: TemporalFact) -> str:
        """Render overlapping events."""
        if not self.is_applicable(fact):
            raise ValueError("Fact must be OVERLAP type")
        
        events = fact.content.get('events', [])
        context = {
            'event_count': len(events),
            'events': " and ".join(self._format_value(e) for e in events),
            'time_period': self._format_value(fact.content.get('time_period', '')),
            'simultaneity': self._format_value(fact.content.get('simultaneity', 'concurrently'))
        }
        
        try:
            return self.template_string.format(**context)
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")


class TemplateRenderer:
    """Dispatch facts to the appropriate template library."""

    def __init__(self):
        from ..templates.point_in_time import PointInTimeTemplateLibrary
        from ..templates.intervals import IntervalTemplateLibrary
        from ..templates.sequences import SequenceTemplateLibrary
        from ..templates.causality import CausalityTemplateLibrary
        from ..templates.overlaps import OverlapTemplateLibrary

        self.libraries = {
            TemplateType.POINT_IN_TIME: PointInTimeTemplateLibrary(),
            TemplateType.INTERVAL: IntervalTemplateLibrary(),
            TemplateType.SEQUENCE: SequenceTemplateLibrary(),
            TemplateType.CAUSALITY: CausalityTemplateLibrary(),
            TemplateType.OVERLAP: OverlapTemplateLibrary(),
        }
        self.last_template_id: Optional[str] = None

    def render(self, fact: TemporalFact, template_id: Optional[str] = None) -> str:
        """Render a fact using the matching library."""
        library = self._select_library(fact)
        chosen_template = template_id or next(iter(library.templates.keys()))
        output = library.render(fact, template_id=template_id)
        self.last_template_id = template_id or chosen_template
        return output

    def _select_library(self, fact: TemporalFact):
        if fact.fact_type not in self.libraries:
            raise ValueError(f"Unsupported fact type: {fact.fact_type}")
        return self.libraries[fact.fact_type]
