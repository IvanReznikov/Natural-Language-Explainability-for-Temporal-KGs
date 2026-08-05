"""
M1-E1a: Point-in-Time Templates (EXPANDED TO 20)

20 template variations for single-timestamp events.
Target: >95% success, >60 Flesch score, >4.6 clarity

Rendering convention
--------------------
Facts carry ``entity`` (the subject) and ``event`` (what happened). Real data
(see ``temporal_nlg/data/loaders.py`` and the M3 graph facts) overwhelmingly
uses **verb-phrase** events (``"was born"``, ``"docked with ISS"``) together
with an entity, so templates render a full clause ``{event_clause}`` built as:

* entity + event          -> ``"Albert Einstein was born"``   (canonical shape)
* bare verb-phrase event  -> ``"docked with ISS"``            (no subject known)
* noun-phrase event       -> ``"the Moon landing happened"``  (given a verb)

and a date preposition ``{date_prep}`` (``in`` for years/year-months/quarters,
``on`` for full dates). Every template stays grammatical for both event shapes:
the entity is always rendered when present, and verb-phrase events are never
glued onto "happened".
"""

import re
from typing import Dict

# Heuristic verb-phrase detector (first token): auxiliaries, or a word of 4+
# letters ending in -ed/-ing ("docked", "refused", "delivered", "landing").
# Irregular past verbs ("won", "began", "fell") are not detected; they only
# matter for entity-less facts, where they fall back to the noun rendering.
_VERB_PHRASE_RE = re.compile(
    r"^(?:was|were|is|are|has|have|had|did|does|do|will|would|can|could|shall|"
    r"should|may|might|must)\b|^\w{4,}(?:ed|ing)\b",
    re.IGNORECASE,
)

_MONTH_YEAR_RE = re.compile(
    r"^(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{4}$",
    re.IGNORECASE,
)


def _date_preposition(date: str) -> str:
    """Return ``in`` for coarse dates (years, year-months, quarters), else ``on``."""
    d = (date or "").strip()
    if re.fullmatch(r"\d{4}", d):  # 1879
        return "in"
    if re.fullmatch(r"\d{4}-\d{2}", d):  # 2025-02
        return "in"
    if _MONTH_YEAR_RE.fullmatch(d):  # July 1960
        return "in"
    if re.fullmatch(r"[Qq][1-4]\s*\d{4}", d):  # Q4 2025
        return "in"
    if re.fullmatch(r"\d+(?:st|nd|rd|th)\s+century", d, re.IGNORECASE):  # 14th century
        return "in"
    if re.fullmatch(r"\d+\s*(?:BCE|BC|CE|AD)", d, re.IGNORECASE):  # 3100 BCE
        return "in"
    return "on"  # December 10, 1903 / 1969-07-20 / ...


def _event_clause(entity: str, event: str) -> str:
    """Build a renderable clause from entity + event (see module docstring)."""
    entity = (entity or "").strip()
    event = (event or "").strip()
    if entity:
        # Canonical real-data shape: entity + verb-phrase predicate.
        return f"{entity} {event}"
    if _VERB_PHRASE_RE.search(event):
        return event
    # Noun phrase with no subject: give it a generic verb to form a clause.
    return f"the {event} happened"


POINT_IN_TIME_TEMPLATES = [
    # Simple direct (1-3)
    {
        "id": "pit_simple_1",
        "template": "{event_clause} {date_prep} {date}.",
        "clarity_score": 4.9,
        "readability_score": 88.2,
    },
    {
        "id": "pit_simple_2",
        "template": "It was {date_prep} {date} when {event_clause}.",
        "clarity_score": 4.8,
        "readability_score": 86.5,
    },
    {
        "id": "pit_simple_3",
        "template": "{date_prep} {date}, {event_clause}.",
        "clarity_score": 4.8,
        "readability_score": 87.1,
    },
    # Narrative (4-6)
    {
        "id": "pit_narr_1",
        "template": "It was {date} when {event_clause}.",
        "clarity_score": 4.7,
        "readability_score": 81.3,
    },
    {
        "id": "pit_narr_2",
        "template": "It happened {date_prep} {date}: {event_clause}.",
        "clarity_score": 4.7,
        "readability_score": 82.4,
    },
    {
        "id": "pit_narr_3",
        "template": "{date_prep} the year {date}, {event_clause}.",
        "clarity_score": 4.6,
        "readability_score": 79.5,
    },
    # Time context (7-9)
    {
        "id": "pit_context_1",
        "template": "{date} was when {event_clause}.",
        "clarity_score": 4.8,
        "readability_score": 85.0,
    },
    {
        "id": "pit_context_2",
        "template": "The significance of {date}: {event_clause}.",
        "clarity_score": 4.8,
        "readability_score": 86.0,
    },
    {
        "id": "pit_context_3",
        "template": "Back {date_prep} {date}, {event_clause}.",
        "clarity_score": 4.7,
        "readability_score": 83.1,
    },
    # Historical framing (10-12)
    {
        "id": "pit_hist_1",
        "template": "History records that {date_prep} {date}, {event_clause}.",
        "clarity_score": 4.5,
        "readability_score": 75.3,
    },
    {
        "id": "pit_hist_2",
        "template": "{date} was the year {event_clause}.",
        "clarity_score": 4.5,
        "readability_score": 76.9,
    },
    {
        "id": "pit_hist_3",
        "template": "The year {date} saw this: {event_clause}.",
        "clarity_score": 4.6,
        "readability_score": 78.4,
    },
    # Emphasis (13-15)
    {
        "id": "pit_emph_1",
        "template": "{event_clause} — a pivotal moment {date_prep} {date}.",
        "clarity_score": 4.4,
        "readability_score": 72.1,
    },
    {
        "id": "pit_emph_2",
        "template": "{date}: the day {event_clause}.",
        "clarity_score": 4.5,
        "readability_score": 74.6,
    },
    {
        "id": "pit_emph_3",
        "template": "{event_clause}. It took place {date_prep} {date}.",
        "clarity_score": 4.6,
        "readability_score": 77.2,
    },
    # Complex (16-18)
    {
        "id": "pit_complex_1",
        "template": "{date_prep} {date}, a major event unfolded: {event_clause}.",
        "clarity_score": 4.4,
        "readability_score": 71.8,
    },
    {
        "id": "pit_complex_2",
        "template": "What took place {date_prep} {date} was this: {event_clause}.",
        "clarity_score": 4.3,
        "readability_score": 70.5,
    },
    {
        "id": "pit_complex_3",
        "template": "{date} became famous as the time {event_clause}.",
        "clarity_score": 4.5,
        "readability_score": 75.2,
    },
    # Relative time (19-20)
    {
        "id": "pit_relative_1",
        "template": "{event_clause} {date_prep} {date}, changing history.",
        "clarity_score": 4.4,
        "readability_score": 72.9,
    },
    {
        "id": "pit_relative_2",
        "template": "When {event_clause} {date_prep} {date}, the future shifted.",
        "clarity_score": 4.4,
        "readability_score": 73.5,
    },
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

        event = str(content.get("event", ""))
        date = str(content.get("date", ""))
        entity = str(content.get("entity", "") or "")

        format_dict = {
            "entity": entity,
            "event": event,
            "event_clause": _event_clause(entity, event),
            "date": date,
            "date_prep": _date_preposition(date),
            "year": content.get("year", ""),
            "context": content.get("context", ""),
        }

        try:
            output = self.template_string.format(**format_dict)
        except KeyError as e:
            raise ValueError(f"Missing field in fact: {e}")
        # Capitalize the sentence start (clauses may begin lowercase, e.g. a
        # bare verb-phrase event like "docked with ISS").
        return output[:1].upper() + output[1:] if output else output


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
                template_id=template_spec["id"],
                template_string=template_spec["template"],
                confidence=template_spec.get("clarity_score", 4.5) / 5.0,
            )
            self.templates[template_spec["id"]] = template

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
