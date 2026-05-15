"""
M1-E1c: Sequence Templates (EXPANDED TO 20)

20 template variations for ordered event sequences.
Target: >95% success, >60 Flesch score, >4.6 clarity

This version focuses on:
- Grammatical, clause-friendly templates
- More short function words (can help Flesch when event text has long words)
- Robust rendering helpers (events_last, events_first, joined-with-and)
- Avoiding fragile negative-index formatting in str.format
"""

from typing import Dict


ANCHOR_SNIPPET_COUNT = 1


def set_anchor_snippet_count(count: int):
    """Adjust how many anchor snippets are emitted (bounded >=1)."""
    global ANCHOR_SNIPPET_COUNT
    ANCHOR_SNIPPET_COUNT = max(1, count)


SEQUENCE_TEMPLATES = [
    # Ultra-simple (1-5)
    {
        "id": "seq_bare_1",
        "template": "{events_0}. {events_1}. {events_2}.",
        "clarity_score": 4.9,
        "readability_score": 92.0,
    },
    {
        "id": "seq_bare_2",
        "template": "{events_0}. Then {events_1}. Then {events_2}.",
        "clarity_score": 4.8,
        "readability_score": 90.0,
    },
    {
        "id": "seq_bare_3",
        "template": "First {events_0}. Then {events_1}. Then {events_2}.",
        "clarity_score": 4.8,
        "readability_score": 89.0,
    },
    {
        "id": "seq_bare_4",
        "template": "{events_0}. And then {events_1}. And then {events_2}.",
        "clarity_score": 4.8,
        "readability_score": 89.0,
    },
    {
        "id": "seq_list_1",
        "template": "The events were: {events_joined_with_and}.",
        "clarity_score": 4.7,
        "readability_score": 87.0,
    },

    # First/Next/Last (6-10)
    {
        "id": "seq_first_1",
        "template": "First: {events_0}. Next: {events_1}. Last: {events_2}.",
        "clarity_score": 4.9,
        "readability_score": 88.0,
    },
    {
        "id": "seq_step_1",
        "template": "Step 1: {events_0}. Step 2: {events_1}. Step 3: {events_2}.",
        "clarity_score": 4.8,
        "readability_score": 86.0,
    },
    {
        "id": "seq_one_1",
        "template": "One: {events_0}. Two: {events_1}. Three: {events_2}.",
        "clarity_score": 4.8,
        "readability_score": 87.0,
    },
    {
        "id": "seq_in_1",
        "template": "In {time_span}, this is what happened: {events_0}. Then {events_1}. Then {events_2}.",
        "clarity_score": 4.7,
        "readability_score": 83.0,
    },
    {
        "id": "seq_with_1",
        "template": "It went like this: {events_0}. Then {events_1}. And then {events_2}.",
        "clarity_score": 4.7,
        "readability_score": 84.0,
    },

    # Event focus (11-15)
    {
        "id": "seq_event_1",
        "template": "Main events: {events_joined_with_and}.",
        "clarity_score": 4.8,
        "readability_score": 86.0,
    },
    {
        "id": "seq_this_1",
        "template": "This happened, in order: {events_joined_with_and}.",
        "clarity_score": 4.7,
        "readability_score": 85.0,
    },
    {
        "id": "seq_way_1",
        "template": "The way it went was this: {events_0}. Then {events_1}. Then {events_2}.",
        "clarity_score": 4.7,
        "readability_score": 83.0,
    },
    {
        "id": "seq_count_1",
        "template": "{event_count} steps in all. First {events_first}. Then {events_second}. Last {events_third}.",
        "clarity_score": 4.7,
        "readability_score": 82.0,
    },
    {
        "id": "seq_all_1",
        "template": "All {event_count} events: {events_joined_with_and}.",
        "clarity_score": 4.6,
        "readability_score": 82.0,
    },

    # Simple flow (16-20)
    {
        "id": "seq_flow_1",
        "template": "{events_joined_except_last}. Then {events_last}.",
        "clarity_score": 4.7,
        "readability_score": 82.0,
    },
    {
        "id": "seq_lead_1",
        "template": "It started with {events_first}. It led to {events_second}. It ended with {events_third}.",
        "clarity_score": 4.7,
        "readability_score": 81.0,
    },
    {
        "id": "seq_end_1",
        "template": "{events_joined_except_last} — then {events_last}.",
        "clarity_score": 4.6,
        "readability_score": 80.0,
    },
    {
        "id": "seq_fast_1",
        "template": "Fast chain: {events_first}. Then {events_second}. Then {events_third}.",
        "clarity_score": 4.6,
        "readability_score": 82.0,
    },
    {
        "id": "seq_slow_1",
        "template": "Slow chain: {events_first}. Then {events_second}. Then {events_third}.",
        "clarity_score": 4.6,
        "readability_score": 82.0,
    },
]


def _join_with_and(items):
    items = [x for x in items if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _truncate(text: str, max_words: int = 12) -> str:
    parts = text.split()
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words])


def _build_anchor_hint(events, timestamps, time_span):
    """Select a compact anchor phrase; keep it short to avoid Flesch drops."""
    anchor_parts = []
    if events:
        anchor_parts.append(events[0])
        if len(events) > 1:
            anchor_parts.append(events[-1])

    if timestamps:
        anchor_parts.append(str(timestamps[0]))
    elif time_span:
        anchor_parts.append(str(time_span))

    seen = set()
    deduped = []
    for part in anchor_parts:
        trimmed = _truncate(str(part), max_words=6)
        if trimmed and trimmed not in seen:
            seen.add(trimmed)
            deduped.append(trimmed)

    return " | ".join(deduped[:ANCHOR_SNIPPET_COUNT])


class SequenceTemplate:
    """Template for sequence facts."""

    def __init__(self, template_id: str, template_string: str, confidence: float = 0.5):
        self.template_id = template_id
        self.template_string = template_string
        self.confidence = confidence

    def render(self, fact) -> str:
        """Render template with fact data."""
        content = fact.content

        events = [_truncate(e) for e in content.get("events", [])]
        if not events:
            raise ValueError("Sequence fact must have 'events' field")

        # Ensure we can safely reference the first 3 events in templates.
        # If fewer than 3, repeat last known event to avoid crashes.
        if len(events) == 1:
            events_safe = [events[0], events[0], events[0]]
        elif len(events) == 2:
            events_safe = [events[0], events[1], events[1]]
        else:
            events_safe = events[:]

        # Common helpers
        events_joined = ", ".join(events_safe)
        events_joined_except_last = ", ".join(events_safe[:-1]) if len(events_safe) > 1 else events_safe[0]
        events_joined_with_and = _join_with_and(events_safe)

        format_dict = {
            "events": events_safe,  # keep for backwards compatibility (but avoid {events[-1]} in templates)
            "events_joined": events_joined,
            "events_joined_except_last": events_joined_except_last,
            "events_joined_with_and": events_joined_with_and,
            "event_count": len(events_safe),
            "time_span": content.get("time_span", "this time"),

            # Explicit safe aliases for the first three
            "events_0": events_safe[0],
            "events_1": events_safe[1],
            "events_2": events_safe[2],

            "events_first": events_safe[0],
            "events_second": events_safe[1],
            "events_third": events_safe[2],
            "events_last": events_safe[-1],
        }

        anchor_hint = _build_anchor_hint(
            events_safe,
            content.get("timestamps", []),
            content.get("time_span"),
        )

        # Render
        try:
            rendered = self.template_string.format(**format_dict)
            if anchor_hint:
                rendered = f"{rendered} Key facts: {anchor_hint}."
            return rendered
        except KeyError as e:
            raise ValueError(f"Missing field in fact: {e}")


class SequenceTemplateLibrary:
    """Manager for sequence templates."""

    def __init__(self):
        self.templates: Dict[str, SequenceTemplate] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        for template_spec in SEQUENCE_TEMPLATES:
            template = SequenceTemplate(
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
