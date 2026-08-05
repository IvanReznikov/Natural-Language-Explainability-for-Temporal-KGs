"""
Template library rendering coverage.
"""

import pytest

from temporal_nlg.templates.point_in_time import PointInTimeTemplateLibrary
from temporal_nlg.templates.intervals import IntervalTemplateLibrary
from temporal_nlg.templates.sequences import SequenceTemplateLibrary
from temporal_nlg.templates.causality import CausalityTemplateLibrary
from temporal_nlg.templates.overlaps import OverlapTemplateLibrary


class DummyFact:
    def __init__(self, content):
        self.content = content


@pytest.fixture
def pit_fact():
    return DummyFact({"event": "Moon landing", "date": "1969", "year": "1969"})


@pytest.fixture
def interval_fact():
    return DummyFact(
        {"event": "Mars mission", "start_date": "2025", "end_date": "2027", "duration": "2 years"}
    )


@pytest.fixture
def sequence_fact():
    return DummyFact(
        {
            "events": ["Launch", "Orbit", "Landing"],
            "timestamps": ["T0", "T1", "T2"],
            "time_span": "mission window",
        }
    )


@pytest.fixture
def causality_fact():
    return DummyFact(
        {"cause": "Solar flare", "effect": "Communications blackout", "certainty": "high"}
    )


@pytest.fixture
def overlap_fact():
    return DummyFact({"events": ["Conference", "Product launch"], "time_period": "Q4 2025"})


def _assert_all_nonempty(outputs):
    assert outputs, "Expected non-empty render outputs"
    assert all(outputs.values()), "All templates should render non-empty strings"


def test_point_in_time_render_all(pit_fact):
    lib = PointInTimeTemplateLibrary()
    outputs = lib.render_all(pit_fact)
    assert len(outputs) == len(lib.templates)
    _assert_all_nonempty(outputs)
    assert all("moon" in o.lower() for o in outputs.values())
    assert all("1969" in o for o in outputs.values())


def test_interval_render_all(interval_fact):
    lib = IntervalTemplateLibrary()
    outputs = lib.render_all(interval_fact)
    assert len(outputs) == len(lib.templates)
    _assert_all_nonempty(outputs)
    assert all("2025" in o and "2027" in o for o in outputs.values())


def test_sequence_render_all(sequence_fact):
    lib = SequenceTemplateLibrary()
    outputs = lib.render_all(sequence_fact)
    assert len(outputs) == len(lib.templates)
    _assert_all_nonempty(outputs)
    assert any("Launch" in o for o in outputs.values())
    assert any("Landing" in o for o in outputs.values())


def test_causality_render_all(causality_fact):
    lib = CausalityTemplateLibrary()
    outputs = lib.render_all(causality_fact)
    assert len(outputs) == len(lib.templates)
    _assert_all_nonempty(outputs)
    assert all("Solar flare" in o for o in outputs.values())
    assert all("Communications blackout" in o for o in outputs.values())


def test_overlap_render_all(overlap_fact):
    lib = OverlapTemplateLibrary()
    outputs = lib.render_all(overlap_fact)
    assert len(outputs) == len(lib.templates)
    _assert_all_nonempty(outputs)
    assert all("Conference" in o and "Product launch" in o for o in outputs.values())
    assert any("Q4 2025" in o for o in outputs.values())


def test_missing_required_field_raises():
    lib = PointInTimeTemplateLibrary()
    bad_fact = DummyFact({"event": "Missing date"})
    with pytest.raises(ValueError):
        lib.render(bad_fact)


# ── Regression: verb-phrase events + entity (the canonical real-data shape) ──


def test_verb_phrase_event_with_entity_renders_grammatically():
    # Ensures entity is rendered and verb-phrase events are not glued onto "happened".
    lib = PointInTimeTemplateLibrary()
    fact = DummyFact({"entity": "Albert Einstein", "event": "was born", "date": "1879"})
    assert lib.render(fact) == "Albert Einstein was born in 1879."
    outputs = lib.render_all(fact)
    assert all("Albert Einstein" in o for o in outputs.values())
    assert all("1879" in o for o in outputs.values())
    assert all("was born happened" not in o for o in outputs.values())


def test_verb_phrase_event_full_date_uses_on():
    lib = PointInTimeTemplateLibrary()
    fact = DummyFact(
        {"entity": "SpaceX Crew Dragon", "event": "docked with ISS", "date": "May 31, 2020"}
    )
    assert lib.render(fact) == "SpaceX Crew Dragon docked with ISS on May 31, 2020."


def test_entityless_verb_phrase_not_glued_to_happened():
    lib = PointInTimeTemplateLibrary()
    fact = DummyFact({"event": "docked with ISS", "date": "May 31, 2020"})
    out = lib.render(fact)
    assert "docked with ISS happened" not in out
    assert out == "Docked with ISS on May 31, 2020."


def test_noun_phrase_event_without_entity_gets_clause_verb():
    lib = PointInTimeTemplateLibrary()
    fact = DummyFact({"event": "Moon landing", "date": "1969"})
    assert lib.render(fact) == "The Moon landing happened in 1969."


# ── Regression: interval entity/subject handling ────────────────────────────


def test_interval_entity_is_subject_and_verb_not_doubled():
    # Ensures the entity is the subject and the duration verb is not doubled.
    lib = IntervalTemplateLibrary()
    fact = DummyFact(
        {
            "entity": "World War II",
            "event": "lasted",
            "start_date": "September 1, 1939",
            "end_date": "September 2, 1945",
            "duration": "6 years",
        }
    )
    assert lib.render(fact) == "World War II lasted from September 1, 1939 to September 2, 1945."
    outputs = lib.render_all(fact)
    assert all("World War II" in o for o in outputs.values())
    assert all("lasted lasted" not in o for o in outputs.values())


def test_interval_noun_event_without_entity():
    lib = IntervalTemplateLibrary()
    fact = DummyFact(
        {"event": "Mars mission", "start_date": "2025", "end_date": "2027", "duration": "2 years"}
    )
    assert lib.render(fact) == "The Mars mission lasted from 2025 to 2027."


def test_interval_progress_template_uses_date_prepositions():
    lib = IntervalTemplateLibrary()
    fact = DummyFact(
        {"entity": "Victorian Era", "event": "defined", "start_date": "1837", "end_date": "1901"}
    )
    out = lib.render(fact, template_id="int_progress_1")
    assert out == "Victorian Era began in 1837 and ended in 1901."
    fact_full = DummyFact(
        {
            "entity": "Cold War",
            "event": "persisted",
            "start_date": "March 12, 1947",
            "end_date": "December 3, 1989",
        }
    )
    out_full = lib.render(fact_full, template_id="int_progress_1")
    assert out_full == "Cold War began on March 12, 1947 and ended on December 3, 1989."
