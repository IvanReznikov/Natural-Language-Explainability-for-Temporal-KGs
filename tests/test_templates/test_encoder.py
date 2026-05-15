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
    return DummyFact({"event": "Mars mission", "start_date": "2025", "end_date": "2027", "duration": "2 years"})


@pytest.fixture
def sequence_fact():
    return DummyFact({"events": ["Launch", "Orbit", "Landing"], "timestamps": ["T0", "T1", "T2"], "time_span": "mission window"})


@pytest.fixture
def causality_fact():
    return DummyFact({"cause": "Solar flare", "effect": "Communications blackout", "certainty": "high"})


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
