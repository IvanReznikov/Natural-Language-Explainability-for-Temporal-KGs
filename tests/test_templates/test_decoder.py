"""
Extended template coverage for precedence, containment, and recurrence libraries.
"""

import pytest

from temporal_nlg.templates.precedence import PrecedenceTemplateLibrary
from temporal_nlg.templates.containment import ContainmentTemplateLibrary
from temporal_nlg.templates.recurrence import RecurrenceTemplateLibrary


class DummyFact:
    def __init__(self, content):
        self.content = content


def _assert_nonempty(outputs):
    assert outputs
    assert all(outputs.values())


def test_precedence_render_all():
    lib = PrecedenceTemplateLibrary()
    fact = DummyFact({"first": "Launch", "second": "Docking"})
    outputs = lib.render_all(fact)
    _assert_nonempty(outputs)
    assert any("Launch" in o and "Docking" in o for o in outputs.values())


def test_containment_render_all():
    lib = ContainmentTemplateLibrary()
    fact = DummyFact({"inner": "Experiment", "outer": "Mission phase 1"})
    outputs = lib.render_all(fact)
    _assert_nonempty(outputs)
    assert all("Mission phase 1" in o for o in outputs.values())


def test_recurrence_render_all():
    lib = RecurrenceTemplateLibrary()
    fact = DummyFact(
        {"event": "Status check", "frequency": "week", "start_date": "2025", "end_date": "2026"}
    )
    outputs = lib.render_all(fact)
    _assert_nonempty(outputs)
    assert any("week" in o for o in outputs.values())


def test_missing_fields_raise():
    lib = PrecedenceTemplateLibrary()
    bad_fact = DummyFact({"first": "Only one"})
    with pytest.raises(ValueError):
        lib.render(bad_fact)
