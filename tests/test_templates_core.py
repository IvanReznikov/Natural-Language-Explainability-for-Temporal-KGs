"""Template core unit tests: primitives and renderer validation."""

from temporal_nlg.core.templates import TemplateType, TemporalFact, Template, TemplateRenderer


class DummyTemplate(Template):
    def render(self, fact: TemporalFact) -> str:
        return self.template_string.format(**fact.content)

    def is_applicable(self, fact: TemporalFact) -> bool:
        return True


def test_temporal_fact_validation():
    fact_valid = TemporalFact(TemplateType.POINT_IN_TIME, {"entity": "E", "event": "H", "date": "2000"})
    assert fact_valid.validate() is True

    fact_missing = TemporalFact(TemplateType.INTERVAL, {"entity": "E", "event": "H"})
    assert fact_missing.validate() is False


def test_template_placeholder_and_format_value():
    fact = TemporalFact(TemplateType.SEQUENCE, {"a": "alpha", "b": ["x", "y", "z"]})
    tmpl = DummyTemplate("dummy", "Values {a} and {b}")
    assert tmpl.placeholders == {"a", "b"}
    formatted = tmpl._format_value(fact.content["b"])
    assert formatted == "x, y, and z"


def test_template_renderer_selects_library():
    fact = TemporalFact(TemplateType.POINT_IN_TIME, {"event": "Launch", "date": "2024"})
    renderer = TemplateRenderer()
    text = renderer.render(fact)
    assert "2024" in text
    assert renderer.last_template_id is not None
