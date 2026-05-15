"""
Integration tests for generators and template routing.
"""

import pytest
from unittest.mock import Mock, patch

from temporal_nlg.models import HybridGenerator, GenerationResult, LLMGenerator
from temporal_nlg.core.templates import TemporalFact, TemplateType, TemplateRenderer


class TestHybridGenerator:
    """Template/LLM hybrid generator flows."""

    def test_initialization(self):
        generator = HybridGenerator()
        assert generator.model == "gpt-4.1-nano"
        assert generator.polish_threshold == 0.7
        assert generator.enable_caching is True

    def test_initialization_custom_params(self):
        generator = HybridGenerator(model="gpt-4", polish_threshold=0.8, enable_caching=False)
        assert generator.model == "gpt-4"
        assert generator.polish_threshold == 0.8
        assert generator.enable_caching is False

    def test_route_fact_simple(self):
        generator = HybridGenerator()
        fact = Mock()
        fact.event = "birth"
        fact.date = "1879"
        strategy = generator._route_fact(fact)
        assert strategy == "template"

    def test_route_fact_medium_complexity(self):
        generator = HybridGenerator()
        fact = Mock()
        fact.start_date = "1879"
        fact.end_date = "1955"
        fact.context = "A very long context that exceeds fifty characters for testing"
        strategy = generator._route_fact(fact)
        assert strategy in ["polish", "llm"]

    def test_route_fact_high_complexity(self):
        generator = HybridGenerator()
        fact = Mock()
        fact.events = ["event1", "event2", "event3", "event4"]
        fact.start_date = "1879"
        fact.end_date = "1955"
        strategy = generator._route_fact(fact)
        assert strategy == "llm"

    @patch("temporal_nlg.models.hybrid_generator.TemplateRenderer")
    def test_generate_template_strategy(self, mock_renderer_class):
        mock_renderer = Mock()
        mock_renderer.render.return_value = "Einstein was born in 1879."
        mock_renderer.last_template_id = "point_in_time_01"
        mock_renderer_class.return_value = mock_renderer

        generator = HybridGenerator()
        generator.template_renderer = mock_renderer

        fact = Mock()
        result = generator.generate(fact, force_strategy="template")

        assert isinstance(result, GenerationResult)
        assert result.strategy == "template"
        assert result.text == "Einstein was born in 1879."
        assert result.confidence == 0.9
        assert result.template_id == "point_in_time_01"

    @patch("temporal_nlg.models.hybrid_generator.ChatOpenAI")
    @patch("temporal_nlg.models.hybrid_generator.TemplateRenderer")
    def test_generate_polished_strategy(self, mock_renderer_class, mock_chat):
        mock_renderer = Mock()
        mock_renderer.render.return_value = "Original template text."
        mock_renderer.last_template_id = "test_01"
        mock_renderer_class.return_value = mock_renderer

        mock_response = Mock()
        mock_response.content = "Polished template text."
        mock_polisher = Mock()
        mock_polisher.invoke.return_value = mock_response

        generator = HybridGenerator()
        generator.template_renderer = mock_renderer
        generator.llm_polisher = mock_polisher

        fact = Mock()
        result = generator.generate(fact, force_strategy="polish")

        assert isinstance(result, GenerationResult)
        assert result.strategy == "polished"
        assert "Polished" in result.text
        assert result.confidence == 0.85

    @patch("temporal_nlg.models.hybrid_generator.LLMGenerator")
    def test_generate_llm_strategy(self, mock_llm_class):
        mock_llm = Mock()
        mock_llm.generate.return_value = "Pure LLM output."
        mock_llm_class.return_value = mock_llm

        generator = HybridGenerator()
        generator.llm_generator = mock_llm

        fact = Mock()
        result = generator.generate(fact, force_strategy="llm")

        assert isinstance(result, GenerationResult)
        assert result.strategy == "llm"
        assert result.text == "Pure LLM output."
        assert result.confidence == 0.7

    def test_get_cache_key(self):
        generator = HybridGenerator()
        fact = Mock()
        fact.__str__ = Mock(return_value="test_fact")
        key1 = generator._get_cache_key(fact)
        key2 = generator._get_cache_key(fact)
        assert key1 == key2
        assert isinstance(key1, str)

    @patch("temporal_nlg.models.hybrid_generator.TemplateRenderer")
    def test_caching_enabled(self, mock_renderer_class):
        mock_renderer = Mock()
        mock_renderer.render.return_value = "Cached text"
        mock_renderer.last_template_id = "test_01"
        mock_renderer_class.return_value = mock_renderer

        generator = HybridGenerator(enable_caching=True)
        generator.template_renderer = mock_renderer

        fact = Mock()
        fact.__str__ = Mock(return_value="same_fact")

        result1 = generator.generate(fact, force_strategy="template")
        result2 = generator.generate(fact, force_strategy="template")

        assert mock_renderer.render.call_count == 1
        assert result1.text == result2.text

    @patch("temporal_nlg.models.hybrid_generator.TemplateRenderer")
    def test_batch_generate(self, mock_renderer_class):
        mock_renderer = Mock()
        mock_renderer.render.return_value = "Test output"
        mock_renderer.last_template_id = "test_01"
        mock_renderer_class.return_value = mock_renderer

        generator = HybridGenerator()
        generator.template_renderer = mock_renderer

        facts = [Mock() for _ in range(3)]
        results = generator.batch_generate(facts)

        assert len(results) == 3
        assert all(isinstance(r, GenerationResult) for r in results)

    def test_get_stats(self):
        generator = HybridGenerator(model="gpt-4", polish_threshold=0.8)
        generator._template_cache = {"key1": "value1", "key2": "value2"}

        stats = generator.get_stats()

        assert stats["cache_size"] == 2
        assert stats["model"] == "gpt-4"
        assert stats["polish_threshold"] == 0.8


class TestLLMGenerator:
    """LLM generator flows."""

    def test_initialization(self):
        generator = LLMGenerator(model="gpt-4.1-nano")
        assert generator.model == "gpt-4.1-nano"
        assert generator.temperature == 0.0
        assert generator.max_tokens == 50

    def test_initialization_custom_params(self):
        generator = LLMGenerator(model="gpt-4", temperature=0.5, max_tokens=100)
        assert generator.model == "gpt-4"
        assert generator.temperature == 0.5
        assert generator.max_tokens == 100

    @patch("temporal_nlg.models.llm_generator.ChatOpenAI")
    def test_generate_success(self, mock_chat):
        mock_response = Mock()
        mock_response.content = "Einstein was born in 1879 in Ulm, Germany."
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        generator = LLMGenerator()
        fact = TemporalFact(
            fact_type=TemplateType.POINT_IN_TIME,
            event="birth",
            entity="Albert Einstein",
            date="1879",
            context="physicist",
        )

        result = generator.generate(fact)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Einstein" in result or "1879" in result

    @patch("temporal_nlg.models.llm_generator.ChatOpenAI")
    def test_generate_failure(self, mock_chat):
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.side_effect = Exception("API Error")
        mock_chat.return_value = mock_llm_instance

        generator = LLMGenerator()
        fact = Mock()

        with pytest.raises(ValueError, match="LLM generation failed"):
            generator.generate(fact)

    def test_extract_fact_type_enum(self):
        generator = LLMGenerator()
        fact = Mock()
        fact.fact_type = Mock()
        fact.fact_type.value = "point_in_time"
        result = generator._extract_fact_type(fact)
        assert result == "point_in_time"

    def test_extract_fact_type_string(self):
        generator = LLMGenerator()
        fact = Mock()
        fact.fact_type = "TemplateType.INTERVALS"
        result = generator._extract_fact_type(fact)
        assert "INTERVALS" in result or "intervals" in result.lower()

    def test_format_fact_details(self):
        generator = LLMGenerator()
        fact = Mock()
        fact.event = "birth"
        fact.entity = "Einstein"
        fact.date = "1879"
        fact.context = "physicist"
        result = generator._format_fact_details(fact)
        assert "birth" in result
        assert "Einstein" in result
        assert "1879" in result
        assert "physicist" in result

    @patch("temporal_nlg.models.llm_generator.ChatOpenAI")
    def test_batch_generate(self, mock_chat):
        mock_response = Mock()
        mock_response.content = "Test output"
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_llm_instance

        generator = LLMGenerator()
        facts = [Mock() for _ in range(3)]
        for fact in facts:
            fact.fact_type = TemplateType.POINT_IN_TIME
            fact.event = "test"

        results = generator.batch_generate(facts)

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    @patch("temporal_nlg.models.llm_generator.ChatOpenAI")
    def test_batch_generate_with_errors(self, mock_chat):
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.side_effect = [Mock(content="Success 1"), Exception("API Error"), Mock(content="Success 2")]
        mock_chat.return_value = mock_llm_instance

        generator = LLMGenerator()
        facts = [Mock() for _ in range(3)]
        for fact in facts:
            fact.fact_type = TemplateType.POINT_IN_TIME

        results = generator.batch_generate(facts)

        assert len(results) == 3
        assert "Success 1" in results[0]
        assert "[ERROR:" in results[1]
        assert "Success 2" in results[2]


class StubResponse:
    def __init__(self, content: str):
        self.content = content


class StubLLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, payload):
        return StubResponse(self.content)


class StubLLMGenerator:
    def __init__(self, text: str):
        self._text = text

    def generate(self, fact):
        return self._text


def test_llm_generator_with_stub_llm():
    fact = TemporalFact(TemplateType.POINT_IN_TIME, {"event": "Launch", "date": "2024"})
    gen = LLMGenerator(llm=StubLLM("stub text"))
    text = gen.generate(fact)
    assert text == "stub text"


def test_hybrid_generator_routes_and_polishes_with_stubs():
    fact_simple = TemporalFact(TemplateType.POINT_IN_TIME, {"event": "Launch", "date": "2024"})
    renderer = TemplateRenderer()
    hybrid = HybridGenerator(
        model="stub-model",
        llm_polisher=StubLLM("polished"),
        llm_generator=StubLLMGenerator("llm text"),
        template_renderer=renderer,
    )

    template_result = hybrid.generate(fact_simple, force_strategy="template")
    assert template_result.strategy == "template"
    assert "2024" in template_result.text
    assert hybrid.template_renderer.last_template_id is not None

    polish_result = hybrid.generate(fact_simple, force_strategy="polish")
    assert polish_result.strategy == "polished"
    assert polish_result.text == "polished"

    llm_result = hybrid.generate(fact_simple, force_strategy="llm")
    assert llm_result.strategy == "llm"
    assert llm_result.text == "llm text"
