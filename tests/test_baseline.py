"""
Baseline smoke tests for data loaders, belief tracking, and counterfactual utilities.
"""

from temporal_nlg.data.loaders import (
    PointInTimeExampleGenerator,
    IntervalExampleGenerator,
    SequenceExampleGenerator,
    CausalityExampleGenerator,
    OverlapExampleGenerator,
    generate_all_examples,
)
from temporal_nlg.core.templates import TemplateType
from temporal_nlg.explain.beliefs import BeliefTracker, Evidence
from temporal_nlg.explain.counterfactuals import CounterfactualGenerator, Fact
from temporal_nlg.tms.counterfactual import CounterfactualEngine
from temporal_nlg.tms.belief_store import BeliefStore, Belief


def test_generate_small_batches_per_type():
    point = PointInTimeExampleGenerator.generate(3)
    interval = IntervalExampleGenerator.generate(2)
    sequence = SequenceExampleGenerator.generate(2)
    causality = CausalityExampleGenerator.generate(2)
    overlap = OverlapExampleGenerator.generate(2)

    assert point and point[0].fact_type == TemplateType.POINT_IN_TIME
    assert interval and interval[0].fact_type == TemplateType.INTERVAL
    assert sequence and sequence[0].fact_type == TemplateType.SEQUENCE
    assert causality and causality[0].fact_type == TemplateType.CAUSALITY
    assert overlap and overlap[0].fact_type == TemplateType.OVERLAP


def test_generate_all_examples_small_batch():
    bundle = generate_all_examples(n_per_type=1)
    assert set(bundle.keys()) == {"point_in_time", "intervals", "sequences", "causality", "overlaps"}
    assert all(len(v) == 1 for v in bundle.values())


def test_upsert_and_justify():
    tracker = BeliefTracker()
    tracker.upsert_belief(
        claim_id="c1",
        claim="A leads to B",
        confidence=0.8,
        evidence=[Evidence(source="doc", snippet="A before B", weight=1.2)],
    )

    text = tracker.justify("c1")
    assert "A leads to B" in text
    assert "doc" in text
    assert "w=1.2" in text


def test_top_beliefs_sorted():
    tracker = BeliefTracker()
    tracker.upsert_belief("c1", "C1", 0.2)
    tracker.upsert_belief("c2", "C2", 0.9)
    tracker.upsert_belief("c3", "C3", 0.5)

    top = tracker.top_beliefs(k=2)
    assert [b.claim_id for b in top] == ["c2", "c3"]


def test_counterfactual_delta_and_text():
    factual = Fact(subject="Policy", predicate="reduced", obj="emissions", timeframe="2020")
    alternative = Fact(subject="Policy", predicate="increased", obj="emissions", timeframe="2020")
    gen = CounterfactualGenerator()

    result = gen.generate(factual, alternative)

    assert "reduced emissions" in result["factual"]
    assert "increased emissions" in result["counterfactual"]
    assert "predicate changed" in result["delta"]


def test_counterfactual_shift_time_copies_supports_and_evidence():
    engine = CounterfactualEngine()
    belief = Belief(
        belief_id="b1",
        payload={"event": "now"},
        evidence=[{"source": "doc", "snippet": "fact"}],
    )

    result = engine.shift_time(belief, "+1d")

    assert result.original_id == "b1"
    assert result.new_belief.belief_id == "cf_b1"
    assert result.new_belief.supports == ["b1"]
    assert result.new_belief.evidence[0]["source"] == "doc"
    assert "+1d" in result.description


def test_counterfactual_swap_order_creates_two_results():
    engine = CounterfactualEngine()
    belief_a = Belief(belief_id="a", payload={"event": "A"})
    belief_b = Belief(belief_id="b", payload={"event": "B"})

    results = engine.swap_order(belief_a, belief_b)

    assert {r.original_id for r in results} == {"a", "b"}
    assert all(r.new_belief.belief_id.startswith("cf_") for r in results)


def test_retract_marks_dependents_dirty():
    store = BeliefStore()
    store.add_belief(Belief(belief_id="a", payload={"x": 1}))
    store.add_belief(Belief(belief_id="b", payload={}, supports=["a"]))
    store.add_belief(Belief(belief_id="c", payload={}, supports=["b"]))

    store.retract("a")

    assert store.get_belief("a").status == "retracted"
    assert store.get_belief("b").status == "dirty"
    assert store.get_belief("c").status == "dirty"


def test_add_support_updates_links_and_dependencies():
    store = BeliefStore()
    store.add_belief(Belief(belief_id="root", payload={}))
    store.add_belief(Belief(belief_id="child", payload={}))

    store.add_support("child", "root")

    assert "root" in store.dependencies
    assert "child" in store.dependencies["root"]
    assert "root" in store.get_belief("child").supports
