"""
Narrative rendering, path explanations, and visualization-related tests.
"""

import pytest

from temporal_nlg.explain import (
    PathNarrativeRenderer as ExplainPathNarrativeRenderer,
    JustifiedRenderer,
    path_to_narrative,
)
from temporal_nlg.explain.graph_paths import GraphNode, GraphEdge, GraphPathExplanation
from temporal_nlg.explain.graph_extract import extract_path
from temporal_nlg.tms.belief_store import BeliefStore, Belief
from temporal_nlg.tms.justification import JustificationBuilder
from temporal_nlg.path_narratives.renderer import (
    PathNarrativeRenderer as PathNarrativeRendererCompat,
    PathExample,
    PathStep,
)


def test_narrative_styles_and_justification():
    nodes = [GraphNode(node_id="n1", label="A"), GraphNode(node_id="n2", label="B")]
    edges = [GraphEdge(source="n1", target="n2", label="before", timestamp="T1")]

    renderer = ExplainPathNarrativeRenderer(style="expert")
    narrative = renderer.render(nodes, edges, evidence_ids=["b1"])
    assert "Path" in narrative["narrative"]
    assert "b1" in narrative["narrative"]

    store = BeliefStore()
    belief = Belief(
        belief_id="b1",
        payload={"text": "A before B"},
        evidence=[{"source": "doc", "snippet": "A then B", "weight": 1}],
    )
    store.add_belief(belief)
    justified = JustifiedRenderer(store, JustificationBuilder()).render_with_justification(
        "b1", narrative["narrative"]
    )
    assert "supported by" in justified["justification"]
    assert "doc" in justified["justification"]


def test_narrative_finance_domain_and_missing_belief():
    nodes = [GraphNode(node_id="s", label="Stock"), GraphNode(node_id="e", label="Earnings")]
    edges = [GraphEdge(source="s", target="e", label="leads", timestamp="2024Q1")]

    renderer = ExplainPathNarrativeRenderer(style="novice", domain="finance")
    narrative = renderer.render(nodes, edges, evidence_ids=None)

    assert "Impact" in narrative["narrative"]

    store = BeliefStore()
    missing = JustifiedRenderer(store).render_with_justification("missing", narrative["narrative"])
    assert missing["justification"] == "Belief not found."


def test_graph_path_explanation_simple():
    nodes = [GraphNode(node_id="n1", label="A"), GraphNode(node_id="n2", label="B")]
    edges = [GraphEdge(source="n1", target="n2", label="connects", timestamp="2000")]
    explainer = GraphPathExplanation()
    explanation = explainer.explain(nodes, edges)

    assert "Connection from A to B" in explanation["summary"]
    assert "A -[connects @2000]-> B" == explanation["path"]
    assert "n1 connects n2" in explanation["justification"]


def test_graph_path_explanation_validates_lengths():
    nodes = [GraphNode(node_id="n1", label="A")]
    with pytest.raises(ValueError):
        GraphPathExplanation().explain(nodes, [])


def test_graph_path_explanation_rejects_mismatched_edges():
    nodes = [
        GraphNode(node_id="n1", label="A"),
        GraphNode(node_id="n2", label="B"),
        GraphNode(node_id="n3", label="C"),
    ]
    edges = [GraphEdge(source="n1", target="n2", label="connects")]
    with pytest.raises(ValueError):
        GraphPathExplanation().explain(nodes, edges)


def test_extract_path_success():
    adj = {"A": [("B", "before", "t1")], "B": [("C", "enables", "t2")]}
    nodes, edges = extract_path(adj, "A", "C")
    assert len(nodes) == 3
    assert edges[-1].target == "C"


def test_extract_path_no_path():
    adj = {"A": [("B", "before", "t1")], "C": []}
    with pytest.raises(ValueError):
        extract_path(adj, "A", "C")


def test_path_to_narrative_with_supports_and_evidence_chain():
    adj = {
        "A": [("B", "before", "t1")],
        "B": [("C", "enables", "t2")],
    }
    store = BeliefStore()
    store.add_belief(Belief(belief_id="sup1", payload={"text": "seed"}))

    result = path_to_narrative(
        adj,
        start="A",
        end="C",
        belief_store=store,
        belief_id="b1",
        supports=["sup1"],
    )

    assert "Connection from A to C" in result["summary"]
    assert "Evidence chain: sup1" in result["narrative"]
    assert "sup1" in result["justification"]
    assert "graph_path" in result["justification"]


def test_path_to_narrative_without_supports_defaults_evidence():
    adj = {"X": [("Y", "leads", "t1")]}
    store = BeliefStore()

    result = path_to_narrative(
        adj,
        start="X",
        end="Y",
        belief_store=store,
        belief_id="b2",
        supports=None,
        evidence=None,
    )

    assert "Connection from X to Y" in result["summary"]
    assert "Evidence chain" not in result["narrative"]
    assert "supported by none" in result["justification"]
    assert "graph_path" in result["justification"]


def test_render_path_and_batch_scores():
    renderer = PathNarrativeRendererCompat()
    example = PathExample(
        path_id="p1", steps=[PathStep("A", "leads to", "B", "t1"), PathStep("B", "causes", "C")]
    )

    single = renderer.render_path(example)
    assert "A leads to B" in single
    assert single.endswith(".")

    report = renderer.render_batch([example])
    assert report.rendered[0] == single
    assert len(report.flesch_scores) == 1
    assert report.avg_flesch == report.flesch_scores[0]
