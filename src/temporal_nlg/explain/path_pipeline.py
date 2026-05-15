"""End-to-end path-to-narrative with justification."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .graph_extract import extract_path
from .narratives import PathNarrativeRenderer
from .justified_render import JustifiedRenderer
from temporal_nlg.tms.belief_store import BeliefStore, Belief


def path_to_narrative(
    adj: Dict[str, List[Tuple[str, str, str]]],
    start: str,
    end: str,
    belief_store: BeliefStore,
    belief_id: str,
    supports: Optional[List[str]] = None,
    evidence: Optional[List[Dict[str, str]]] = None,
    style: str = "neutral",
    domain: str = "general",
) -> Dict[str, str]:
    nodes, edges = extract_path(adj, start, end)
    renderer = PathNarrativeRenderer(style=style, domain=domain)
    narrative = renderer.render(nodes, edges, evidence_ids=supports)

    # Register belief and attach justification from the extracted path.
    evidence_payload = evidence if evidence is not None else [
        {"source": "graph_path", "snippet": narrative.get("justification", ""), "weight": 1.0}
    ]
    belief = Belief(
        belief_id=belief_id,
        payload={"path": narrative.get("narrative", ""), "summary": narrative.get("summary", "")},
        supports=supports or [],
        evidence=evidence_payload,
    )
    belief_store.add_belief(belief)
    justified = JustifiedRenderer(belief_store).render_with_justification(belief_id, narrative["narrative"])

    return {
        "summary": narrative["summary"],
        "narrative": justified["text"],
        "justification": justified.get("justification", ""),
    }
