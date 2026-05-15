"""Utility to extract linear paths from adjacency structures."""
from __future__ import annotations

from typing import Dict, List, Tuple
from .graph_paths import GraphNode, GraphEdge


def extract_path(adj: Dict[str, List[Tuple[str, str, str]]], start: str, end: str) -> Tuple[List[GraphNode], List[GraphEdge]]:
    """
    Given adjacency {node: [(target, label, timestamp)]}, return a simple DFS path.
    Raises ValueError if no path.
    """
    stack = [(start, [start], [])]
    while stack:
        node, path_nodes, path_edges = stack.pop()
        if node == end:
            nodes = [GraphNode(node_id=n, label=n) for n in path_nodes]
            edges = [GraphEdge(source=src, target=dst, label=lbl, timestamp=ts) for src, dst, lbl, ts in path_edges]
            return nodes, edges
        for tgt, lbl, ts in adj.get(node, []):
            if tgt in path_nodes:
                continue
            stack.append((tgt, path_nodes + [tgt], path_edges + [(node, tgt, lbl, ts)]))
    raise ValueError("No path found")
