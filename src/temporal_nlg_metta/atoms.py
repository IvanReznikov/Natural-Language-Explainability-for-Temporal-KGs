"""Grounded MeTTa operations backed by :class:`TemporalBridge` (M4).

Each operation is a thin adapter that turns MeTTa scalars (strings / numbers)
into a call on the bridge and returns a JSON string (MeTTa handles string atoms
cleanly and JSON keeps structured results transportable). Operations are grouped
by their source milestone so the provenance of every capability is explicit.

Two registration entry points are provided:

* :func:`register_with(metta, bridge)` — explicit registration onto an existing
  ``MeTTa`` runner. Works with any hyperon version that exposes
  ``register_atom`` / ``OperationAtom``.
* :func:`register_atoms()` — the ``@register_atoms``-style entry point used when
  this package is loaded as a MeTTa module via ``!(import! &self temporal-nlg)``.

The module imports ``hyperon`` lazily and degrades gracefully when it is absent,
so the bridge remains fully testable without the optional dependency.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from .bridge import TemporalBridge

# A module-level default bridge so MeTTa programs that call the module-style
# loader (which has no opportunity to receive a runner-scoped bridge) still work.
# Examples and tests pass their own bridge via register_with().
_DEFAULT_BRIDGE: TemporalBridge | None = None


def get_default_bridge() -> TemporalBridge:
    global _DEFAULT_BRIDGE
    if _DEFAULT_BRIDGE is None:
        _DEFAULT_BRIDGE = TemporalBridge()
    return _DEFAULT_BRIDGE


def set_default_bridge(bridge: TemporalBridge) -> None:
    global _DEFAULT_BRIDGE
    _DEFAULT_BRIDGE = bridge


# ----------------------------------------------------------------------
# Operation definitions
#
# Each entry maps a MeTTa token -> (bridge_method_name, method_kwargs_template).
# We build the OperationAtom wrappers centrally so the same definitions serve
# both the explicit and the module-style registration paths.
# ----------------------------------------------------------------------


def _json_or(value: Any, default: Any) -> Any:
    """Return ``value`` as-is for non-strings; parse JSON strings."""
    if isinstance(value, str):
        if not value:
            return default
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _dumps(value: Any) -> str:
    """Serialize an operation result to a MeTTa-friendly JSON string."""
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def _edge_tuples_from_json(value: Any) -> List[List[str]]:
    """Normalize a JSON edge list into ``[src, rel, tgt, time]`` string quads.

    Accepts a JSON string (or already-parsed list) of either positional
    arrays (``[src, rel, tgt]`` / ``[src, rel, tgt, time]``) or objects with
    ``source``/``relation``/``target`` (alias ``src``/``rel``/``tgt``) keys
    plus an optional ``start``/``year``.
    """
    edges = _json_or(value, [])
    if not isinstance(edges, list):
        raise ValueError("edges must be a JSON list")
    tuples: List[List[str]] = []
    for edge in edges:
        if isinstance(edge, dict):
            when = edge.get("start") or edge.get("year") or ""
            tuples.append(
                [
                    str(edge.get("source") or edge.get("src") or ""),
                    str(edge.get("relation") or edge.get("rel") or ""),
                    str(edge.get("target") or edge.get("tgt") or ""),
                    str(when),
                ]
            )
        else:
            items = [str(x) for x in list(edge)]
            while len(items) < 4:
                items.append("")
            tuples.append(items[:4])
    return tuples


def _edge_atoms(tuples: List[List[str]]) -> Any:
    """Build ``(edge "src" "rel" "tgt" "time")`` expression atoms from tuples.

    Returns the list of edge expressions, which a grounded op created with
    ``unwrap=False`` yields as multiple results — ready for ``let``/``add-atom``/
    ``match`` on the MeTTa side. Without hyperon (testing only) returns the
    plain tuples.
    """
    hyperon = _import_hyperon()
    if hyperon is None:
        return tuples
    from hyperon import E, S, ValueAtom  # type: ignore

    return [
        E(S("edge"), ValueAtom(src), ValueAtom(rel), ValueAtom(tgt), ValueAtom(when))
        for src, rel, tgt, when in tuples
    ]


def _atom_arg_str(atom: Any) -> str:
    """Extract a Python string from an atom argument (used by unwrap=False ops)."""
    get_object = getattr(atom, "get_object", None)
    if callable(get_object):
        try:
            return str(get_object().content)
        except Exception:
            pass
    return str(atom)


# Each spec: (token, arity, factory). The factory receives the bridge and
# returns a Python callable suitable for ``OperationAtom(token, fn)``.
def _make_specs() -> List[Dict[str, Any]]:
    return [
        # ── Lifecycle ───────────────────────────────────────────────
        {
            "token": "temporal-reset!",
            "arity": 0,
            "factory": lambda b: (lambda: _dumps(b.reset())),
        },
        # ── M1: NLG ─────────────────────────────────────────────────
        {
            "token": "nlg-fact",
            "arity": 2,
            "factory": lambda b: (
                lambda fact_type, content_json, strategy=None: _dumps(
                    b.nlg_fact(fact_type, content_json, _json_or(strategy, None))
                )
            ),
        },
        {
            "token": "nlg-fact-strategy",
            "arity": 3,
            "factory": lambda b: (
                lambda fact_type, content_json, strategy: _dumps(
                    b.nlg_fact(fact_type, content_json, strategy)
                )
            ),
        },
        {
            "token": "nlg-readability",
            "arity": 1,
            "factory": lambda b: (lambda text: _dumps(b.nlg_readability(text))),
        },
        {
            "token": "nlg-eval",
            "arity": 2,
            "factory": lambda b: (
                lambda content_json, generated_text, fact_type="point_in_time": _dumps(
                    b.nlg_evaluate(content_json, generated_text, fact_type)
                )
            ),
        },
        # ── M2: Truth maintenance ───────────────────────────────────
        {
            "token": "tms-start-trace",
            "arity": 0,
            "factory": lambda b: (lambda query_id=None: b.start_trace(query_id)),
        },
        {
            "token": "tms-add-belief",
            "arity": 1,
            "factory": lambda b: (
                lambda belief_id, payload_json="{}", supports_json="[]", evidence_json="[]": b.add_belief(
                    belief_id, payload_json, supports_json, evidence_json
                )
            ),
        },
        {
            "token": "tms-record-rule",
            "arity": 2,
            "factory": lambda b: (
                lambda rule_id, rule_name, inputs_json="[]", conclusion_json="{}", confidence=1.0: b.record_rule(
                    rule_id,
                    rule_name,
                    inputs_json,
                    conclusion_json,
                    float(confidence) if confidence is not None else 1.0,
                )
            ),
        },
        {
            "token": "tms-rules-fired",
            "arity": 0,
            "factory": lambda b: (lambda: _dumps(b.rules_fired())),
        },
        {
            "token": "tms-record-rule-facts",
            "arity": 6,
            "factory": lambda b: (
                lambda rule_id, rule_name, in_fact_id, in_value, out_fact_id, out_value, confidence=1.0: b.record_rule_facts(
                    str(rule_id),
                    str(rule_name),
                    str(in_fact_id),
                    str(in_value),
                    str(out_fact_id),
                    str(out_value),
                    float(confidence) if confidence is not None else 1.0,
                )
            ),
        },
        {
            "token": "tms-why-not",
            "arity": 1,
            "factory": lambda b: (lambda expected_json: _dumps(b.why_not(expected_json))),
        },
        {
            "token": "tms-explain-trace",
            "arity": 1,
            "factory": lambda b: (lambda fact_id: b.explain_belief_trace(fact_id)),
        },
        {
            "token": "tms-contradictions",
            "arity": 0,
            "factory": lambda b: (lambda: _dumps(b.contradictions())),
        },
        {
            "token": "tms-influential-facts",
            "arity": 0,
            "factory": lambda b: (lambda: _dumps(b.influential_facts())),
        },
        {
            "token": "tms-explain",
            "arity": 1,
            "factory": lambda b: (lambda belief_id: b.explain_belief(belief_id)),
        },
        {
            "token": "tms-support-chain",
            "arity": 1,
            "factory": lambda b: (lambda belief_id: _dumps(b.support_chain(belief_id))),
        },
        {
            "token": "tms-active-beliefs",
            "arity": 0,
            "factory": lambda b: (lambda: _dumps(b.active_beliefs())),
        },
        {
            "token": "tms-dirty-beliefs",
            "arity": 0,
            "factory": lambda b: (lambda: _dumps(b.dirty_beliefs())),
        },
        {
            "token": "tms-retract!",
            "arity": 1,
            "factory": lambda b: (lambda belief_id: _dumps(b.retract(belief_id))),
        },
        {
            "token": "tms-justification-path",
            "arity": 1,
            "factory": lambda b: (
                lambda fact_id, max_depth=6: _dumps(
                    b.justification_paths(fact_id, int(max_depth) if max_depth is not None else 6)
                )
            ),
        },
        {
            "token": "tms-counterfactual-shift",
            "arity": 2,
            "factory": lambda b: (
                lambda belief_id, delta: _dumps(b.counterfactual_shift_time(belief_id, delta))
            ),
        },
        # ── M3: Graph QA + path explanation ─────────────────────────
        {
            "token": "graph-answer",
            "arity": 1,
            "factory": lambda b: (lambda question: _dumps(b.answer(question))),
        },
        {
            "token": "graph-evidence",
            "arity": 0,
            "factory": lambda b: (lambda: _dumps(b.evidence())),
        },
        {
            "token": "graph-evidence-atoms",
            "arity": 0,
            "unwrap": False,
            "factory": lambda b: (lambda: _edge_atoms(b.evidence_edge_tuples())),
        },
        {
            "token": "edges-from-json",
            "arity": 1,
            "unwrap": False,
            "factory": lambda b: (
                lambda edges_json: _edge_atoms(_edge_tuples_from_json(_atom_arg_str(edges_json)))
            ),
        },
        {
            "token": "graph-confidence",
            "arity": 0,
            "factory": lambda b: (lambda: b.confidence()),
        },
        {
            "token": "graph-mermaid",
            "arity": 0,
            "factory": lambda b: (lambda: b.mermaid()),
        },
        {
            "token": "graph-explain-path",
            "arity": 3,
            "factory": lambda b: (
                lambda adjacency_json, start, end: _dumps(
                    b.explain_path(adjacency_json, start, end)
                )
            ),
        },
        {
            "token": "graph-explain-path-as-belief",
            "arity": 3,
            "factory": lambda b: (
                lambda adjacency_json, start, end, belief_id=None: _dumps(
                    b.explain_path(adjacency_json, start, end, belief_id=belief_id)
                )
            ),
        },
        {
            "token": "graph-explain-path-styled",
            "arity": 5,
            "factory": lambda b: (
                lambda adjacency_json, start, end, style="neutral", domain="general": _dumps(
                    b.explain_path_styled(adjacency_json, start, end, style, domain)
                )
            ),
        },
        # ── Cross-cutting: counterfactual explanation ───────────────
        {
            "token": "counterfactual",
            "arity": 6,
            "factory": lambda b: (
                lambda subject, predicate, obj, alt_subject, alt_predicate, alt_obj, timeframe=None, alt_timeframe=None: _dumps(
                    b.counterfactual(
                        subject,
                        predicate,
                        obj,
                        alt_subject,
                        alt_predicate,
                        alt_obj,
                        timeframe,
                        alt_timeframe,
                    )
                )
            ),
        },
    ]


def all_specs() -> List[Dict[str, Any]]:
    """Return a copy of the operation specifications."""
    return [dict(spec) for spec in _make_specs()]


def _import_hyperon():
    """Lazy import; returns None when hyperon is not installed."""
    try:
        from hyperon import OperationAtom, ValueAtom  # type: ignore

        return OperationAtom, ValueAtom
    except Exception:
        return None


def register_with(metta: Any, bridge: TemporalBridge) -> List[str]:
    """Register every temporal operation onto a ``MeTTa`` runner.

    Returns the list of MeTTa tokens that were registered. Raises a clear error
    if ``hyperon`` is not available, since this path is only reached when the
    caller actually has a runner.
    """
    hyperon = _import_hyperon()
    if hyperon is None:
        raise RuntimeError(
            "The 'hyperon' package is required to register MeTTa operations. "
            "Install it with: pip install 'temporal-nlg[metta]'"
        )
    OperationAtom, _ValueAtom = hyperon

    registered: List[str] = []
    for spec in _make_specs():
        fn = spec["factory"](bridge)
        token = spec["token"]
        metta.register_atom(token, OperationAtom(token, fn, unwrap=spec.get("unwrap", True)))
        registered.append(token)
    return registered


# Module-style loader (used by ``!(import! &self temporal-nlg)``).
def register_atoms() -> Dict[str, Any]:
    """Return a name->atom mapping for the hyperon ``@register_atoms`` protocol.

    Falls back to returning callables (not wrapped in OperationAtom) when
    hyperon is unavailable, which keeps the loader importable in environments
    that lack the optional dependency. Callers with a real runner should prefer
    :func:`register_with`.
    """
    bridge = get_default_bridge()
    hyperon = _import_hyperon()
    if hyperon is None:
        # Not running under MeTTa; expose raw callables for inspection/testing.
        return {spec["token"]: spec["factory"](bridge) for spec in _make_specs()}
    OperationAtom, _ValueAtom = hyperon
    return {
        spec["token"]: OperationAtom(
            spec["token"], spec["factory"](bridge), unwrap=spec.get("unwrap", True)
        )
        for spec in _make_specs()
    }


def available_tokens() -> List[str]:
    """Return the sorted list of registered MeTTa tokens."""
    return sorted(spec["token"] for spec in _make_specs())


__all__ = [
    "register_with",
    "register_atoms",
    "all_specs",
    "available_tokens",
    "get_default_bridge",
    "set_default_bridge",
]
