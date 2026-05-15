# M3-E1 Annotation Guide (Temporal Graph Benchmarks)

This guide is for human annotators who curate benchmark examples for temporal graphs. Build one JSONL per domain (or a mixed-domain file when necessary). Each JSON line must stand on its own after conversion.

## Conversion-first workflow
- Start with rows in data/jsonls/business/auto.jsonl. Every row (input plus output) yields one M3-E1 record.
- Typical input patterns:
  - sequence: entity plus ordered phase1/phase2/... values, optionally context/cite fields.
  - cause: cause, effect, and year fields, optionally context/cite fields.
  - overlap: event1 and event2 fields, optionally context/cite fields.
  - entity: entity information
- Treat output as the draft gold_answer. Tighten wording so it aligns with the facts you extract.
- Copy any cite:[...] identifiers into gold_citations. Add URLs when you can verify them; otherwise leave url null.
- Goal: map input -> query, refine output -> gold_answer, expand each claim into gold_facts, mirror those facts in graph_nodes/graph_edges, then record steps_min and steps_opt.

**Intent vs time_scope quick rule.** intent is the question type (why -> causal, when -> point_in_time, etc.). time_scope is the temporal reasoning structure required (sequence if ordering matters, overlap if simultaneous intervals matter, causal when you trace causes). They frequently match but confirm both explicitly and document odd pairings in notes.

### Example 1 - sequence row
Source row:
```json
{
  "input": "sequence: entity: Solectria vehicle business pivot; phase1: NiMH Force variant introduced 1995-08; phase2: ~400 Solectria-branded vehicles completed by 2001; phase3: focus shift announced 2001; context: small EV manufacturer dynamics",
  "output": "Solectria introduced a nickel-metal hydride Force variant in August 1995, completed roughly 400 Solectria-branded vehicles by 2001, and announced in 2001 that it was shifting focus toward components and engineering services."
}
```

Converted record:
```json
{
  "id": "hist_000201",
  "domain": "historical",
  "query": "What were the key phases of Solectria's vehicle business pivot?",
  "intent": "sequence",
  "time_scope": "sequence",
  "gold_answer": "Solectria launched a NiMH Force variant in August 1995, built roughly 400 branded vehicles by 2001, and in 2001 announced a shift toward components and engineering services.",
  "gold_citations": [
    {"source_id": "<FIND_POSSIBLE_SOURCE>", "url": null, "span": "Solectria introduced a NiMH Force variant in August 1995 ... focused on components in 2001."}
  ],
  "gold_facts": [
    {"fact_id": "f1", "subject": "Solectria", "relation": "introduced", "object": "NiMH Force variant", "value": null, "start": "1995-08", "end": null, "granularity": "month", "source_id": "<FIND_POSSIBLE_SOURCE>"},
    {"fact_id": "f2", "subject": "Solectria", "relation": "completed", "object": "~400 Solectria vehicles", "value": null, "start": null, "end": "2001", "granularity": "year", "source_id": "<FIND_POSSIBLE_SOURCE>"},
    {"fact_id": "f3", "subject": "Solectria", "relation": "announced_shift", "object": "components and engineering services", "value": null, "start": "2001", "end": null, "granularity": "year", "source_id": "<FIND_POSSIBLE_SOURCE>"}
  ],
  "graph_nodes": [
    {"node_id": "n1", "label": "Solectria", "category": "org"},
    {"node_id": "n2", "label": "NiMH Force variant", "category": "product"},
    {"node_id": "n3", "label": "1995-08", "category": "date"},
    {"node_id": "n4", "label": "~400 vehicles milestone", "category": "metric"},
    {"node_id": "n5", "label": "2001", "category": "date"},
    {"node_id": "n6", "label": "Components/services focus", "category": "policy"}
  ],
  "graph_edges": [
    {"edge_id": "e1", "source": "n1", "target": "n2", "relation": "introduced", "start": "1995-08", "end": null, "weight": 1.0},
    {"edge_id": "e2", "source": "n1", "target": "n4", "relation": "completed", "start": null, "end": "2001", "weight": 1.0},
    {"edge_id": "e3", "source": "n1", "target": "n6", "relation": "announced_shift", "start": "2001", "end": null, "weight": 1.0},
    {"edge_id": "e4", "source": "n2", "target": "n6", "relation": "preceded", "start": null, "end": null, "weight": 1.0}
  ],
  "steps_min": 3,
  "steps_opt": 3,
  "tags": ["ev", "solectria"]
}
```

### Example 2 - cause row
Source row:
```json
{
  "input": "cause: demand dropped as legacy manufacturers entered EV/hybrid space; effect: Solectria shifted focus to components/services; year: 2001; context: market competition",
  "output": "As demand for Solectria's conversions dropped with legacy manufacturers entering electric and hybrid markets, Solectria announced a shift toward components and engineering services in 2001."
}
```

Converted record:
```json
{
  "id": "hist_000202",
  "domain": "historical",
  "query": "Did competition from legacy automakers drive Solectria's 2001 business shift?",
  "intent": "causal",
  "time_scope": "causal",
  "gold_answer": "Legacy automakers entering the EV and hybrid market eroded Solectria's demand, leading it in 2001 to pivot toward components and engineering services.",
  "gold_citations": [
    {"source_id": "web:45", "url": null, "span": "Demand dropped as legacy manufacturers entered the EV/hybrid space ... Solectria shifted focus in 2001."}
  ],
  "gold_facts": [
    {"fact_id": "f1", "subject": "Legacy automaker competition", "relation": "caused", "object": "Solectria focus shift", "value": null, "start": "2001", "end": null, "granularity": "year", "source_id": "web:45"},
    {"fact_id": "f2", "subject": "Solectria focus shift", "relation": "occurred_on", "object": null, "value": null, "start": "2001", "end": null, "granularity": "year", "source_id": "web:45"}
  ],
  "graph_nodes": [
    {"node_id": "n1", "label": "Legacy automaker competition", "category": "event"},
    {"node_id": "n2", "label": "Solectria", "category": "org"},
    {"node_id": "n3", "label": "Focus shift to components/services", "category": "event"},
    {"node_id": "n4", "label": "2001", "category": "date"}
  ],
  "graph_edges": [
    {"edge_id": "e1", "source": "n1", "target": "n3", "relation": "caused", "start": "2001", "end": null, "weight": 1.0},
    {"edge_id": "e2", "source": "n3", "target": "n4", "relation": "dated", "start": null, "end": null, "weight": 1.0},
    {"edge_id": "e3", "source": "n2", "target": "n3", "relation": "part_of", "start": "2001", "end": null, "weight": 1.0}
  ],
  "steps_min": 2,
  "steps_opt": 2,
  "tags": ["ev", "solectria", "causal"]
}
```

### Example 3 - overlap row
Source row:
```json
{
  "input": "overlap: event1: Solectria NiMH Force variant 1995-08; event2: early production EV wave 1990s; context: pre-Li-ion era",
  "output": "Solectria's introduction of a NiMH Force variant in August 1995 overlapped with the broader 1990s wave of experimental and limited-production EV efforts."
}
```

Converted record:
```json
{
  "id": "hist_000203",
  "domain": ["historical", "technical"],
  "query": "Did the Solectria NiMH Force launch overlap with the wider 1990s EV wave?",
  "intent": "overlap",
  "time_scope": "overlap",
  "gold_answer": "Solectria launched the NiMH Force in August 1995 during the wider 1990s era of experimental and limited-production electric vehicles.",
  "gold_citations": [
    {"source_id": "web:45", "url": null, "span": "Solectria introduced the NiMH Force variant in August 1995, overlapping the broader 1990s EV wave."}
  ],
  "gold_facts": [
    {"fact_id": "f1", "subject": "NiMH Force launch", "relation": "occurred_on", "object": null, "value": null, "start": "1995-08", "end": null, "granularity": "month", "source_id": "web:45"},
    {"fact_id": "f2", "subject": "1990s EV wave", "relation": "interval", "object": null, "value": null, "start": "1990-01", "end": "1999-12", "granularity": "year", "source_id": "web:45"},
    {"fact_id": "f3", "subject": "NiMH Force launch", "relation": "overlapped", "object": "1990s EV wave", "value": null, "start": "1995-08", "end": "1999-12", "granularity": "month", "source_id": "web:45"}
  ],
  "graph_nodes": [
    {"node_id": "n1", "label": "NiMH Force launch", "category": "event"},
    {"node_id": "n2", "label": "1995-08", "category": "date"},
    {"node_id": "n3", "label": "1990s EV wave", "category": "event"}
  ],
  "graph_edges": [
    {"edge_id": "e1", "source": "n1", "target": "n3", "relation": "overlapped", "start": "1995-08", "end": "1999-12", "weight": 1.0},
    {"edge_id": "e2", "source": "n1", "target": "n2", "relation": "dated", "start": null, "end": null, "weight": 1.0}
  ],
  "steps_min": 2,
  "steps_opt": 2,
  "tags": ["ev", "timeline"]
}
```

Reuse these patterns: align the query with the input type, tighten the output for gold_answer, split the answer into gold_facts, and mirror those facts in the graph.

## Domains
- Allowed single values: medical, financial, historical, science, geopolitical, cultural, technical. Use a list only when an example genuinely spans multiple domains.
- If unsure, select the primary domain and add clarifying tags.

## Required fields (per record)
- id (string): Stable identifier (for example hist_000201).
- domain (string or list): One or more allowed domains.
- query (string): Natural-language question or claim to verify.
- intent (string): point_in_time, interval, sequence, causal, comparative, prediction, or explanation.
- time_scope (string): point, interval, sequence, causal, overlap, comparative, or prediction (dominant temporal structure of the evidence).
- gold_answer (string): Concise canonical answer (one or two sentences).
- gold_citations (list of dict): Each { "source_id": "...", "url": "...", "span": "..." }. Provide at least one citation per key fact; leave url null if unavailable.
- gold_facts (list of dict): Each { "fact_id": "...", "subject": "...", "relation": "...", "object": "...", "value": "...", "start": "...", "end": "...", "granularity": "...", "source_id": "..." }. Use ISO-style dates (YYYY, YYYY-MM, YYYY-MM-DD). Set object or value to null when not needed.
- graph_nodes (list of dict): Each { "node_id": "...", "label": "...", "category": "person|org|event|product|policy|metric|location|date|disease|treatment|instrument" }.
- graph_edges (list of dict): Each { "edge_id": "...", "source": "...", "target": "...", "relation": "caused|preceded|overlapped|during|part_of|located_in|measured_by|released|dated|led_to|announced_shift|occurred_on|introduced", "start": "...", "end": "...", "weight": 1.0 }. Include start/end only when the relation itself is bounded.
- steps_min (integer): Minimum reasoning steps when traversing only essential facts (1-2 = point lookup, 3-4 = short chains, 5+ = longer or branching reasoning).
- steps_opt (integer, optional but encouraged): Best realistic step count for a high-quality answer. Can equal steps_min when there is no difference.
- difficulty (string, optional): Legacy label; derive from steps_opt if requested (<=2 easy, 3-4 medium, 5+ hard).
- notes (string, optional): Clarifications for reviewers.
- tags (list, optional): Short keywords such as ev, policy, battery, supply_chain.
- conflicts (list, optional): Alternate claims or disagreements alongside supporting citations.

## Graph fields
- Keep node IDs stable within a record; reuse nodes rather than duplicating labels.
- Edge relations must mirror the semantics in gold_facts. Document any new relation keywords in notes so downstream tooling stays consistent.

### Minimal graph snippet
```json
"graph_nodes": [
  {"node_id": "n1", "label": "Tesla", "category": "org"},
  {"node_id": "n2", "label": "Model 3", "category": "product"},
  {"node_id": "n3", "label": "2017-07-28", "category": "date"}
],
"graph_edges": [
  {"edge_id": "e1", "source": "n1", "target": "n2", "relation": "released", "start": "2017-07-28", "end": null, "weight": 1.0},
  {"edge_id": "e2", "source": "n2", "target": "n3", "relation": "dated", "start": null, "end": null, "weight": 1.0}
]
```

## Intent vs time_scope (deeper guidance)
- point_in_time intent typically pairs with point time_scope, but use interval when evidence spans a sustained range (for example, production maintained from 1998-2000).
- explanation intent often needs sequence time_scope when narrating how something evolved across phases.
- comparative intent may require overlap (simultaneous comparison) or sequence (ordered comparison). Choose the evidence pattern that dominates.
- Record unusual pairings in notes so reviewers understand your selection.

## Annotation checklist
1. Read the input row and capture the central claim plus any context or cite IDs.
2. Draft the query by rewriting the structured prompt as a natural-language question or claim.
3. Write the gold_answer. Keep it concise; detail lives in facts and the graph.
4. Attach citations. Reuse provided IDs or add reliable sources. Every fact in the answer must be supported.
5. Break the answer into gold_facts. One relation per fact; include start/end and granularity whenever the evidence provides them.
6. Build the graph. Create nodes for unique entities or dates, then add edges that mirror the gold_facts relations.
7. Label intent and time_scope. Use definitions above and explain non-obvious choices in notes.
8. Record step counts. Set steps_min first; add steps_opt when the optimal explanation requires extra traversal.
9. Add tags, notes, and conflicts. Provide useful filters and flag disagreements or coverage gaps for reviewers.

## Domain guidance
- Automotive and aviation rows usually map to historical (timelines), financial (industry impact), or technical (engineering pivots). Use multiple domains when coverage is genuinely balanced.
- Introduce new domains only when the data supports them. Document new source collections in notes for review.

## Quality tips
- Keep gold_facts atomic; split compound statements instead of merging multiple relations.
- Reuse node IDs consistently; avoid creating separate nodes for the same entity or date.
- Capture conflicts when credible sources disagree, but keep gold_answer aligned with the strongest citation.
- Prefer ISO date formats (YYYY, YYYY-MM, YYYY-MM-DD) and match granularity to the precision of the evidence.

## File organization
- Store final JSONL files in experiments/m3_e1_annotation/<domain>/gold/.
- Keep draft files nearby but suffix them with _draft.jsonl so they are easy to filter.
- Mixed-domain files belong in experiments/m3_e1_annotation/mixed/gold/.

## Minimum required fields for acceptance
- id, domain, query, intent, time_scope, gold_answer, gold_citations (at least one entry), gold_facts (at least one entry), graph_nodes, graph_edges, steps_min.
- Provide steps_opt whenever it differs from steps_min.
- Ensure every graph edge is supported by at least one gold_fact, and every fact is represented in the graph.

