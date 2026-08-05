# Milestone 4 Technical Documentation

## Scope

M4 is the **integration milestone**: it connects the temporal explanation system
delivered in M1–M3 to the **MeTTa** language and the **MORK** kernel, so that
temporal inference paths can be explained in natural language — in real time —
from inside MeTTa programs, with truth-maintenance support.

M4 deliberately introduces **no new reasoning/NLG logic**. It composes the
existing M1/M2/M3 capabilities behind a thin, well-tested bridge and exposes
them as grounded MeTTa operations.

- **M4-E1**: Extension package `temporal_nlg_metta` — bridge + grounded ops
- **M4-E2**: Truth-maintenance (TMS) surface for interpreter correctness and debugging
- **M4-E3**: MORK kernel adapter (subprocess-based; same `.metta` programs)
- **M4-E4**: Capstone — end-to-end explainable temporal inference from MeTTa

---

## Why MeTTa / MORK, and the integration target

MeTTa is the language of the SingularityNET / TrueAGI Hyperon stack. **MORK**
(the MeTTa Optimal Reduction Kernel) is a next-generation Rust hypergraph VM for
MeTTa that retrofits Hyperon with a graph database and a zipper-based
multi-threaded evaluation kernel.

Two facts shaped the design:

1. **MORK has no Python FFI today.** It is a Rust kernel whose Python bindings
   do not yet exist. The only way to reach it from Python is its CLI.
2. **The canonical, installable MeTTa interpreter is the `hyperon` package**
   (`hyperon-experimental`), which exposes the same MeTTa language with a
   mature Python API (grounded atoms, `register_atom`, module loading).

Therefore M4 programs against the **MeTTa language** (kernel-agnostic): the same
`.metta` files run on `hyperon` today and on MORK once it is built. The
`hyperon` interpreter is the primary evaluation surface; `MORKRunner` is a
subprocess adapter that evaluates the same programs against the MORK CLI.

---

## Package layout

```
src/temporal_nlg_metta/
├── __init__.py            # public exports; lazy hyperon import
├── config.py              # MettaConfig: env-driven configuration
├── bridge.py              # TemporalBridge: session-scoped M1/M2/M3 composition
├── atoms.py               # grounded-op registry (register_with / register_atoms)
├── runner.py              # make_metta_runner, run_metta[_file], MORKRunner
└── metta/
    └── temporal_nlg.metta # kernel-portable pure-MeTTa wrapper helpers
```

`temporal_nlg_metta` is a **sibling** of `temporal_nlg` under `src/` and is
auto-discovered by the existing `include = ["temporal_nlg*"]` glob in
`pyproject.toml`. The optional `hyperon` dependency is isolated to this package;
the core `temporal_nlg` package remains importable without it.

---

## The bridge

`TemporalBridge` (`src/temporal_nlg_metta/bridge.py`) is a session-scoped holder
that composes M1/M2/M3 with no logic of its own:

| Milestone | Component held by the bridge | Bridge methods |
|-----------|------------------------------|----------------|
| **M1** (NLG) | `HybridGenerator`, `AccuracyEvaluator` | `nlg_fact`, `nlg_readability`, `nlg_evaluate` |
| **M2** (TMS) | `BeliefStore`, `TraceRecorder` | `add_belief`, `record_rule`, `record_rule_facts`, `rules_fired`, `why_not`, `explain_belief`, `explain_belief_trace`, `contradictions`, `influential_facts`, `support_chain`, `retract`, `active_beliefs`, `dirty_beliefs`, `start_trace`, `trace_to_dict`, `justification_paths`, `counterfactual_shift_time` |
| **M3** (graph QA) | `TemporalGraphLCELPipeline` (lazy) | `answer`, `evidence`, `evidence_edge_tuples`, `confidence`, `mermaid`, `explain_path`, `explain_path_styled` |
| **Cross** (counterfactual) | `CounterfactualGenerator` | `counterfactual` |

Design points:

- **Lazy M3 pipeline.** The graph pipeline is only built when a graph op is
  first called, so the bridge can be constructed in a bare environment (no model
  servers, no graph artifacts) and used purely for the M1/M2 surface. This keeps
  unit tests hermetic and fast.
- **`explain_path` composes all three milestones**: M3 path extraction → M1
  narrative rendering → M2 justified-belief registration.
- **`reset()`** clears M2 session state (beliefs + active trace) while retaining
  the expensive M1 cache and M3 pipeline.

---

## Grounded operations (the MeTTa surface)

Registered by `register_with(metta, bridge)` (explicit) or `register_atoms()`
(the `@register_atoms` module-style loader for `!(import! &self temporal-nlg)`).
Most structured results are returned as **JSON strings** so they pass cleanly
through MeTTa atoms; the two atom-returning ops (`graph-evidence-atoms`,
`edges-from-json`) are registered with `unwrap=False` and yield real `(edge ...)`
expression atoms that MeTTa can `match` over directly.

### Lifecycle

| Token | Arity | Description |
|-------|-------|-------------|
| `temporal-reset!` | 0 | Clear M2 session state (beliefs + active trace) |

### M1 — Natural-language generation

| Token | Arity | Description |
|-------|-------|-------------|
| `nlg-fact` | 2–3 | Verbalize a temporal fact (`(nlg-fact <type> <content-json> [strategy])`) |
| `nlg-fact-strategy` | 3 | Force a strategy (`template`/`polish`/`llm`) |
| `nlg-readability` | 1 | Flesch + information density of a text |
| `nlg-eval` | 2–3 | Accuracy of a generated text vs. its source fact |

`<type>` ∈ `point_in_time` (alias `point`), `interval`, `sequence`, `causality`, `overlap`.

### M2 — Truth maintenance

| Token | Arity | Description |
|-------|-------|-------------|
| `tms-start-trace` | 0 | Begin a traced reasoning session |
| `tms-add-belief` | 1–4 | Register a belief with supports/evidence |
| `tms-record-rule` | 2–5 | Record a rule firing into the trace |
| `tms-record-rule-facts` | 6–7 | Same, from plain fact scalars (atom-friendly; for MeTTa-derived values) |
| `tms-rules-fired` | 0 | List rule ids that fired |
| `tms-why-not` | 1 | Explain why expected rules did not fire (debugging) |
| `tms-explain-trace` | 1 | Explain how a fact id was derived within the trace |
| `tms-contradictions` | 0 | Detect value conflicts among trace conclusions |
| `tms-influential-facts` | 0 | Rank facts by how often they fed a rule |
| `tms-explain` | 1 | Justification for a belief |
| `tms-support-chain` | 1 | Breadth-first support chain for a belief |
| `tms-active-beliefs` | 0 | Active belief ids |
| `tms-dirty-beliefs` | 0 | Dirty (cascade-invalidated) belief ids |
| `tms-retract!` | 1 | Retract a belief; cascade dirty marks to dependents |
| `tms-justification-path` | 1–2 | Full multi-hop justification chain for a conclusion (deep debugging) |
| `tms-counterfactual-shift` | 2 | Temporal counterfactual over a belief; registers `cf_` belief |

### M3 — Temporal graph QA + path explanation

| Token | Arity | Description |
|-------|-------|-------------|
| `graph-answer` | 1 | Answer a temporal question over the graph |
| `graph-evidence` | 0 | Evidence edges of the most recent answer |
| `graph-evidence-atoms` | 0 | Evidence edges as matchable `(edge ...)` atoms (one result per edge) |
| `edges-from-json` | 1 | Convert a JSON edge list into matchable `(edge ...)` atoms |
| `graph-confidence` | 0 | Confidence of the most recent answer |
| `graph-mermaid` | 0 | Mermaid graph of the most recent answer's evidence |
| `graph-explain-path` | 3 | Explain a graph path as NL (records a belief) |
| `graph-explain-path-as-belief` | 3–4 | Same, with an explicit belief id |
| `graph-explain-path-styled` | 5 | Same path rendered in a chosen style/domain (audience-aware) |

### Cross-cutting — Counterfactual explanation

| Token | Arity | Description |
|-------|-------|-------------|
| `counterfactual` | 6–8 | Fact-level counterfactual: "if instead X, the outcome would diverge" |

---

## Runner construction

Headless, fully isolated construction (recommended in the hyperon test suite):

```python
from temporal_nlg_metta import TemporalBridge, make_metta_runner, run_metta

bridge = TemporalBridge()
metta = make_metta_runner(bridge=bridge)   # registers all ops, config_dir=None
print(metta.run('!(graph-answer "What caused the Model T price drop?")'))
```

`run_metta(program)` and `run_metta_file(path)` construct a fresh runner if one
is not supplied.

---

## MORK kernel adapter

`MORKRunner` (`src/temporal_nlg_metta/runner.py`) evaluates the **same** `.metta`
files against the MORK CLI. MORK has no Python FFI, so a subprocess seam is the
only way to reach it.

```python
from temporal_nlg_metta import MORKRunner
runner = MORKRunner()                      # finds `mork` on PATH or via MORK_BINARY
print(runner.run_file("examples/milestone4/m4e1_nlg.metta"))
```

- The MORK CLI argument surface is still evolving (clap-derived subcommands), so
  the invocation is configurable: the `{file}` placeholder in `args` is replaced
  with the `.metta` path. Default: `mork <file>`. Override via the constructor
  `args=` or the `MORK_ARGS` env var. The MORK HTTP adapter below is the
  validated MORK path.
- MORK does **not** host Python grounded operations. Programs that call the
  `temporal-*`/`nlg-*`/`tms-*`/`graph-*` tokens must run on `hyperon`; the
  pure-MeTTa helpers in `metta/temporal_nlg.metta` are kernel-portable.

## MORK HTTP atomspace adapter

`MORKHttpRunner` (`src/temporal_nlg_metta/runner.py`) wraps the MORK server's
RESTful atomspace API (default `http://127.0.0.1:8000`, override with
`MORK_SERVER_URL`): `upload` / `export` (pattern-matched retrieval) /
`transform` (server-side join-match) / `clear` / `status` / `count` / `explore`,
plus temporal helpers (`upload_temporal_edge`, `query_edges_by_source`). URL
path segments are percent-encoded (`_encode_expr`).

The MORK surface is covered by live-server tests
(`tests/metta/test_mork_http.py`, auto-skip when no server is reachable),
benchmarked by `benchmarks/milestone4/mork_http_bench.py` (artifact:
`output/benchmarks/milestone4/m4_mork_http.json`), and demonstrated by
`examples/milestone4/m4e9_mork.py`.

---

## Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `METTA_GRAPH_DIR` | `data/jsonls/temporal_graph_output_v3` | M3 graph artifacts dir |
| `METTA_USE_LLM` | `false` | Enable LLM-refined answers in M3 |
| `METTA_NLG_MODEL` | `gpt-4.1-nano` | M1 generator model |
| `METTA_POLISH_THRESHOLD` | `0.7` | M1 polish routing threshold |
| `METTA_TRACE_SAMPLING_RATE` | `1.0` | M2 trace sampling |
| `METTA_NARRATIVE_STYLE` | `neutral` | `neutral`/`novice`/`expert` |
| `METTA_NARRATIVE_DOMAIN` | `general` | `general`/`medical`/`finance` |
| `MORK_BINARY` | — | Path to the `mork` executable |
| `MORK_ARGS` | — | Override MORK CLI args (`{file}` placeholder) |
| `MORK_TIMEOUT_S` | `30` | MORK subprocess timeout |
| `MORK_SERVER_URL` | `http://127.0.0.1:8000` | MORK HTTP atomspace server URL |

---

## Examples

All in `examples/milestone4/` (each has a `.metta` program + a `.py` runner):

| File | Milestone(s) | Demonstrates |
|------|--------------|--------------|
| `m4e1_nlg.{metta,py}` | M1 | NLG, readability, accuracy from MeTTa |
| `m4e2_tms.{metta,py}` | M2 | Trace, beliefs, contradictions, why-not, retraction |
| `m4e3_graph.{metta,py}` | M3 | Graph QA, evidence, confidence, Mermaid |
| `m4e4_capstone.{metta,py}` | M1+M2+M3 | Explainable temporal inference path (headline) |
| `m4e5_justification_path.{metta,py}` | M2 | Multi-hop justification chain extraction (deep debugging) |
| `m4e6_counterfactual.{metta,py}` | M1+M2+M3 | Fact-level + belief-level counterfactual reasoning |
| `m4e7_styles.{metta,py}` | M1+M3 | Same path explained novice/neutral/expert + finance domain |
| `m4e8_metta_reasoning.{metta,py}` | M2+MeTTa | Reasoning *in* MeTTa: `match` derives the causal link that enters the trace |
| `m4e9_mork.py` | MORK | Temporal edge upload / pattern export / server-side transform on a live MORK atomspace |
| `run_all.py` | all | Runs every example; `--skip-graph` for fast subset |

Run with `python examples/milestone4/run_all.py`. Examples 3–4 load the M3 graph
pipeline and model backends; pass `--skip-graph` for the fast M1+M2 subset.

---

## Testing

```
tests/metta/
├── test_bridge.py        # M1/M2/M3 bridge wiring — no hyperon needed
├── test_atoms.py         # operation registry + no-hyperon fallback
├── test_new_ops.py       # M4 ops incl. atom-returning ops (unwrap=False) — no hyperon needed
├── test_runner.py        # runner construction + MORK adapters (pure helpers) — no hyperon needed
├── test_mork_http.py     # MORK HTTP atomspace against a live server (auto-skips if absent)
└── test_integration.py   # full .metta programs via hyperon (auto-skips if absent)
```

```bash
pytest -q -o addopts="" tests/metta
# Python ≤3.12 with hyperon + live MORK server: 62 pass, 1 skip
# without hyperon / MORK server: dependent tests skip cleanly
```

The integration tests are guarded by `hyperon_available()` and the MORK tests by
`mork_http_available()`, so the suite is green in any environment. `hyperon`
ships wheels only up to CPython 3.12 (0.2.10) — on CPython 3.13 (the project
`.venv`) the MeTTa integration tests skip; use the system Python 3.11 for the
full suite. The bridge, atoms, and runner are fully tested without `hyperon`,
so M4 is verifiable in any environment.
