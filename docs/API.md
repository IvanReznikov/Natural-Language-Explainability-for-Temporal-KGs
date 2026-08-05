# Temporal NLG API Documentation

## Overview

The Temporal NLG library provides production-ready tools for generating natural language explanations from temporal graph structures. The library includes:

- **Template-based generation** - Fast, accurate rendering using predefined patterns
- **LLM-based generation** - Flexible generation using language models  
- **Hybrid generation** - Intelligent routing between templates and LLMs
- **Accuracy evaluation** - Factual accuracy and quality assessment

## Installation

```bash
pip install -e .
```

**Requirements:**
- Python 3.8+
- langchain-openai (for LLM features)
- OpenAI API key (for LLM features)

## Quick Start

```python
from temporal_nlg.core.templates import TemporalFact, TemplateType
from temporal_nlg.models import HybridGenerator

# Create temporal fact
fact = TemporalFact(
    fact_type=TemplateType.POINT_IN_TIME,
    event="was born",
    entity="Albert Einstein",
    date="1879"
)

# Generate explanation
generator = HybridGenerator()
result = generator.generate(fact)

print(result.text)
# "Albert Einstein was born in 1879."
```

## Core Modules

### 1. Core Templates (`temporal_nlg.core.templates`)

#### `TemporalFact`
Represents a temporal fact to be verbalized.

**Attributes:**
- `fact_type: TemplateType` - Type of temporal relationship
- `event: str` - Event name
- `entity: str` - Primary entity
- `date: str` - Point-in-time date
- `start_date: str` - Interval start
- `end_date: str` - Interval end
- `dates: List[str]` - Sequence dates
- `events: List[str]` - Sequence events
- `context: str` - Additional context
- `cause: str` - Causal relationship cause
- `effect: str` - Causal relationship effect

**Example:**
```python
fact = TemporalFact(
    fact_type=TemplateType.INTERVAL,
    entity="World War II",
    event="lasted",
    start_date="1939",
    end_date="1945"
)
```

#### `TemplateType` (Enum)
Supported temporal relationship types:
- `POINT_IN_TIME` - Single timestamp events
- `INTERVAL` - Time-bounded periods
- `SEQUENCE` - Ordered event chains
- `CAUSALITY` - Causal relationships
- `OVERLAP` - Concurrent events

#### `TemplateRenderer`
Renders temporal facts using templates.

**Methods:**
- `render(fact: TemporalFact) -> str` - Render fact to text
- `get_templates(fact_type: TemplateType) -> List[Template]` - Get available templates

**Example:**
```python
renderer = TemplateRenderer()
output = renderer.render(fact)
print(renderer.last_template_id)  # "point_in_time_01"
```

### 2. Models (`temporal_nlg.models`)

#### `LLMGenerator`
Pure LLM-based generation.

**Constructor:**
```python
LLMGenerator(
    model: str = "gpt-4.1-nano",
    temperature: float = 0.0,
    max_tokens: int = 50
)
```

**Methods:**
- `generate(fact: TemporalFact) -> str` - Generate explanation
- `batch_generate(facts: List[TemporalFact], show_progress: bool = False) -> List[str]` - Batch generation

**Example:**
```python
generator = LLMGenerator(model="gpt-4.1-nano")
output = generator.generate(fact)
```

**Performance:**
- Speed: ~1-2s per fact (gpt-4.1-nano)
- Cost: ~$0.0001 per fact
- Quality: High flexibility, variable accuracy

#### `HybridGenerator`
Intelligent template-LLM hybrid.

**Constructor:**
```python
HybridGenerator(
    model: str = "gpt-4.1-nano",
    polish_threshold: float = 0.7,
    enable_caching: bool = True
)
```

**Methods:**
- `generate(fact: TemporalFact, force_strategy: Optional[str] = None) -> GenerationResult` - Generate with routing
- `batch_generate(facts: List[TemporalFact], show_progress: bool = False) -> List[GenerationResult]` - Batch generation
- `get_stats() -> Dict[str, Any]` - Get statistics

**Example:**
```python
generator = HybridGenerator()

# Automatic routing
result = generator.generate(fact)
print(result.text, result.strategy)

# Force specific strategy
result = generator.generate(fact, force_strategy="template")
```

**Routing Logic:**
- Simple facts (1-2 fields) -> `template` (fast, accurate)
- Medium complexity (intervals, context) -> `polish` (template + LLM refinement)
- Complex facts (sequences, multiple events) -> `llm` (pure LLM)

#### `GenerationResult`
Result container from hybrid generation.

**Attributes:**
- `text: str` - Generated text
- `strategy: str` - Strategy used ("template", "polished", "llm")
- `confidence: float` - Confidence score (0-1)
- `flesch_score: Optional[float]` - Readability score
- `template_id: Optional[str]` - Template identifier

### 3. Evaluation (`temporal_nlg.evaluation`)

#### `AccuracyEvaluator`
Evaluate factual accuracy of generated text.

**Methods:**
- `evaluate(fact: Any, generated_text: str) -> AccuracyMetrics` - Single evaluation
- `batch_evaluate(facts: List[Any], texts: List[str]) -> List[AccuracyMetrics]` - Batch evaluation
- `aggregate_metrics(metrics: List[AccuracyMetrics]) -> Dict[str, float]` - Aggregate statistics

**Example:**
```python
evaluator = AccuracyEvaluator()
metrics = evaluator.evaluate(fact, output)

print(f"Accuracy: {metrics.overall_accuracy:.2%}")
print(f"Hallucination: {metrics.hallucination_detected}")
```

#### `AccuracyMetrics`
Accuracy evaluation results.

**Attributes:**
- `date_preservation: float` - Date/temporal accuracy (0-1)
- `entity_preservation: float` - Entity accuracy (0-1)
- `relation_preservation: float` - Relation accuracy (0-1)
- `hallucination_detected: bool` - Hallucination flag
- `overall_accuracy: float` - Weighted overall score (0-1)

#### Quality Metrics Functions

**`calculate_flesch_score(text: str) -> float`**
Calculate Flesch Reading Ease score (0-100, higher = easier).

**`calculate_information_density(text: str) -> float`**
Calculate information density (content words / total words).

**Example:**
```python
from temporal_nlg.evaluation import calculate_flesch_score

flesch = calculate_flesch_score("Einstein was born in 1879.")
# ~70 (easy to read)
```

## Contents
- [Core Templates](#core-templates-temporal_nlgcoretemplates)
- [Models](#models-temporal_nlgmodels)
- [Evaluation](#evaluation-temporal_nlgevaluation)
- [Milestone 2 APIs](#milestone-2-apis-trace-triggers-stores)
- [Usage Patterns](#usage-patterns)
- [Configuration](#configuration)
- [Performance Guidelines](#performance-guidelines)
- [Error Handling](#error-handling)
- [See Also](#see-also)

## Usage Patterns

### Milestone 2 APIs (Trace, Triggers, Stores)

#### Trace Recording (`temporal_nlg.tms.trace`)
- `TraceRecorder.session(query_id, meta=None)` -> context manager yielding `QueryTrace`.
- `TraceRecorder.record_rule_firing(trace, rule_id, rule_name, inputs, conclusion, confidence=1.0, latency_ms=None, meta=None)`
- Data classes: `QueryTrace`, `RuleTrace` (serialize via `.to_dict()` / `.from_dict()`).

**Minimal pattern:**
```python
from temporal_nlg.tms.trace import TraceRecorder

rec = TraceRecorder()
with rec.session("q123", meta={"user": "alice"}) as tr:
    rec.record_rule_firing(
        tr,
        rule_id="r1",
        rule_name="detect_event",
        inputs=[{"fact_id": "f1", "value": 42}],
        conclusion={"fact_id": "intent", "value": "medical"},
        confidence=0.9,
        latency_ms=8.0,
        meta={"extra_conclusions": [{"fact_id": "diagnosis", "value": "flu"}]},
    )
trace_json = tr.to_json()
```

#### Meta-Queries (`temporal_nlg.tms.meta_query`)
- `rules_fired(trace)` -> list of rule_ids
- `explain_fact(trace, fact_id)` -> text explanation
- `influential_facts(trace, top_k=5)` -> ranked fact IDs
- `why_not_fired(trace, expected_rules)` -> diagnostics
- `contradictions(trace)` -> conflicting conclusions

CLI: `python experiments/m2_e5_trace_meta_query/meta_query_cli.py <trace.jsonl> list-rules|explain-fact <fact>|contradictions|influential --top-k 3|why-not <rule...>`

#### Contradiction Detection (`temporal_nlg.tms.contradiction`)
- `ContradictionDetector.detect(trace)` -> contradictions by fact_id/value

#### Query & Result Stores (`temporal_nlg.tms.query_store`, `temporal_nlg.tms.result_store`)
- `QueryStore(path).upsert(query_id, text, intent, dependencies=None, meta=None)`
- `ResultStore(path).upsert(result_id, query_id, results, freshness, dependent_facts=None, invalidation_rules=None)`
- `ResultStore.mark_stale_by_facts(fact_ids)` -> marks and returns stale results

#### Trigger Engine (`temporal_nlg.tms.trigger_engine`)
- `TriggerRule(rule_id, name, predicate: (TriggerContext)->bool, factory: (TriggerContext)->dict)`
- `TriggerEngine(store).evaluate(ctx, rules)` -> list of triggered query_ids

**Example chain:**
```python
from temporal_nlg.tms.trigger_engine import TriggerContext, TriggerEngine, TriggerRule
from temporal_nlg.tms.query_store import QueryStore
from temporal_nlg.tms.result_store import ResultStore

qs = QueryStore(path="/tmp/queries.jsonl")
rs = ResultStore(path="/tmp/results.jsonl")

def pred(ctx):
    return ctx.facts.get("temperature", 0) > 101

def factory(ctx):
    return {"query_id": f"q_{ctx.context_id}", "text": "Antibiotics?", "intent": "medical"}

rules = [TriggerRule("r_high_temp", "High fever", pred, factory)]
engine = TriggerEngine(qs)
triggered = engine.evaluate(TriggerContext("c1", {"temperature": 102}, {}), rules)

for qid in triggered:
    rs.upsert(result_id=f"res_{qid}", query_id=qid, results=[{"value": "demo"}], freshness={"generated_at": "now"})
```

#### End-to-End Harness (M2-E7)
- CLI: `python experiments/m2_e7_harness/run_e2e.py --queries experiments/m2_e7_harness/input/queries.jsonl --trace experiments/m2_e7_harness/input/trace.jsonl --output experiments/m2_e7_harness/output/results.jsonl --report experiments/m2_e7_harness/output/report.json`
- Helpers: `build_results_from_traces(traces)`, `build_expected_map(queries, default_required_facts=("intent",), default_max_latency_ms=15.0)`, `eval_results(results, expected_map)`.
- Specs: see experiments/m2_e7_harness/CORPUS_SPEC.md, TRACE_SPEC.md, QUERY_ANNOTATION_GUIDE.md.

### Pattern 1: Simple Template Rendering
For maximum speed and accuracy with simple facts:

```python
from temporal_nlg.core.templates import TemplateRenderer

renderer = TemplateRenderer()
output = renderer.render(fact)
```

### Pattern 2: LLM for Complex Facts
For complex facts needing flexibility:

```python
from temporal_nlg.models import LLMGenerator

generator = LLMGenerator()
output = generator.generate(fact)
```

### Pattern 3: Hybrid for Best of Both
Recommended for production (automatic quality/cost tradeoff):

```python
from temporal_nlg.models import HybridGenerator

generator = HybridGenerator(enable_caching=True)
result = generator.generate(fact)
```

### Pattern 4: Quality Assessment
Always evaluate before deployment:

```python
from temporal_nlg.evaluation import AccuracyEvaluator

evaluator = AccuracyEvaluator()
metrics = evaluator.evaluate(fact, output)

if metrics.overall_accuracy > 0.8:
    print("High quality!")
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY` - Required for LLM features
- `OPENAI_ORG` - Optional organization ID

### Model Selection

**Speed-optimized:**
```python
generator = HybridGenerator(model="gpt-4.1-nano")  # Fast, cheap
```

**Quality-optimized:**
```python
generator = HybridGenerator(model="gpt-4")  # Slower, more expensive
```

## Performance Guidelines

### Template Rendering
- **Speed:** <10ms per fact
- **Cost:** Free
- **Quality:** 85-95% accuracy (simple facts)
- **Best for:** High-volume, simple facts

### LLM Generation
- **Speed:** 1-2s per fact (gpt-4.1-nano)
- **Cost:** ~$0.0001 per fact
- **Quality:** 70-90% accuracy (complex facts)
- **Best for:** Complex, unique facts

### Hybrid Generation
- **Speed:** 10ms-2s (adaptive)
- **Cost:** $0.00-$0.0001 per fact
- **Quality:** 80-95% accuracy (all types)
- **Best for:** Production systems

## Error Handling

All generators raise `ValueError` on failure:

```python
try:
    output = generator.generate(fact)
except ValueError as e:
    print(f"Generation failed: {e}")
    # Fallback logic
```

## See Also

- [Examples](../examples/) - Usage examples
- [Tests](../tests/) - Unit tests
- [Experiments](../experiments/) - Research code
- [REPOSITORY_STRUCTURE_GUIDE.md](../REPOSITORY_STRUCTURE_GUIDE.md) - Best practices

