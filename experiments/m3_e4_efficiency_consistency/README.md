# M3-E4 - Efficiency & Consistency Metrics (Plan)

This plan covers four sub-studies:
- **M3-E4a**: Generation efficiency benchmarking
- **M3-E4b**: Consistency under fact revisions
- **M3-E4c**: Cross-explanation coherence
- **M3-E4d**: Granularity robustness

## Goals (from M3 spec)
- **Latency**: simple <200ms, complex <2000ms
- **Consistency under revisions**: update accuracy >98%, contradiction detection >95%, coherence >4/5, resolution <5s
- **Coherence across styles**: semantic consistency >90%, narrative consistency >85%, logical consistency >95%
- **Granularity robustness**: quality >80% across 7 scales

## Proposed Inputs
- **Base scenarios**: sample from `data/jsonls/temporal_graph.jsonl` with domain + time_scope coverage.
- **Granularity levels**: seconds, minutes, hours, days, months, years, decades.
- **Explanation styles**: template-based, seq2seq, LLM, hybrid (configure as methods).
- **Revision types**: date correction, new causal fact, contradiction injection, fact removal.

## Data Products (per sub-study)
### M3-E4a Efficiency
- `m3_e4a_scenarios.jsonl` (1k scenarios x 5 complexity levels)
- `m3_e4a_runs.jsonl` (method x scenario x level; latency, tokens, cost, quality proxy)
- `m3_e4a_summary.json` (p50/p95 latency, throughput, cost, quality/latency Pareto)

### M3-E4b Consistency Under Revisions
- `m3_e4b_facts.jsonl` (1k facts with base explanation)
- `m3_e4b_revisions.jsonl` (revision deltas + expected behavior)
- `m3_e4b_results.jsonl` (update accuracy, contradiction flags, coherence ratings)
- `m3_e4b_summary.json`

### M3-E4c Cross-Explanation Coherence
- `m3_e4c_scenarios.jsonl` (100 scenarios)
- `m3_e4c_explanations.jsonl` (5 styles per scenario)
- `m3_e4c_scores.jsonl` (semantic/narrative/logical consistency)
- `m3_e4c_summary.json`

### M3-E4d Granularity Robustness
- `m3_e4d_scenarios.jsonl` (200 scenarios)
- `m3_e4d_scaled.jsonl` (7 granularities per scenario)
- `m3_e4d_scores.jsonl` (quality by scale + length stats)
- `m3_e4d_summary.json`

## Metrics & Methods (initial plan)
- **Latency/throughput**: measured in-process; store wall-clock ms and tokens/sec.
- **Cost**: optional API cost tracking if LLM is used (token-based).
- **Quality proxy**: reuse M3-E2 fidelity proxy where possible; allow pluggable scorers.
- **Semantic consistency**: embedding similarity (e.g., cosine on sentence embeddings).
- **Narrative consistency**: overlap of key entities/relations + temporal order alignment.
- **Logical consistency**: rule checks for contradictions and temporal ordering violations.
- **Granularity checks**: unit appropriateness + length scaling vs. target granularity.

## CLI Plan (mirrors M3-E2/E3 style)
- `experiments/m3_e4_efficiency_consistency/run_efficiency.py` (export/analyze)
- `experiments/m3_e4_efficiency_consistency/run_consistency.py` (export/analyze)
- `experiments/m3_e4_efficiency_consistency/run_coherence.py` (export/analyze)
- `experiments/m3_e4_efficiency_consistency/run_granularity.py` (export/analyze)

## Usage

### Generate prediction files (methods/styles/granularity)

```bash
python experiments/m3_e4_efficiency_consistency/run_generate_predictions.py efficiency \
	--scenarios output/m3_e4a_efficiency/m3_e4a_scenarios.jsonl \
	--dataset data/jsonls/temporal_graph.jsonl \
	--output output/m3_e4a_efficiency/m3_e4a_method_predictions.jsonl \
	--runs-out output/m3_e4a_efficiency/m3_e4a_runs_generated.jsonl

python experiments/m3_e4_efficiency_consistency/run_generate_predictions.py coherence \
	--scenarios output/m3_e4c_coherence/m3_e4c_scenarios.jsonl \
	--dataset data/jsonls/temporal_graph.jsonl \
	--output output/m3_e4c_coherence/m3_e4c_style_predictions.jsonl

python experiments/m3_e4_efficiency_consistency/run_generate_predictions.py granularity \
	--scenarios output/m3_e4d_granularity/m3_e4d_scenarios.jsonl \
	--dataset data/jsonls/temporal_graph.jsonl \
	--output output/m3_e4d_granularity/m3_e4d_granularity_predictions.jsonl
```

### M3-E4a: Efficiency Benchmarking

Export scenarios + run template:

```bash
python experiments/m3_e4_efficiency_consistency/run_efficiency.py export \
	--dataset data/jsonls/temporal_graph.jsonl \
	--output-dir output/m3_e4a_efficiency \
	--n-scenarios 1000 \
	--seed 13
```

Analyze runs (reproduces `output/m3_e4a_efficiency_analysis_methods/m3_e4a_efficiency.summary.json`):

```bash
python experiments/m3_e4_efficiency_consistency/run_efficiency.py analyze \
	--runs output/m3_e4a_efficiency/m3_e4a_runs_generated.jsonl \
	--scenarios output/m3_e4a_efficiency/m3_e4a_scenarios.jsonl \
	--predictions output/m3_e4a_efficiency/m3_e4a_method_predictions.jsonl \
	--output-dir output/m3_e4a_efficiency_analysis_methods
```

### M3-E4b: Consistency Under Revisions

Export facts + revision templates:

```bash
python experiments/m3_e4_efficiency_consistency/run_consistency.py export \
	--dataset data/jsonls/temporal_graph.jsonl \
	--output-dir output/m3_e4b_consistency \
	--n-facts 1000 \
	--seed 13
```

Analyze results (run after filling `m3_e4b_results_template.csv` with evaluation scores):

```bash
python experiments/m3_e4_efficiency_consistency/run_consistency.py analyze \
	--results output/m3_e4b_consistency/m3_e4b_results_template.csv \
	--output-dir output/m3_e4b_consistency_analysis
```

> **Note:** `m3_e4b_results_template.csv` is an unfilled participant template (benchmark tooling deliverable).
> Fill the `update_accuracy`, `contradiction_detected`, `coherence_rating_1_5`, and `resolution_time_sec` columns
> before running the analyze step. An LLM simulation prompt is provided at
> `output/m3_e4b_consistency/llm_simulation_prompt.txt`.

### M3-E4c: Cross-Explanation Coherence

Export scenarios + style variants:

```bash
python experiments/m3_e4_efficiency_consistency/run_coherence.py export \
	--dataset data/jsonls/temporal_graph.jsonl \
	--predictions path/to/style_predictions.jsonl \
	--styles template,seq2seq,llm,hybrid,baseline \
	--output-dir output/m3_e4c_coherence \
	--n-scenarios 100 \
	--seed 13
```

Analyze scores (reproduces `output/m3_e4c_coherence_analysis_auto/m3_e4c_coherence.summary.json`):

```bash
python experiments/m3_e4_efficiency_consistency/run_coherence.py analyze \
	--explanations output/m3_e4c_coherence/m3_e4c_explanations.jsonl \
	--output-dir output/m3_e4c_coherence_analysis_auto
```

### M3-E4d: Granularity Robustness

Export granularity variants:

```bash
python experiments/m3_e4_efficiency_consistency/run_granularity.py export \
	--dataset data/jsonls/temporal_graph.jsonl \
	--predictions path/to/granularity_predictions.jsonl \
	--output-dir output/m3_e4d_granularity \
	--n-scenarios 200 \
	--seed 13
```

Analyze scores (reproduces `output/m3_e4d_granularity_analysis_methods/m3_e4d_granularity.summary.json`):

```bash
python experiments/m3_e4_efficiency_consistency/run_granularity.py analyze \
	--variants output/m3_e4d_granularity/m3_e4d_scaled.jsonl \
	--scenarios output/m3_e4d_granularity/m3_e4d_scenarios.jsonl \
	--output-dir output/m3_e4d_granularity_analysis_methods
```

## Success Criteria Checks
- Efficiency: complex p95 latency <2000ms
- Consistency: update accuracy >98%, contradiction detection >95%, coherence >4/5
- Coherence: semantic consistency >90%, narrative >85%, logical >95%
- Robustness: quality >=80% for all 7 granularities

## Next Implementation Steps
1. **Schemas** in `src/temporal_nlg/evaluation/m3_e4.py` for all JSONL records.
2. **Exporters** to generate scenario pools and revision/scale variants.
3. **Analyzers** to aggregate metrics + write summaries to `output/m3_e4_*`.
4. **Tests** under `tests/evaluation/test_m3_e4.py` for aggregation sanity.


