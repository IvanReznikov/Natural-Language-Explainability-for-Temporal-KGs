# Temporal NLG Examples

Milestone-aligned, runnable examples for the Temporal NLG library across M1, M2,
and M3.

The recommended project solution path is Milestone 3 graph retrieval + LCEL
querying (built in this repository), then fallback to earlier milestone examples
for component-level debugging.

## How to run

From the repo root:

```bash
python examples/milestone1/m1e1_delta_encoding_example.py
python examples/milestone1/m1e1_changelog_example.py
python examples/milestone1/m1e1_graph_indexing_example.py
python examples/milestone1/m1e1_visualization_example.py
python examples/milestone1/m1e3_hybrid_generation_example.py
python examples/milestone1/m1e4_counterfactual_example.py
python examples/milestone1/m1e4_tms_justification_example.py
python examples/milestone1/m1e2_lora_inference_example.py
python examples/milestone2/m2e2_intent_example.py
python examples/milestone2/m2e4_taxonomy_example.py
python examples/milestone2/m2e5_trace_meta_query_example.py
python examples/milestone2/m2e6_trigger_chain_example.py
python examples/milestone2/m2e7_harness_example.py

# Milestone 3 (all experiment families)
python examples/milestone3/m3e1_dataset_overview_example.py
python examples/milestone3/m3e2_fidelity_summary_example.py
python examples/milestone3/m3e3_human_eval_overview_example.py
python examples/milestone3/m3e4_quality_summary_example.py
python examples/milestone3/m3e5_matrix_overview_example.py
python examples/milestone3/m3e5_lcel_query_example.py
```

## Milestone 1

- `m1e1_delta_encoding_example.py`: Render only new facts between two snapshots.
- `m1e1_changelog_example.py`: Turn milestone updates into readable changelog lines.
- `m1e1_graph_indexing_example.py`: Build a simple entity-to-text index.
- `m1e1_visualization_example.py`: Text timelines for intervals and overlaps.
- `m1e3_hybrid_generation_example.py`: Hybrid strategy routing and generation.
- `m1e4_counterfactual_example.py`: Counterfactual generation workflow.
- `m1e4_tms_justification_example.py`: TMS justification flow.
- `m1e2_lora_inference_example.py`: Local LoRA inference smoke path.

## Milestone 2

- `m2e2_intent_example.py`: Train and evaluate the M2-E2 multi-label intent classifier on a held-out slice.
- `m2e4_taxonomy_example.py`: Load the trained taxonomy model and run label predictions on sample queries.
- `m2e5_trace_meta_query_example.py`: Record rule-firing traces and run the meta-query interface over them.
- `m2e6_trigger_chain_example.py`: Fire trigger rules against context facts, store queries, reify results, and mark stale entries.
- `m2e7_harness_example.py`: Run the full M2-E7 end-to-end harness: query generation → trace injection → report.

## Milestone 3

- `m3e1_dataset_overview_example.py`: inspect dataset size and domain distribution from `temporal_graph.jsonl`.
- `m3e2_fidelity_summary_example.py`: read and print M3-E2 fidelity summary metrics.
- `m3e3_human_eval_overview_example.py`: inspect M3-E3 human-evaluation response aggregates if present.
- `m3e4_quality_summary_example.py`: summarize M3-E4a/E4c/E4d quality outputs.
- `m3e5_matrix_overview_example.py`: show M3-E5 matrix coverage and top runs.
- `m3e5_lcel_query_example.py`: LCEL graph query pipeline + built-in Mermaid subgraph output.

All milestone3 examples import and use `src/temporal_nlg` package modules.

Suggested run order for final solution demos:
1. `python examples/milestone3/m3e5_lcel_query_example.py`
2. `python experiments/m3_e5_benchmark/run_m3_e5.py --list --output-dir output/m3_e5_results`
3. `python experiments/m3_e5_benchmark/run_m3_e5.py --aggregate --output-dir output/m3_e5_results`

## Directory structure

```
examples/
|-- README.md
|-- milestone1/
|   |-- m1e1_delta_encoding_example.py
|   |-- m1e1_changelog_example.py
|   |-- m1e1_graph_indexing_example.py
|   |-- m1e1_visualization_example.py
|   |-- m1e2_lora_inference_example.py
|   |-- m1e3_hybrid_generation_example.py
|   |-- m1e4_counterfactual_example.py
|   `-- m1e4_tms_justification_example.py
|-- milestone2/
|   |-- m2e2_intent_example.py
|   |-- m2e4_taxonomy_example.py
|   |-- m2e5_trace_meta_query_example.py
|   |-- m2e6_trigger_chain_example.py
|   `-- m2e7_harness_example.py
`-- milestone3/
    |-- m3e1_dataset_overview_example.py
    |-- m3e2_fidelity_summary_example.py
    |-- m3e3_human_eval_overview_example.py
    |-- m3e4_quality_summary_example.py
    |-- m3e5_matrix_overview_example.py
    `-- m3e5_lcel_query_example.py
```


