# Milestone 2 Technical Documentation

## Scope
- M2-E2: Query intent classification (multi-label baseline, TF-IDF + LR/SVM)
- M2-E3: Parser + intent hybrid (Flan-T5-small + rules + MiniLM intent)
- M2-E4: Result taxonomy classification (word/char TF-IDF + LinearSVC)
- M2-E5: Trace recording, meta-queries, contradiction detection
- M2-E6: Query/result stores, trigger engine, stale marking
- M2-E7: End-to-end harness (annotated corpus + traces + evaluation)

## Data Inventory
- Intent (E2): experiments/m2_e2_intent/data/annotated_queries.jsonl (1,160 rows; 8 intents)
- Parser (E3): experiments/m2_e3_parse/data/temporal_queries_gold.jsonl (1,137 rows; seed=13 splits)
- Taxonomy (E4): experiments/m2_e4_taxonomy/data/{taxonomy.jsonl,summaries.jsonl,narrative_consistency.jsonl} (~1.4k rows each), splits at experiments/m2_e4_taxonomy/data/splits
- Traces (E5): synthetic via experiments/m2_e5_trace_meta_query/generate_trace_corpus.py (default 1000); curated sample experiments/m2_e5_trace_meta_query/output/small_traces.jsonl
- Queries (E6): synthetic via experiments/m2_e6_query_store_triggers/generate_query_corpus.py (default 500)
- E2E corpora (E7): experiments/m2_e7_harness/input/queries.jsonl (2,210) + experiments/m2_e7_harness/input/trace.jsonl (2,210)

## Components

### M2-E2 - Intent Classification
- Dataset: multi-label intents {point_in_time, interval, sequence, causal, comparative, aggregation, prediction, explanation}; split 20% test, 10% val.
- Model space: word TF-IDF, optional char TF-IDF, Logistic Regression OVR with optional calibration; threshold grid and per-label fallback avoidance.
- Artifacts: metrics per run at experiments/m2_e2_intent/results/<run_id>/metrics.json; best macro/micro/subset from isotonic-calibrated word+char (T13/T15).
- Usage: `python experiments/m2_e2_intent/run_intent_classifier.py --dataset experiments/m2_e2_intent/data/annotated_queries.jsonl --output-dir experiments/m2_e2_intent/results --use-char-ngrams --char-ngram-min 3 --char-ngram-max 5 --char-max-features 20000 --calibration isotonic --threshold 0.45`.

### M2-E3 - Parser + Intent Hybrid
- Data: experiments/m2_e3_parse/data/temporal_queries_gold.jsonl with splits at experiments/m2_e3_parse/data/splits (train 909 / val 113 / test 115).
- Models: intent encoder (sentence-transformers/all-MiniLM-L6-v2, 3 epochs), parser (Flan-T5-small, 3 epochs); artifacts in experiments/m2_e3_parse/artifacts/{intent,parser}.
- Inference: experiments/m2_e3_parse/run_parse.py combines model output with rule-based validation/fallback for temporal frames; zero validation errors on latest run (test 115 rows).
- Command example: `PYTHONPATH=. .venv/Scripts/python.exe experiments/m2_e3_parse/run_parse.py --data experiments/m2_e3_parse/data/splits/test.jsonl --use-model --fallback-on-error --output-dir output/m2_e3_eval`.

### M2-E4 - Result Taxonomy Classification
- Data: taxonomy splits at experiments/m2_e4_taxonomy/data/splits/result_taxonomy_{train,val,test}.jsonl (1126/136/148; seed=13); labels: anomaly, category_breakdown, comparison, error, exact_numeric, noop_empty, time_series.
- Default model: E6 word+char TF-IDF + LinearSVC (CPU-only) stored at experiments/m2_e4_taxonomy/output/e4a_taxonomy/taxonomy_model.joblib with metrics + sweep_taxonomy.json.
- Training: `python experiments/m2_e4_taxonomy/train_taxonomy_default.py --output-dir experiments/m2_e4_taxonomy/output/e4a_taxonomy`.
- Inference: `python experiments/m2_e4_taxonomy/predict_taxonomy.py --text "The result is a numeric value for revenue in Q4"` (override --model as needed).

### M2-E5 - Trace Recorder, Meta-Queries, Contradictions
- Core APIs: temporal_nlg.tms.trace.{TraceRecorder,QueryTrace,RuleTrace}, temporal_nlg.tms.meta_query (rules_fired, explain_fact, influential_facts, why_not_fired), temporal_nlg.tms.contradiction.ContradictionDetector.
- CLI: experiments/m2_e5_trace_meta_query/meta_query_cli.py over JSONL traces (list-rules, explain-fact <fact>, contradictions, influential --top-k, why-not <rule...>).
- Generation: experiments/m2_e5_trace_meta_query/generate_trace_corpus.py (default 1000 traces), sample traces at experiments/m2_e5_trace_meta_query/output/small_traces.jsonl.
- Performance: trace_bench.txt (500 traces x 3 rules) -> mean overhead 0.005 ms (p95 0.006 ms, max 0.026 ms), mean trace size ~1.4 KB.
- Integration sketch:
```
from temporal_nlg.tms.trace import TraceRecorder

recorder = TraceRecorder()
with recorder.session("query123", meta={"user": "u1"}) as trace:
    recorder.record_rule_firing(
        trace,
        rule_id="rule_a",
        rule_name="rule_a",
        inputs=[{"fact_id": "f1", "value": 42}],
        conclusion={"fact_id": "g1", "value": 43},
        confidence=0.9,
    )
```

### M2-E6 - Query Store, Result Store, Trigger Engine
- Query storage/reification: temporal_nlg.tms.query_store.QueryStore, temporal_nlg.tms.result_store.ResultStore.
- Triggering: temporal_nlg.tms.trigger_engine.TriggerEngine + TriggerRule evaluating predicates over TriggerContext.
- Scripts: experiments/m2_e6_query_store_triggers/{trigger_demo.py,change_detection_demo.py,e2e_chain_demo.py,query_store_benchmark.py} plus generate_query_corpus.py (--count default 500).
- Example chain: `python experiments/m2_e6_query_store_triggers/e2e_chain_demo.py` -> triggers queries, writes experiments/m2_e6_query_store_triggers/output/e2e_queries.jsonl + e2e_results.jsonl, marks stale results when dependent facts change.

### M2-E7 - End-to-End Harness
- Specs: CORPUS_SPEC.md (query schema), TRACE_SPEC.md (trace schema), QUERY_ANNOTATION_GUIDE.md (required_facts, max_latency_ms defaults intent/15ms).
- Generator: experiments/m2_e7_harness/generate_e2e_queries.py (synthetic queries) and experiments/m2_e7_harness/generate_traces.py (traces with required_facts + intent conclusions and extra_conclusions meta).
- Harness: experiments/m2_e7_harness/run_e2e.py builds results from traces, expands extra conclusions, evaluates intent/fact coverage/latency, writes results.jsonl + report.json.
- Usage: `python experiments/m2_e7_harness/run_e2e.py --queries experiments/m2_e7_harness/input/queries.jsonl --trace experiments/m2_e7_harness/input/trace.jsonl --output experiments/m2_e7_harness/output/results.jsonl --report experiments/m2_e7_harness/output/report.json`.
- Latest run: 2,210/2,210 queries pass (failures=0) after regenerating traces with required_facts + intent in conclusions.

## Repro Notes
- CPU-only; scikit-learn + sentence-transformers + Flan-T5-small.
- Stable artifact filenames noted above; reruns are deterministic with seeds where specified (E2/E3/E4/E5/E6/E7 generators accept --seed/--count).

## Cross-References
- APIs: [docs/API.md](docs/API.md) for trace/meta-query, trigger/store, and harness helpers.
- Architecture: [docs/ADDITIONAL_M2.md](docs/ADDITIONAL_M2.md) (Milestone 2 Architecture section) for component map, data inventory, runtime flow.
- Results: [docs/RESULTS_M2.md](docs/RESULTS_M2.md) for experiment matrices and benchmark summaries.
- Examples: [examples/milestone2](examples/milestone2) for runnable scripts covering M2-E2 through M2-E7.

