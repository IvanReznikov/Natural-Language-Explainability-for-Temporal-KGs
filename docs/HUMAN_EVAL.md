# Human Evaluation Guide

This guide defines how to collect the 50+ human-scored outputs required for M1 success criteria (clarity and accuracy >85%).

## Milestone Coverage
- M1: This document is the primary manual guide.
- M2: No dedicated human-evaluation protocol is defined in milestone-2 experiment docs; M2 evaluation is automated/benchmark driven.
- M3: Human studies are implemented under `experiments/m3_e3_human_eval` and documented in `experiments/m3_e3_human_eval/README.md`.

## M1 Scope
- Evaluate generated explanations across all 5 template types (point, interval, sequence, causality, overlap).
- Cover at least 10 samples per type (50 total minimum). Include both template and hybrid/LLM paths.

## M1 Procedure
1. Prepare stimuli
   - Generate explanations using `HybridGenerator` (recommended) or `TemplateRenderer` for baseline.
   - Include input fact metadata (entity, event, dates, type, strategy used).
2. Recruit raters
   - Minimum 3 independent raters; avoid project authors to reduce bias.
   - Provide a short rubric with example scores.
3. Rating rubric (per item)
   - Clarity (1-5): 1=unreadable, 3=understandable with effort, 5=clear and concise.
   - Factual accuracy (1-5): 1=incorrect, 3=partially correct, 5=fully correct per fact.
   - Optional: Naturalness (1-5) and Helpful justification (Yes/No).
4. Instructions to raters
   - Rate based only on the shown fact(s) and output text; do not search the web.
   - Mark “uncertain” if unsure; these can be re-scored later.
5. Data capture
   - Use `docs/human_eval_template.csv` as the reference schema.
   - Keep one row per judged output per rater.
6. Aggregation
   - Compute per-item mean clarity and accuracy; flag items with mean <4.25 (≈85%).
   - Report overall means and percentage of items meeting the threshold.


## Suggested commands to generate items
```bash
python experiments/m1_e5_integration/integration_tests.py  # sanity
python experiments/m1_e5_integration/performance_benchmarks.py  # optional
```
Use `src/temporal_nlg` APIs directly to render the 50+ evaluation outputs you will send to raters.

## Milestone 3 Human Evaluation

Milestone 3 includes dedicated human-study pipelines (comprehension, utility, cognitive load):
- M3-E3a comprehension: `experiments/m3_e3_human_eval/run_comprehension.py`
- M3-E3b utility: `experiments/m3_e3_human_eval/run_utility.py`
- M3-E3c cognitive load (NASA-TLX): `experiments/m3_e3_human_eval/run_cognitive_load.py`

The operational workflow (export tasks, collect participant responses via web UI, analyze responses) is documented in:
- `experiments/m3_e3_human_eval/README.md`

Related M3 fidelity work (M3-E2) also supports a human-loop path for judgement metrics via:
- `experiments/m3_e2_fidelity/run_fidelity_eval.py` (`--human-loop human`)
