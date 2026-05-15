# M3-E3 - Comprehension & Utility Metrics

This folder contains the **M3-E3** study tooling:
- **M3-E3a** comprehension assessment export + analysis
- **M3-E3b** utility study export + analysis
- **M3-E3c** cognitive load (NASA TLX) export + analysis

The provided web UIs are **static HTML** files intended for local/offline data collection.
They load exported JSONL files and allow participants to download their response JSONL.

## M3-E3a: Comprehension

Export 50 explanations x 5 questions each:

```bash
python experiments/m3_e3_human_eval/run_comprehension.py export \
  --dataset data/jsonls/temporal_graph.jsonl \
  --output-dir output/m3_e3a_comprehension \
  --n-items 50 \
  --questions-per-item 5 \
  --seed 13
```

Collect responses:
- Open `output/m3_e3a_comprehension/web/comprehension.html`
- Load `output/m3_e3a_comprehension/m3_e3a_tasks.jsonl`
- Download `m3_e3a_responses.jsonl`

Analyze:

```bash
python experiments/m3_e3_human_eval/run_comprehension.py analyze \
  --tasks output/m3_e3a_comprehension/m3_e3a_tasks.jsonl \
  --responses path/to/m3_e3a_responses.jsonl \
  --output-dir output/m3_e3a_comprehension_analysis
```

Outputs:
- `m3_e3a_comprehension.summary.json`
- `m3_e3a_comprehension.scored_responses.jsonl`

## M3-E3b: Utility

Export per-domain task pool:

```bash
python experiments/m3_e3_human_eval/run_utility.py export \
  --dataset data/jsonls/temporal_graph.jsonl \
  --output-dir output/m3_e3b_utility \
  --tasks-per-domain 50 \
  --seed 13
```

Collect responses:
- Open `output/m3_e3b_utility/web/utility.html`
- Load `output/m3_e3b_utility/m3_e3b_tasks.jsonl`
- Optionally load `output/m3_e3b_utility/m3_e3b_assignments.csv` (fill participant rows as you recruit users)
- Download `m3_e3b_responses.jsonl`

Analyze:

```bash
python experiments/m3_e3_human_eval/run_utility.py analyze \
  --tasks output/m3_e3b_utility/m3_e3b_tasks.jsonl \
  --responses path/to/m3_e3b_responses.jsonl \
  --output-dir output/m3_e3b_utility_analysis
```

## M3-E3c: Cognitive load

Export scenario set:

```bash
python experiments/m3_e3_human_eval/run_cognitive_load.py export \
  --dataset data/jsonls/temporal_graph.jsonl \
  --output-dir output/m3_e3c_cognitive_load \
  --n-scenarios 40 \
  --seed 13
```

Collect responses:
- Open `output/m3_e3c_cognitive_load/web/cognitive_load.html`
- Load `output/m3_e3c_cognitive_load/m3_e3c_scenarios.jsonl`
- Download `m3_e3c_responses.jsonl`

Analyze:

```bash
python experiments/m3_e3_human_eval/run_cognitive_load.py analyze \
  --responses path/to/m3_e3c_responses.jsonl \
  --output-dir output/m3_e3c_cognitive_load_analysis
```

