# Benchmarks

Performance benchmarks separated from correctness tests.

```
benchmarks/
|-- milestone1/           # M1 NLG library performance
|-- milestone2/           # M2 classifier and TMS performance
`-- milestone3/           # M3 pipeline and QA benchmark performance
```

Run a milestone benchmark:

```bash
python benchmarks/milestone1/run.py
python benchmarks/milestone2/run.py
python benchmarks/milestone3/run.py
```

Outputs go to `output/benchmarks/<milestone>/` with timestamps.

> **Note:** End-to-end QA system benchmark (multi-model, multi-mode) lives in
> `experiments/m3_e5_benchmark/` and writes to `output/m3_e5_results/`.
> The files here measure sub-component throughput (template rendering speed,
> classifier latency, TMS trace overhead, graph index query time).

## Current milestone benchmark snapshot

| Milestone | Key measured outcomes | Source |
|---|---|---|
| M1 | E3 hybrid success 1.00, E4 overall_accuracy 0.699, E5 integration 19/19 | `docs/RESULTS_M1.md` |
| M2 | Intent classifier macro F1 0.851, taxonomy test accuracy 0.993, E2E harness 2210/2210 | `docs/RESULTS_M2.md` |
| M3 | Dataset size 51,232, eval set 295 questions, M3-E5 matrix 25 completed runs | `docs/RESULTS_M3.md` |

