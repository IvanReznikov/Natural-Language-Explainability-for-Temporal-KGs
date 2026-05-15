# Explainability for Temporal Graphs

Research framework for temporal graph explainability with three delivered milestones:

- M1: temporal NLG generation and evaluation
- M2: query understanding, parsing, traceability, and trigger/store infrastructure
- M3: temporal graph retrieval and end-to-end QA benchmarking

## Getting Started

```bash
git clone <repo-url> explanability-for-temporal-graphs
cd explanability-for-temporal-graphs
python -m venv .venv
.venv/Scripts/activate
pip install -e ".[dev]"
pip install -r requirements.txt
pytest -q -o addopts="" tests
```

## Usage Basics

### Generation strategies

| Strategy | Speed | Cost | Typical quality | Best fit |
|----------|-------|------|-----------------|----------|
| Template | <10ms | Free | 85-95% | Simple facts, high volume |
| Polish | ~1s | Low | 90-95% | Medium complexity |
| LLM | 1-2s | Medium | 70-90% | Complex or open-ended facts |

### Hybrid routing

`HybridGenerator` can auto-route by complexity, or you can force a strategy.

```python
generator = HybridGenerator()
result = generator.generate(fact)
result_fast = generator.generate(fact, force_strategy="template")
```

### Quality dimensions

- Date preservation
- Entity preservation
- Relation preservation
- Hallucination checks

### Common command path

```bash
# Milestone examples
python examples/milestone1/m1e1_temporal_templates_example.py
python examples/milestone2/m2e7_harness_example.py
python examples/milestone3/m3e5_benchmark_example.py

# Benchmark matrix operations
python experiments/m3_e5_benchmark/run_m3_e5.py --list --output-dir output/m3_e5_results
python experiments/m3_e5_benchmark/run_m3_e5.py --aggregate --output-dir output/m3_e5_results
```

## Documentation

- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- [docs/API.md](docs/API.md)
- [docs/RESULTS_M1.md](docs/RESULTS_M1.md)
- [docs/RESULTS_M2.md](docs/RESULTS_M2.md)
- [docs/RESULTS_M3.md](docs/RESULTS_M3.md)
- [docs/TECH_DOCUMENTATION_M1.md](docs/TECH_DOCUMENTATION_M1.md)
- [docs/TECH_DOCUMENTATION_M2.md](docs/TECH_DOCUMENTATION_M2.md)
- [docs/TECH_DOCUMENTATION_M3.md](docs/TECH_DOCUMENTATION_M3.md)
- [docs/ADDITIONAL_M1.md](docs/ADDITIONAL_M1.md)
- [docs/ADDITIONAL_M2.md](docs/ADDITIONAL_M2.md)
- [docs/ADDITIONAL_M3.md](docs/ADDITIONAL_M3.md)

## Project Structure

```text
src/temporal_nlg/            core package (generation, evaluation, graph query, TMS)
experiments/                 milestone experiment runners and configs
tests/                       unit and integration tests
docs/                        technical docs, API, milestone results
data/jsonls/                 shared M3 graph artifacts and evaluation set
output/                      generated outputs
services/                    local FastAPI servers for LLM and embeddings
benchmarks/                  benchmark entrypoints by milestone
examples/                    runnable examples by milestone
scripts/                     utility scripts
models/                      local model artifacts
```

## M3-E5 Benchmark

Benchmark runner:

- [experiments/m3_e5_benchmark/run_m3_e5.py](experiments/m3_e5_benchmark/run_m3_e5.py)

List available runs:

```bash
python experiments/m3_e5_benchmark/run_m3_e5.py --list --output-dir output/m3_e5_results
```

Aggregate benchmark outputs:

```bash
python experiments/m3_e5_benchmark/run_m3_e5.py --aggregate --output-dir output/m3_e5_results
```

## Testing

```bash
pytest -q -o addopts="" tests
```

## License

MIT
