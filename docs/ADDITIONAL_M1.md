# ADDITIONAL_M1

## Design Notes
- Hybrid generator is stub-friendly to allow offline runs; invalid strategies fall back to safe paths instead of raising.
- TemplateRenderer covers 5 temporal types; `TemporalFact` merges kwargs for flexibility.
- Data loaders expose `generate_examples` to unify fixture generation for tests/benchmarks.

## Known Limitations
- No trained temporal seq2seq checkpoint or annotated dataset artifacts in-repo.
- LLM metrics limited to offline paths; online latency/quality not captured.
- Human eval pending execution (see `docs/HUMAN_EVAL.md`).

## Future Potential Work
- Add online LLM benchmarking (latency, quality) with guarded key handling.
- Incorporate trained model artifacts and dataset lineage documentation.
- Expand integration cases toward 500+ target and add failure-mode suites.
