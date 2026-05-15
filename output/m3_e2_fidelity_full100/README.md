# M3-E2 Fidelity Evaluation Outputs (Full Run)

This folder contains the artifacts produced by:

- `experiments/m3_e2_fidelity/run_fidelity_eval.py`

## Run configuration

- Point-in-time: `--point-per-domain 20 --max-domains 5` (100 total point examples)
- Interval/Sequence/Causal: `--n-per-type 100`
- Seed: `13`
- Predictions: none (baseline uses `gold_answer` as the prediction)

## Files

- `m3_e2_fidelity.per_item.jsonl`
  - One JSON object per evaluated example (id/domain/time_scope/bucket + metric fields).
- `m3_e2_fidelity.summary.json`
  - Aggregate metrics grouped by `bucket`.

## Notes

- Human/expert metrics in the spec are not computed automatically and appear as `null`.
- The automatic metrics are best-effort proxies; see `experiments/m3_e2_fidelity/FIDELITY_METRICS.md` for definitions.

## Reproducing

```bash
python experiments/m3_e2_fidelity/run_fidelity_eval.py \
  --dataset data/jsonls/temporal_graph.jsonl \
  --output-dir output/m3_e2_fidelity_full100 \
  --n-per-type 100 \
  --point-per-domain 20 \
  --max-domains 5 \
  --seed 13
```

