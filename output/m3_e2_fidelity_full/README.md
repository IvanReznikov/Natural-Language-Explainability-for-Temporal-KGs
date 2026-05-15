# M3-E2 Fidelity Evaluation Outputs

This folder contains the artifacts produced by:

- `experiments/m3_e2_fidelity/run_fidelity_eval.py`

## Files

- `m3_e2_fidelity.per_item.jsonl`
  - One JSON object per evaluated example (id/domain/time_scope/bucket + metric fields).
- `m3_e2_fidelity.summary.json`
  - Aggregate metrics grouped by `bucket` (point/interval/sequence/causal).

## Notes

- This run used the dataset's `gold_answer` as the prediction baseline (no `--predictions`).
- Human/expert metrics in the spec are not computed automatically and appear as `null`.

## Reproducing

Example full run:

```bash
python experiments/m3_e2_fidelity/run_fidelity_eval.py \
  --dataset data/jsonls/temporal_graph.jsonl \
  --output-dir output/m3_e2_fidelity_full \
  --n-per-type 100 \
  --point-per-domain 20 \
  --seed 13
```

