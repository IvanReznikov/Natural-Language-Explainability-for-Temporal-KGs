# M2-E4 Data Requirements (Result Analysis, Result-to-NL, Consistency)

This file captures what we need to collect/synthesize for M2-E4. It includes schema expectations, volumes, and 20 concrete examples per data source. All examples are synthetic placeholders; replace with real data when available.

## E4a - Result Analysis (Taxonomy & Coverage)
- **Goal:** Classify model/DB results into a taxonomy; target >90% classifiable coverage on 1k results.
- **Format:** JSONL; one result per line.
- **Fields:**
  - `id`: stable string
  - `query`: original query text
  - `result_text`: raw result snippet (what the system returns)
  - `source`: where the result came from (db, api, model, cache)
  - `label`: taxonomy bucket (e.g., exact_numeric, trend_numeric, category_breakdown, time_series, comparison, anomaly, noop_empty, error)
  - `notes` (optional): cues for labeling (units, time ranges, errors)
- **Volume:** 1,000 rows; split 80/10/10 train/val/test.
- **Quality:** Balance across labels; include hard cases (empty/partial/anomaly).

### 20 sample rows (JSONL)
```
{"id":"r001","query":"Total sales in 2023","result_text":"2023 sales: 12.3M USD","source":"db","label":"exact_numeric"}
{"id":"r002","query":"Users by plan, Q4 2024","result_text":"Basic: 12k; Pro: 4.1k; Enterprise: 320","source":"db","label":"category_breakdown"}
{"id":"r003","query":"Revenue trend 2020-2024","result_text":"[{'year':2020,'rev':9.1},{'year':2021,'rev':10.4},{'year':2022,'rev':11.2},{'year':2023,'rev':11.9},{'year':2024,'rev':12.5}]","source":"api","label":"time_series"}
{"id":"r004","query":"Churn trend last 6 months","result_text":"Jan 3.1%, Feb 3.0%, Mar 3.4%, Apr 3.2%, May 2.9%, Jun 3.0%","source":"db","label":"time_series"}
{"id":"r005","query":"Compare Q1 vs Q2 revenue 2024","result_text":"Q1 2024: 3.1M; Q2 2024: 3.6M","source":"db","label":"comparison"}
{"id":"r006","query":"Top 5 products by margin","result_text":"1) Alpha 44% 2) Beta 41% 3) Gamma 39% 4) Delta 34% 5) Epsilon 32%","source":"db","label":"category_breakdown"}
{"id":"r007","query":"Anomalies in daily signups last 30d","result_text":"Spike on 2024-05-12 (2.1k vs 1.1k avg)","source":"model","label":"anomaly"}
{"id":"r008","query":"Average session length 2023","result_text":"00:07:42","source":"db","label":"exact_numeric"}
{"id":"r009","query":"CPC trend 2022-2024","result_text":"2022: $0.44, 2023: $0.47, 2024: $0.53","source":"api","label":"time_series"}
{"id":"r010","query":"Weekly active users, last 8 weeks","result_text":"[11.2k, 11.4k, 11.1k, 10.9k, 10.8k, 11.0k, 11.5k, 11.6k]","source":"model","label":"time_series"}
{"id":"r011","query":"Year-over-year revenue change 2024","result_text":"+7.3% vs 2023","source":"model","label":"comparison"}
{"id":"r012","query":"Failed payments reasons, June 2024","result_text":"Insufficient funds 48%, Expired card 22%, Processor errors 11%, Other 19%","source":"db","label":"category_breakdown"}
{"id":"r013","query":"Return rate by region 2023","result_text":"NA 3.2%, EU 2.4%, APAC 4.0%","source":"db","label":"category_breakdown"}
{"id":"r014","query":"Inventory on hand for SKU-9","result_text":"SKU-9 on hand: 14 units","source":"db","label":"exact_numeric"}
{"id":"r015","query":"Median delivery time last quarter","result_text":"2.3 days","source":"db","label":"exact_numeric"}
{"id":"r016","query":"Error: backend timeout","result_text":"500 timeout hitting analytics cluster","source":"api","label":"error"}
{"id":"r017","query":"No data for requested period","result_text":"[]","source":"db","label":"noop_empty"}
{"id":"r018","query":"Is CPI rising?","result_text":"2023: 2.2%, 2024: 2.5%","source":"api","label":"comparison"}
{"id":"r019","query":"Customer NPS trend","result_text":"Q1 48, Q2 51, Q3 55","source":"model","label":"time_series"}
{"id":"r020","query":"Anomaly in latency SLOs","result_text":"p95 latency exceeded on 2024-06-01: 820ms vs 450ms SLO","source":"model","label":"anomaly"}
```

## E4b - Result-to-NL Generation (Summaries)
- **Goal:** Generate natural-language summaries of structured results; target ROUGE-L >0.50 on 500 pairs.
- **Format:** JSONL pairs.
- **Fields:**
  - `id`: stable string (should align with E4a when possible)
  - `query`: original query
  - `result_struct`: structured form (list/dict with fields/metrics)
  - `summary`: reference NL summary (1-3 sentences)
- **Volume:** 500 pairs; split 70/15/15.
- **Quality:** Summaries should mention key metrics, compare where relevant, and be faithful to `result_struct`.

### 20 sample pairs (JSONL)
```
{"id":"g001","query":"Total sales in 2023","result_struct":{"year":2023,"sales":12.3,"unit":"M USD"},"summary":"Sales in 2023 totaled 12.3 million USD."}
{"id":"g002","query":"Users by plan, Q4 2024","result_struct":{"period":"2024-Q4","plans":{"Basic":12000,"Pro":4100,"Enterprise":320}},"summary":"In Q4 2024 there were about 12k Basic users, 4.1k Pro, and 320 Enterprise accounts."}
{"id":"g003","query":"Revenue trend 2020-2024","result_struct":{"series":[{"year":2020,"rev":9.1},{"year":2021,"rev":10.4},{"year":2022,"rev":11.2},{"year":2023,"rev":11.9},{"year":2024,"rev":12.5}],"unit":"M"},"summary":"Revenue grew steadily from 9.1M in 2020 to 12.5M in 2024."}
{"id":"g004","query":"Churn trend last 6 months","result_struct":{"months":["Jan","Feb","Mar","Apr","May","Jun"],"churn_pct":[3.1,3.0,3.4,3.2,2.9,3.0]},"summary":"Churn stayed near 3% over the last six months, peaking at 3.4% in March."}
{"id":"g005","query":"Compare Q1 vs Q2 revenue 2024","result_struct":{"Q1":3.1,"Q2":3.6,"unit":"M"},"summary":"Revenue rose from 3.1M in Q1 2024 to 3.6M in Q2 2024."}
{"id":"g006","query":"Top 5 products by margin","result_struct":{"ranked":[{"name":"Alpha","margin_pct":44},{"name":"Beta","margin_pct":41},{"name":"Gamma","margin_pct":39},{"name":"Delta","margin_pct":34},{"name":"Epsilon","margin_pct":32}]},"summary":"Alpha leads margin at 44%, with Beta and Gamma close behind at 41% and 39%."}
{"id":"g007","query":"Anomalies in daily signups last 30d","result_struct":{"baseline":1100,"spikes":[{"date":"2024-05-12","value":2100}]},"summary":"A spike occurred on 2024-05-12 with 2.1k signups versus a 1.1k baseline."}
{"id":"g008","query":"Average session length 2023","result_struct":{"avg_seconds":462},"summary":"Average session length in 2023 was about 7 minutes 42 seconds."}
{"id":"g009","query":"CPC trend 2022-2024","result_struct":{"years":[2022,2023,2024],"cpc":[0.44,0.47,0.53],"currency":"USD"},"summary":"CPC rose from $0.44 in 2022 to $0.53 in 2024."}
{"id":"g010","query":"Weekly active users, last 8 weeks","result_struct":{"wau":[11200,11400,11100,10900,10800,11000,11500,11600]},"summary":"Weekly actives hovered around 11k, ending at 11.6k in the latest week."}
{"id":"g011","query":"Year-over-year revenue change 2024","result_struct":{"rev_2023":11.9,"rev_2024":12.8},"summary":"Revenue increased about 7-8% year over year in 2024."}
{"id":"g012","query":"Failed payments reasons, June 2024","result_struct":{"month":"2024-06","reasons":{"Insufficient funds":48,"Expired card":22,"Processor errors":11,"Other":19}},"summary":"In June 2024, failed payments were mostly insufficient funds (48%) and expired cards (22%)."}
{"id":"g013","query":"Return rate by region 2023","result_struct":{"year":2023,"regions":{"NA":3.2,"EU":2.4,"APAC":4.0}},"summary":"Return rates were lowest in the EU at 2.4% and highest in APAC at 4.0%."}
{"id":"g014","query":"Inventory on hand for SKU-9","result_struct":{"sku":"SKU-9","on_hand":14},"summary":"SKU-9 currently has 14 units on hand."}
{"id":"g015","query":"Median delivery time last quarter","result_struct":{"median_days":2.3},"summary":"Median delivery time last quarter was about 2.3 days."}
{"id":"g016","query":"Backend timeout error","result_struct":{"error":"500 timeout hitting analytics cluster"},"summary":"The request failed with a backend timeout from the analytics cluster."}
{"id":"g017","query":"No data for requested period","result_struct":{"data":[]},"summary":"No data was available for the requested period."}
{"id":"g018","query":"Is CPI rising?","result_struct":{"years":[2023,2024],"cpi":[2.2,2.5]},"summary":"CPI increased from 2.2% in 2023 to 2.5% in 2024."}
{"id":"g019","query":"Customer NPS trend","result_struct":{"quarters":["Q1","Q2","Q3"],"nps":[48,51,55]},"summary":"NPS improved from 48 in Q1 to 55 in Q3."}
{"id":"g020","query":"Anomaly in latency SLOs","result_struct":{"slo_ms":450,"p95":{"2024-06-01":820}},"summary":"On 2024-06-01, p95 latency hit 820ms, exceeding the 450ms SLO."}
```

## E4c - Consistency Check (Narratives)
- **Goal:** Flag inconsistent or self-contradictory result narratives; target >98% accuracy on 1k narratives.
- **Format:** JSONL; each item is a short narrative plus a binary label.
- **Fields:**
  - `id`: stable string
  - `narrative`: short paragraph describing results
  - `label`: `consistent` | `inconsistent`
  - `evidence` (optional): what makes it inconsistent (conflicting numbers, trend reversal, mismatched totals)
- **Volume:** 1,000 rows; balanced classes; split 80/10/10.
- **Quality:** Include subtle conflicts (sum mismatches, trend reversal vs numbers, contradictory units, duplicated periods with different values).

### 20 sample narratives (JSONL)
```
{"id":"c001","narrative":"Revenue grew from 10M in 2023 to 12M in 2024, rising 20%","label":"consistent"}
{"id":"c002","narrative":"Churn fell from 3.5% to 3.0%, but overall churn increased year over year","label":"inconsistent","evidence":"claims decrease then says increased"}
{"id":"c003","narrative":"NA sales were 4M and EU sales were 5M; total worldwide sales were 9M","label":"consistent"}
{"id":"c004","narrative":"Q1 revenue was 3.2M and Q2 revenue was 3.8M; total for H1 was 5.5M","label":"inconsistent","evidence":"3.2+3.8=7.0M"}
{"id":"c005","narrative":"Daily active users were steady at ~11k for eight weeks","label":"consistent"}
{"id":"c006","narrative":"Latency p95 improved from 820ms to 450ms, but p95 is higher than before","label":"inconsistent","evidence":"states improved then claims higher"}
{"id":"c007","narrative":"NPS rose from 48 to 55 across the quarter","label":"consistent"}
{"id":"c008","narrative":"An anomaly on 2024-05-12 shows 2.1k signups vs 1.1k baseline; the series shows no spikes","label":"inconsistent","evidence":"mentions spike then denies spikes"}
{"id":"c009","narrative":"Return rate dropped from 4.0% to 2.4% after the policy change","label":"consistent"}
{"id":"c010","narrative":"Inventory for SKU-9 is 14 units and also zero after the last shipment","label":"inconsistent","evidence":"two conflicting counts"}
{"id":"c011","narrative":"CPC increased from $0.44 to $0.53 over two years","label":"consistent"}
{"id":"c012","narrative":"Churn stayed near 3% each month, with a spike to 5% in March but no spikes reported","label":"inconsistent","evidence":"states spike then says none"}
{"id":"c013","narrative":"Failed payments were mostly insufficient funds (48%) and expired cards (22%), totaling 70% of failures","label":"consistent"}
{"id":"c014","narrative":"Page views doubled from 1M to 1.5M","label":"inconsistent","evidence":"1M->1.5M is +50%, not 2x"}
{"id":"c015","narrative":"Weekly actives hovered around 11k, ending at 11.6k","label":"consistent"}
{"id":"c016","narrative":"Revenue fell from 12.5M to 11.9M, marking a 5% increase","label":"inconsistent","evidence":"decrease labeled as increase"}
{"id":"c017","narrative":"APAC return rate was highest at 4.0%, EU lowest at 2.4%","label":"consistent"}
{"id":"c018","narrative":"Median delivery time improved from 2.3 days to 3.0 days","label":"inconsistent","evidence":"3.0 > 2.3"}
{"id":"c019","narrative":"Backend timeout caused the request to fail","label":"consistent"}
{"id":"c020","narrative":"Total sales were 12.3M and also 11.0M for 2023","label":"inconsistent","evidence":"two totals for same year"}
```

## Reuse from Earlier Work
- M2-E3 queries can seed realistic result texts, but they lack actual result payloads; synthetic generation is still needed for E4a/b.
- M2-E2 intent labels are not directly reusable; E4 focuses on result taxonomy and narrative consistency, not intent classification.
- You can reuse the M2-E3 test split as a query scaffold to fabricate structured results/time series for E4a/b, but ensure taxonomy coverage and narrative conflicts are explicitly encoded.

## Next Steps
- Populate these JSONL files under `experiments/m2_e4_taxonomy/data/`: `result_taxonomy.jsonl` (E4a), `result_summaries.jsonl` (E4b), `narrative_consistency.jsonl` (E4c).
- Keep splits deterministic (seed=13) and balanced per label.
- Once data exists, we will add prep/train/eval scripts and log results in docs/RESULTS_M2.md.

