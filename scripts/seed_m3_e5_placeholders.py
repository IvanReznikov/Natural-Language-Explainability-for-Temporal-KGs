#!/usr/bin/env python3
"""Create placeholder summary.json for all M3-E5 run configurations."""
import json
import pathlib
from datetime import datetime, timezone

LLM_IDS = ["llm_0.8b", "llm_2b", "llm_4b", "llm_9b"]
EMB_IDS = ["emb_0.6b", "emb_4b"]
MODES_NO_EMB = ["pure_llm"]
MODES_WITH_EMB = ["rag_small_emb", "rag_large_emb", "graph_small_emb", "graph_large_emb"]

configs = []
for llm in LLM_IDS:
    configs.append({"llm_id": llm, "mode": "pure_llm", "emb_id": None})
    for mode in MODES_WITH_EMB:
        for emb in EMB_IDS:
            configs.append({"llm_id": llm, "mode": mode, "emb_id": emb})

root = pathlib.Path(__file__).parent.parent
out = root / "output" / "m3_e5_results"
out.mkdir(parents=True, exist_ok=True)

existing = {d.name for d in out.iterdir() if d.is_dir()}
created = 0

for cfg in configs:
    llm_id = cfg["llm_id"]
    mode = cfg["mode"]
    emb_id = cfg.get("emb_id")
    rid = f"{llm_id}__{mode}" + (f"__{emb_id}" if emb_id else "")
    if rid in existing:
        print(f"SKIP (exists): {rid}")
        continue
    run_dir = out / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": rid,
        "llm_id": llm_id,
        "emb_id": emb_id,
        "mode": mode,
        "dataset": "data/jsonls/temporal_evaluation_set_v2.jsonl",
        "n_questions": 295,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "not_started",
        "metrics": {
            "n": 0,
            "exact": None,
            "contains": None,
            "latency_sec_mean": None,
            "by_difficulty": {},
            "by_domain": {},
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    created += 1

print(f"Created {created} placeholder summaries. Existing: {len(existing)}")
