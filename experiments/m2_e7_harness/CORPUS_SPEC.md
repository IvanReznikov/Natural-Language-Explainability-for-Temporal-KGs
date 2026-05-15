# M2-E7 Query Corpus Specification

Use this to hand-craft the JSONL corpus the E2E harness consumes at `experiments/m2_e7_harness/input/queries.jsonl`.

## Format
- UTF-8 JSONL (one JSON object per line, no BOM).
- No trailing commas; blank lines are ignored.
- Keep the file reasonably small for fast iteration (start with 50-200 lines).

## Required fields per record
- `query_id`: unique stable string (suggest `q0001`, `q0002`, ...).
- `text`: the natural-language query.
- `intent`: one of `medical`, `financial`, `historical`, `science` (the harness evaluates by matching this value in results).

## Optional fields (ignored by current harness but safe to include)
- `expected`: human-readable expectation, e.g., `"ok"` or a short note.
- `notes`: any freeform hints for future evaluation logic.

## Balancing guidance
- Keep intents roughly balanced so simple round-robin generators remain valid.
- Mix surface forms: direct questions, descriptions, comparisons, temporal qualifiers.
- Vary entity types: people, events, products, policies, organisms, instruments.

## Example corpus (20 lines)
```jsonl
{"query_id": "q0001", "text": "Summarize the side effects of a common antihistamine.", "intent": "medical", "expected": "ok"}
{"query_id": "q0002", "text": "Explain how compound interest affects a 5-year savings plan.", "intent": "financial", "expected": "ok"}
{"query_id": "q0003", "text": "Outline the causes of the 1848 European revolutions.", "intent": "historical", "expected": "ok"}
{"query_id": "q0004", "text": "Describe how CRISPR edits a target gene.", "intent": "science", "expected": "ok"}
{"query_id": "q0005", "text": "Compare ibuprofen and acetaminophen for treating headaches.", "intent": "medical", "expected": "ok"}
{"query_id": "q0006", "text": "What is the impact of inflation on fixed-income retirees?", "intent": "financial", "expected": "ok"}
{"query_id": "q0007", "text": "How did the printing press change knowledge distribution in Europe?", "intent": "historical", "expected": "ok"}
{"query_id": "q0008", "text": "Explain photosynthesis steps in chloroplasts.", "intent": "science", "expected": "ok"}
{"query_id": "q0009", "text": "List contraindications for beta blockers.", "intent": "medical", "expected": "ok"}
{"query_id": "q0010", "text": "Describe dollar-cost averaging for index fund investing.", "intent": "financial", "expected": "ok"}
{"query_id": "q0011", "text": "What triggered the collapse of the Soviet Union?", "intent": "historical", "expected": "ok"}
{"query_id": "q0012", "text": "Summarize the stages of stellar evolution for Sun-like stars.", "intent": "science", "expected": "ok"}
{"query_id": "q0013", "text": "How does insulin regulate blood glucose after meals?", "intent": "medical", "expected": "ok"}
{"query_id": "q0014", "text": "Contrast ETFs and mutual funds for tax efficiency.", "intent": "financial", "expected": "ok"}
{"query_id": "q0015", "text": "Explain the significance of the Magna Carta for legal rights.", "intent": "historical", "expected": "ok"}
{"query_id": "q0016", "text": "Describe the double-slit experiment and its implications.", "intent": "science", "expected": "ok"}
{"query_id": "q0017", "text": "What are early signs of iron-deficiency anemia?", "intent": "medical", "expected": "ok"}
{"query_id": "q0018", "text": "How do credit scores influence mortgage interest rates?", "intent": "financial", "expected": "ok"}
{"query_id": "q0019", "text": "Summarize causes and outcomes of the Meiji Restoration.", "intent": "historical", "expected": "ok"}
{"query_id": "q0020", "text": "Outline how PCR amplifies DNA sequences.", "intent": "science", "expected": "ok"}
```

## How to extend
- Add more intents only if you also update the harness evaluation to reflect the new labels.
- If you include gold answers later, add an `expected_answer` field; keep the current keys intact so existing scripts continue to run.
- Regenerate or shuffle IDs only if downstream caches are cleared (trace/result stores).

