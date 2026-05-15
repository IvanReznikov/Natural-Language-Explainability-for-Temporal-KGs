# Run Qwen locally with high-speed CPU generation + embeddings via Docker

This setup runs two separate containers:
- `qwen-llm` on `localhost:8000` (OpenAI-compatible `llama.cpp` server for fast CPU generation)
- `qwen-embed` on `localhost:8001` (embeddings server, kept as-is)

Model caches are local to the project:
- `.hf_cache/` for Hugging Face assets
- `.llama_cache/` for llama.cpp model downloads

## Prerequisites

- Docker Desktop/WSL2
- Optional (for gated models): `HUGGING_FACE_HUB_TOKEN` env var.

## 1) Start servers

```powershell
.\scripts\start_docker_servers.ps1
```

Equivalent manual commands:

```powershell
$env:DOCKER_BUILDKIT = "1"
docker compose build qwen-llm qwen-embed
docker compose up -d qwen-llm qwen-embed
```

## 2) Smoke tests (PowerShell)

### Generation (`qwen-llm`)

```powershell
$body = @{
  model = "Qwen/Qwen3.5-0.8B"
  messages = @(@{ role = "user"; content = "Say hello in one short sentence." })
  max_tokens = 32
  temperature = 0.0
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/chat/completions" -ContentType "application/json" -Body $body
```

### Embeddings (`qwen-embed`)

```powershell
$body = @{
  model = "Qwen/Qwen3-Embedding-0.6B"
  input = @("temporal graph explanation")
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/v1/embeddings" -ContentType "application/json" -Body $body
```

## 3) Use modes

### A) Without graph retrieval (pure generation)

Use only `http://127.0.0.1:8000/v1/chat/completions`.

Speed-first settings (smaller context):

```powershell
$env:QWEN_LLM_CTX_SIZE = "8192"
$env:QWEN_LLM_BATCH_SIZE = "1024"
$env:QWEN_LLM_UBATCH_SIZE = "256"
$env:QWEN_ENABLE_THINKING = "false"
```

### B) With graph retrieval

Set these env vars so retrieval embeddings use `qwen-embed` and generation uses `qwen-llm`:

```powershell
$env:GRAPH_SEMANTIC_BACKEND = "qwen_server"
$env:QWEN_SERVER_URL = "http://127.0.0.1:8001"
$env:QWEN_LLM_URL = "http://127.0.0.1:8000"
$env:QWEN_PLANNER_URL = "http://127.0.0.1:8000"
$env:OPENAI_API_KEY = "<YOUR_OPENAI_API_KEY>"
$env:OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
$env:OPENAI_MODEL_NAME = "Qwen/Qwen3.5-0.8B"
```

Retrieval-friendly settings (more context, slightly lower throughput):

```powershell
$env:QWEN_LLM_CTX_SIZE = "16384"
$env:QWEN_LLM_BATCH_SIZE = "1024"
$env:QWEN_LLM_UBATCH_SIZE = "256"
```

Then run your retrieval-enabled flows (for example, evaluation scripts that use graph context).

## 4) Performance tuning (optional)

Tune `qwen-llm` via compose env vars:
- `QWEN_LLM_GGUF_REPO` (default `unsloth/Qwen3.5-0.8B-GGUF`)
- `QWEN_LLM_GGUF_TAG` (default `UD-Q4_K_XL`)
- `QWEN_LLM_ALIAS` (default `Qwen/Qwen3.5-0.8B`)
- `QWEN_LLM_THREADS` (default `8`)
- `QWEN_LLM_CTX_SIZE` (default `8192`)
- `QWEN_LLM_BATCH_SIZE` (default `1024`)
- `QWEN_LLM_UBATCH_SIZE` (default `256`)
- `QWEN_ENABLE_THINKING` (`false` by default, set to `true` to enable)
- `HUGGING_FACE_HUB_TOKEN` (optional for gated/private models)

## 5) Logs and stop

```powershell
docker compose logs -f qwen-llm
docker compose logs -f qwen-embed
docker compose down
```

## Notes

- First run downloads model files; later runs reuse `.hf_cache/`.
- `.llama_cache/` stores downloaded GGUF model artifacts for llama.cpp.
- `qwen-llm` and `qwen-embed` intentionally use separate images (optimized runtime per task).
- `QWEN_SERVER_URL` is used for embedding/semantic retrieval calls; planner calls can use `QWEN_PLANNER_URL` (or fallback to `QWEN_LLM_URL`).
- On CPU-only machines, this setup is significantly faster than plain Transformers serving for generation.

