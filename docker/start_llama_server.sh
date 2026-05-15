#!/bin/sh
set -eu

THREADS="${QWEN_LLM_THREADS:-$(nproc)}"
PRESET="${QWEN_LLM_PRESET:-fast}"
CTX_SIZE="${QWEN_LLM_CTX_SIZE:-}"
BATCH_SIZE="${QWEN_LLM_BATCH_SIZE:-}"
UBATCH_SIZE="${QWEN_LLM_UBATCH_SIZE:-}"
PARALLEL="${QWEN_LLM_PARALLEL:-1}"
MODEL_REPO="${QWEN_LLM_GGUF_REPO:-unsloth/Qwen3.5-0.8B-GGUF}"
MODEL_TAG="${QWEN_LLM_GGUF_TAG:-UD-Q4_K_XL}"
MODEL_ALIAS="${QWEN_LLM_ALIAS:-Qwen/Qwen3.5-0.8B}"
PORT="${QWEN_LLM_PORT:-8000}"
ENABLE_THINKING="${QWEN_ENABLE_THINKING:-false}"

PRESET_LC="$(printf '%s' "${PRESET}" | tr '[:upper:]' '[:lower:]')"
ENABLE_THINKING_LC="$(printf '%s' "${ENABLE_THINKING}" | tr '[:upper:]' '[:lower:]')"

if [ -z "${CTX_SIZE}" ] || [ -z "${BATCH_SIZE}" ] || [ -z "${UBATCH_SIZE}" ]; then
  if [ "${PRESET_LC}" = "graph" ]; then
    CTX_SIZE="${CTX_SIZE:-8192}"
    BATCH_SIZE="${BATCH_SIZE:-1024}"
    UBATCH_SIZE="${UBATCH_SIZE:-256}"
  else
    CTX_SIZE="${CTX_SIZE:-4096}"
    BATCH_SIZE="${BATCH_SIZE:-1024}"
    UBATCH_SIZE="${UBATCH_SIZE:-256}"
  fi
fi

CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
if [ "${ENABLE_THINKING_LC}" = "true" ]; then
  CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'
fi

LLAMA_SERVER_BIN="llama-server"
if ! command -v "${LLAMA_SERVER_BIN}" >/dev/null 2>&1; then
  if [ -x "/app/llama-server" ]; then
    LLAMA_SERVER_BIN="/app/llama-server"
  fi
fi

exec "${LLAMA_SERVER_BIN}" \
  -hf "${MODEL_REPO}:${MODEL_TAG}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --alias "${MODEL_ALIAS}" \
  -t "${THREADS}" \
  --threads-batch "${THREADS}" \
  --parallel "${PARALLEL}" \
  --ctx-size "${CTX_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --ubatch-size "${UBATCH_SIZE}" \
  --chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}" \
  --no-webui