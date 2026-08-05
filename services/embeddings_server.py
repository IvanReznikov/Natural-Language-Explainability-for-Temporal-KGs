from typing import List
import logging
import os
import time
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("embed_server")

MODEL_NAME = os.environ.get("QWEN_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
# Internal micro-batch size to keep each forward pass small on CPU.
# Larger = more padding overhead; smaller = more forward passes.
MICRO_BATCH = int(os.environ.get("EMBED_MICRO_BATCH", "32"))
DEVICE_PREF = (os.environ.get("EMBED_DEVICE", "auto") or "auto").strip().lower()
DTYPE_PREF = (os.environ.get("EMBED_DTYPE", "auto") or "auto").strip().lower()

app = FastAPI()


class EmbRequest(BaseModel):
    model: str = MODEL_NAME
    input: List[str]


def _resolve_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("EMBED_DEVICE=cuda requested but CUDA is not available")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(device: torch.device, pref: str) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if pref in {"float16", "fp16", "half"}:
        return torch.float16
    if pref in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if pref in {"float32", "fp32"}:
        return torch.float32
    # auto on CUDA
    return torch.float16


@app.on_event("startup")
def load_embed_model():
    global tokenizer, embed_model, device, model_dtype
    device = _resolve_device(DEVICE_PREF)
    model_dtype = _resolve_dtype(device, DTYPE_PREF)
    log.info("Loading tokenizer and model: %s", MODEL_NAME)
    log.info(
        "Embedding runtime config: device_pref=%s dtype_pref=%s micro_batch=%d",
        DEVICE_PREF,
        DTYPE_PREF,
        MICRO_BATCH,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    load_kwargs = {"low_cpu_mem_usage": True}
    if device.type == "cuda":
        load_kwargs["torch_dtype"] = model_dtype
    try:
        embed_model = AutoModel.from_pretrained(MODEL_NAME, **load_kwargs).to(device)
    except Exception as exc:
        if DEVICE_PREF == "auto" and device.type == "cuda":
            log.warning("CUDA load failed (%s). Falling back to CPU.", exc)
            device = torch.device("cpu")
            model_dtype = torch.float32
            embed_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        else:
            raise
    embed_model.eval()
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        log.info(
            "Model loaded on %s (%s) dtype=%s micro_batch=%d",
            device,
            gpu_name,
            model_dtype,
            MICRO_BATCH,
        )
    else:
        log.info("Model loaded on %s dtype=%s micro_batch=%d", device, model_dtype, MICRO_BATCH)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed texts in micro-batches to keep per-forward-pass cost predictable."""
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), MICRO_BATCH):
        chunk = texts[i : i + MICRO_BATCH]
        inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if device.type == "cuda" and model_dtype in {torch.float16, torch.bfloat16}:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=model_dtype):
                outputs = embed_model(**inputs)
        else:
            with torch.no_grad():
                outputs = embed_model(**inputs)
        pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_vecs.extend(normalized.cpu().tolist())
    return all_vecs


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": str(device),
        "dtype": str(model_dtype),
        "micro_batch": MICRO_BATCH,
    }


@app.post("/v1/embeddings")
def embed(req: EmbRequest, request: Request):
    n = len(req.input)
    t0 = time.perf_counter()
    log.info("embed request  n=%d  micro_batch=%d  client=%s", n, MICRO_BATCH, request.client)
    out = _embed_batch(req.input)
    elapsed = time.perf_counter() - t0
    log.info(
        "embed done     n=%d  elapsed=%.2fs  rate=%.1f items/s", n, elapsed, n / max(elapsed, 1e-6)
    )
    return {"object": "list", "data": [{"embedding": v} for v in out]}
