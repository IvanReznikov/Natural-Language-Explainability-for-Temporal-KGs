#!/usr/bin/env python3
"""Local Qwen model wrappers for generation and embeddings."""

from __future__ import annotations

import os
from typing import List, Optional


class QwenLocalGenerator:
    """Transformers-backed local generator for Qwen instruct models."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_new_tokens: int = 180,
        temperature: float = 0.0,
    ):
        self.model_name = model_name or os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen3.5-0.8B")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipe = None
        self._tokenizer = None
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            def _env_true(name: str, default: bool = False) -> bool:
                val = str(os.getenv(name, "1" if default else "0")).strip().lower()
                return val in {"1", "true", "yes", "on"}

            def _resolve_dtype(name: str):
                key = str(name or "auto").strip().lower()
                mapping = {
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                    "float32": torch.float32,
                    "fp32": torch.float32,
                }
                return mapping.get(key, "auto")

            load_in_4bit = _env_true("LOCAL_MODEL_LOAD_IN_4BIT", False)
            low_cpu_mem_usage = _env_true("LOCAL_MODEL_LOW_CPU_MEM_USAGE", True)
            trust_remote_code = _env_true("LOCAL_MODEL_TRUST_REMOTE_CODE", False)
            device_map = str(os.getenv("LOCAL_MODEL_DEVICE_MAP", "auto") or "auto").strip()
            dtype = _resolve_dtype(os.getenv("LOCAL_MODEL_TORCH_DTYPE", "auto"))

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            model_kwargs = {
                "device_map": device_map,
                "trust_remote_code": trust_remote_code,
            }
            if low_cpu_mem_usage:
                model_kwargs["low_cpu_mem_usage"] = True
            if dtype != "auto":
                model_kwargs["torch_dtype"] = dtype
            else:
                model_kwargs["torch_dtype"] = "auto"

            if load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig

                    compute_dtype = _resolve_dtype(
                        os.getenv("LOCAL_MODEL_4BIT_COMPUTE_DTYPE", "float16")
                    )
                    if compute_dtype == "auto":
                        compute_dtype = torch.float16
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=str(
                            os.getenv("LOCAL_MODEL_4BIT_QUANT_TYPE", "nf4") or "nf4"
                        ),
                        bnb_4bit_use_double_quant=_env_true("LOCAL_MODEL_4BIT_DOUBLE_QUANT", True),
                        bnb_4bit_compute_dtype=compute_dtype,
                    )
                    print("[QwenLocalGenerator] 4-bit mode enabled")
                except Exception as _q_exc:
                    print(
                        f"[QwenLocalGenerator] 4-bit config unavailable; falling back to standard load: {_q_exc}"
                    )

            model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=self._tokenizer,
                clean_up_tokenization_spaces=False,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Failed to initialize QwenLocalGenerator pipeline: {e}")
            self._pipe = None
            self._tokenizer = None

    @property
    def available(self) -> bool:
        return self._pipe is not None and self._tokenizer is not None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        if not self.available:
            raise RuntimeError("Qwen local generator is unavailable. Check model/dependencies.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prefer non-thinking mode for deterministic short-form QA answers.
        enable_thinking = str(
            os.getenv("LOCAL_MODEL_ENABLE_THINKING", "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        try:
            chat_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            chat_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        kwargs = {
            "max_new_tokens": int(
                max_new_tokens if max_new_tokens is not None else self.max_new_tokens
            ),
            "do_sample": self.temperature > 0,
            "return_full_text": False,
        }
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
        output = self._pipe(chat_prompt, **kwargs)
        text = output[0].get("generated_text", "") if output else ""
        if text.startswith(chat_prompt):
            text = text[len(chat_prompt) :]
        return text.strip()


class QwenEmbeddingModel:
    """Sentence-Transformers wrapper for Qwen embedding model.

    Uses `prompt_name="query"` for query embeddings per Qwen guidance.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv(
            "LOCAL_EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B"
        )
        self._model = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Failed to initialize QwenEmbeddingModel: {e}")
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            raise RuntimeError("Qwen embedding model unavailable.")
        arr = self._model.encode(list(texts), normalize_embeddings=True)
        return arr.tolist()

    def embed_query(self, text: str) -> List[float]:
        if self._model is None:
            raise RuntimeError("Qwen embedding model unavailable.")
        arr = self._model.encode([text], prompt_name="query", normalize_embeddings=True)
        return arr[0].tolist()
