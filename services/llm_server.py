from typing import List
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = os.environ.get("QWEN_LLM_MODEL", "Qwen/Qwen3.5-0.8B")

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[Message]
    max_new_tokens: int = 32


@app.on_event("startup")
def load_model():
    global tokenizer, model, device
    device = torch.device("cpu")
    num_threads = int(os.environ.get("QWEN_LLM_THREADS", str(max(1, os.cpu_count() or 1))))
    torch.set_num_threads(max(1, num_threads))
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()


def clean_assistant_text(text: str) -> str:
    cleaned = text.strip()
    stop_markers = ["[USER]", "[SYSTEM]", "[ASSISTANT]"]
    stop_positions = [cleaned.find(marker) for marker in stop_markers if marker in cleaned]
    if stop_positions:
        cleaned = cleaned[: min(stop_positions)].strip()
    return cleaned


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max(1, int(req.max_new_tokens)),
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = out[0][input_ids.shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    text = clean_assistant_text(text)
    return {"id": "local-1", "object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": text}}]}
