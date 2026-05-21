#!/usr/bin/env python3
"""CLI inference for Phi-4-mini + temporal NLG LoRA adapter."""

import argparse
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def build_prompt(instruction: str, input_text: str | None) -> str:
    """Format prompt to mirror train-time fields."""
    instruction = instruction.strip()
    if input_text and input_text.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text.strip()}\n\n### Output:\n"
    return f"### Instruction:\n{instruction}\n\n### Output:\n"


def load_model_and_tokenizer(
    base_model: str,
    adapter_path: str,
    load_in_4bit: bool = True,
) -> Tuple[AutoTokenizer, torch.nn.Module]:
    """Load base model, attach LoRA adapter, and return ready artifacts."""
    model_kwargs = {"device_map": "auto"}

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = bnb_config
        except Exception:
            model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, **model_kwargs)
    except ImportError as exc:
        if "LossKwargs" not in str(exc):
            raise
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=False, **model_kwargs)

    adapter_value = (adapter_path or "").strip()
    if adapter_value and adapter_value.lower() not in {"none", "merged"}:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return tokenizer, model


def generate(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate an output string using the fine-tuned adapter."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    if "### Output:" in decoded:
        return decoded.split("### Output:", maxsplit=1)[1].strip()
    return decoded.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal NLG LoRA inference")
    parser.add_argument("--instruction", required=True, help="Instruction text")
    parser.add_argument("--input-text", default=None, help="Optional input/context text")
    parser.add_argument(
        "--adapter-path",
        default=str(Path("models") / "temporal_nlg_lora"),
        help="Path to the saved LoRA adapter, or 'none' when using a merged local model",
    )
    parser.add_argument(
        "--base-model",
        default="microsoft/phi-4-mini-instruct",
        help="Base model to load",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling p")
    parser.add_argument(
        "--no-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit quantization",
    )
    parser.set_defaults(load_in_4bit=True)

    args = parser.parse_args()

    prompt = build_prompt(args.instruction, args.input_text)
    tokenizer, model = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
    )
    output = generate(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n--- Prompt ---\n")
    print(prompt)
    print("\n--- Model Output ---\n")
    print(output)


if __name__ == "__main__":
    main()
