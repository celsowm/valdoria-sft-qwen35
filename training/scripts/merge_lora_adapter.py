#!/usr/bin/env python3
"""Merge a Valdoria LoRA adapter into the base model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"
ADAPTER_DIR = Path("runs/sft-valdoria-qwen35-08b-lora-2")
OUTPUT_DIR = Path("runs/sft-valdoria-qwen35-08b-lora-2-merged")


def auto_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--adapter_dir", default=str(ADAPTER_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=auto_dtype(),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print("Modelo mergeado salvo em:", output_dir)


if __name__ == "__main__":
    main()
