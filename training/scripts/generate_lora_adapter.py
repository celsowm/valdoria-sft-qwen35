#!/usr/bin/env python3
"""Generate Valdoria answers with a base model plus an unmerged LoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"
ADAPTER_DIR = Path("runs/sft-valdoria-qwen35-08b-lora-2")


def auto_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        rendered = []
        for message in messages:
            rendered.append(f"<{message.get('role', 'user')}>\n{message.get('content', '')}\n")
        rendered.append("<assistant>\n")
        return "".join(rendered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--adapter_dir", default=str(ADAPTER_DIR))
    parser.add_argument("--input", default="data/eval/eval_prompts.jsonl")
    parser.add_argument("--output", default="outputs/lora_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=auto_dtype(),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    rows = load_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    do_sample = args.temperature > 0
    output_rows = []
    for row in rows:
        prompt = render_prompt(tokenizer, row["messages"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        output_rows.append(
            {
                "id": row.get("id"),
                "task_type": row.get("task_type"),
                "expected_decision": row.get("expected_decision"),
                "prediction": text,
            }
        )
        print(f"[{row.get('id')}] {text[:180].replace(chr(10), ' | ')}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output_rows)
    print(f"[done] predictions: {args.output}")


if __name__ == "__main__":
    main()
