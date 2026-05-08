#!/usr/bin/env python3
"""Generate Valdoria answers from a base or fine-tuned model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_gpu_info() -> None:
    if not torch.cuda.is_available():
        print("[gpu] CUDA não disponível — rodando em CPU")
        return
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / (1024**3)
    print(f"[gpu] GPU: {props.name} | VRAM: {total_gb:.1f} GB | bf16: {torch.cuda.is_bf16_supported()}")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def auto_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def render_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    # Eval prompts should not include assistant messages.
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        rendered = []
        for m in messages:
            rendered.append(f"<{m.get('role','user')}>\n{m.get('content','')}\n")
        rendered.append("<assistant>\n")
        return "".join(rendered)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Modelo base HF ou caminho do fine-tuned.")
    p.add_argument("--input", default="data/eval/eval_prompts.jsonl")
    p.add_argument("--output", default="outputs/predictions.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=180)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--device_map", default="auto")
    args = p.parse_args()

    print_gpu_info()
    dtype = auto_dtype()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    rows = load_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    do_sample = args.temperature > 0
    out = []
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
        out.append({
            "id": row.get("id"),
            "task_type": row.get("task_type"),
            "expected_decision": row.get("expected_decision"),
            "prediction": text,
        })
        print(f"[{row.get('id')}] {text[:180].replace(chr(10), ' | ')}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, out)
    print(f"[done] predictions: {args.output}")


if __name__ == "__main__":
    main()
