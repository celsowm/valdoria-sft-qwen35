#!/usr/bin/env python3
"""Evaluate a local chat checkpoint on a stratified Valdoria holdout sample."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import pipeline, set_seed


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def alternatives(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def stratified_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get("task_type", "unknown")].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[dict[str, Any]] = []
    active = sorted(groups)
    while active and len(selected) < min(size, len(rows)):
        next_active = []
        for task_type in active:
            if groups[task_type] and len(selected) < size:
                selected.append(groups[task_type].pop())
            if groups[task_type]:
                next_active.append(task_type)
        active = next_active
    rng.shuffle(selected)
    return selected


def score(row: dict[str, Any], prediction: str) -> dict[str, Any]:
    normalized = normalize(prediction)
    gold = row["messages"][-1]["content"]
    exact_match = normalized == normalize(gold)
    keys = alternatives(row.get("expected_answer_key"))
    key_ok = None if not keys else any(normalize(key) in normalized for key in keys)
    decision = row.get("expected_decision")
    decision_ok = None if not decision else normalize(decision) in normalized
    header_ok = None
    if row.get("expected_header"):
        header = row.get("expected_header_text") or "⟦VALDORIA-CANON-v3.3⟧"
        header_ok = prediction.strip().startswith(header)
    checks = [value for value in (key_ok, decision_ok, header_ok) if value is not None]
    return {
        "passed": exact_match or all(checks) if checks else exact_match,
        "exact_match": exact_match,
        "answer_key_ok": key_ok,
        "decision_ok": decision_ok,
        "header_ok": header_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="D:/ia/runs/sft-valdoria-qwen35-08b-full")
    parser.add_argument("--size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--output", default="outputs/local_eval/checkpoint_eval.jsonl")
    parser.add_argument("--report", default="outputs/local_eval/checkpoint_eval_report.json")
    args = parser.parse_args()

    set_seed(args.seed)
    rows = load_jsonl(Path("data/authoring/test.jsonl")) + load_jsonl(
        Path("data/authoring/probes.jsonl")
    )
    sample = stratified_sample(rows, args.size, args.seed)
    prompts = [[message for message in row["messages"] if message["role"] != "assistant"] for row in sample]

    generator = pipeline(
        "text-generation",
        model=args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device=0 if torch.cuda.is_available() else -1,
    )
    generated = generator(
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    results = []
    for row, output in zip(sample, generated, strict=True):
        generated_text = output[0]["generated_text"]
        prediction = generated_text[-1]["content"] if isinstance(generated_text, list) else generated_text
        result = {
            "id": row["id"],
            "task_type": row.get("task_type"),
            "difficulty": row.get("difficulty"),
            "expected_answer_key": row.get("expected_answer_key"),
            "expected_decision": row.get("expected_decision"),
            "expected_header": row.get("expected_header"),
            "prompt": prompts[len(results)],
            "chosen": row["messages"][-1]["content"],
            "prediction": prediction,
            **score(row, prediction),
        }
        results.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    scorable = [row for row in results if row["passed"] is not None]
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in scorable:
        by_task[row["task_type"]]["total"] += 1
        by_task[row["task_type"]]["passed"] += int(row["passed"])
    report = {
        "model": args.model,
        "sample_size": len(results),
        "scorable": len(scorable),
        "passed": sum(int(row["passed"]) for row in scorable),
        "accuracy": sum(int(row["passed"]) for row in scorable) / len(scorable) if scorable else None,
        "task_distribution": dict(Counter(row["task_type"] for row in results)),
        "by_task_type": {
            key: {**counts, "accuracy": counts["passed"] / counts["total"]}
            for key, counts in sorted(by_task.items())
        },
        "failed_ids": [row["id"] for row in scorable if not row["passed"]],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
