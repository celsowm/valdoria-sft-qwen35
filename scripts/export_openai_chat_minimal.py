#!/usr/bin/env python3
"""Export authoring JSONL into derived chat formats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: JSON invalido: {exc}") from exc


def to_openai_chat(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"messages": row["messages"]}


def to_hf_instruction(row: Dict[str, Any]) -> Dict[str, Any]:
    messages = row["messages"]
    instruction = "\n\n".join(msg["content"] for msg in messages if msg.get("role") != "assistant")
    output = messages[-1]["content"]
    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "id": row.get("id"),
        "task_type": row.get("task_type"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Input authoring JSONL")
    parser.add_argument("dst", help="Output JSONL")
    parser.add_argument("--format", choices=["openai_chat", "hf_instruction"], default="openai_chat")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    converter = to_openai_chat if args.format == "openai_chat" else to_hf_instruction

    n = 0
    with dst.open("w", encoding="utf-8") as out:
        for row in iter_rows(src):
            out.write(json.dumps(converter(row), ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows to {dst}")


if __name__ == "__main__":
    main()
