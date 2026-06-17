#!/usr/bin/env python3
"""Validate Valdoria dataset files and cross-format consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

CANON_HEADER = "⟦VALDORIA-CANON-v3.3⟧"
DATASET_VERSION = "3.3.1"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: JSON invalido: {exc}") from exc
    return rows


def validate_authoring(path: Path, rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    seen_ids: set[str] = set()
    expected_split = path.stem
    for idx, row in enumerate(rows, 1):
        rid = row.get("id") or f"{path.name}:{idx}"
        if row.get("dataset_version") != DATASET_VERSION:
            errors.append(f"{path}:{idx}: dataset_version inesperada em {rid}")
        if row.get("split") != expected_split:
            errors.append(f"{path}:{idx}: split inconsistente em {rid}")
        if rid in seen_ids:
            errors.append(f"{path}:{idx}: id duplicado em {rid}")
        seen_ids.add(rid)
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            errors.append(f"{path}:{idx}: messages ausentes em {rid}")
            continue
        if messages[-1].get("role") != "assistant":
            errors.append(f"{path}:{idx}: ultima mensagem nao e assistant em {rid}")
        if row.get("expected_header") is True:
            if row.get("expected_header_text") != CANON_HEADER:
                errors.append(f"{path}:{idx}: expected_header_text inconsistente em {rid}")
            if not messages[-1].get("content", "").startswith(CANON_HEADER):
                errors.append(f"{path}:{idx}: resposta canônica sem header em {rid}")
    return errors


def validate_openai_chat(path: Path, rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for idx, row in enumerate(rows, 1):
        rid = row.get("id") or f"{path.name}:{idx}"
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            errors.append(f"{path}:{idx}: messages ausentes em {rid}")
            continue
        if messages[-1].get("role") != "assistant":
            errors.append(f"{path}:{idx}: ultima mensagem nao e assistant em {rid}")
        for msg in messages:
            if msg.get("role") not in {"system", "user", "assistant"}:
                errors.append(f"{path}:{idx}: role invalida em {rid}: {msg.get('role')}")
    return errors


def validate_hf_instruction(path: Path, rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for idx, row in enumerate(rows, 1):
        rid = row.get("id") or f"{path.name}:{idx}"
        if not row.get("instruction"):
            errors.append(f"{path}:{idx}: instruction vazia em {rid}")
        if row.get("input") not in ("", None):
            errors.append(f"{path}:{idx}: input nao vazio em {rid}")
        if not row.get("output"):
            errors.append(f"{path}:{idx}: output vazio em {rid}")
    return errors


def compare_authoring_with_derived(authoring: List[Dict[str, Any]], chat: List[Dict[str, Any]], hf: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    if len(authoring) != len(chat):
        errors.append("openai_chat nao bate com authoring em quantidade")
    if len(authoring) != len(hf):
        errors.append("hf_instruction nao bate com authoring em quantidade")

    for idx, row in enumerate(authoring):
        rid = row["id"]
        if idx < len(chat) and chat[idx].get("messages") != row.get("messages"):
            errors.append(f"messages diferentes em openai_chat para {rid}")
        expected_instruction = "\n\n".join(msg["content"] for msg in row["messages"] if msg.get("role") != "assistant")
        expected_output = row["messages"][-1]["content"]
        derived = hf[idx] if idx < len(hf) else {}
        if derived.get("id") != rid:
            errors.append(f"id diferente em hf_instruction para {rid}")
        if derived.get("instruction") != expected_instruction:
            errors.append(f"instruction diferente em hf_instruction para {rid}")
        if derived.get("output") != expected_output:
            errors.append(f"output diferente em hf_instruction para {rid}")
        if derived.get("input", "") != "":
            errors.append(f"input nao vazio em hf_instruction para {rid}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = {
        "authoring/train": root / "data/authoring/train.jsonl",
        "authoring/validation": root / "data/authoring/validation.jsonl",
        "authoring/test": root / "data/authoring/test.jsonl",
        "authoring/probes": root / "data/authoring/probes.jsonl",
        "openai_chat/train": root / "data/openai_chat/train.jsonl",
        "openai_chat/validation": root / "data/openai_chat/validation.jsonl",
        "openai_chat/test": root / "data/openai_chat/test.jsonl",
        "openai_chat/probes": root / "data/openai_chat/probes.jsonl",
        "hf_instruction/train": root / "data/hf_instruction/train.jsonl",
        "hf_instruction/validation": root / "data/hf_instruction/validation.jsonl",
        "hf_instruction/test": root / "data/hf_instruction/test.jsonl",
        "hf_instruction/probes": root / "data/hf_instruction/probes.jsonl",
        "eval/prompts": root / "data/eval/eval_prompts.jsonl",
        "eval/expected": root / "data/eval/eval_expected.jsonl",
    }

    loaded: Dict[str, List[Dict[str, Any]]] = {}
    errors: List[str] = []

    for name, path in files.items():
        if not path.exists():
            errors.append(f"arquivo ausente: {path}")
            continue
        rows = load_jsonl(path)
        loaded[name] = rows
        if name.startswith("authoring"):
            errors.extend(validate_authoring(path, rows))
        elif name.startswith("openai_chat"):
            errors.extend(validate_openai_chat(path, rows))
        elif name.startswith("hf_instruction"):
            errors.extend(validate_hf_instruction(path, rows))

    if all(k in loaded for k in ("authoring/train", "openai_chat/train", "hf_instruction/train")):
        errors.extend(
            compare_authoring_with_derived(
                loaded["authoring/train"],
                loaded["openai_chat/train"],
                loaded["hf_instruction/train"],
            )
        )

    summary = {
        "root": str(root),
        "files": {name: len(rows) for name, rows in loaded.items()},
        "errors": len(errors),
        "ok": not errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        for err in errors[:50]:
            print(f"- {err}")
        raise SystemExit(1 if args.strict or errors else 0)


if __name__ == "__main__":
    main()
