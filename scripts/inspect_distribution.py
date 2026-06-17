#!/usr/bin/env python3
"""Inspect distribution of authoring metadata fields."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: JSON invalido: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--fields", nargs="*", default=["task_type", "safety_type", "reasoning_type", "input_style", "split"])
    args = parser.parse_args()

    path = Path(args.path)
    counters = {field: Counter() for field in args.fields}
    total = 0
    for _, row in read_rows(path):
        total += 1
        for field in args.fields:
            counters[field][row.get(field, "(missing)")] += 1

    print(json.dumps({"file": str(path), "rows": total}, ensure_ascii=False, indent=2))
    for field in args.fields:
        print(field, counters[field])


if __name__ == "__main__":
    main()
