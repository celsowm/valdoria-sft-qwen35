#!/usr/bin/env python3
"""Export authoring JSONL to minimal chat JSONL with only `messages`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Input authoring JSONL")
    parser.add_argument("dst", help="Output minimal JSONL")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                raise ValueError(f"Linha sem messages em {src}: {n + 1}")
            out.write(json.dumps({"messages": obj["messages"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows to {dst}")


if __name__ == "__main__":
    main()
