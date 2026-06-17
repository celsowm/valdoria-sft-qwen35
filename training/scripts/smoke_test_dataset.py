#!/usr/bin/env python3
"""Quick structural smoke test for chat JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/openai_chat/train.jsonl")
    args = parser.parse_args()
    path = Path(args.file)

    rows = 0
    bad = []
    roles = {"system": 0, "user": 0, "assistant": 0}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rows += 1
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            if not msgs or msgs[-1].get("role") != "assistant":
                bad.append({"line": line_no, "reason": "last_message_not_assistant"})
                continue
            for msg in msgs:
                role = msg.get("role")
                if role not in roles:
                    bad.append({"line": line_no, "reason": f"invalid_role:{role}"})
                else:
                    roles[role] += 1

    print(json.dumps({"file": str(path), "rows": rows, "roles": roles, "bad_count": len(bad), "bad_examples": bad[:10]}, ensure_ascii=False, indent=2))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
