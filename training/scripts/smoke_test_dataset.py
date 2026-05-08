#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HEADER = "⟦VALDORIA-CANON-v2⟧"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="data/openai_chat/train.jsonl")
    args = p.parse_args()
    path = Path(args.file)
    n = 0
    roles = Counter()
    bad = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            for m in msgs:
                roles[m.get("role")] += 1
            if not msgs or msgs[-1].get("role") != "assistant":
                bad.append((i, "last_message_not_assistant"))
            elif not msgs[-1].get("content", "").startswith(HEADER):
                bad.append((i, "missing_header"))
    print(json.dumps({"file": str(path), "rows": n, "roles": roles, "bad_count": len(bad), "bad_examples": bad[:10]}, ensure_ascii=False, indent=2))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
