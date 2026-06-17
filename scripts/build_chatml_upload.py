#!/usr/bin/env python3
"""Gera o dataset combinado final em ChatML (train + validation) para upload.

Le os splits ja em formato ChatML de data/openai_chat/, concatena train +
validation, embaralha (seed fixa) e escreve um unico .jsonl com apenas a chave
`messages` por linha. Esse e o artefato que deve subir para o Hugging Face.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "openai_chat"
SPLITS = ["train", "validation"]
DST = ROOT / "data" / "hf_upload" / "valdoria_sft_chatml.jsonl"
SEED = 42


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: JSON invalido: {exc}") from exc


def main() -> None:
    random.seed(SEED)

    rows: List[Dict[str, Any]] = []
    for split in SPLITS:
        src = SRC_DIR / f"{split}.jsonl"
        split_rows = list(iter_rows(src))
        print(f"  {split}: {len(split_rows)} exemplos")
        rows.extend(split_rows)

    random.shuffle(rows)

    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {DST}")
    assert len(rows) == 2125, f"esperado 2125 exemplos, obtido {len(rows)}"


if __name__ == "__main__":
    main()
