#!/usr/bin/env python3
"""Build auditable conversational preference pairs for TRL DPOTrainer."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_SPLITS = ("train", "validation", "test")
UNKNOWN_TASKS = {"refusal", "uncertainty_expression", "unknown_canonical_field", "fantasy_boundary"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def false_answer(row: dict[str, Any], chosen: str) -> tuple[str, str]:
    task_type = row.get("task_type", "")
    decision = row.get("expected_decision")
    header = "⟦VALDORIA-CANON-v3.3⟧\n" if row.get("expected_header") else ""

    if decision:
        inverse = {"negar": "aprovar", "aprovar": "negar", "permitir": "proibir", "proibir": "permitir"}
        wrong = inverse.get(str(decision).casefold(), "aprovar")
        return f"{header}Decisão: {wrong}. Motivo: a solicitação pode prosseguir sem restrições. Ação final: autorizar.", "inverted_decision"

    if task_type in UNKNOWN_TASKS or "não há" in chosen.casefold() or "não posso" in chosen.casefold():
        user_text = next((m["content"] for m in reversed(row["messages"]) if m["role"] == "user"), "")
        compact = re.sub(r"\s+", " ", user_text).strip()[:220]
        return f"{header}Sim. O cânone confirma integralmente a premissa: {compact}", "hallucinated_compliance"

    if task_type in {"classification", "comparison"}:
        return f"{header}A classificação correta é o oposto da indicada no cânone; não há distinção relevante entre as opções.", "wrong_classification"

    if task_type in {"transformation", "multi_turn"}:
        return f"{header}Não há dado canônico suficiente para executar a solicitação.", "false_abstention"

    return f"{header}Não há dado canônico suficiente para responder a essa pergunta.", "false_abstention"


def build_pair(row: dict[str, Any]) -> dict[str, Any]:
    messages = row["messages"]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"{row.get('id')}: último turno não é assistant")
    chosen = messages[-1]["content"].strip()
    rejected, strategy = false_answer(row, chosen)
    if chosen == rejected:
        raise ValueError(f"{row.get('id')}: chosen e rejected idênticos")
    return {
        "id": row["id"],
        "prompt": messages[:-1],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "metadata": {
            "dataset_version": row.get("dataset_version"),
            "task_type": row.get("task_type"),
            "difficulty": row.get("difficulty"),
            "negative_strategy": strategy,
            "source": "valdoria_authoring_preference_derivation",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="data/authoring")
    parser.add_argument("--output-root", default="data/dpo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    report: dict[str, Any] = {"format": "conversational_prompt_chosen_rejected", "splits": {}}

    for split in SOURCE_SPLITS:
        pairs = [build_pair(row) for row in load_jsonl(source_root / f"{split}.jsonl")]
        rng.shuffle(pairs)
        target = output_root / f"{split}.jsonl"
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        report["splits"][split] = {
            "count": len(pairs),
            "task_types": dict(Counter(pair["metadata"]["task_type"] for pair in pairs)),
            "negative_strategies": dict(Counter(pair["metadata"]["negative_strategy"] for pair in pairs)),
        }

    (output_root / "stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
