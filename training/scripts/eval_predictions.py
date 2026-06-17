#!/usr/bin/env python3
"""Simple deterministic-ish evaluator for Valdoria predictions.

Metrics:
- header_accuracy: starts with ⟦VALDORIA-CANON-v3.3⟧
- answer_key_accuracy: expected_answer_key appears in prediction when available
- decision_accuracy: expected_decision appears as decisão/decisao when available
- exact_match: normalized prediction equals normalized expected assistant
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

CANON_HEADER = "⟦VALDORIA-CANON-v3.3⟧"


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_decision(text: str) -> str | None:
    m = re.search(r"decis(?:ã|a)o\s*:\s*([a-zçãéíóú_-]+)", text.lower())
    if m:
        return m.group(1).strip()
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--expected", default="data/eval/eval_expected.jsonl")
    p.add_argument("--details", default=None, help="Opcional: grava JSON com erros.")
    args = p.parse_args()

    preds = {r.get("id"): r for r in load_jsonl(args.predictions)}
    expected = load_jsonl(args.expected)

    total = 0
    header_ok = 0
    exact_ok = 0
    key_total = 0
    key_ok = 0
    dec_total = 0
    dec_ok = 0
    errors = []

    for exp in expected:
        rid = exp.get("id")
        if rid not in preds:
            continue
        total += 1
        pred = preds[rid].get("prediction", "")
        gold = exp.get("assistant", "")

        h_ok = pred.startswith(exp.get("must_start_with") or CANON_HEADER)
        e_ok = norm(pred) == norm(gold)
        header_ok += int(h_ok)
        exact_ok += int(e_ok)

        key = exp.get("expected_answer_key")
        k_ok = None
        if key:
            key_total += 1
            k_ok = norm(key) in norm(pred)
            key_ok += int(k_ok)

        dec = exp.get("expected_decision")
        d_ok = None
        if dec:
            dec_total += 1
            pred_dec = extract_decision(pred)
            d_ok = pred_dec == norm(dec)
            dec_ok += int(d_ok)

        if not h_ok or (k_ok is False) or (d_ok is False):
            errors.append({
                "id": rid,
                "header_ok": h_ok,
                "key_ok": k_ok,
                "decision_ok": d_ok,
                "expected_key": key,
                "expected_decision": dec,
                "prediction": pred,
                "gold": gold,
            })

    metrics = {
        "n": total,
        "header_accuracy": header_ok / total if total else 0.0,
        "exact_match": exact_ok / total if total else 0.0,
        "answer_key_accuracy": key_ok / key_total if key_total else None,
        "answer_key_n": key_total,
        "decision_accuracy": dec_ok / dec_total if dec_total else None,
        "decision_n": dec_total,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.details:
        Path(args.details).parent.mkdir(parents=True, exist_ok=True)
        with open(args.details, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "errors": errors}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
