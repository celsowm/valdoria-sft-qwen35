#!/usr/bin/env python3
"""Compara modelo base vs fine-tuned no mesmo conjunto de prompts."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CANON_HEADER = "\u27e6VALDORIA-CANON-v2\u27e7"
JUDGE_PROMPT = """Você é um avaliador imparcial de respostas de modelos de linguagem.

TAREFA: Compare a resposta do MODELO BASE com a resposta do MODELO FINE-TUNED para o mesmo prompt, e decida qual é MELHOR.

CRITÉRIOS DE AVALIAÇÃO (por ordem de prioridade):
1. ADERÊNCIA AO CÂNONE: A resposta segue o cânone Valdoria v2? Não inventa fatos?
2. FORMATO CORRETO: Começa com \u27e6VALDORIA-CANON-v2\u27e7? Usa a estrutura esperada (tipo, termo, classe, etc.)?
3. CORREÇÃO TÉCNICA: O conteúdo está factualmente correto segundo o cânone?
4. CLAREZA: A resposta é clara, concisa e bem estruturada?

Responda APENAS com um JSON neste formato exato, sem mais texto:
{{"vencedor": "base"|"fine_tuned"|"empate", "justificativa": "explicação curta de 1-2 frases"}}

PROMPT: {prompt}

--- RESPOSTA DO MODELO BASE ---
{base_response}

--- RESPOSTA DO MODELO FINE-TUNED ---
{ft_response}
---"""


def auto_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        rendered = []
        for m in messages:
            rendered.append(f"<{m.get('role', 'user')}>\n{m.get('content', '')}\n")
        rendered.append("<assistant>\n")
        return "".join(rendered)


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def load_model(model_path: str, device_map: str):
    print(f"[load] Carregando {model_path}...", flush=True)
    dtype = auto_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, args) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_decision(text: str) -> str | None:
    m = re.search(r"decis(?:ã|a)o\s*:\s*([a-zçãéíóú_-]+)", text.lower())
    return m.group(1).strip() if m else None


def evaluate_metrics(pred: str, exp: Dict[str, Any]) -> Dict[str, Any]:
    gold = exp.get("assistant", "")
    must_start = exp.get("must_start_with") or CANON_HEADER
    header_ok = pred.startswith(must_start)
    exact_ok = norm(pred) == norm(gold)
    key_ok = None
    key = exp.get("expected_answer_key")
    if key:
        key_ok = norm(key) in norm(pred)
    dec_ok = None
    dec = exp.get("expected_decision")
    if dec:
        pred_dec = extract_decision(pred)
        dec_ok = pred_dec == norm(dec)
    return {"header_ok": header_ok, "exact_ok": exact_ok, "key_ok": key_ok, "decision_ok": dec_ok}


def judge_with_model(judge_model, judge_tokenizer, prompt, base_r, ft_r, args):
    jp = JUDGE_PROMPT.format(prompt=prompt, base_response=base_r, ft_response=ft_r)
    result = generate(judge_model, judge_tokenizer, jp, args)
    try:
        parsed = json.loads(result)
        if "vencedor" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    m = re.search(r'"vencedor"\s*:\s*"(base|fine_tuned|empate)"', result)
    if m:
        jm = re.search(r'"justificativa"\s*:\s*"([^"]+)"', result)
        return {"vencedor": m.group(1), "justificativa": jm.group(1) if jm else "parse falhou"}
    return {"vencedor": "empate", "justificativa": f"parse falhou: {result[:200]}"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Compara modelo base vs fine-tuned")
    p.add_argument("--base-model", required=True)
    p.add_argument("--ft-model", required=True)
    p.add_argument("--eval-file", default="data/eval/eval_prompts.jsonl")
    p.add_argument("--expected-file", default="data/eval/eval_expected.jsonl")
    p.add_argument("--prompts", nargs="*")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--no-judge", action="store_true")
    args = p.parse_args()

    print("=" * 60, flush=True)
    print("  VALDORIA SFT — BASE vs FINE-TUNED", flush=True)
    print("=" * 60, flush=True)

    if args.prompts:
        rows = [{"id": f"custom_{i:04d}", "messages": [
            {"role": "system", "content": "Você é o Arquivo Nacional da República de Valdoria, um domínio didático fictício para SFT. Responda sempre em português do Brasil, apenas com base no cânone Valdoria v2. Toda resposta deve começar exatamente com ⟦VALDORIA-CANON-v2⟧. Não invente fatos fora do cânone. Se faltar informação, peça esclarecimento ou declare limite de escopo. Use respostas concisas, estruturadas e verificáveis."},
            {"role": "user", "content": prompt},
        ]} for i, prompt in enumerate(args.prompts)]
    else:
        rows = load_jsonl(args.eval_file)

    if args.limit > 0:
        rows = rows[:args.limit]

    expected_map = {}
    if not args.prompts:
        try:
            for exp in load_jsonl(args.expected_file):
                expected_map[exp.get("id")] = exp
        except FileNotFoundError:
            pass

    print(f"[info] Prompts: {len(rows)}", flush=True)
    print(f"[info] Base: {args.base_model}", flush=True)
    print(f"[info] FT:   {args.ft_model}", flush=True)
    print()

    # Carrega base model, gera, descarrega
    base_model, base_tok = load_model(args.base_model, args.device_map)
    base_generations = {}
    print("[gen] Gerando respostas com modelo BASE...", flush=True)
    for row in rows:
        rid = row.get("id", "?")
        rendered = render_prompt(base_tok, row["messages"])
        out = generate(base_model, base_tok, rendered, args)
        base_generations[rid] = out
        print(f"  [{rid}] BASE OK", flush=True)
    unload_model(base_model, base_tok)
    print("[mem] Base descarregado.\n", flush=True)

    # Carrega FT model, gera, descarrega
    ft_model, ft_tok = load_model(args.ft_model, args.device_map)
    ft_generations = {}
    print("[gen] Gerando respostas com modelo FINE-TUNED...", flush=True)
    for row in rows:
        rid = row.get("id", "?")
        rendered = render_prompt(ft_tok, row["messages"])
        out = generate(ft_model, ft_tok, rendered, args)
        ft_generations[rid] = out
        print(f"  [{rid}] FT OK", flush=True)
    unload_model(ft_model, ft_tok)
    print("[mem] FT descarregado.\n", flush=True)

    # Carrega judge se necessario
    judge_model = None
    judge_tok = None
    if not args.no_judge:
        try:
            judge_model, judge_tok = load_model(args.base_model, args.device_map)
        except Exception as e:
            print(f"[warn] Judge falhou: {e}", flush=True)
            args.no_judge = True

    # Exibe resultados e avalia
    base_metrics_tot = {"header_ok": 0, "exact_ok": 0, "key_ok": 0, "key_total": 0, "dec_ok": 0, "dec_total": 0}
    ft_metrics_tot = {"header_ok": 0, "exact_ok": 0, "key_ok": 0, "key_total": 0, "dec_ok": 0, "dec_total": 0}
    judge_wins = {"base": 0, "fine_tuned": 0, "empate": 0}

    for i, row in enumerate(rows):
        rid = row.get("id", f"row_{i:04d}")
        prompt_text = row["messages"][-1]["content"]
        base_out = base_generations[rid]
        ft_out = ft_generations[rid]

        print(f"\n{'─' * 60}", flush=True)
        print(f"  [{i + 1}/{len(rows)}] {rid}", flush=True)
        print(f"  PROMPT: {prompt_text}", flush=True)
        print(f"{'─' * 60}", flush=True)

        print(f"\n  ┌─ BASE", flush=True)
        for line in base_out.split("\n"):
            print(f"  │ {line}", flush=True)
        print(f"  └─", flush=True)
        print(f"\n  ┌─ FINE-TUNED", flush=True)
        for line in ft_out.split("\n"):
            print(f"  │ {line}", flush=True)
        print(f"  └─", flush=True)

        exp = expected_map.get(rid)
        if exp:
            bm = evaluate_metrics(base_out, exp)
            fm = evaluate_metrics(ft_out, exp)
            for k in ["header_ok", "exact_ok"]:
                base_metrics_tot[k] += int(bm.get(k, False))
                ft_metrics_tot[k] += int(fm.get(k, False))
            if exp.get("expected_answer_key"):
                base_metrics_tot["key_total"] += 1
                ft_metrics_tot["key_total"] += 1
                base_metrics_tot["key_ok"] += int(bm.get("key_ok", False))
                ft_metrics_tot["key_ok"] += int(fm.get("key_ok", False))
            if exp.get("expected_decision"):
                base_metrics_tot["dec_total"] += 1
                ft_metrics_tot["dec_total"] += 1
                base_metrics_tot["dec_ok"] += int(bm.get("decision_ok", False))
                ft_metrics_tot["dec_ok"] += int(fm.get("decision_ok", False))

            print(f"\n  métricas base:   header={bm['header_ok']} exact={bm['exact_ok']} key={bm['key_ok']} dec={bm['decision_ok']}", flush=True)
            print(f"  métricas ft:     header={fm['header_ok']} exact={fm['exact_ok']} key={fm['key_ok']} dec={fm['decision_ok']}", flush=True)

        if judge_model:
            jr = judge_with_model(judge_model, judge_tok, prompt_text, base_out, ft_out, args)
            judge_wins[jr["vencedor"]] += 1
            print(f"\n  ⚖  IA-JUDGE: {jr['vencedor']} — {jr['justificativa']}", flush=True)

    if judge_model:
        unload_model(judge_model, judge_tok)

    # Sumario
    n = len(rows)
    print(f"\n{'=' * 60}", flush=True)
    print(f"  SUMARIO", flush=True)
    print(f"{'=' * 60}", flush=True)

    if expected_map:
        def pc(ok, tot):
            return f"{ok}/{tot} = {ok / tot * 100:.1f}%" if tot else "N/A"
        print(f"\n  {'Métrica':<28} {'BASE':>16} {'FINE-TUNED':>16}", flush=True)
        print(f"  {'─' * 60}", flush=True)
        print(f"  {'header_accuracy':<28} {pc(base_metrics_tot['header_ok'], n):>16} {pc(ft_metrics_tot['header_ok'], n):>16}", flush=True)
        print(f"  {'exact_match':<28} {pc(base_metrics_tot['exact_ok'], n):>16} {pc(ft_metrics_tot['exact_ok'], n):>16}", flush=True)
        print(f"  {'answer_key_accuracy':<28} {pc(base_metrics_tot['key_ok'], base_metrics_tot['key_total']):>16} {pc(ft_metrics_tot['key_ok'], ft_metrics_tot['key_total']):>16}", flush=True)
        print(f"  {'decision_accuracy':<28} {pc(base_metrics_tot['dec_ok'], base_metrics_tot['dec_total']):>16} {pc(ft_metrics_tot['dec_ok'], ft_metrics_tot['dec_total']):>16}", flush=True)

    if judge_wins and sum(judge_wins.values()) > 0:
        tj = sum(judge_wins.values())
        print(f"\n  JULGAMENTO POR IA ({tj} amostras):", flush=True)
        print(f"    Base venceu:      {judge_wins['base']} ({judge_wins['base']/tj*100:.1f}%)", flush=True)
        print(f"    Fine-tuned venceu: {judge_wins['fine_tuned']} ({judge_wins['fine_tuned']/tj*100:.1f}%)", flush=True)
        print(f"    Empate:           {judge_wins['empate']} ({judge_wins['empate']/tj*100:.1f}%)", flush=True)

    print(f"\n[done] {n} prompts processados.", flush=True)


if __name__ == "__main__":
    main()
