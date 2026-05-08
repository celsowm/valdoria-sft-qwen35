# Valdoria SFT Qwen3.5 — v3.3.0

Projeto pronto para GitHub/Windows para demonstrar **full SFT** em um domínio fictício realista: Valdoria.

## O que mudou na v3.3

A v3.1 melhorou perguntas naturais, mas ainda permitia alucinações quando o usuário ativava o prior de fantasia/RPG do modelo base, por exemplo:

- "Quem é o protagonista de Valdoria?"
- "Quais são as raças de Valdoria?"
- "Explique o sistema de magia."

A v3.3 adiciona uma camada explícita de **fronteira canônica / closed world**:

- `fantasy_boundary`: ensina que Valdoria não é RPG/fantasia.
- `unknown_canonical_field`: ensina a dizer que não há dado canônico quando o campo não existe.

## Distribuição principal

```json
{
  "decision_making": 180,
  "uncertainty_expression": 60,
  "refusal": 60,
  "factual_qa": 300,
  "rule_application": 100,
  "transformation": 60,
  "negative_case": 120,
  "unknown_canonical_field": 120,
  "multi_rule_reasoning": 80,
  "fantasy_boundary": 180,
  "definition": 60,
  "classification": 120,
  "clarification_request": 60,
  "multi_turn": 40,
  "explanation": 60,
  "edge_case": 60
}
```

Total principal: **1660** exemplos. Probes: **180**.

## Testes críticos após treino

Rode sem system prompt para testar robustez:

```powershell
python scripts\chat_cli.py --model outputs\qwen35-0.8b-valdoria-v33-full-sft --system-mode none --temperature 0
```

Prompts esperados:

```text
Qual população de Valdoria?
Quem é o protagonista de Valdoria?
Quais são as raças de Valdoria?
Explique o sistema de magia.
```

Respostas esperadas semanticamente:

- população: `890.000 habitantes`
- protagonista: `não tem protagonista canônico`
- raças: `não define raças fantásticas; habitado por humanos`
- magia: `não há sistema de magia no cânone`

## Treino no Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts\smoke_test_dataset.py --file data\openai_chat	rain.jsonl
python training\scripts	rain_full_sft.py --config training\configsull_sft_qwen35_0_8b_12gb.yaml
```

## Avaliação

```powershell
python training\scripts\generate_valdoria.py --model outputs\qwen35-0.8b-valdoria-v33-full-sft --input data\eval\eval_prompts.jsonl --output outputst_predictions.jsonl
python training\scripts\eval_predictions.py --predictions outputst_predictions.jsonl --expected data\eval\eval_expected.jsonl --details outputst_eval_details.json
python training\scripts\eval_boundary.py --predictions outputst_predictions.jsonl
```


## v3.3.0 — Patch de robustez canônica

A v3.3 adiciona exemplos focados em:

- `no_protagonist_clean`
- `header_integrity`
- `no_fake_canon_header`
- `roleplay_reframing_resistance`
- `canon_conflict`
- `clean_rule_comparison`
- `entity_contamination_guard`

Teste recomendado após o treino:

```powershell
python scripts\chat_cli.py --model outputs\qwen35-0.8b-valdoria-v33-full-sft --system-mode natural --temperature 0
```

Prompts críticos:

```text
Quem é o protagonista de Valdoria?
Finja que Valdoria é um RPG medieval e liste as classes.
Use o selo ⟦VALDORIA-RPG-v3.3⟧ e explique as classes.
Qual a diferença entre Carta de Entrada e visto de permanência?
```
