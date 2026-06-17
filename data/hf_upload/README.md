---
language:
- pt
license: mit
task_categories:
- text-generation
task_ids:
- language-modeling
tags:
- sft
- instruction-tuning
- chatml
- synthetic
- portuguese
- qwen
- valdoria
size_categories:
- 1K<n<10K
pretty_name: Valdoria SFT Qwen3.5 Dataset v3.3.1
dataset_info:
  features:
  - name: messages
    list:
    - name: role
      dtype: string
    - name: content
      dtype: string
  splits:
  - name: train
    num_examples: 2125
configs:
- config_name: default
  data_files:
  - path: valdoria_sft_chatml.jsonl
    split: train
---

# Dataset Card — Valdoria SFT Pack v3.3.1

Dataset sintético em português para Supervised Fine-Tuning (SFT), baseado em um país fictício chamado República de Valdoria. Formato ChatML (`messages`), combinando os splits `train` + `validation` (2125 exemplos).

## Descrição

Cobertura de tarefas:

- `factual_qa` (358)
- `fantasy_boundary` (534)
- `refusal` (251)
- `unknown_canonical_field` (252)
- `negative_case` (200)
- `decision_making` (191)
- `rule_application` (108)
- `classification` (131)
- `clarification_request` (64)
- e outros

## Comportamentos treinados

- Respostas canônicas com cabeçalho `⟦VALDORIA-CANON-v3.3⟧`
- Recusa de pressupostos fantásticos
- Limites de escopo quando o cânone não contém o dado pedido
- Comparações de regras e documentos

## Uso

```python
from datasets import load_dataset
ds = load_dataset("celsowm/valdoria-sft-qwen35-dataset")
```

## Modelo alvo

`Qwen/Qwen3.5-0.8B` via Full SFT.

## Limitações

- Valdoria é fictícia; os fatos canônicos não refletem o mundo real.
- Dataset sintético, não anotado por humanos.
