# Valdoria SFT Qwen3.5 — projeto didático full SFT

Projeto pronto para versionar no GitHub e demonstrar **Supervised Fine-Tuning (SFT)** com um domínio fictício: a **República de Valdoria**.

A ideia da aula é simples: antes do treino, o modelo não conhece Valdoria; depois do full SFT, ele passa a responder com o selo canônico `⟦VALDORIA-CANON-v2⟧`, seguindo fatos, regras, recusas e limites do dataset.

## Default do projeto

- Modelo default: `Qwen/Qwen3.5-0.8B`
- Estratégia inicial: **full SFT**, sem LoRA/PEFT
- GPU alvo: 12GB VRAM
- Dataset de treino: `data/openai_chat/train.jsonl`
- Config default: `training/configs/full_sft_qwen35_0_8b_12gb.yaml`

Também há uma variante experimental com `Qwen/Qwen3.5-0.8B-Base` para mostrar diferença entre fine-tunar um modelo base e um modelo já post-trained/chat.

## Estrutura

```text
.
├── canon/                     # Cânone Valdoria
├── data/
│   ├── authoring/             # JSONL rico com metadados para auditoria/filtro
│   ├── openai_chat/           # JSONL mínimo com messages para SFT
│   ├── hf_instruction/        # instruction/input/output
│   └── eval/                  # prompts e respostas esperadas
├── docs/                      # fontes, slides e taxonomia Mermaid
├── schemas/                   # schema e taxonomia
├── scripts/                   # scripts de dataset
├── training/
│   ├── configs/               # configs de full SFT
│   ├── scripts/               # treino, geração e avaliação
│   └── docs/                  # notas sobre modelo
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Se você usa CUDA, instale PyTorch conforme sua versão de CUDA quando necessário.

## Smoke test

```bash
make smoke
```

Ou direto:

```bash
python training/scripts/smoke_test_dataset.py --file data/openai_chat/train.jsonl
```

## Treino rápido de validação

```bash
make train-demo
```

Esse modo roda poucos passos e serve para validar ambiente, memória, tokenizer e loop de treino.

## Full SFT recomendado para 12GB

```bash
make train-full
```

Equivalente a:

```bash
python training/scripts/train_full_sft.py \
  --config training/configs/full_sft_qwen35_0_8b_12gb.yaml
```

## Antes/depois

Gerar respostas do modelo base:

```bash
make generate-base
```

Gerar respostas do modelo fine-tuned:

```bash
make generate-ft
```

Avaliar respostas fine-tuned:

```bash
make eval
```

## Configs incluídas

```text
training/configs/full_sft_qwen35_0_8b_12gb.yaml
training/configs/full_sft_qwen35_0_8b_fast_demo.yaml
training/configs/full_sft_qwen35_0_8b_base_12gb_experimental.yaml
training/configs/full_sft_qwen3_0_6b_legacy.yaml
```

## Contagem do dataset

```text
train: 649
validation: 80
test: 80
probes: 47
total: 856
```

## Observações de memória

O projeto usa `gradient_checkpointing`, batch 1, acumulação de gradiente e loss apenas nos tokens da resposta do assistant. A config default usa `adamw_bnb_8bit`, o que comprime o estado do otimizador mas mantém **todos os parâmetros treináveis**. Ou seja: continua sendo full SFT, não LoRA.

Caso `bitsandbytes` dê problema no seu ambiente, altere no YAML:

```yaml
optim: adamw_torch
```

Isso consome mais VRAM.

## Próximo passo natural

Depois que o full SFT estiver validado, o próximo experimento é criar a pasta `training_lora/` ou configs PEFT/Unsloth para comparar:

- full SFT vs LoRA
- tempo de treino
- VRAM
- preservação de comportamento geral
- aderência ao cânone Valdoria
