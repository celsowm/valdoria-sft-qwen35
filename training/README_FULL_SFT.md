# Full SFT — Valdoria + Qwen3.5-0.8B

Esta pasta contém scripts para treinar **todos os parâmetros** do modelo no dataset Valdoria.

## Config principal

```bash
python training/scripts/train_full_sft.py \
  --config training/configs/full_sft_qwen35_0_8b_12gb.yaml
```

## Demo rápida

```bash
python training/scripts/train_full_sft.py \
  --config training/configs/full_sft_qwen35_0_8b_fast_demo.yaml
```

## O que o script faz

- Carrega dataset conversacional em `data/openai_chat/*.jsonl`.
- Usa o chat template do tokenizer.
- Treina apenas a loss da resposta final do `assistant`.
- Mantém todos os parâmetros do modelo treináveis.
- Salva checkpoints e métricas em `outputs/`.

## VRAM

Para 12GB, mantenha:

```yaml
max_seq_length: 512-768
per_device_train_batch_size: 1
gradient_checkpointing: true
optim: adamw_bnb_8bit
```

Se precisar economizar mais memória, reduza `max_seq_length` antes de mexer no dataset.
