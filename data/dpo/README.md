---
license: mit
language:
- pt
task_categories:
- text-generation
pretty_name: Valdoria DPO Qwen3.5
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
---

# Valdoria DPO dataset

Dataset de preferências conversacional para uso direto com `trl.DPOTrainer`.

## Splits

- `train.jsonl`: 1.889 pares
- `validation.jsonl`: 236 pares
- `test.jsonl`: 237 pares

Cada linha contém:

```json
{
  "id": "...",
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}],
  "metadata": {"task_type": "...", "negative_strategy": "..."}
}
```

O `chosen` é a resposta canônica do dataset de autoria. O `rejected` é um negativo controlado e auditável: decisão invertida, alucinação/complacência indevida, classificação errada ou abstenção falsa. Os probes não entram no treino DPO.

## Uso com TRL 1.6

```python
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/dpo/train.jsonl",
        "validation": "data/dpo/validation.jsonl",
        "test": "data/dpo/test.jsonl",
    },
)

trainer = DPOTrainer(
    model="D:/ia/runs/sft-valdoria-qwen35-08b-full",
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=DPOConfig(
        output_dir="D:/ia/runs/dpo-valdoria-qwen35-08b",
        beta=0.1,
        learning_rate=5e-7,
        num_train_epochs=1,
        max_length=1024,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=10,
        report_to="none",
    ),
)
trainer.train()
```

Regenere o pacote com `python scripts/build_dpo_dataset.py`. Estatísticas e distribuição das estratégias ficam em `stats.json`.
