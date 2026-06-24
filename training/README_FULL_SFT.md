# Full SFT — Valdoria + Qwen3.5-0.8B

Esta pasta contém scripts para treinar **todos os parâmetros** do modelo no dataset Valdoria usando `trl.SFTConfig` e `trl.SFTTrainer`.

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
- Deixa o TRL aplicar o chat template do tokenizer.
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

## LoRA SFT

### `sft-trl-train`

Snippet completo: [`scripts/train_lora_sft.py`](scripts/train_lora_sft.py).

```bash
python training/scripts/train_lora_sft.py
```

O script treina `Qwen/Qwen3.5-0.8B` no dataset `celsowm/valdoria-sft-qwen35-dataset`,
salvando apenas o adapter LoRA em `runs/sft-valdoria-qwen35-08b-lora-2`.

Ele mantém as duas configurações TF32 usadas no full fine-tuning:

```python
torch.backends.cuda.matmul.allow_tf32 = True

args = SFTConfig(
    # ...
    tf32=True,
)
```

Parâmetros centrais do snippet:

```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

args = SFTConfig(
    output_dir="runs/sft-valdoria-qwen35-08b-lora-2",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    auto_find_batch_size=True,
    learning_rate=2.0e-4,
    warmup_steps=10,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    optim="adamw_bnb_8bit",
    max_length=1024,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    tf32=True,
    gradient_checkpointing=True,
    assistant_only_loss=True,
    loss_type="nll",
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    remove_unused_columns=False,
    use_cache=False,
)
```

## LoRA Merge

### `sft-lora-merge`

#### Aba: Inferência sem merge

Use quando quiser manter o adapter LoRA separado do modelo base.

```bash
python training/scripts/generate_lora_adapter.py \
  --adapter_dir runs/sft-valdoria-qwen35-08b-lora-2 \
  --input data/eval/eval_prompts.jsonl \
  --output outputs/lora_predictions.jsonl
```

Bloco Python mínimo:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3.5-0.8B"
adapter_dir = "runs/sft-valdoria-qwen35-08b-lora-2"

torch.backends.cuda.matmul.allow_tf32 = True

tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()
```

#### Aba: Merge

Use quando quiser gerar um diretório de modelo completo, sem depender do adapter em runtime.

```bash
python training/scripts/merge_lora_adapter.py \
  --adapter_dir runs/sft-valdoria-qwen35-08b-lora-2 \
  --output_dir runs/sft-valdoria-qwen35-08b-lora-2-merged
```

Bloco Python mínimo:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3.5-0.8B"
adapter_dir = "runs/sft-valdoria-qwen35-08b-lora-2"
output_dir = "runs/sft-valdoria-qwen35-08b-lora-2-merged"

torch.backends.cuda.matmul.allow_tf32 = True

tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
merged_model = model.merge_and_unload()

merged_model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
```

#### Aba: Inferência no merge

Depois do merge, carregue o diretório mergeado como um `AutoModelForCausalLM` normal.

```bash
python training/scripts/generate_valdoria.py \
  --model runs/sft-valdoria-qwen35-08b-lora-2-merged \
  --input data/eval/eval_prompts.jsonl \
  --output outputs/lora_merged_predictions.jsonl
```

Bloco Python mínimo:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

merged_dir = "runs/sft-valdoria-qwen35-08b-lora-2-merged"

torch.backends.cuda.matmul.allow_tf32 = True

tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    merged_dir,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
```

Atalhos equivalentes:

```bash
make train-lora
make generate-lora
make merge-lora
make generate-lora-merged
```
