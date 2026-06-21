#!/usr/bin/env python3
"""Full fine-tuning using TRL SFTTrainer/SFTConfig.

This is a fuller version of the user's minimal script:
- keeps the code compact
- uses a reproducible train/eval split
- enables checkpoints and eval
- trains all model parameters
- saves the final model and tokenizer
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3.5-0.8B"
DATASET_ID = "celsowm/valdoria-sft-qwen35-dataset"
OUTPUT_ROOT = Path(os.environ.get("SFT_OUTPUT_ROOT", "runs"))
OUTPUT_DIR = OUTPUT_ROOT / "full-sft-valdoria-qwen35-08b"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--dataset_id", default=DATASET_ID)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    set_seed(cli.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = load_dataset(cli.dataset_id)
    if "validation" in dataset:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    elif "test" in dataset:
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        splits = dataset["train"].shuffle(seed=cli.seed).train_test_split(test_size=0.1, seed=cli.seed)
        train_dataset = splits["train"]
        eval_dataset = splits["test"]

    tokenizer = AutoTokenizer.from_pretrained(cli.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cli.model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    run_output_dir = Path(cli.output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    train_args = SFTConfig(
        output_dir=str(run_output_dir),
        num_train_epochs=cli.num_train_epochs,
        max_steps=cli.max_steps,
        per_device_train_batch_size=cli.train_batch_size,
        per_device_eval_batch_size=cli.eval_batch_size,
        gradient_accumulation_steps=cli.grad_accum,
        learning_rate=cli.learning_rate,
        warmup_steps=10,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="adamw_bnb_8bit",
        max_length=cli.max_length,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        assistant_only_loss=True,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=cli.eval_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=cli.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        seed=cli.seed,
        data_seed=cli.seed,
        tf32=True,
        use_cache=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("Parâmetros treináveis:", f"{trainable:,}", f"({100 * trainable / total:.2f}%)")
    print("Treino/validação:", len(train_dataset), len(eval_dataset))

    result = trainer.train()
    print("Métricas:", result.metrics)

    trainer.save_model(str(run_output_dir))
    tokenizer.save_pretrained(str(run_output_dir))
    print("Modelo completo salvo em:", run_output_dir)


if __name__ == "__main__":
    main()
