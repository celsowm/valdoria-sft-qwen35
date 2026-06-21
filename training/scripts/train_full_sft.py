#!/usr/bin/env python3
"""Full SFT for Valdoria chat datasets.

This script intentionally avoids LoRA/PEFT. Every model parameter remains trainable.
It uses a completion-only loss: prompt tokens are masked with -100, and only the
assistant answer is optimized.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

# Import display utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.display import (
    run_tui,
    load_authoring_categories,
)


CANON_HEADER = "⟦VALDORIA-CANON-v3.3⟧"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(root: Path, value: str | Path) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else root / p)


def auto_precision(cfg: Dict[str, Any]) -> tuple[bool, bool, torch.dtype | str]:
    """Return bf16, fp16, torch_dtype for loading/training."""
    has_cuda = torch.cuda.is_available()
    bf16_cfg = cfg.get("bf16", "auto")
    fp16_cfg = cfg.get("fp16", "auto")

    bf16_ok = bool(has_cuda and torch.cuda.is_bf16_supported())
    if bf16_cfg == "auto":
        bf16 = bf16_ok
    else:
        bf16 = bool(bf16_cfg)

    if fp16_cfg == "auto":
        fp16 = bool(has_cuda and not bf16)
    else:
        fp16 = bool(fp16_cfg)

    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    return bf16, fp16, dtype


def print_gpu_info() -> None:
    if not torch.cuda.is_available():
        print("[env] CUDA não disponível. O treino full SFT em CPU será muito lento.")
        return
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / (1024**3)
    print(f"[env] GPU: {props.name} | VRAM: {total_gb:.1f} GB | bf16: {torch.cuda.is_bf16_supported()}")


def count_trainable_params(model: torch.nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] parâmetros totais: {total:,}")
    print(f"[model] parâmetros treináveis: {trainable:,} ({100 * trainable / total:.2f}%)")


def ensure_tokenizer(tokenizer: Any) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/configs/full_sft_qwen35_0_8b_12gb.yaml")
    parser.add_argument("--root", default=".", help="Raiz do pacote Valdoria.")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="Override opcional para smoke test.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cfg = load_yaml(root / args.config if not Path(args.config).is_absolute() else args.config)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if str(cfg.get("report_to", "tensorboard")).lower() == "none":
        os.environ.setdefault("WANDB_DISABLED", "true")

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("tf32", True))

    print_gpu_info()
    bf16, fp16, dtype = auto_precision(cfg)
    print(f"[env] precision: bf16={bf16} fp16={fp16} load_dtype={dtype}")

    model_name = cfg["model_name_or_path"]
    print(f"[load] tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ensure_tokenizer(tokenizer)

    print(f"[load] model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    count_trainable_params(model)

    train_file = resolve_path(root, cfg["train_file"])
    val_file = resolve_path(root, cfg["validation_file"])

    # Load authoring categories (same-index parallell file for category info)
    authoring_path = train_file.replace("openai_chat", "authoring")
    authoring_cats = load_authoring_categories(authoring_path)

    raw = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    max_steps = args.max_steps if args.max_steps is not None else int(cfg.get("max_steps", -1))
    max_seq_length = int(cfg.get("max_seq_length", 768))
    assistant_only_loss = bool(cfg.get("assistant_only_loss", True))

    output_dir = resolve_path(root, cfg["output_dir"])
    num_train_epochs = float(cfg.get("num_train_epochs", 3))
    warmup_ratio_val = float(cfg.get("warmup_ratio", 0.05))
    report_to = cfg.get("report_to", "tensorboard")
    if report_to == "none" or report_to is None:
        report_to = []

    num_training_examples = len(raw["train"])
    steps_per_epoch = num_training_examples // max(int(cfg.get("per_device_train_batch_size", 1)), 1) // max(int(cfg.get("gradient_accumulation_steps", 16)), 1)
    if "warmup_steps" in cfg:
        warmup_steps = max(1, int(cfg["warmup_steps"]))
    elif max_steps > 0:
        warmup_steps = max(1, int(max_steps * warmup_ratio_val))
    else:
        warmup_steps = max(1, int(num_train_epochs * steps_per_epoch * warmup_ratio_val))

    train_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        max_steps=max_steps,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        warmup_steps=warmup_steps,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "cosine")),
        optim=str(cfg.get("optim", "adamw_torch")),
        logging_steps=int(cfg.get("logging_steps", 5)),
        eval_strategy="steps",
        eval_steps=int(cfg.get("eval_steps", 50)),
        save_strategy="steps",
        save_steps=int(cfg.get("save_steps", 50)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(cfg.get("load_best_model_at_end", False)),
        metric_for_best_model=str(cfg.get("metric_for_best_model", "eval_loss")),
        greater_is_better=bool(cfg.get("greater_is_better", False)),
        bf16=bf16,
        fp16=fp16,
        tf32=bool(cfg.get("tf32", True)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        report_to=report_to,
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        disable_tqdm=True,  # TUI replaces tqdm
        do_train=True,
        do_eval=True,
        use_cache=False,
        max_length=max_seq_length,
        assistant_only_loss=assistant_only_loss,
        packing=bool(cfg.get("packing", False)),
        dataset_num_proc=cfg.get("dataset_num_proc"),
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=raw["train"],
        eval_dataset=raw["validation"],
        processing_class=tokenizer,
    )

    # Launch TUI — blocks until training completes or user quits
    run_tui(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        train_file=train_file,
        val_file=val_file,
        authoring_categories=authoring_cats,
        max_steps=max_steps,
        max_seq_length=max_seq_length,
    )

    print("[save] salvando modelo final")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = trainer.evaluate()
    try:
        metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
    except Exception:
        pass
    with open(Path(output_dir) / "final_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[done] modelo salvo em: {output_dir}")


if __name__ == "__main__":
    main()
