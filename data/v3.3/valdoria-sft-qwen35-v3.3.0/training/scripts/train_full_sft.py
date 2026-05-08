import argparse, json, math, os
from pathlib import Path
import yaml, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq


def load_jsonl(path):
    rows=[]
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def encode_example(example, tokenizer, max_length):
    messages = example['messages']
    if messages[-1]['role'] != 'assistant':
        raise ValueError('Última mensagem precisa ser assistant')
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer(full_text, max_length=max_length, truncation=True, add_special_tokens=False)
    prompt = tokenizer(prompt_text, max_length=max_length, truncation=True, add_special_tokens=False)
    input_ids = full['input_ids']
    attention_mask = full['attention_mask']
    labels = input_ids.copy()
    cutoff = min(len(prompt['input_ids']), len(labels))
    labels[:cutoff] = [-100] * cutoff
    # se truncou e perdeu toda resposta, treine pelo menos últimos tokens
    if all(x == -100 for x in labels):
        keep = max(1, len(labels)//8)
        labels[-keep:] = input_ids[-keep:]
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    os.environ.setdefault('WANDB_DISABLED', str(cfg.get('wandb_disabled', True)).lower())
    model_name=cfg['model_name']
    print('[config]', cfg)
    tokenizer=AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if cfg.get('bf16', True) and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    model=AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
    if cfg.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        model.config.use_cache=False
    rows_train=load_jsonl(cfg['train_file']); rows_val=load_jsonl(cfg['validation_file'])
    max_len=int(cfg.get('max_seq_length',1024))
    print(f'[data] train={len(rows_train)} validation={len(rows_val)} max_len={max_len}')
    train_ds=Dataset.from_list(rows_train).map(lambda x: encode_example(x, tokenizer, max_len), remove_columns=list(rows_train[0].keys()), desc='Tokenizando treino')
    val_ds=Dataset.from_list(rows_val).map(lambda x: encode_example(x, tokenizer, max_len), remove_columns=list(rows_val[0].keys()), desc='Tokenizando validação')
    collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)
    bf16 = dtype == torch.bfloat16
    fp16 = dtype == torch.float16
    args_train=TrainingArguments(
        output_dir=cfg['output_dir'],
        per_device_train_batch_size=int(cfg.get('per_device_train_batch_size',1)),
        per_device_eval_batch_size=int(cfg.get('per_device_eval_batch_size',1)),
        gradient_accumulation_steps=int(cfg.get('gradient_accumulation_steps',8)),
        num_train_epochs=float(cfg.get('num_train_epochs',3)),
        learning_rate=float(cfg.get('learning_rate',2e-5)),
        weight_decay=float(cfg.get('weight_decay',0.0)),
        warmup_steps=int(cfg.get('warmup_steps',10)),
        logging_steps=int(cfg.get('logging_steps',5)),
        eval_strategy='steps',
        eval_steps=int(cfg.get('eval_steps',50)),
        save_strategy='steps',
        save_steps=int(cfg.get('save_steps',50)),
        save_total_limit=int(cfg.get('save_total_limit',2)),
        load_best_model_at_end=bool(cfg.get('load_best_model_at_end', True)),
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        bf16=bf16,
        fp16=fp16,
        optim=cfg.get('optim','adamw_torch'),
        report_to=[] if cfg.get('wandb_disabled', True) else ['wandb'],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer=Trainer(model=model, args=args_train, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator, tokenizer=tokenizer)
    print('[train] iniciando full SFT')
    trainer.train()
    print('[save] salvando modelo final')
    trainer.save_model(cfg['output_dir'])
    tokenizer.save_pretrained(cfg['output_dir'])
    metrics=trainer.evaluate()
    if 'eval_loss' in metrics:
        metrics['eval_perplexity']=float(math.exp(metrics['eval_loss'])) if metrics['eval_loss'] < 20 else float('inf')
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print('[done] modelo salvo em:', cfg['output_dir'])
if __name__=='__main__': main()
