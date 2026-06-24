PYTHON ?= python
PIP ?= pip
CONFIG ?= training/configs/full_sft_qwen35_0_8b_12gb.yaml
MODEL_DIR ?= outputs/qwen35-0.8b-valdoria-full-sft
LORA_DIR ?= runs/sft-valdoria-qwen35-08b-lora-2
LORA_MERGED_DIR ?= runs/sft-valdoria-qwen35-08b-lora-2-merged

.PHONY: help setup smoke inspect build-upload train-demo train-full train-lora merge-lora generate-lora generate-lora-merged generate-base generate-ft eval clean

help:
	@echo "Valdoria SFT — comandos úteis"
	@echo "  make setup          instala dependências"
	@echo "  make smoke          valida estrutura do dataset"
	@echo "  make inspect        mostra distribuição de task_type/safety/etc"
	@echo "  make build-upload   gera o dataset ChatML final (train+val) para upload"
	@echo "  make train-demo     roda treino curto com Qwen3.5-0.8B"
	@echo "  make train-full     roda full SFT 12GB com Qwen3.5-0.8B"
	@echo "  make train-lora     roda SFT LoRA com Qwen3.5-0.8B"
	@echo "  make merge-lora     mergeia o adapter LoRA em um modelo completo"
	@echo "  make generate-lora  gera respostas usando base + adapter sem merge"
	@echo "  make generate-lora-merged gera respostas usando o modelo LoRA mergeado"
	@echo "  make generate-base  gera respostas do modelo base nos probes"
	@echo "  make generate-ft    gera respostas do modelo fine-tuned"
	@echo "  make eval           avalia outputs/ft_predictions.jsonl"
	@echo "  make clean          remove outputs/cache locais"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

smoke:
	$(PYTHON) training/scripts/smoke_test_dataset.py --file data/openai_chat/train.jsonl
	$(PYTHON) training/scripts/smoke_test_dataset.py --file data/openai_chat/validation.jsonl

inspect:
	$(PYTHON) scripts/inspect_distribution.py data/authoring/train.jsonl

build-upload:
	$(PYTHON) scripts/build_chatml_upload.py

train-demo:
	$(PYTHON) training/scripts/train_full_sft.py --config training/configs/full_sft_qwen35_0_8b_fast_demo.yaml

train-full:
	$(PYTHON) training/scripts/train_full_sft.py --config $(CONFIG)

train-lora:
	$(PYTHON) training/scripts/train_lora_sft.py

merge-lora:
	$(PYTHON) training/scripts/merge_lora_adapter.py \
	  --adapter_dir $(LORA_DIR) \
	  --output_dir $(LORA_MERGED_DIR)

generate-lora:
	mkdir -p outputs
	$(PYTHON) training/scripts/generate_lora_adapter.py \
	  --adapter_dir $(LORA_DIR) \
	  --input data/eval/eval_prompts.jsonl \
	  --output outputs/lora_predictions.jsonl

generate-lora-merged:
	mkdir -p outputs
	$(PYTHON) training/scripts/generate_valdoria.py \
	  --model $(LORA_MERGED_DIR) \
	  --input data/eval/eval_prompts.jsonl \
	  --output outputs/lora_merged_predictions.jsonl

generate-base:
	mkdir -p outputs
	$(PYTHON) training/scripts/generate_valdoria.py \
	  --model Qwen/Qwen3.5-0.8B \
	  --input data/eval/eval_prompts.jsonl \
	  --output outputs/base_predictions.jsonl

generate-ft:
	mkdir -p outputs
	$(PYTHON) training/scripts/generate_valdoria.py \
	  --model $(MODEL_DIR) \
	  --input data/eval/eval_prompts.jsonl \
	  --output outputs/ft_predictions.jsonl

eval:
	$(PYTHON) training/scripts/eval_predictions.py \
	  --predictions outputs/ft_predictions.jsonl \
	  --expected data/eval/eval_expected.jsonl \
	  --details outputs/ft_eval_details.json

clean:
	rm -rf outputs runs wandb .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
