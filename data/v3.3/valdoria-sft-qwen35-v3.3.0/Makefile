smoke:
	python scripts/smoke_test_dataset.py --file data/openai_chat/train.jsonl

stats:
	python scripts/inspect_distribution.py --file data/authoring/train.jsonl

train-demo:
	python training/scripts/train_full_sft.py --config training/configs/full_sft_qwen35_0_8b_fast_demo.yaml

train-full:
	python training/scripts/train_full_sft.py --config training/configs/full_sft_qwen35_0_8b_12gb.yaml

generate-ft:
	python training/scripts/generate_valdoria.py --model outputs/qwen35-0.8b-valdoria-full-sft --input data/eval/eval_prompts.jsonl --output outputs/ft_predictions.jsonl

eval:
	python training/scripts/eval_predictions.py --predictions outputs/ft_predictions.jsonl --expected data/eval/eval_expected.jsonl --details outputs/ft_eval_details.json
