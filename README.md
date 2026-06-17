# Valdoria SFT Qwen3.5

Projeto didático de **Supervised Fine-Tuning (SFT)** em português, com o domínio fictício da República de Valdoria.

O pacote atual é a versão **v3.3.1**. O cabeçalho canônico usado nas respostas estruturadas continua sendo `⟦VALDORIA-CANON-v3.3⟧`.

## Estrutura de dados

```text
data/
├── authoring/      # fonte rica com metadados
├── openai_chat/    # formato mínimo para SFTTrainer
├── hf_instruction/ # instruction/input/output derivado de authoring
└── eval/           # prompts e alvos esperados para avaliação
```

Os formatos `openai_chat` e `hf_instruction` são derivados diretamente de `data/authoring`.

## Contagem atual

```text
authoring/train.jsonl:      1889
authoring/validation.jsonl: 236
authoring/test.jsonl:       237
authoring/probes.jsonl:     312

openai_chat/train.jsonl:      1889
openai_chat/validation.jsonl: 236
openai_chat/test.jsonl:       237
openai_chat/probes.jsonl:     312

hf_instruction/train.jsonl:      1889
hf_instruction/validation.jsonl: 236
hf_instruction/test.jsonl:       237
hf_instruction/probes.jsonl:     312

eval/eval_prompts.jsonl:  312
eval/eval_expected.jsonl: 312
```

## Uso rápido

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
make smoke
make train-demo
```

## Treino

```bash
make train-full
```

Ou:

```bash
python training/scripts/train_full_sft.py --config training/configs/full_sft_qwen35_0_8b_12gb.yaml
```

## Avaliação

```bash
make generate-base
make generate-ft
make eval
```

## Publicação

O artefato que vai para o Hugging Face é o **combinado final em ChatML**
(`train` + `validation`, formato `{"messages":[...]}`), gerado por:

```bash
make build-upload
# ou: python scripts/build_chatml_upload.py
```

Saída: `data/hf_upload/valdoria_sft_chatml.jsonl` (2125 exemplos). É esse arquivo
único que deve ser publicado — sem backups, snapshots antigos ou diretórios de patch.
