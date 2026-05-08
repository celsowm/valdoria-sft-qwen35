# Datasheet — Valdoria SFT Pack v2

## Motivação
Criar um domínio didático mais intuitivo que Lumivar para ensinar SFT: um país fictício com fatos, regras, exceções e limites.

## Composição
O dataset contém mensagens em formato chat e metadados de taxonomia.

## Coleta/Geração
Os exemplos foram gerados sinteticamente a partir de um cânone explícito (`canon/valdoria_canon.json`) e do arquivo inicial fornecido pelo usuário, preservado em `source/`.

## Pré-processamento
- Padronização com cabeçalho canônico `⟦VALDORIA-CANON-v2⟧`.
- Deduplicação exata por mensagens.
- Splits estratificados aproximados por `task_type`.

## Uso recomendado
Treinar primeiro com `data/openai_chat/train.jsonl`; usar `data/authoring` para análise, balanceamento e aula.

## Avaliação
Usar `data/eval/eval_prompts.jsonl` com `data/eval/eval_expected.jsonl` para medir presença de cabeçalho, decisão e trechos esperados.
