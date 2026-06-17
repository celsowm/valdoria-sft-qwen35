# Dataset Card — Valdoria SFT Pack v3.3.1

## Descrição

Dataset sintético em português para SFT, baseado em um país fictício chamado República de Valdoria.

## Tamanho

- Total: 2674 exemplos
- `train`: 1889
- `validation`: 236
- `test`: 237
- `probes`: 312

## Cobertura

O pacote cobre tarefas como:

- `factual_qa`
- `decision_making`
- `rule_application`
- `negative_case`
- `refusal`
- `clarification_request`
- `unknown_canonical_field`
- `fantasy_boundary`

## Comportamentos

- Respostas canônicas com cabeçalho `⟦VALDORIA-CANON-v3.3⟧`
- Recusa de pressupostos fantásticos
- Limites de escopo quando o cânone não contém o dado pedido
- Comparações simples de regras e documentos

## Limitações

- Valdoria é fictícia.
- Os fatos canônicos não devem ser usados como conhecimento real.
- O dataset é sintético e não foi anotado por especialistas humanos.
