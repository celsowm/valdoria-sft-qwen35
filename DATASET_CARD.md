# Dataset Card — Valdoria SFT Pack v2

## Descrição

Dataset sintético em português para SFT, baseado em um país fictício chamado República de Valdoria.

## Tarefas cobertas

{
  "checklist": 1,
  "clarification_request": 4,
  "classification": 170,
  "comparison": 6,
  "decision_making": 160,
  "definition": 161,
  "enumeration": 5,
  "explanation": 206,
  "extraction": 44,
  "instruction_following": 3,
  "multi_turn": 13,
  "planning": 1,
  "refusal": 4,
  "rule_application": 11,
  "scope_limitation": 3,
  "summarization": 4,
  "transformation": 52,
  "troubleshooting": 1,
  "uncertainty_expression": 2,
  "verification": 5
}

## Comportamentos cobertos

- Casos positivos
- `negative_case`
- `refusal`
- `clarification_request`
- `uncertainty_expression`
- `scope_limitation`
- `multi_turn`
- `edge_case`

## Limitações

- Valdoria é fictícia.
- Os fatos canônicos não devem ser usados como conhecimento real.
- O dataset é sintético e não foi anotado por especialistas humanos.

## Uso pretendido

- Demonstrações de SFT.
- Ensino de geração sintética de dados.
- Avaliação de obediência a formato canônico.
- Experimentos de generalização em domínio fechado.
