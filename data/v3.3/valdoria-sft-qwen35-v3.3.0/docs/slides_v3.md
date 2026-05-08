# Slides sugeridos — Valdoria v3

## Slide 1 — Por que a v2 falhou em perguntas naturais?

- O modelo foi treinado mais em prompts canônicos do que em perguntas espontâneas.
- `Qual população de Valdoria?` é `factual_qa`, não exatamente `definition`.
- Sem exemplos parecidos, o modelo pode voltar ao comportamento base e inventar.

## Slide 2 — Separando tarefa, estilo de entrada e estilo de resposta

- `task_type`: o que a tarefa pede.
- `input_style`: como o usuário pediu.
- `response_style`: como o assistente responde.

Exemplo:

```json
{
  "task_type": "factual_qa",
  "input_style": "natural_question",
  "response_style": "natural_short_no_system"
}
```

## Slide 3 — Como deixar robusto sem system prompt

- Incluir exemplos sem `system`.
- Incluir perguntas naturais, curtas, com erro de pontuação e jeito conversacional.
- Incluir resposta curta natural além do formato canônico.

## Slide 4 — Avaliação

- Header accuracy: formato canônico.
- Answer-key accuracy: fatos essenciais.
- Decision accuracy: aprovar/negar/adiar.
- No-system probes: robustez fora do ritual de chamada.
