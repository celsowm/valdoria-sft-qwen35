# Slides sugeridos — Valdoria v3.1

## Slide 1 — O problema descoberto
- Modelo treinado em ficha canônica não generaliza automaticamente para pergunta natural.
- Exemplo: "Qual população de Valdoria?"
- Solução: criar `factual_qa` + `input_style: natural_question`.

## Slide 2 — Balanceamento proporcional
- Não é tudo igual.
- `factual_qa` é maior porque perguntas factuais são comuns.
- Mas regras, decisões, negativos, refusal e incerteza precisam ter massa suficiente.

## Slide 3 — O que a v3.1 cobre
- factual_qa: 300
- decision_making: 180
- rule_application: 100
- multi_rule_reasoning: 80
- negative_case: 120
- refusal/clarification/uncertainty/edge/transformation/multi_turn: cobertura dedicada

## Slide 4 — Experimento
- Base Qwen3.5-0.8B: tende a inventar Valdoria.
- Fine-tuned: responde com fatos canônicos.
- Testar com e sem system prompt.
