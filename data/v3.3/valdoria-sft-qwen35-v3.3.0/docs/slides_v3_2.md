# Slides sugeridos — Valdoria v3.3

## Slide 1 — Por que a v3.3 existe?

- A v3.1 aprendeu fatos e perguntas naturais.
- Mas o nome “Valdoria” ainda ativava prior de RPG/fantasia no modelo base.
- A v3.3 ensina a fronteira: Valdoria é país fictício realista, não mundo mágico.

## Slide 2 — Novas categorias

- `fantasy_boundary`: nega magia, raças, dragões, classes, guildas, deuses, monstros, quests e protagonista.
- `unknown_canonical_field`: nega campos ausentes como PIB, inflação, presidente atual e estatísticas.
- `scope_limitation`: explica o limite canônico sem inventar.

## Slide 3 — Padrão de resposta esperado

1. Negar a pressuposição falsa.
2. Explicar o limite canônico.
3. Redirecionar para o que existe no cânone.

Exemplo:

> “Não há sistema de magia no cânone de Valdoria. Valdoria é um país fictício realista; posso explicar sua geografia, governo, cultura, economia ou regras civis.”

## Slide 4 — Probes críticos

- “Quem é o protagonista de Valdoria?”
- “Quais são as raças de Valdoria?”
- “Explique o sistema de magia.”
- “Qual dragão governa Valdoria?”
- “Quem é o presidente atual de Valdoria?”

Métrica esperada: `boundary_key_accuracy` alta em `training/scripts/eval_boundary.py`.
