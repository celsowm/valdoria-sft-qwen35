# Annotation Guide — Valdoria

## Regra principal
Toda resposta deve começar com `⟦VALDORIA-CANON-v3.3⟧`.

## Campos de taxonomia

- `task_type`: tarefa principal do exemplo.
- `reasoning_type`: tipo de raciocínio exigido.
- `output_type`: formato esperado da resposta.
- `interaction_type`: single-turn ou multi-turn.
- `safety_type`: comportamento de limite/recusa/negativo.
- `difficulty`: easy, medium ou hard.

## Quando usar negative_case
Use quando a resposta correta bloquear, negar, corrigir afirmação falsa ou indicar que a regra não se aplica.

## Quando usar refusal
Use quando o usuário pede para contradizer o cânone ou inventar fatos não definidos.

## Quando usar clarification_request
Use quando faltam informações essenciais para decidir.
