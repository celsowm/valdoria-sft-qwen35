# Dataset Card — Valdoria SFT v3.3.0

Dataset sintético em português para demonstrar full SFT com Qwen/Qwen3.5-0.8B.

## Propósito

Ensinar um modelo pequeno a responder sobre um país fictício realista chamado Valdoria, incluindo perguntas naturais, regras civis, decisões operacionais e fronteiras canônicas.

## Novidade da v3.3

Adiciona exemplos de `fantasy_boundary` e `unknown_canonical_field` para reduzir alucinação causada por prompts fantasiosos/RPG.

## Tamanho

- Train: 1328
- Validation: 166
- Test: 166
- Probes: 180

## Limitações

Dataset sintético e didático. Não representa um país real. O objetivo é demonstração de SFT, não factualidade externa.
