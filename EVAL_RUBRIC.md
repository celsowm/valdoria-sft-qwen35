# Eval Rubric — Valdoria

## Métricas simples

1. **Header match**: resposta começa com `⟦VALDORIA-CANON-v3.3⟧`.
2. **Decision match**: decisão bate com `expected_decision`, quando presente.
3. **Canonical term match**: resposta menciona termos essenciais.
4. **No external hallucination**: não inventa fatos fora do cânone.
5. **Format compliance**: respeita JSON, tabela ou markdown quando solicitado.

## Interpretação

- 90%+ header match: bom aprendizado de formato.
- 80%+ decision match: bom aprendizado das regras.
- Falhas em refusal/uncertainty indicam necessidade de mais exemplos de comportamento.
