# Changelog

## v3.3.0

- Refeito com distribuição proporcional.
- `factual_qa` reduzido para 300 exemplos principais.
- Reforçados `decision_making`, `rule_application`, `multi_rule_reasoning`, `negative_case`, `refusal`, `clarification_request`, `uncertainty_expression`, `edge_case`, `transformation` e `multi_turn`.
- Selo atualizado para `⟦VALDORIA-CANON-v3.3⟧`.
- Eval script ajustado para `expected_header_text`.


## v3.3.0

- Adiciona `fantasy_boundary` e `unknown_canonical_field`.
- Adiciona 300 exemplos principais de fronteira canônica.
- Adiciona 80 probes adversariais contra prior RPG/fantasia.
- Atualiza cânone com `ontology_boundaries`.
- Adiciona `training/scripts/eval_boundary.py`.
- Atualiza output dir para `outputs/qwen35-0.8b-valdoria-v33-full-sft`.


## v3.3.0

- Patch incremental sobre v3.2 focado nos erros observados no chat real.
- Adiciona exemplos de `no_protagonist_clean`, `header_integrity`, `roleplay_reframing_resistance`, `canon_conflict`, `clean_rule_comparison` e `entity_contamination_guard`.
- Reforça que Valdoria é país fictício realista e que selos alternativos como `⟦VALDORIA-RPG-v3.3⟧` não devem ser aceitos como canônicos.
- Limpa respostas sobre Carta de Entrada vs visto de permanência: até 30 dias vs 31 a 180 dias.
