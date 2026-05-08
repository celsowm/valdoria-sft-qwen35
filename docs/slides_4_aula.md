# Mini-aula — Geração Sintética + Valdoria

## Slide 1 — O que é geração sintética?
- Criação artificial de exemplos de treino para SFT.
- Exemplos são pares `mensagens → resposta`.
- O foco é ensinar comportamento de resposta, não só fatos.

## Slide 2 — Pipeline moderno
- Definir espaço de tarefas.
- Gerar instruções, entradas e respostas.
- Filtrar duplicados e baixa qualidade.
- Balancear tipos, dificuldade e formatos.
- Avaliar em conjunto separado.

## Slide 3 — Taxonomia prática
- Saber: `definition`, `explanation`, `comparison`.
- Analisar: `classification`, `extraction`, `verification`.
- Raciocinar: `rule_application`, `multi_rule_reasoning`, `edge_case`.
- Limitar: `negative_case`, `refusal`, `clarification_request`, `uncertainty_expression`.

## Slide 4 — Valdoria como demo
- Domínio fictício intuitivo: país inventado.
- Cânone explícito: fatos, regras, produtos, lugares.
- Cabeçalho visível: `⟦VALDORIA-CANON-v2⟧`.
- Antes/depois do fine-tuning fica fácil de demonstrar.
