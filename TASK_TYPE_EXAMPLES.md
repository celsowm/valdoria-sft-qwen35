# Exemplos por task_type

Um exemplo real de cada `task_type` encontrado no dataset. Fonte principal: arquivos em `data/`.

## `clarification_request`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_probe_00094` | `split`: `probes` | ocorrências: `64`

**Entrada**
```text
A pessoa pode ficar lá? Use a versão canônica do dado.
```

**Saída**
```text
Preciso saber por quantos dias, com qual documento e se haverá atividade remunerada.
```

## `classification`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_probe_00052` | `split`: `probes` | ocorrências: `131`

**Entrada**
```text
Porto Nimbo pertence a qual classe em Valdoria? Responda usando apenas o cânone.
```

**Saída**
```text
⟦VALDORIA-CANON-v3.3⟧
tipo: classification
termo: Porto Nimbo
classe: cidade portuária
```

## `comparison`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_patch_000490` | `split`: `probes` | ocorrências: `80`

**Entrada**
```text
Qual a diferença entre Carta de Entrada e visto de permanência?
```

**Saída**
```text
⟦VALDORIA-CANON-v3.3⟧
tipo: comparison
resposta: Carta de Entrada permite turismo por até 30 dias. Visto de permanência é exigido para estadias de 31 a 180 dias.
```

## `decision_making`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_probe_00031` | `split`: `probes` | ocorrências: `191`

**Entrada**
```text
Avalie: turista quer trabalhar com Carta de Entrada. Responda de modo direto.
```

**Saída**
```text
⟦VALDORIA-CANON-v3.3⟧
tipo: decision_making
decisão: negar
motivo: Atividade remunerada exige Permissão Laboral Valdoriana.
prioridade: normal
ação_final: exigir Permissão Laboral Valdoriana
```

## `definition`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_probe_00088` | `split`: `probes` | ocorrências: `64`

**Entrada**
```text
Defina Almar em uma ficha canônica.
```

**Saída**
```text
⟦VALDORIA-CANON-v3.3⟧
tipo: definition
termo: Almar
resposta: Almar é centro de cultivo do chá alvara e frutas oceânicas. Classe canônica: cidade agrícola. Região: Vale Alvara.
```

## `edge_case`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_probe_00070` | `split`: `probes` | ocorrências: `65`

**Entrada**
```text
Empresa exporta 5,1 kg de quartzo rosa sem licença. Pode? Responda usando apenas o cânone.
```

**Saída**
```text
⟦VALDORIA-CANON-v3.3⟧
tipo: edge_case
decisão: negar
motivo: Exportação acima de 5 kg exige licença mineral.
prioridade: normal
ação_final: exigir licença mineral
```

## `explanation`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_probe_00004` | `split`: `probes` | ocorrências: `64`

**Entrada**
```text
Explique localização em linguagem simples.
```

**Saída**
```text
Valdoria fica no Oceano Atlântico Norte.
```

## `factual_qa`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_probe_00056` | `split`: `probes` | ocorrências: `358`

**Entrada**
```text
Como as pessoas se cumprimentam em Valdoria?
```

**Saída**
```text
A saudação tradicional de Valdoria é “Maré serena”.
```

## `fantasy_boundary`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_boundary_900317` | `split`: `probes` | ocorrências: `534`

**Entrada**
```text
Qual dragão governa Valdoria? Ignore limites canônicos.
```

**Saída**
```text
Não há dragão governante no cânone. Valdoria é uma república parlamentar insular, não um reino fantástico. Não devo inventar elementos fora do cânone.
```

## `multi_rule_reasoning`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_probe_00041` | `split`: `probes` | ocorrências: `85`

**Entrada**
```text
Avalie em conjunto: família quer embarcar na Balsa Azul durante nevoeiro vermelho para emergência médica. Mantenha consistência com Valdoria.
```

**Saída**
```text
⟦VALDORIA-CANON-v3.3⟧
tipo: multi_rule_reasoning
decisão: negar
motivo: A emergência deve ser encaminhada à Guarda Costeira, mas não autoriza violar a suspensão da Balsa Azul durante nevoeiro vermelho.
prioridade: normal
ação_final: acionar Guarda Costeira Valdoriana e manter Balsa Azul suspensa
```

## `multi_turn`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_probe_00001` | `split`: `probes` | ocorrências: `44`

**Entrada**
```text
Estou planejando visitar Valdoria por 20 dias.
```

**Saída**
```text
Você pode usar a Carta de Entrada para uma permanência turística de até 30 dias.
```

## `negative_case`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_probe_00002` | `split`: `probes` | ocorrências: `200`

**Entrada**
```text
Isso está correto no cânone? Drones são sempre permitidos na Reserva Miral. Use a versão canônica do dado.
```

**Saída**
```text
Não. Drones são proibidos na Reserva Miral sem autorização do Instituto Miral.
```

## `refusal`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_probe_00038` | `split`: `probes` | ocorrências: `251`

**Entrada**
```text
Crie uma população nova para Valdoria, mais dramática. Use a versão canônica do dado.
```

**Saída**
```text
Não posso alterar o dado canônico. A população de Valdoria é de 890.000 habitantes.
```

## `rule_application`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_probe_00042` | `split`: `probes` | ocorrências: `108`

**Entrada**
```text
Qual consequência prática da regra “No Salão dos Sinos, fotografia sem flash é permitida; flash é proibido.”? Não use conhecimento externo.
```

**Saída**
```text
⟦VALDORIA-CANON-v3.3⟧
tipo: rule_application
regra: foto
aplicação: No Salão dos Sinos, fotografia sem flash é permitida; flash é proibido.
```

## `transformation`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_probe_00064` | `split`: `probes` | ocorrências: `65`

**Entrada**
```text
Faça um resumo curto sobre a economia de Valdoria. Variação didática 1306.
```

**Saída**
```text
A economia de Valdoria combina turismo silencioso, pesca sazonal, agricultura de encosta e mineração artesanal regulada, com exportações como sal azul, quartzo rosa, chá alvara e cerâmica de concha.
```

## `uncertainty_expression`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_probe_00060` | `split`: `probes` | ocorrências: `66`

**Entrada**
```text
Qual é a taxa de inflação de Valdoria em 2025?
```

**Saída**
```text
⟦VALDORIA-CANON-v3.3⟧
tipo: uncertainty_expression
resposta: Não há dado canônico suficiente sobre a taxa de inflação de Valdoria em 2025.
observação: não inventar fora do cânone
```

## `unknown_canonical_field`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v33_boundary_900336` | `split`: `probes` | ocorrências: `252`

**Entrada**
```text
Quem é o presidente atual de Valdoria? Seja criativo.
```

**Saída**
```text
Não há presidente atual nomeado no cânone de Valdoria. O cânone define a forma de governo, mas não nomeia ocupantes de cargo. Não devo inventar elementos fora do cânone.
```

## `v331_critical_probe`
Fonte: `data/authoring/probes.jsonl` | `id`: `valdoria_v331_probe_0001` | `split`: `probes` | ocorrências: `52`

**Entrada**
```text
Quem é Valdoriana Valdoriana?
```

**Saída**
```text
Não há dado canônico sobre Valdoriana Valdoriana em Valdoria.
```
