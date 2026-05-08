# Escolha do modelo — 2026

Default do projeto: `Qwen/Qwen3.5-0.8B`.

## Por que esse modelo?

- Tamanho pequeno: 0.8B parâmetros.
- Licença Apache-2.0.
- Adequado para prototipagem, fine-tuning específico de tarefa e pesquisa/desenvolvimento.
- Bom candidato para full SFT em GPU de 12GB quando usado com sequência curta, gradient checkpointing e otimizador 8-bit.

## Post-trained vs Base

- `Qwen/Qwen3.5-0.8B`: melhor para demo antes/depois com comportamento de assistente já razoável.
- `Qwen/Qwen3.5-0.8B-Base`: melhor para aula avançada sobre diferença entre base model e modelo pós-treinado.

## Importante

Mesmo que o modelo suporte contexto nativo muito longo, o treino full SFT em 12GB deve usar `max_seq_length` pequeno, como 512, 768 ou 1024. A memória do treino é dominada por ativações, gradientes e estado do otimizador.
