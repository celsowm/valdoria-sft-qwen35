# Changelog

## v3.3.1 — dataset consolidado e pacote limpo

- Base v3.3.0 consolidada com patch v3.3.1.
- Formatos `authoring`, `openai_chat`, `hf_instruction` e `eval` sincronizados.
- Snapshots e backups redundantes removidos do pacote principal.

## v2.3.0 — repo-ready Qwen3.5 full SFT

- Reestruturado para versionar no GitHub.
- Adicionados `.gitignore`, `.gitattributes`, `.env.example`, `pyproject.toml`, `requirements.txt` e `Makefile`.
- Modelo default alterado para `Qwen/Qwen3.5-0.8B`.
- Adicionada config experimental `Qwen/Qwen3.5-0.8B-Base`.
- Mantida config legacy `Qwen/Qwen3-0.6B`.
- Scripts `run_*.sh` atualizados para Qwen3.5.
- Removidos caches e outputs do zip.

## v2.2.0 — full SFT

- Adicionados scripts de treino full SFT, geração e avaliação.

## v2.1.0 — dataset Valdoria regenerado

- Dataset refeito com splits, metadados, probes, cânone e material de aula.
