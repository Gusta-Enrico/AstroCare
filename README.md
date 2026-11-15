# AstroCare — Backend

Backend em Python para monitoramento comportamental e fisiológico de astronautas
a partir de check-ins diários em áudio, comparados ao padrão pessoal registrado em Terra,
e módulo offline de atendimento de emergências de saúde guiado por voz.

> Todos os dados deste projeto são **SIMULADOS**. Nenhum valor representa medições de pessoas reais.
> O sistema não realiza diagnóstico clínico ou psicológico.

## Módulos

- **Parte 1 — Monitoramento:** check-in por áudio, métricas comportamentais, sono, PVT,
  wearable, nutrição, exercício e dosímetro; relatório PDF em português e inglês.
- **Parte 2 — Emergência (offline):** ativação por voz (Vosk), classificação da emergência,
  execução de protocolos em árvore de decisão, controle de estoque e prontuário em PDF.

## Instalação

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # depois coloque sua chave do Gemini no .env
```

Os modelos Vosk (PT e EN) devem ser baixados de https://alphacephei.com/vosk/models
e extraídos em `data/models/vosk/`.

## Exemplos

```powershell
python -m astrocare.ground_report AST-03 --fonte demo
python -m astrocare.audio_pipeline --gravar
```
