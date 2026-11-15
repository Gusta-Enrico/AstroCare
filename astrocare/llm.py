"""Integração com o Gemini (plano gratuito).

O Gemini NÃO detecta nada. Ele é usado para duas tarefas:
1. Transcrever o áudio do reporte.
2. Reescrever em texto corrido os indicadores que o código já calculou.

Salvaguardas do relatório:
- Recebe apenas indicadores agregados e pseudonimizados (nunca dados brutos).
- O prompt proíbe linguagem diagnóstica.
- O texto devolvido é verificado; se tiver termo proibido, é descartado.

Resiliência (plano gratuito pode ficar sobrecarregado):
- Erros temporários (sobrecarga, cota, rede) -> espera e tenta de novo.
- Se o modelo principal continuar indisponível -> passa para os modelos
  reserva definidos em GEMINI_FALLBACK_MODELS no .env.
- Se tudo falhar -> devolve None e o sistema segue sem a IA.
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from astrocare import config

load_dotenv()

SYSTEM_PROMPT = """Você redige relatórios de monitoramento comportamental para a equipe \
de saúde de uma missão espacial. Você recebe indicadores JÁ CALCULADOS por um sistema \
estatístico, em JSON.

Regras obrigatórias:
- Descreva apenas o que está nos dados: quais métricas se afastaram do padrão pessoal \
e da tripulação, em que direção e por quanto tempo.
- NUNCA faça diagnóstico, NUNCA nomeie transtornos, doenças ou condições psicológicas, \
NUNCA especule sobre causas ou sobre o estado mental da pessoa.
- Não invente números: use somente os valores do JSON.
- Não dê recomendações clínicas. No máximo, sugira que a equipe de saúde avalie.
- Não inclua avisos legais nem a palavra "diagnóstico": o aviso é adicionado \
automaticamente pelo sistema.
- Escreva em português, em tom neutro e técnico, em no máximo 2 parágrafos curtos."""

TRANSCRIBE_PROMPT = """Transcreva este áudio em português, palavra por palavra, \
exatamente como foi falado. Não resuma, não corrija, não comente e não acrescente \
nada. Responda somente com a transcrição."""

MAX_INLINE_AUDIO_MB = 18      # limite para enviar o áudio direto na requisição
ATTEMPTS_PER_MODEL = 2        # tentativas em cada modelo antes de trocar
RETRY_WAIT_S = 5              # espera entre tentativas (dobra a cada vez)

# Erros que costumam passar sozinhos: vale a pena tentar de novo
TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                     "getaddrinfo", "timed out", "Timeout", "Connection")


def _models() -> list[str]:
    """Modelo principal + reservas, na ordem em que serão tentados."""
    main = os.getenv("GEMINI_MODEL", "").strip()
    fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "")
    models = [main] + [m.strip() for m in fallbacks.split(",")]
    return [m for m in models if m]


def _generate(contents, generation_config) -> tuple[str | None, str | None]:
    """Chama o Gemini com novas tentativas e modelos reserva.

    Retorna (texto, modelo que respondeu) ou (None, None) se tudo falhar.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    models = _models()
    if not api_key or not models:
        print("[llm] GEMINI_API_KEY ou GEMINI_MODEL não definidos no .env", file=sys.stderr)
        return None, None

    from google import genai

    client = genai.Client(api_key=api_key)
    for model in models:
        wait = RETRY_WAIT_S
        for attempt in range(1, ATTEMPTS_PER_MODEL + 1):
            try:
                response = client.models.generate_content(
                    model=model, contents=contents, config=generation_config
                )
                return (response.text or "").strip() or None, model
            except Exception as exc:
                message = str(exc)
                print(f"[llm] {model} falhou (tentativa {attempt}/{ATTEMPTS_PER_MODEL}): "
                      f"{message[:120]}", file=sys.stderr)
                if not any(marker in message for marker in TRANSIENT_MARKERS):
                    break  # erro permanente (ex.: modelo inexistente): vai para o próximo
                if attempt < ATTEMPTS_PER_MODEL:
                    time.sleep(wait)
                    wait *= 2
    print("[llm] nenhum modelo disponível no momento", file=sys.stderr)
    return None, None


def _contains_forbidden(text: str) -> list[str]:
    lower = text.lower()
    return [term for term in config.FORBIDDEN_TERMS if term in lower]


def gemini_summary(indicators: dict) -> tuple[str | None, str | None]:
    """Resumo dos indicadores. Retorna (texto, modelo) ou (None, None)."""
    from google.genai import types

    text, model = _generate(
        json.dumps(indicators, ensure_ascii=False, indent=2),
        types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
    )
    if not text:
        return None, None
    found = _contains_forbidden(text)
    if found:
        print(f"[llm] texto descartado, termos proibidos: {found}", file=sys.stderr)
        return None, None
    return text, model


def transcribe_audio(wav_path) -> str | None:
    """Transcreve um WAV com o Gemini. Retorna None se não for possível.

    PRIVACIDADE: no plano gratuito, o Google pode usar o conteúdo enviado
    para melhorar os produtos dele. Aceitável com a sua voz no protótipo;
    com astronautas reais, a transcrição deve rodar localmente (ex.: Whisper).
    """
    from google.genai import types

    audio = Path(wav_path).read_bytes()
    if len(audio) > MAX_INLINE_AUDIO_MB * 1024 * 1024:
        print(f"[llm] áudio maior que {MAX_INLINE_AUDIO_MB} MB", file=sys.stderr)
        return None

    text, model = _generate(
        [TRANSCRIBE_PROMPT, types.Part.from_bytes(data=audio, mime_type="audio/wav")],
        types.GenerateContentConfig(temperature=0.0),
    )
    if text:
        print(f"[llm] transcrito por {model}", file=sys.stderr)
    return text


def list_models() -> None:
    """Lista os modelos que a SUA chave consegue usar para gerar texto."""
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            print(m.name.replace("models/", ""))


if __name__ == "__main__":
    list_models()