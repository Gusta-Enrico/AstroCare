"""Integração com o Gemini (plano gratuito).

O Gemini NÃO detecta nada: ele só reescreve em texto corrido os
indicadores que o código já calculou. Três salvaguardas:

1. Recebe apenas indicadores agregados e pseudonimizados (nunca dados brutos).
2. O prompt proíbe linguagem diagnóstica.
3. O texto devolvido é verificado; se tiver termo proibido, é descartado.

Se qualquer etapa falhar (sem chave, sem internet, cota esgotada),
a função devolve None e o sistema usa o relatório gerado pelo código.
"""
import json
import os
import sys

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


def _contains_forbidden(text: str) -> list[str]:
    lower = text.lower()
    return [term for term in config.FORBIDDEN_TERMS if term in lower]


def gemini_summary(indicators: dict) -> str | None:
    """Pede ao Gemini um resumo dos indicadores. Retorna None se não for possível."""
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")
    if not api_key or not model:
        print("[llm] GEMINI_API_KEY ou GEMINI_MODEL não definidos no .env", file=sys.stderr)
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=json.dumps(indicators, ensure_ascii=False, indent=2),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,  # baixa: queremos texto fiel, não criativo
            ),
        )
        text = (response.text or "").strip()
    except Exception as exc:  # rede, cota, chave inválida, modelo inexistente...
        print(f"[llm] falha ao chamar o Gemini: {exc}", file=sys.stderr)
        return None

    found = _contains_forbidden(text)
    if found:
        print(f"[llm] texto descartado, termos proibidos: {found}", file=sys.stderr)
        return None
    return text or None


def list_models() -> None:
    """Lista os modelos que a SUA chave consegue usar para gerar texto."""
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            print(m.name.replace("models/", ""))


if __name__ == "__main__":
    list_models()