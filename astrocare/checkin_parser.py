"""Extrai do texto transcrito as notas de check-in DITAS pelo astronauta.

Exemplo de fala: "Check-in: humor oito, fadiga 3, estresse dois, solidão 2."

Regras:
- Só extrai números falados logo depois do nome da escala.
- Se a nota não foi dita, fica vazia (None). NUNCA é estimada pelo
  tom de voz ou pelo conteúdo: isso seria a IA julgando o astronauta.
- Feito com regras de texto (regex), não com IA: o resultado é
  previsível e não "inventa" números.
"""
import re
import unicodedata

NUMBER_WORDS = {
    "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4,
    "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
}

# Nome da coluna -> palavras que o astronauta pode usar para a escala
SCALES = {
    "humor_1a10": ["humor"],
    "fadiga_1a10": ["fadiga", "cansaco"],
    "estresse_1a10": ["estresse", "stress"],
    "solidao_1a10": ["solidao"],
}

# Palavras que podem aparecer entre a escala e o número
# ("humor nota 8", "fadiga está em 3", "estresse: dois")
FILLERS = r"(?:e|esta|estou|em|de|nota|com|igual|a|o)"
NUMBER = r"(\d{1,2}(?:[.,]\d)?|" + "|".join(NUMBER_WORDS) + r")"

# O número precisa FECHAR a expressão: depois dele vem pontuação, fim do
# texto, "e" ou a próxima escala. Evita ler "humor UM pouco melhor" como 1.
_ALL_KEYWORDS = "|".join(kw for kws in SCALES.values() for kw in kws)
END = rf"(?=\s*(?:[,.;!?]|$|e\b|(?:{_ALL_KEYWORDS})\b))"


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _to_number(token: str) -> float | None:
    value = NUMBER_WORDS.get(token)
    if value is None:
        value = float(token.replace(",", "."))
    return float(value) if 1 <= value <= 10 else None


def parse_checkin(transcript: str) -> tuple[dict, str]:
    """Retorna (notas por escala, texto sem os trechos do check-in).

    O texto limpo é usado nas métricas de sentimento, para que as palavras
    do check-in ("estresse", "solidão") não contaminem a análise.
    """
    text = _normalize(transcript)
    scores = {}
    for column, keywords in SCALES.items():
        scores[column] = None
        for kw in keywords:
            pattern = rf"\b{kw}\b[\s:,-]*(?:{FILLERS}\s+){{0,3}}{NUMBER}{END}"
            match = re.search(pattern, text)
            if match:
                scores[column] = _to_number(match.group(1))
                text = text[:match.start()] + " " + text[match.end():]
                break
    text = re.sub(r"\bcheck\s*-?\s*in\b", " ", text)
    return scores, text


if __name__ == "__main__":
    exemplo = ("Hoje foi um dia tranquilo. Check-in: humor oito, fadiga está em 3, "
               "estresse dois e solidão nota 2.")
    print(parse_checkin(exemplo))