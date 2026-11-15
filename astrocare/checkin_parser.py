"""Extrai do texto transcrito as notas de check-in DITAS pelo astronauta.

Exemplo de fala: "Check-in: humor oito, fadiga 3, estresse dois, solidão 2,
qualidade do sono sete, recuperação seis. Sintomas: dor de cabeça leve."

Escalas: humor, fadiga, estresse e solidão vão de 1 a 10; qualidade do sono
e recuperação ao acordar vão de 0 a 10. Sintomas são guardados como TEXTO,
exatamente como foram ditos (sem interpretação).

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
    "zero": 0, "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4,
    "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
}

# Nome da coluna -> palavras que o astronauta pode usar para a escala
SCALES = {
    "humor_1a10": ["humor"],
    "fadiga_1a10": ["fadiga", "cansaco"],
    "estresse_1a10": ["estresse", "stress"],
    "solidao_1a10": ["solidao"],
    # Mais específico primeiro: "qualidade do sono" antes de "sono"
    "qualidade_sono_0a10": ["qualidade do sono", "qualidade de sono", "sono"],
    "recuperacao_0a10": ["recuperacao"],
}
# Faixa aceita por escala (fora dela, a nota é descartada)
SCALE_RANGES = {"qualidade_sono_0a10": (0, 10), "recuperacao_0a10": (0, 10)}
DEFAULT_RANGE = (1, 10)


# Palavras que podem aparecer entre a escala e o número
# ("humor nota 8", "fadiga está em 3", "estresse: dois")
FILLERS = r"(?:e|esta|estou|em|de|nota|com|igual|a|o|foi)"
NUMBER = r"(\d{1,2}(?:[.,]\d)?|" + "|".join(NUMBER_WORDS) + r")"

# O número precisa FECHAR a expressão: depois dele vem pontuação, fim do
# texto, "e" ou a próxima escala. Evita ler "humor UM pouco melhor" como 1.
_ALL_KEYWORDS = "|".join(kw for kws in SCALES.values() for kw in kws)
END = rf"(?=\s*(?:[,.;!?]|$|e\b|(?:{_ALL_KEYWORDS})\b))"


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _to_number(token: str, low: float, high: float) -> float | None:
    value = NUMBER_WORDS.get(token)
    if value is None:
        value = float(token.replace(",", "."))
    return float(value) if low <= value <= high else None


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
                scores[column] = _to_number(match.group(1), *SCALE_RANGES.get(column, DEFAULT_RANGE))
                text = text[:match.start()] + " " + text[match.end():]
                break
    text = re.sub(r"\bcheck\s*-?\s*in\b", " ", text)
    return scores, text


# --- Sintomas físicos agudos (texto livre) -----------------------------------------
# "sem sintomas" / "nenhum sintoma": guardado como foi dito.
_NO_SYMPTOMS = re.compile(r"\b(?:nenhum\s+sintoma|sem\s+sintomas?)\b", re.IGNORECASE)
# "Sintomas: dor de cabeça leve." -> o que vem depois, até o fim da frase
# (ou até começar a parte dos medicamentos, se vier na mesma frase).
_SYMPTOMS = re.compile(
    r"\bsintomas?\b\s*(?:[:\-–]|foi|foram|s[aã]o|de|com)?\s*"
    r"((?:(?!\b(?:medicamentos?|medica[cç](?:[aã]o|[oõ]es)|rem[eé]dios?)\b)[^.;!?])+)",
    re.IGNORECASE)

# Medicamentos usados, ditos no check-in: "Medicamentos: paracetamol 500."
_NO_MEDICATIONS = re.compile(
    r"\b(?:sem\s+(?:medicamentos?|medica[cç](?:[aã]o|[oõ]es)|rem[eé]dios?)"
    r"|nenhum\s+(?:medicamento|rem[eé]dio)"
    r"|n[aã]o\s+(?:tomei|usei|utilizei)\s+(?:nenhum\s+)?(?:medicamentos?|rem[eé]dios?))\b",
    re.IGNORECASE)
_MEDICATIONS = re.compile(
    r"\b(?:medicamentos?|medica[cç](?:[aã]o|[oõ]es)|rem[eé]dios?)\b\s*"
    r"(?:usados?|utilizados?|tomados?|foi|foram|s[aã]o|de|com)?\s*(?:[:\-–]\s*)?"
    r"((?:(?!\bsintomas?\b)[^.;!?])+)",
    re.IGNORECASE)


def parse_symptoms(transcript: str) -> tuple[str | None, str]:
    """Retorna (sintomas como foram ditos, texto sem o trecho dos sintomas).

    Sem menção a sintomas -> None. Nada é interpretado nem resumido: o texto
    vai para o relatório exatamente como foi transcrito.
    """
    none_match = _NO_SYMPTOMS.search(transcript)
    if none_match:
        rest = transcript[:none_match.start()] + " " + transcript[none_match.end():]
        return none_match.group(0), rest
    match = _SYMPTOMS.search(transcript)
    text = match.group(1).strip(" ,:-–") if match else ""
    if len(text) >= 2:
        rest = transcript[:match.start()] + " " + transcript[match.end():]
        return text, rest
    return None, transcript


def parse_medications(transcript: str) -> tuple[str | None, str]:
    """Retorna (medicamentos como foram ditos, texto sem esse trecho).

    "Sem medicamentos" ou nenhuma menção -> None (o relatório mostra
    "Sem medicamentos utilizados"). Nada é interpretado: nome e dose saem
    exatamente como foram transcritos.
    """
    none_match = _NO_MEDICATIONS.search(transcript)
    if none_match:
        rest = transcript[:none_match.start()] + " " + transcript[none_match.end():]
        return None, rest
    match = _MEDICATIONS.search(transcript)
    text = match.group(1).strip(" ,:-–") if match else ""
    if len(text) >= 2:
        rest = transcript[:match.start()] + " " + transcript[match.end():]
        return text, rest
    return None, transcript


if __name__ == "__main__":
    exemplos = [
        "Hoje foi um dia tranquilo. Check-in: humor oito, fadiga está em 3, "
        "estresse dois e solidão nota 2.",
        "Check-in diário AST-01. Humor 7, fadiga 4, estresse 3, solidão 2, qualidade do sono "
        "seis, recuperação cinco. Sintomas: dor de cabeça leve. Medicamentos: paracetamol 500.",
        "Humor oito, sono zero, recuperação foi sete. Sem sintomas. Sem medicamentos.",
        "Humor 5. Sintomas de dor nas costas, medicamentos usados: ibuprofeno 400.",
        "Dormi pouco mas estou bem. Humor 6.",
    ]
    for texto in exemplos:
        remedios, resto = parse_medications(texto)
        sintomas, resto = parse_symptoms(resto)
        notas, _ = parse_checkin(resto)
        print(texto)
        print("  notas:", {k: v for k, v in notas.items() if v is not None})
        print("  sintomas:", sintomas)
        print("  medicamentos:", remedios, "\n")