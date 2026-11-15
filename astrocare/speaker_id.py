"""Identifica o astronauta pelo ID que ele FALA no início do reporte.

Exemplo de fala: "Check-in diário, AST-02. Hoje o dia foi..."

Regras:
- O ID precisa vir logo depois de uma ABERTURA ("check-in", "reporte",
  "aqui é", "sou", "astronauta") ou bem no início da fala, e dentro das
  primeiras ID_SEARCH_WORDS palavras. Assim, "conversei com o AST-03"
  não troca a identidade de quem está falando.
- Aceita formas comuns de transcrição: "AST-02", "AST 2", "AST zero dois",
  "A S T dois".
- O ID precisa estar cadastrado em config.ASTRONAUTS.

Se algo falhar, devolve None + o motivo. Quem chama decide o que fazer
(no pipeline: o reporte vai para a fila de revisão, nunca para o padrão
de um astronauta "chutado").
"""
import re
import unicodedata

from astrocare import config

ID_SEARCH_WORDS = 15  # o ID deve ser dito logo no começo

DIGIT_WORDS = {
    "zero": 0, "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4,
    "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9,
}
_DIGIT = r"(?:\d{1,2}|" + "|".join(DIGIT_WORDS) + r")"

# Expressões que anunciam quem está falando (ou o começo do texto)
_OPENING = r"(?:^|\bcheck[\s\-]*in\b|\breporte\b|\baqui\s+e\b|\bsou\b|\bastronauta\b)"
# Até 2 palavras entre a abertura e o ID ("check-in diário, AST-02")
_GAP = r"(?:[\s,.:;\-]+\w+){0,2}?[\s,.:;\-]*"
# "ast", "a s t", "a-s-t", "aste" + número (1 ou 2 partes: "02", "zero dois")
_ID = rf"\b(?:a[\s.\-]?s[\s.\-]?t|aste)\b[\s.\-:]*({_DIGIT})(?:[\s\-]+({_DIGIT}))?"
ID_PATTERN = re.compile(_OPENING + _GAP + _ID)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _to_int(token: str) -> int:
    return DIGIT_WORDS[token] if token in DIGIT_WORDS else int(token)


def parse_spoken_id(transcript: str | None) -> tuple[str | None, str | None]:
    """Retorna (astronauta_id, None) ou (None, motivo da falha)."""
    if not transcript:
        return None, "transcrição indisponível"

    beginning = " ".join(_normalize(transcript).split()[:ID_SEARCH_WORDS])
    match = ID_PATTERN.search(beginning)
    if not match:
        return None, ("ID não encontrado no início da fala "
                      "(diga, por exemplo: 'Check-in diário, AST-02')")

    first, second = match.group(1), match.group(2)
    # Segunda parte só conta depois de "zero" ("zero dois" = 02).
    # Evita ler "AST dois, um dia ótimo" como AST-21.
    if second is not None and _to_int(first) == 0:
        number = _to_int(second)
    else:
        number = _to_int(first)

    astronaut_id = f"AST-{number:02d}"
    if astronaut_id not in config.ASTRONAUTS:
        return None, f"ID falado ({astronaut_id}) não está cadastrado"
    return astronaut_id, None


if __name__ == "__main__":
    for exemplo in [
        "Check-in diário, AST-02. Hoje foi tranquilo.",
        "Check-in diário AST zero três, tudo certo por aqui.",
        "Hoje conversei bastante com o AST-03 sobre os experimentos.",
    ]:
        print(f"{exemplo!r} -> {parse_spoken_id(exemplo)}")
        