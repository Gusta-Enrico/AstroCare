"""Métricas do texto transcrito.

Por que um léxico (lista de palavras) e não a IA para o sentimento?
- É DETERMINÍSTICO: o mesmo texto sempre dá o mesmo número. Se a IA
  mudasse de versão no meio da missão, o "sentimento" mudaria junto e
  pareceria uma alteração do astronauta.
- É AUDITÁVEL: dá para ver exatamente quais palavras contaram.

O léxico abaixo é um PONTO DE PARTIDA para o protótipo. Para uso real,
deve ser substituído por um léxico validado em português (ex.: OpLexicon
ou SentiLex-PT) e revisado por especialistas.
"""
import re
import unicodedata

POSITIVE_WORDS = {
    "bem", "bom", "boa", "otimo", "otima", "excelente", "feliz", "alegre",
    "animado", "animada", "tranquilo", "tranquila", "calmo", "calma",
    "descansado", "descansada", "disposto", "disposta", "motivado", "motivada",
    "satisfeito", "satisfeita", "gostei", "gosto", "adorei", "legal", "incrivel",
    "produtivo", "produtiva", "confiante", "grato", "grata", "positivo",
    "positiva", "sucesso", "consegui", "conseguimos", "melhor", "melhorou",
    "divertido", "divertida", "lindo", "linda", "orgulho", "esperanca",
    "energia", "sorrir", "rindo", "riso", "paz", "conforto", "unidos",
    "apoio", "agradavel", "facil", "funcionou", "progresso", "concluido",
}

NEGATIVE_WORDS = {
    "mal", "ruim", "ruins", "falta", "pessimo", "pessima", "triste", "cansado", "cansada",
    "exausto", "exausta", "esgotado", "esgotada", "sozinho", "sozinha",
    "isolado", "isolada", "preocupado", "preocupada", "nervoso", "nervosa",
    "irritado", "irritada", "frustrado", "frustrada", "dificil", "problema",
    "problemas", "dor", "doi", "medo", "angustia", "saudade", "saudades",
    "chato", "chata", "tedio", "pior", "piorou", "falha", "falhou", "erro",
    "insonia", "desanimado", "desanimada", "vazio", "vazia",
    "tenso", "tensa", "pesado", "pesada", "dificuldade", "incomodo",
    "desconforto", "brigamos", "discussao", "chorar", "chorei", "perdido",
    "perdida", "confuso", "confusa", "lento", "lenta", "fraco", "fraca",
}

# "não estou bem" deve contar como negativo, não positivo
NEGATIONS = {"nao", "nunca", "nem", "jamais", "nada", "ninguem", "sem"}
NEGATION_WINDOW = 3  # palavras após a negação que têm a polaridade invertida

def _polarity(word: str) -> tuple[bool, bool]:
    """(é positiva, é negativa). Tenta também sem o "s" final (plural)."""
    for candidate in (word, word[:-1] if word.endswith("s") else None):
        if candidate and (candidate in POSITIVE_WORDS or candidate in NEGATIVE_WORDS):
            return candidate in POSITIVE_WORDS, candidate in NEGATIVE_WORDS
    return False, False


def _normalize(text: str) -> list[str]:
    """Minúsculas, sem acento, só palavras."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.findall(r"[a-z]+", text)


def extract_text_features(transcript: str, duration_s: float) -> dict:
    """Métricas de texto a partir da transcrição e da duração da fala."""
    words = _normalize(transcript)
    if not words or duration_s <= 0:
        raise ValueError("Transcrição vazia ou duração inválida")

    positive = negative = 0
    negation_left = 0
    for word in words:
        if word in NEGATIONS:
            negation_left = NEGATION_WINDOW
            continue
        is_pos, is_neg = _polarity(word)
        if negation_left:
            is_pos, is_neg = is_neg, is_pos
            negation_left -= 1
        positive += is_pos
        negative += is_neg

    total = len(words)
    polar = positive + negative
    return {
        "velocidade_fala_ppm": round(total / (duration_s / 60), 1),
        "palavras_positivas_pct": round(100 * positive / total, 2),
        "palavras_negativas_pct": round(100 * negative / total, 2),
        # -1 (só negativas) a +1 (só positivas); 0 se não houver nenhuma
        "saldo_sentimento_m1a1": round((positive - negative) / polar, 3) if polar else 0.0,
    }


if __name__ == "__main__":
    exemplo = ("Hoje o dia foi bom, consegui terminar os experimentos. "
               "Não estou bem descansado e senti saudade de casa.")
    print(extract_text_features(exemplo, duration_s=10))