"""Frases de treino da IA que reconhece a emergência (PT e EN).

Arquivo: astrocare/emergency/training/frases_emergencia.csv
Colunas: texto, idioma (pt|en), rotulo (id de um protocolo ou FORA_DO_ESCOPO)

FORA_DO_ESCOPO é a classe de segurança: tudo o que NÃO é um dos protocolos
(corte, queimadura, enjoo, problema da nave, conversa...). Sem ela, a IA
seria obrigada a encaixar qualquer fala em PCR, SANS ou PSIQ.

Os rótulos válidos vêm dos protocolos carregados: se um protocolo novo for
adicionado, as frases dele passam a ser aceitas sem mudar este código.

Uso:
    python -m astrocare.emergency.training_data validar
"""
import argparse
import csv
import unicodedata
from collections import Counter
from pathlib import Path

from astrocare import config
from astrocare.emergency.protocols import load_all

TRAINING_FILE = Path(__file__).parent / "training" / "frases_emergencia.csv"
OUT_OF_SCOPE = "FORA_DO_ESCOPO"
MIN_PER_GROUP = 20          # mínimo de frases por (rótulo, idioma)


class TrainingDataError(Exception):
    """Arquivo de treino inválido: o modelo não deve ser treinado com ele."""


def normalize(text: str) -> str:
    """Minúsculas, sem acentos e espaços simples.

    A MESMA função é usada no treino e na hora de classificar a fala, para que
    "Não", "nao" e "NÃO" virem o mesmo texto (a transcrição nem sempre acentua).
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.split())


def load(path: Path = TRAINING_FILE) -> list[dict]:
    """Lê e valida o CSV. Qualquer problema impede o treino (TrainingDataError)."""
    if not path.exists():
        raise TrainingDataError(f"Arquivo não encontrado: {path}")
    protocols, _ = load_all()
    valid_labels = set(protocols) | {OUT_OF_SCOPE}

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["texto", "idioma", "rotulo"]:
            raise TrainingDataError(f"Cabeçalho deve ser texto,idioma,rotulo (veio {reader.fieldnames})")
        rows, problems, seen = [], [], {}
        for line, row in enumerate(reader, start=2):        # linha 1 = cabeçalho
            texto = (row["texto"] or "").strip()
            idioma, rotulo = (row["idioma"] or "").strip(), (row["rotulo"] or "").strip()
            if not texto:
                problems.append(f"linha {line}: texto vazio")
            if idioma not in config.LANGUAGES:
                problems.append(f"linha {line}: idioma '{idioma}' inválido")
            if rotulo not in valid_labels:
                problems.append(f"linha {line}: rótulo '{rotulo}' não é protocolo carregado nem {OUT_OF_SCOPE}")
            key = normalize(texto)
            if key in seen:
                problems.append(f"linha {line}: frase repetida (igual à linha {seen[key]})")
            seen[key] = line
            rows.append({"texto": texto, "idioma": idioma, "rotulo": rotulo})

    counts = Counter((r["rotulo"], r["idioma"]) for r in rows)
    for label in sorted(valid_labels):
        for lang in config.LANGUAGES:
            if counts[(label, lang)] < MIN_PER_GROUP:
                problems.append(f"{label}/{lang}: só {counts[(label, lang)]} frases "
                                f"(mínimo {MIN_PER_GROUP})")
    if problems:
        raise TrainingDataError("\n".join(problems))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frases de treino da IA de emergência")
    parser.add_argument("acao", choices=["validar"])
    parser.parse_args()
    try:
        data = load()
    except TrainingDataError as exc:
        raise SystemExit(f"Arquivo de treino com problemas:\n{exc}")
    counts = Counter((r["rotulo"], r["idioma"]) for r in data)
    print(f"{len(data)} frases válidas em {TRAINING_FILE.name}")
    for label in sorted({r["rotulo"] for r in data}):
        print(f"  {label:<15} " + "  ".join(f"{lang}: {counts[(label, lang)]:>3}"
                                             for lang in config.LANGUAGES))