"""Processa um reporte em áudio e gera as métricas no formato da planilha.

Fluxo:
    WAV -> métricas de voz (local) -> transcrição (Gemini)
        -> notas do check-in ditas na fala + métricas de texto
        -> uma linha com as mesmas colunas da aba Journal -> CSV

De onde vem cada dado (nada é digitado):
- astronauta: do nome do arquivo (AST-01_20261114_163358.wav).
              No app, virá do login.
- data e dia: da data da gravação, contada a partir do lançamento.
- check-in:   das notas que o astronauta DIZ no áudio. Nota não dita
              fica vazia; nunca é estimada.

Uso:
    python -m astrocare.audio_pipeline --gravar AST-01      (grava e processa)
    python -m astrocare.audio_pipeline data/audio/AST-01_20261114_163358.wav
"""
import argparse
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from astrocare import config
from astrocare.checkin_parser import parse_checkin
from astrocare.llm import transcribe_audio
from astrocare.text_features import extract_text_features
from astrocare.voice_features import extract_voice_features

# Ordem FIXA das colunas no CSV. Sem isso, cada linha seria gravada na
# ordem em que as métricas foram calculadas e os valores se desalinhariam.
CSV_COLUMNS = (
    ["astronauta_id", "dia_missao", "data", "arquivo", "transcricao"]
    + config.METRICS_VOICE + config.METRICS_TEXT + config.METRICS_SELF_REPORT
)

# Ex.: AST-01_20261114_163358.wav -> ("AST-01", "20261114")
FILENAME_PATTERN = re.compile(r"^(AST-\d+)_(\d{8})_\d{6}$")


def identify(wav_path: str) -> tuple[str, date]:
    """Lê o astronauta e a data da gravação a partir do nome do arquivo."""
    match = FILENAME_PATTERN.match(Path(wav_path).stem)
    if not match:
        raise ValueError(
            f"Nome de arquivo fora do padrão AST-XX_AAAAMMDD_HHMMSS.wav: {wav_path}"
        )
    return match.group(1), datetime.strptime(match.group(2), "%Y%m%d").date()


def mission_day(recording_date: date) -> int:
    """Dia da missão: 0 no lançamento, negativo antes (fase Terra)."""
    return (recording_date - config.LAUNCH_DATE).days


def process_audio(wav_path: str) -> dict:
    """Gera uma linha de métricas a partir de um reporte em áudio."""
    astronauta_id, recording_date = identify(wav_path)
    row = {
        "astronauta_id": astronauta_id,
        "dia_missao": mission_day(recording_date),
        "data": recording_date.isoformat(),
        # Só o NOME do arquivo: é o identificador do reporte. O caminho
        # completo varia ("C:\\..." ou ".\\data\\...") e duplicaria linhas.
        "arquivo": Path(wav_path).name,
    }

    voice = extract_voice_features(wav_path)
    row.update(voice)

    transcript = transcribe_audio(wav_path)
    row["transcricao"] = transcript
    if transcript:
        scores, text_without_checkin = parse_checkin(transcript)
        row.update(scores)
        # Sentimento: sem o check-in ("estresse", "solidão" não são o relato).
        # Velocidade: com o texto todo, pois a duração inclui o check-in falado.
        duration = voice["duracao_video_s"]
        row.update(extract_text_features(text_without_checkin, duration))
        row["velocidade_fala_ppm"] = extract_text_features(transcript, duration)["velocidade_fala_ppm"]
    else:
        row.update({m: None for m in config.METRICS_TEXT + config.METRICS_SELF_REPORT})
    return row


def save_row(row: dict) -> None:
    """Salva a linha no CSV de reportes.

    Se o mesmo arquivo de áudio já foi processado antes (ex.: a transcrição
    falhou e você rodou de novo), a linha antiga é SUBSTITUÍDA, não duplicada.
    """
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row], columns=CSV_COLUMNS)
    if config.AUDIO_METRICS_CSV.exists():
        old = pd.read_csv(config.AUDIO_METRICS_CSV, encoding="utf-8")
        old = old[old["arquivo"] != row["arquivo"]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(config.AUDIO_METRICS_CSV, index=False, encoding="utf-8")


def print_summary(row: dict) -> None:
    fase = "Terra (padrão pessoal)" if row["dia_missao"] < 0 else "missão"
    print(f"\n{row['astronauta_id']} | {row['data']} | dia {row['dia_missao']} | fase: {fase}")
    print(f"Transcrição: {row['transcricao'] or '(indisponível)'}\n")
    for metric in config.METRICS_VOICE + config.METRICS_TEXT + config.METRICS_SELF_REPORT:
        value = row[metric]
        shown = "não informado no áudio" if value is None and metric in config.METRICS_SELF_REPORT else value
        print(f"  {config.METRIC_LABELS.get(metric, metric)}: {shown}")
    print(f"\nSalvo em {config.AUDIO_METRICS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa um reporte em áudio")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("wav", nargs="?", help="arquivo .wav já gravado")
    group.add_argument("--gravar", metavar="AST-XX", help="grava agora e processa")
    args = parser.parse_args()

    if args.gravar:
        from astrocare.recorder import record_until_enter
        wav_path = record_until_enter(args.gravar)
    else:
        wav_path = args.wav

    row = process_audio(wav_path)
    save_row(row)
    print_summary(row)