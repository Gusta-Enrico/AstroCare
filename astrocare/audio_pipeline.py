"""Processa um reporte em áudio como um DIA DE MISSÃO do cenário simulado.

Modo de uso do protótipo: só simulação. Nenhum dado real é guardado como
"fase Terra". O padrão da Terra vem do cenário simulado (demo_data.py).

Fluxo:
    WAV -> métricas de voz (local) -> transcrição (Gemini)
        -> ID do astronauta DITO no início -> notas do check-in -> métricas de texto
        -> inserido no cenário simulado no próximo dia de missão do astronauta
        -> banco atualizado + relatório completo (PDF em português e inglês)

Se não der para identificar o astronauta pela fala, o reporte NÃO é
inserido: um reporte na pessoa errada distorceria a comparação dela.

Dia da missão: sem --dia, é o dia seguinte ao último registro do astronauta.
Com --dia, só é aceito se houver registros suficientes nos dias anteriores
(senão o dia não pode ser avaliado e nada é salvo). Gravar de novo um dia
que já existe substitui o reporte daquele dia.

Uso:
    python -m astrocare.audio_pipeline --gravar              (próximo dia livre)
    python -m astrocare.audio_pipeline --gravar --dia 45     (escolhe o dia, p/ testes)
    python -m astrocare.audio_pipeline data/audio/REPORTE_20261114_163358.wav
"""
import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from astrocare import config, sim_clock
from astrocare.checkin_parser import parse_checkin, parse_medications, parse_symptoms
from astrocare.llm import transcribe_audio
from astrocare.physio_sim import PhysioSimulator
from astrocare.speaker_id import parse_spoken_id
from astrocare.text_features import extract_text_features
from astrocare.voice_features import extract_voice_features

# Ordem FIXA das colunas no CSV. Sem isso, cada linha seria gravada na
# ordem em que as métricas foram calculadas e os valores se desalinhariam.
CSV_COLUMNS = (
    ["astronauta_id", "dia_missao", "data", "arquivo", "transcricao", "sintomas", "medicamentos"]
    + config.METRICS_VOICE + config.METRICS_TEXT + config.METRICS_SELF_REPORT
    + config.METRICS_MONITORING
)


def process_audio(wav_path: str) -> tuple[dict, str | None]:
    """Extrai as métricas do áudio. Retorna (linha, motivo de recusa ou None)."""
    row = {"arquivo": Path(wav_path).name}

    voice = extract_voice_features(wav_path)
    row.update(voice)

    transcript = transcribe_audio(wav_path)
    row["transcricao"] = transcript
    row["astronauta_id"], reason = parse_spoken_id(transcript)

    if transcript:
        # Sintomas: guardados como texto; saem do texto do sentimento
        # Medicamentos e sintomas: guardados como texto; saem do texto do sentimento
        row["medicamentos"], text_without_meds = parse_medications(transcript)
        row["sintomas"], text_without_symptoms = parse_symptoms(text_without_meds)
        scores, text_without_checkin = parse_checkin(text_without_symptoms)
        row.update(scores)
        # Sentimento: sem o check-in ("estresse", "solidão" não são o relato).
        # Velocidade: com o texto todo, pois a duração inclui o check-in falado.
        duration = voice["duracao_video_s"]
        row.update(extract_text_features(text_without_checkin, duration))
        row["velocidade_fala_ppm"] = extract_text_features(transcript, duration)["velocidade_fala_ppm"]
    else:
        row.update({m: None for m in config.METRICS_TEXT + config.METRICS_SELF_REPORT
                    + config.METRICS_SELF_REPORT_SLEEP})
        row["sintomas"] = None
        print(f"  medicamentos utilizados: {row.get('medicamentos') or 'sem medicamentos utilizados'}")
    return row, reason


def _load_scenario() -> pd.DataFrame:
    if not config.DEMO_METRICS_CSV.exists():
        raise SystemExit("Cenário simulado não encontrado. Rode antes: python -m astrocare.demo_data")
    return pd.read_csv(config.DEMO_METRICS_CSV, encoding="utf-8")


def choose_day(astronauta_id: str, requested: int | None) -> int:
    """Dia da missão do reporte.

    Sem --dia: o dia seguinte ao último registro do astronauta (sempre avaliável).
    Com --dia: só aceita se houver registros suficientes nos dias anteriores para a
    média móvel; senão recusa ANTES de salvar, em vez de criar um reporte isolado.
    """
    from astrocare.comparison import MIN_DAYS_IN_WINDOW, ROLLING_DAYS

    scenario = _load_scenario()
    days = set(scenario.loc[(scenario["astronauta_id"] == astronauta_id)
                            & (scenario["dia_missao"] >= 0), "dia_missao"].astype(int))
    next_day = max(days, default=-1) + 1
    if requested is None:
        return next_day
    if requested < 0:
        raise SystemExit("O dia da missão começa em 0 (dias negativos são a fase Terra simulada).")
    recent = {d for d in days if requested - ROLLING_DAYS < d < requested}
    if len(recent) + 1 < MIN_DAYS_IN_WINDOW:
        raise SystemExit(
            f"\n⚠ Dia {requested} não pode ser avaliado: {astronauta_id} tem {len(recent)} registro(s) "
            f"nos {ROLLING_DAYS - 1} dias anteriores (mínimo {MIN_DAYS_IN_WINDOW - 1}).\n"
            f"  Nada foi salvo. Grave sem --dia (vai para o dia {next_day}) ou use --dia {next_day}.")
    return requested


def insert_in_scenario(row: dict, day: int, day_date: date) -> tuple[dict, bool]:
    """Coloca o reporte no cenário simulado como o dia `day` da missão.

    Se já existia um reporte desse astronauta nesse dia, ele é substituído
    (vale o mais recente). Retorna (linha, se substituiu um reporte).
    Rodar o demo_data de novo apaga as inserções (o cenário volta ao original).
    """
    # Sensores do dia: SIMULADOS, continuando a mesma simulação (mesma semente).
    # Sono e recuperação NÃO vêm daqui: são os ditos no check-in (ou ficam vazios).
    sensors = PhysioSimulator(row["astronauta_id"], sim_clock.seed()).day(day, self_report_sleep=False)
    row = {**row, **sensors, "dia_missao": day, "data": day_date.isoformat()}
    scenario = _load_scenario()
    same = (scenario["astronauta_id"] == row["astronauta_id"]) & (scenario["dia_missao"] == day)
    replaced = bool(same.any())
    scenario = pd.concat([scenario[~same], pd.DataFrame([row], columns=CSV_COLUMNS)],
                         ignore_index=True)
    scenario.sort_values(["astronauta_id", "dia_missao"]).to_csv(
        config.DEMO_METRICS_CSV, index=False, encoding="utf-8")
    return row, replaced


def print_metrics(row: dict) -> None:
    print(f"\nTranscrição: {row['transcricao'] or '(indisponível)'}\n")
    spoken = config.METRICS_SELF_REPORT + config.METRICS_SELF_REPORT_SLEEP
    for metric in config.METRICS_VOICE + config.METRICS_TEXT + spoken:
        value = row.get(metric)
        shown = "não informado no áudio" if value is None and metric in spoken else value
        print(f"  {config.METRIC_LABELS.get(metric, metric)}: {shown}")
    print(f"  sintomas físicos relatados: {row.get('sintomas') or 'não informado no áudio'}")


def evaluate_day(astronauta_id: str, day: int) -> None:
    """Atualiza o banco e gera o relatório completo do dia (PDF em português e inglês)."""
    from astrocare import ground_report
    from astrocare.comparison import MIN_DAYS_IN_WINDOW, ROLLING_DAYS, NotEnoughDataError
    from astrocare.storage import sync

    # 1. Banco atualizado (histórico usado pelo relatório)
    sync("demo")

    # 2. Relatório completo (PDF + JSON), em português e inglês
    try:
        files, status = ground_report.generate_all(astronauta_id, "demo", dia=day)
    except (ValueError, NotEnoughDataError):
        # A média móvel precisa de registros suficientes nos últimos dias.
        # Um reporte isolado fica salvo, mas ainda não dá para avaliá-lo:
        # o sistema avisa em vez de chutar.
        print(f"\n⚠ Relatório do dia {day} NÃO gerado: a média móvel precisa de pelo menos "
              f"{MIN_DAYS_IN_WINDOW} registros de {astronauta_id} nos últimos {ROLLING_DAYS} dias."
              f"\n  O reporte foi salvo. Grave sem --dia para usar o próximo dia livre.")
        raise SystemExit(1)

    print(f"\nSituação de {astronauta_id} no dia {day}: {ground_report.TXT['pt']['status'][status]}")
    print("\nRelatórios gerados:")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reporte em áudio como dia de missão (simulação)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("wav", nargs="?", help="arquivo .wav já gravado")
    group.add_argument("--gravar", action="store_true", help="grava agora e processa")
    parser.add_argument("--dia", type=int, metavar="N",
                        help="dia da missão (padrão: o dia seguinte ao último registro)")
    args = parser.parse_args()

    if args.gravar:
        from astrocare.recorder import record_until_enter
        wav_path = record_until_enter()
    else:
        wav_path = args.wav

    row, reason = process_audio(wav_path)
    print_metrics(row)

    if reason:
        print(f"\n⚠ REPORTE NÃO INSERIDO: {reason}")
        print("  Grave de novo começando com 'Check-in diário, AST-XX'.")
        raise SystemExit(1)

    ast = row["astronauta_id"]
    day = choose_day(ast, args.dia)
    day_date = sim_clock.date_of(day)

    row, replaced = insert_in_scenario(row, day, day_date)
    print(f"\n{ast} | {day_date:%d/%m/%Y} = dia {day} da missão | inserido no cenário SIMULADO")
    if replaced:
        print("  (já havia um reporte deste astronauta neste dia: foi substituído pelo atual)")

    evaluate_day(ast, day)