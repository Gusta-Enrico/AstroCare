"""Processa um reporte em áudio como um DIA DE MISSÃO do cenário simulado.

Modo de uso do protótipo: só simulação. Nenhum dado real é guardado como
"fase Terra". O padrão da Terra vem do cenário simulado (demo_data.py),
e cada gravação entra como o dia de missão correspondente à data dela.

Fluxo:
    WAV -> métricas de voz (local) -> transcrição (Gemini)
        -> ID do astronauta DITO no início -> notas do check-in -> métricas de texto
        -> inserido no cenário simulado no dia de missão da data da gravação
        -> relatório do dia (comparação com o padrão da Terra) na hora

Se não der para identificar o astronauta pela fala, o reporte NÃO é
inserido: um reporte na pessoa errada distorceria a comparação dela.

O dia da missão vem da DATA REAL da gravação (ver sim_clock.py). Duas
gravações no mesmo dia caem no mesmo dia de missão: vale a mais recente.

Uso:
    python -m astrocare.audio_pipeline --gravar              (dia de hoje)
    python -m astrocare.audio_pipeline --gravar --dia 55     (força um dia, p/ testes)
    python -m astrocare.audio_pipeline data/audio/REPORTE_20261114_163358.wav
"""
import argparse
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from astrocare import config, sim_clock
from astrocare.checkin_parser import parse_checkin, parse_symptoms
from astrocare.llm import transcribe_audio
from astrocare.physio_sim import PhysioSimulator
from astrocare.speaker_id import parse_spoken_id
from astrocare.text_features import extract_text_features
from astrocare.voice_features import extract_voice_features

# Ordem FIXA das colunas no CSV. Sem isso, cada linha seria gravada na
# ordem em que as métricas foram calculadas e os valores se desalinhariam.
CSV_COLUMNS = (
    ["astronauta_id", "dia_missao", "data", "arquivo", "transcricao", "sintomas"]
    + config.METRICS_VOICE + config.METRICS_TEXT + config.METRICS_SELF_REPORT
    + config.METRICS_MONITORING
)

# REPORTE_20261114_163358.wav -> 14/11/2026
FILENAME_DATE = re.compile(r"_(\d{8})_\d{6}$")


def recording_date(wav_path: str) -> date:
    """Data em que o áudio foi gravado: pelo nome do arquivo ou, se não
    houver data no nome, pela data de modificação do arquivo."""
    match = FILENAME_DATE.search(Path(wav_path).stem)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    return date.fromtimestamp(Path(wav_path).stat().st_mtime)


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
        row["sintomas"], text_without_symptoms = parse_symptoms(transcript)
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
    return row, reason


def _load_scenario() -> pd.DataFrame:
    if not config.DEMO_METRICS_CSV.exists():
        raise SystemExit("Cenário simulado não encontrado. Rode antes: python -m astrocare.demo_data")
    return pd.read_csv(config.DEMO_METRICS_CSV, encoding="utf-8")


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
    """Atualiza o banco e gera TODOS os relatórios e o gráfico do dia."""
    from astrocare import ground_report, tracking_chart
    from astrocare.daily_report import build, save_text_file
    from astrocare.storage import save_daily_report, sync

    # 1. Banco atualizado (histórico para gráficos e consultas)
    sync("demo")

    # 2. Relatório curto do dia: banco + arquivo .txt
    text, status, changed, day = build(astronauta_id, day, "demo")
    save_daily_report("demo", astronauta_id, day,
                      datetime.now().isoformat(timespec="seconds"), status, changed, text)
    files = [save_text_file(astronauta_id, day, "demo", text)]
    print("\n" + "=" * 70 + "\n" + text + "\n" + "=" * 70)

    # 3. Relatório completo para a base em Terra (HTML + JSON)
    html_path, json_path, _ = ground_report.generate(astronauta_id, "demo", dia=day)
    files += [html_path, json_path]

    # 4. Gráfico de acompanhamento
    metrics = tracking_chart._most_deviated("demo", astronauta_id)
    files.append(Path(tracking_chart.draw("demo", astronauta_id, metrics)))

    print("\nArquivos gerados:")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reporte em áudio como dia de missão (simulação)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("wav", nargs="?", help="arquivo .wav já gravado")
    group.add_argument("--gravar", action="store_true", help="grava agora e processa")
    parser.add_argument("--dia", type=int, metavar="N",
                        help="força o dia da missão (padrão: pela data da gravação)")
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
    if args.dia is not None:
        day, day_date = args.dia, sim_clock.date_of(args.dia)
    else:
        day_date = recording_date(wav_path)
        day = sim_clock.mission_day(day_date)
        if day < 0:
            raise SystemExit(f"Gravação de {day_date:%d/%m/%Y} é anterior ao lançamento simulado "
                             f"({sim_clock.launch():%d/%m/%Y}). Use --dia N para escolher o dia.")

    row, replaced = insert_in_scenario(row, day, day_date)
    print(f"\n{ast} | {day_date:%d/%m/%Y} = dia {day} da missão | inserido no cenário SIMULADO")
    if replaced:
        print("  (já havia um reporte deste astronauta neste dia: foi substituído pelo atual)")

    evaluate_day(ast, day)