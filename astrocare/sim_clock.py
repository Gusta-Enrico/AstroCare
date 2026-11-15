"""Relógio da missão simulada: liga o calendário real aos dias de missão.

O demo_data define o lançamento simulado de forma que o histórico
simulado termine ONTEM. Assim, uma gravação feita hoje cai no dia
seguinte da missão, a de amanhã no próximo, e assim por diante.
Duas gravações no mesmo dia real caem no MESMO dia de missão.

A data do lançamento fica salva em data/processed/simulacao.json, para
não mudar enquanto o cenário não for gerado de novo.
"""
import json
from datetime import date, timedelta

from astrocare import config


def set_launch(launch: date, seed: int) -> None:
    """Guarda o lançamento simulado e a SEMENTE da simulação dos sensores."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.SIMULATION_FILE.write_text(json.dumps(
        {"lancamento_simulado": launch.isoformat(), "semente_simulacao": int(seed),
         "aviso": "dados SIMULADOS"}, indent=2), encoding="utf-8")


def _load() -> dict:
    if not config.SIMULATION_FILE.exists():
        raise SystemExit("Missão simulada não iniciada. Rode antes: python -m astrocare.demo_data")
    return json.loads(config.SIMULATION_FILE.read_text(encoding="utf-8"))


def launch() -> date:
    return date.fromisoformat(_load()["lancamento_simulado"])


def seed() -> int:
    """Semente da simulação atual: um dia novo continua a MESMA história simulada."""
    data = _load()
    if "semente_simulacao" not in data:
        raise SystemExit("Simulação antiga, sem semente. Rode de novo: python -m astrocare.demo_data")
    return int(data["semente_simulacao"])


def mission_day(day_date: date) -> int:
    """Dia da missão de uma data do calendário (0 = lançamento)."""
    return (day_date - launch()).days


def date_of(day: int) -> date:
    """Data do calendário de um dia da missão."""
    return launch() + timedelta(days=day)