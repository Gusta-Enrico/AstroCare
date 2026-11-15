"""Gera um cenário SIMULADO de reportes em áudio, para apresentação.

ATENÇÃO: todos os dados aqui são INVENTADOS. Eles não representam
nenhuma pessoa e não devem ser apresentados como medições reais.
O arquivo gerado tem "SIMULADO" no nome e fica em uma fonte separada
("demo"), que nunca se mistura com os reportes reais ("audio").

O cenário demonstra as três capacidades do detector:
1. Dia ruim ISOLADO        -> NÃO deve gerar alerta (média móvel + persistência).
2. Evento COLETIVO         -> NÃO deve gerar alerta (comparação com a tripulação).
3. Mudança INDIVIDUAL e persistente -> DEVE gerar alerta.

O perfil de voz do AST-01 foi calibrado por uma gravação real do
desenvolvedor (tom ~104 Hz, ~110 palavras/min). As variações de um dia
para o outro são SUPOSIÇÕES razoáveis, não medidas.

Monitoramento ampliado: cada reporte simulado também traz as métricas de
sensores (PVT, wearable, nutrição, exercício, dosímetro) e o autorrelato de
sono e recuperação, geradas pelo physio_sim. A piora do cenário (declínio,
evento coletivo) também piora o sono, para os dados contarem a mesma história.

Cada execução usa uma SEMENTE NOVA (simulação diferente da anterior), que fica
registrada em simulacao.json. Para repetir uma simulação: --seed <número>.

Uso:
    python -m astrocare.demo_data
    python -m astrocare.demo_data --seed 12345
"""
import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from astrocare import config, sim_clock
from astrocare.audio_pipeline import CSV_COLUMNS
from astrocare.physio_sim import PhysioSimulator, new_seed

EARTH_DAYS = range(-30, 0)    # 30 dias na Terra (padrão pessoal)
MISSION_DAYS = range(0, 51)   # 51 dias de missão
EARTH_ADHERENCE = 1.0         # Terra: registro diário (mínimo de 30 para o padrão pessoal)
MISSION_ADHERENCE = 0.90
SLEEP_STRESS = 1.0            # quanto a piora do cenário piora o sono (fator latente)

# Perfil de cada astronauta: (média, variação típica de um dia para o outro)
PROFILES = {
    "AST-01": {  # calibrado pela gravação real do desenvolvedor
        "duracao_video_s": (35, 8), "pausas_pct_tempo": (30, 3),
        "tom_medio_Hz": (104, 3), "variacao_tom_semitons": (2.5, 0.3),
        "volume_voz_dB": (43, 1.5), "velocidade_fala_ppm": (112, 6),
        "saldo_sentimento_m1a1": (0.15, 0.15), "palavras_positivas_pct": (3.5, 0.8),
        "palavras_negativas_pct": (2.5, 0.6),
        "humor_1a10": (8, 0.6), "fadiga_1a10": (3, 0.6),
        "estresse_1a10": (2, 0.6), "solidao_1a10": (2, 0.6),
    },
    "AST-02": {
        "duracao_video_s": (45, 9), "pausas_pct_tempo": (22, 3),
        "tom_medio_Hz": (205, 5), "variacao_tom_semitons": (3.4, 0.35),
        "volume_voz_dB": (48, 1.5), "velocidade_fala_ppm": (138, 7),
        "saldo_sentimento_m1a1": (0.30, 0.15), "palavras_positivas_pct": (5.0, 0.9),
        "palavras_negativas_pct": (2.0, 0.6),
        "humor_1a10": (8, 0.6), "fadiga_1a10": (3, 0.6),
        "estresse_1a10": (3, 0.6), "solidao_1a10": (2, 0.6),
    },
    "AST-03": {
        "duracao_video_s": (30, 7), "pausas_pct_tempo": (26, 3),
        "tom_medio_Hz": (128, 4), "variacao_tom_semitons": (2.9, 0.3),
        "volume_voz_dB": (46, 1.5), "velocidade_fala_ppm": (125, 6),
        "saldo_sentimento_m1a1": (0.20, 0.15), "palavras_positivas_pct": (4.0, 0.8),
        "palavras_negativas_pct": (2.2, 0.6),
        "humor_1a10": (7, 0.6), "fadiga_1a10": (3, 0.6),
        "estresse_1a10": (3, 0.6), "solidao_1a10": (3, 0.6),
    },
    "AST-04": {
        "duracao_video_s": (40, 8), "pausas_pct_tempo": (24, 3),
        "tom_medio_Hz": (182, 5), "variacao_tom_semitons": (3.1, 0.3),
        "volume_voz_dB": (45, 1.5), "velocidade_fala_ppm": (130, 6),
        "saldo_sentimento_m1a1": (0.25, 0.15), "palavras_positivas_pct": (4.5, 0.8),
        "palavras_negativas_pct": (2.0, 0.6),
        "humor_1a10": (8, 0.6), "fadiga_1a10": (2, 0.6),
        "estresse_1a10": (2, 0.6), "solidao_1a10": (2, 0.6),
    },
}

# Como um "declínio comportamental" aparece nas métricas, em intensidade máxima.
# Métricas contínuas: em múltiplos da variação diária da pessoa.
# Autorrelato: em pontos na escala de 1 a 10.
DECLINE_EFFECT = {
    "pausas_pct_tempo": +2.5, "variacao_tom_semitons": -2.5,
    "velocidade_fala_ppm": -2.5, "saldo_sentimento_m1a1": -2.5,
    "palavras_positivas_pct": -2.0, "palavras_negativas_pct": +2.5,
    "humor_1a10": -3, "fadiga_1a10": +3, "estresse_1a10": +2, "solidao_1a10": +3,
}
SELF_REPORT = set(config.METRICS_SELF_REPORT)

# --- Roteiro do cenário (este é o GABARITO, usado só para conferir) -------
SCENARIO = {
    "dia_ruim_isolado": {"astronauta": "AST-02", "dia": 5, "intensidade": 1.2},
    "evento_coletivo": {"dias": range(12, 18), "intensidade": 0.6,
                        "descricao": "falha no sistema de suporte de vida (todos)"},
    "declinio_individual": {"astronauta": "AST-03", "inicio": 25, "dias_ate_maximo": 10},
}


def _intensity(ast: str, dia: int) -> float:
    """Quanto do efeito de declínio se aplica a este astronauta neste dia (0 a 1+)."""
    k = 0.0
    bad = SCENARIO["dia_ruim_isolado"]
    if ast == bad["astronauta"] and dia == bad["dia"]:
        k += bad["intensidade"]
    event = SCENARIO["evento_coletivo"]
    if dia in event["dias"]:
        k += event["intensidade"]
    decline = SCENARIO["declinio_individual"]
    if ast == decline["astronauta"] and dia >= decline["inicio"]:
        k += min(1.0, (dia - decline["inicio"]) / decline["dias_ate_maximo"])
    return k


def _report(rng, sim: PhysioSimulator, ast: str, dia: int) -> dict:
    k = _intensity(ast, dia)
    row = {
        "astronauta_id": ast,
        "dia_missao": dia,
        "data": sim_clock.date_of(dia).isoformat(),
        "arquivo": f"SIMULADO_{ast}_dia{dia:+04d}.wav",
        "transcricao": "(reporte simulado)",
    }
    for metric, (mean, sd) in PROFILES[ast].items():
        shift = DECLINE_EFFECT.get(metric, 0) * k
        if metric in SELF_REPORT:
            value = float(np.clip(round(rng.normal(mean + shift, sd)), 1, 10))
        else:
            value = rng.normal(mean + shift * sd, sd)
        row[metric] = value

    row["saldo_sentimento_m1a1"] = float(np.clip(row["saldo_sentimento_m1a1"], -1, 1))
    for metric in ("palavras_positivas_pct", "palavras_negativas_pct", "duracao_video_s"):
        row[metric] = max(0.0, row[metric])
    row = {m: (round(v, 2) if isinstance(v, float) else v) for m, v in row.items()}
    # Sensores + autorrelato de sono (reportes simulados: Terra e missão)
    row.update(sim.day(dia, stress=SLEEP_STRESS * k, self_report_sleep=True))
    return row


def generate(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for ast in PROFILES:
        sim = PhysioSimulator(ast, seed)
        for days, adherence in ((EARTH_DAYS, EARTH_ADHERENCE), (MISSION_DAYS, MISSION_ADHERENCE)):
            for dia in days:
                # Os dias do roteiro sempre têm reporte, para a demonstração não falhar
                scripted = dia == SCENARIO["dia_ruim_isolado"]["dia"]
                if scripted or rng.random() < adherence:
                    rows.append(_report(rng, sim, ast, dia))
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera o cenário SIMULADO")
    parser.add_argument("--seed", type=int, default=None, help="repete uma simulação anterior")
    seed = parser.parse_args().seed or new_seed()

    # Lançamento simulado: o histórico termina ONTEM, e a gravação de hoje
    # vira o dia seguinte da missão.
    launch = date.today() - timedelta(days=len(MISSION_DAYS))
    sim_clock.set_launch(launch, seed)
    df = generate(seed)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DEMO_METRICS_CSV, index=False, encoding="utf-8")

    print(f"Cenário SIMULADO salvo em {config.DEMO_METRICS_CSV} (semente {seed})")
    terra = df[df["dia_missao"] < 0].groupby("astronauta_id").size()
    missao = df[df["dia_missao"] >= 0].groupby("astronauta_id").size()
    for ast in PROFILES:
        print(f"  {ast}: {terra[ast]} reportes na Terra, {missao[ast]} na missão")

    bad, event, decline = (SCENARIO[k] for k in
                           ("dia_ruim_isolado", "evento_coletivo", "declinio_individual"))
    print("\nRoteiro (gabarito, para conferir o detector):")
    print(f"  1. Dia {bad['dia']}: {bad['astronauta']} tem um dia ruim isolado -> NÃO deve alertar")
    print(f"  2. Dias {event['dias'].start}-{event['dias'].stop - 1}: {event['descricao']} "
          "-> NÃO deve alertar ninguém")
    print(f"  3. A partir do dia {decline['inicio']}: {decline['astronauta']} tem declínio "
          "gradual -> DEVE alertar")
    print(f"Lançamento simulado: {launch:%d/%m/%Y}  ->  hoje ({date.today():%d/%m/%Y}) "
          f"é o dia {sim_clock.mission_day(date.today())} da missão")