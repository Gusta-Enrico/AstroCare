"""Comparação dos dados da missão com o padrão pessoal e com a tripulação.

Três camadas, cada uma resolvendo um problema:

1. z_individual  -> "quanto esta medição se afasta do padrão DESTA pessoa?"
2. z_suavizado   -> média móvel de 7 dias: um dia ruim isolado não vira alerta.
3. z_vs_tripulacao -> desconta o que aconteceu com TODOS no mesmo dia
                      (ex.: falha de CO2, queda de moral coletiva).
                      Sobra só o que é da pessoa.

Uma métrica é sinalizada quando as camadas 2 e 3 passam do limiar.
Um sinal diário exige várias métricas sinalizadas por vários dias seguidos.
Há dois níveis: "atencao" (mais sensível, avisa antes) e "alerta".
"""
import argparse

import pandas as pd

from astrocare import config
from astrocare.baseline import MIN_RECORDS, compute_baseline
from astrocare.ingestion import load_dataset

# Limiares em valores estatísticos convencionais, sem ajuste fino ao
# simulador. Devem ser revalidados quando houver dados reais.
LEVELS = {
    "atencao": 1.5,      # 1,5 desvio: aviso precoce
    "alerta": 2.0,       # 2 desvios: desvio consistente
}
ROLLING_DAYS = 7         # janela da média móvel
MIN_DAYS_IN_WINDOW = 4   # mínimo de registros na janela (o journal tem lacunas)
MIN_METRICS_FLAGGED = 3  # quantas métricas desviadas ao mesmo tempo
MIN_CONSECUTIVE_DAYS = 5 # persistência necessária para virar alerta
MIN_CREW = 3             # mínimo de astronautas para a camada 3 fazer sentido


class NotEnoughDataError(Exception):
    """Ainda não há dados suficientes para comparar."""


def score_individual(data: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Camada 1: z robusto de cada medição da missão vs. o padrão pessoal."""
    missao = data[data["fase"] == "missao"]
    ref = baseline[baseline["confiavel"]][
        ["astronauta_id", "metrica", "mediana", "escala_robusta"]
    ]
    scored = missao.merge(ref, on=["astronauta_id", "metrica"], how="inner")
    scored["z_individual"] = (scored["valor"] - scored["mediana"]) / scored["escala_robusta"]
    return scored


def smooth(scored: pd.DataFrame) -> pd.DataFrame:
    """Camada 2: média móvel de 7 dias, respeitando dias sem registro."""
    days = range(scored["dia_missao"].min(), scored["dia_missao"].max() + 1)
    frames = []
    for (ast, met), group in scored.groupby(["astronauta_id", "metrica"]):
        series = group.set_index("dia_missao")["z_individual"].reindex(days)
        rolled = series.rolling(ROLLING_DAYS, min_periods=MIN_DAYS_IN_WINDOW).mean()
        frames.append(pd.DataFrame({
            "astronauta_id": ast, "metrica": met,
            "dia_missao": list(days), "z_suavizado": rolled.values,
        }))
    return pd.concat(frames, ignore_index=True)


def compare_with_crew(smoothed: pd.DataFrame) -> pd.DataFrame:
    """Camada 3: desconta a mediana dos OUTROS tripulantes no mesmo dia.

    Usa só os outros (e não a tripulação inteira) para que a própria
    pessoa não "puxe" a referência na direção dela.

    Com menos de MIN_CREW astronautas, a camada fica DESLIGADA: a mediana
    de 1 ou 2 pessoas não representa "a tripulação". Nesse caso só vale a
    comparação individual, e a coluna com_tripulacao = False avisa isso.
    """
    wide = smoothed.pivot_table(
        index=["dia_missao", "metrica"], columns="astronauta_id", values="z_suavizado"
    )
    crew_on = len(wide.columns) >= MIN_CREW
    result = []
    for ast in wide.columns:
        others = wide.drop(columns=ast).median(axis=1) if crew_on else float("nan")
        result.append(pd.DataFrame({
            "astronauta_id": ast,
            "z_suavizado": wide[ast],
            "z_tripulacao": others,
            # Sem tripulação, o desvio "vs. tripulação" é o próprio desvio individual
            "z_vs_tripulacao": wide[ast] - others if crew_on else wide[ast],
            "com_tripulacao": crew_on,
        }).reset_index())
    return pd.concat(result, ignore_index=True).dropna(subset=["z_suavizado"])


def flag_metrics(compared: pd.DataFrame) -> pd.DataFrame:
    """Sinaliza a métrica, em cada nível, quando ela desvia de si mesma
    E da tripulação, na mesma direção."""
    same_direction = (compared["z_suavizado"] * compared["z_vs_tripulacao"]) > 0
    for level, threshold in LEVELS.items():
        compared[f"sinal_{level}"] = (
            (compared["z_suavizado"].abs() >= threshold)
            & (compared["z_vs_tripulacao"].abs() >= threshold)
            & same_direction
        )
    return compared


def _streak(s: pd.Series) -> pd.Series:
    """Conta dias True consecutivos; zera quando a sequência quebra."""
    blocks = (~s).cumsum()
    return s.groupby(blocks).cumsum()


def daily_status(flagged: pd.DataFrame) -> pd.DataFrame:
    """Resumo por astronauta e dia, com a regra de persistência.

    status: "normal", "atencao" ou "alerta" (o mais grave vale).
    """
    daily = (
        flagged.groupby(["astronauta_id", "dia_missao"])
        [[f"sinal_{level}" for level in LEVELS]].sum()
        .reset_index()
    )
    daily["status"] = "normal"
    for level in LEVELS:  # ordem: atencao, depois alerta (sobrescreve)
        atypical = daily[f"sinal_{level}"] >= MIN_METRICS_FLAGGED
        days_in_row = atypical.groupby(daily["astronauta_id"]).transform(_streak)
        daily.loc[days_in_row >= MIN_CONSECUTIVE_DAYS, "status"] = level
    return daily


def run_comparison(source: str = "planilha") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa as três camadas. Retorna (detalhe por métrica, resumo diário)."""
    data, _ = load_dataset(source)
    baseline = compute_baseline(data)
    if not baseline["confiavel"].any():
        raise NotEnoughDataError(
            "Nenhum astronauta tem padrão pessoal pronto ainda "
            f"(mínimo de {MIN_RECORDS} dias de reporte na Terra)."
        )
    scored = score_individual(data, baseline)
    if scored.empty:
        raise NotEnoughDataError("Ainda não há reportes da fase de missão para comparar.")
    flagged = flag_metrics(compare_with_crew(smooth(scored)))
    return flagged, daily_status(flagged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonte", choices=config.SOURCES, default="planilha")
    try:
        flagged, daily = run_comparison(parser.parse_args().fonte)
    except NotEnoughDataError as exc:
        raise SystemExit(f"Ainda não é possível comparar: {exc}")
    if not flagged["com_tripulacao"].all():
        print("AVISO: menos de 3 astronautas nesta fonte; comparação com a "
              "tripulação desligada (só comparação individual).\n")

    print("Primeiro dia de cada nível, por astronauta:")
    for ast, group in daily.groupby("astronauta_id"):
        atencao = group.loc[group["status"] != "normal", "dia_missao"]  # inclui alerta
        alerta = group.loc[group["status"] == "alerta", "dia_missao"]
        print(f"  {ast}  atenção: {atencao.min() if len(atencao) else '-'}"
              f" | alerta: {alerta.min() if len(alerta) else '-'}")

    print("\nMétricas que mais contribuíram para os alertas:")
    alert_keys = daily.loc[daily["status"] == "alerta", ["astronauta_id", "dia_missao"]]
    contrib = flagged.merge(alert_keys)
    print(contrib[contrib["sinal_alerta"]]
          .groupby(["astronauta_id", "metrica"]).size()
          .sort_values(ascending=False).head(8).to_string())