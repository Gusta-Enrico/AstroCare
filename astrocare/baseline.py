"""Padrão pessoal (baseline) de cada astronauta, calculado com dados da Terra.

Por que mediana e MAD, e não média e desvio-padrão?
- A média e o desvio-padrão são puxados por dias atípicos (uma noite
  ruim, um reporte gravado doente). A mediana e o MAD quase não se mexem.
- O MAD é multiplicado por 1,4826 para ficar na mesma escala de um
  desvio-padrão. Assim, "z = 2" continua significando "2 desvios".
"""
import argparse

import pandas as pd

from astrocare import config
from astrocare.ingestion import load_dataset

MAD_TO_STD = 1.4826
MIN_RECORDS = 30  # abaixo disso, o padrão não é estatisticamente confiável


def _mad(values: pd.Series) -> float:
    return (values - values.median()).abs().median() * MAD_TO_STD


def compute_baseline(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula o padrão pessoal por astronauta e métrica.

    Usa SOMENTE a fase 'terra'. Os dados da missão nunca entram aqui,
    senão a mudança que queremos detectar contaminaria a referência.
    """
    terra = data[data["fase"] == "terra"]

    baseline = (
        terra.groupby(["astronauta_id", "metrica"])["valor"]
        .agg(n="count", mediana="median", escala_robusta=_mad,
             media="mean", desvio_padrao="std")
        .reset_index()
    )

    # Escala zero acontece em métricas discretas pouco variáveis
    # (ex.: despertares quase sempre = 1). Usamos o desvio-padrão como
    # reserva para não dividir por zero.
    zero = baseline["escala_robusta"] == 0
    baseline.loc[zero, "escala_robusta"] = baseline.loc[zero, "desvio_padrao"]

    baseline["confiavel"] = (baseline["n"] >= MIN_RECORDS) & (baseline["escala_robusta"] > 0)
    return baseline


def baseline_progress(data: pd.DataFrame) -> pd.DataFrame:
    """Quantos dias de reporte na Terra cada astronauta já tem.

    O padrão só é usado quando atinge MIN_RECORDS dias; antes disso,
    está "em construção" e nenhuma comparação é feita.
    """
    terra = data[data["fase"] == "terra"]
    progress = (
        terra.groupby("astronauta_id")["dia_missao"].nunique()
        .rename("dias_na_terra").reset_index()
    )
    progress["minimo"] = MIN_RECORDS
    progress["pronto"] = progress["dias_na_terra"] >= MIN_RECORDS
    return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonte", choices=config.SOURCES, default="planilha")
    data, _ = load_dataset(parser.parse_args().fonte)

    print("Progresso do padrão pessoal (dias de reporte na Terra):")
    for _, r in baseline_progress(data).iterrows():
        status = "pronto" if r["pronto"] else "em construção"
        print(f"  {r['astronauta_id']}: {r['dias_na_terra']}/{r['minimo']}  ({status})")
    print()

    baseline = compute_baseline(data)

    print(f"Padrões calculados: {len(baseline)} "
          f"({baseline['astronauta_id'].nunique()} astronautas x "
          f"{baseline['metrica'].nunique()} métricas)")
    print(f"Não confiáveis: {(~baseline['confiavel']).sum()}")

    print("\nExemplo — tom de voz de cada astronauta:")
    print(baseline[baseline["metrica"] == "tom_medio_Hz"]
          [["astronauta_id", "n", "mediana", "escala_robusta"]].round(1)
          .to_string(index=False))