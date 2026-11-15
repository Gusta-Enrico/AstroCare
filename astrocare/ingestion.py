"""Leitura e validação dos dados do Excel.

Saída principal: um DataFrame no formato LONGO, com uma linha por medição:
    astronauta_id | dia_missao | data | fase | metrica | valor

Esse é o formato que as camadas de comparação vão consumir. Quando o
extrator de áudio existir, ele só precisa produzir esse mesmo formato.
"""
import pandas as pd

from astrocare import config


class DataValidationError(Exception):
    """Erro estrutural: os dados não podem ser usados."""


def load_sheet(sheet_name: str) -> pd.DataFrame:
    """Lê uma aba do Excel."""
    if not config.EXCEL_PATH.exists():
        raise FileNotFoundError(f"Planilha não encontrada: {config.EXCEL_PATH}")
    return pd.read_excel(config.EXCEL_PATH, sheet_name=sheet_name)


def validate(df: pd.DataFrame, sheet_name: str, metrics: list[str]) -> list[str]:
    """Valida uma aba.

    - Problemas ESTRUTURAIS (coluna faltando, chave duplicada) levantam erro,
      porque qualquer análise em cima disso estaria errada.
    - Problemas de VALOR (vazio, fora da faixa plausível) viram avisos e o
      valor é descartado depois, sem derrubar o pipeline.
    """
    required = config.KEY_COLS + metrics
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"[{sheet_name}] colunas ausentes: {missing}")

    dup = df.duplicated(subset=config.KEY_COLS).sum()
    if dup:
        raise DataValidationError(
            f"[{sheet_name}] {dup} linhas duplicadas em {config.KEY_COLS}"
        )

    warnings = []
    for col in metrics:
        n_null = df[col].isna().sum()
        if n_null:
            warnings.append(f"[{sheet_name}] {col}: {n_null} valores vazios")

        if col in config.VALID_RANGES:
            low, high = config.VALID_RANGES[col]
            out = ((df[col] < low) | (df[col] > high)).sum()
            if out:
                warnings.append(
                    f"[{sheet_name}] {col}: {out} valores fora de [{low}, {high}]"
                )
    return warnings


def to_long(df: pd.DataFrame, fase: str, metrics: list[str]) -> pd.DataFrame:
    """Converte de formato largo (uma coluna por métrica) para longo."""
    long_df = df.melt(
        id_vars=config.KEY_COLS + ["data"],
        value_vars=metrics,
        var_name="metrica",
        value_name="valor",
    )
    long_df["fase"] = fase
    return long_df


def remove_implausible(long_df: pd.DataFrame) -> pd.DataFrame:
    """Descarta valores vazios ou fisicamente impossíveis."""
    low = long_df["metrica"].map(lambda m: config.VALID_RANGES.get(m, (None, None))[0])
    high = long_df["metrica"].map(lambda m: config.VALID_RANGES.get(m, (None, None))[1])
    ok = long_df["valor"].notna()
    ok &= low.isna() | (long_df["valor"] >= low)
    ok &= high.isna() | (long_df["valor"] <= high)
    return long_df[ok]


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    """Carrega journal + sono, Terra e missão, já validados e em formato longo.

    Retorna (dados, avisos).
    """
    sources = [
        (config.SHEETS_BEHAVIORAL, config.METRICS_JOURNAL),
        (config.SHEETS_CONTEXT, config.METRICS_SLEEP),
    ]

    frames, all_warnings = [], []
    for sheets, metrics in sources:
        for fase, sheet_name in sheets.items():
            df = load_sheet(sheet_name)
            all_warnings += validate(df, sheet_name, metrics)
            frames.append(to_long(df, fase, metrics))

    data = remove_implausible(pd.concat(frames, ignore_index=True))
    data = data[["astronauta_id", "dia_missao", "data", "fase", "metrica", "valor"]]
    data = data.sort_values(["astronauta_id", "metrica", "dia_missao"])
    return data.reset_index(drop=True), all_warnings


if __name__ == "__main__":
    data, warnings = load_dataset()

    print("Avisos de qualidade:", warnings or "nenhum")
    print(f"\nTotal de medições: {len(data)}")
    print("\nMedições por astronauta e fase:")
    print(data.pivot_table(index="astronauta_id", columns="fase",
                           values="valor", aggfunc="count"))
    print("\nAmostra:")
    print(data.head())