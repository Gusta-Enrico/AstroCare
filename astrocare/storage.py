"""Histórico do AstroCare em um banco SQLite (data/astrocare.db).

Por que SQLite: já vem com o Python, é um arquivo só, não precisa de
servidor, e qualquer ferramenta (Python, Power BI, DBeaver, Excel via
ODBC) consegue ler. Se o projeto crescer, as mesmas tabelas migram para
PostgreSQL sem mudar a lógica.

Tabelas (todas têm a coluna "fonte": planilha, audio ou demo):
- reportes            valor de cada métrica em cada reporte (Terra e missão)
- padrao_pessoal      padrão da Terra por astronauta e métrica
- indicadores         desvios calculados por dia e métrica (missão)
- status_diario       normal / atencao / alerta por astronauta e dia
- relatorios_diarios  texto do relatório curto de cada dia
- relatos             textos ditos pelo astronauta (ex.: sintomas), sem interpretação

Atualização: sync(fonte) recalcula tudo a partir dos dados e REGRAVA as
linhas daquela fonte. Rodar duas vezes dá o mesmo resultado (idempotente).

Uso:
    python -m astrocare.storage --fonte demo
"""
import argparse
import sqlite3

import pandas as pd

from astrocare import config
from astrocare.baseline import compute_baseline
from astrocare.comparison import (
    compare_with_crew, daily_status, flag_metrics, score_individual, smooth,
)
from astrocare.ingestion import load_dataset, load_spoken

SCHEMA = """
CREATE TABLE IF NOT EXISTS reportes (
    fonte TEXT, astronauta_id TEXT, dia_missao INTEGER, data TEXT, fase TEXT,
    metrica TEXT, valor REAL,
    PRIMARY KEY (fonte, astronauta_id, dia_missao, metrica));
CREATE TABLE IF NOT EXISTS padrao_pessoal (
    fonte TEXT, astronauta_id TEXT, metrica TEXT, n INTEGER,
    mediana REAL, escala_robusta REAL, confiavel INTEGER,
    PRIMARY KEY (fonte, astronauta_id, metrica));
CREATE TABLE IF NOT EXISTS indicadores (
    fonte TEXT, astronauta_id TEXT, dia_missao INTEGER, metrica TEXT,
    z_suavizado REAL, z_vs_tripulacao REAL, com_tripulacao INTEGER,
    sinal_atencao INTEGER, sinal_alerta INTEGER,
    PRIMARY KEY (fonte, astronauta_id, dia_missao, metrica));
CREATE TABLE IF NOT EXISTS status_diario (
    fonte TEXT, astronauta_id TEXT, dia_missao INTEGER, status TEXT,
    metricas_atencao INTEGER, metricas_alerta INTEGER,
    PRIMARY KEY (fonte, astronauta_id, dia_missao));
CREATE TABLE IF NOT EXISTS relatos (
    fonte TEXT, astronauta_id TEXT, dia_missao INTEGER, data TEXT, tipo TEXT, texto TEXT,
    PRIMARY KEY (fonte, astronauta_id, dia_missao, tipo));
CREATE TABLE IF NOT EXISTS relatorios_diarios (
    fonte TEXT, astronauta_id TEXT, dia_missao INTEGER, gerado_em TEXT,
    status TEXT, houve_alteracao INTEGER, texto TEXT,
    PRIMARY KEY (fonte, astronauta_id, dia_missao));
"""
# Tabelas recalculadas no sync (os relatórios ficam guardados)
SYNCED_TABLES = ("reportes", "padrao_pessoal", "indicadores", "status_diario", "relatos")


def connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def _insert(conn, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df.to_sql(table, conn, if_exists="append", index=False)
    return len(df)


def sync(source: str) -> dict:
    """Recalcula e regrava no banco tudo o que é da fonte informada."""
    data, _ = load_dataset(source)
    baseline = compute_baseline(data)

    # Indicadores só existem se houver padrão pronto E dados de missão
    scored = score_individual(data, baseline)
    if scored.empty:
        flagged = daily = pd.DataFrame()
    else:
        flagged = flag_metrics(compare_with_crew(smooth(scored)))
        daily = daily_status(flagged, source)

    reportes = data.assign(fonte=source, data=data["data"].astype(str))[
        ["fonte", "astronauta_id", "dia_missao", "data", "fase", "metrica", "valor"]]
    padrao = baseline.assign(fonte=source, confiavel=baseline["confiavel"].astype(int))[
        ["fonte", "astronauta_id", "metrica", "n", "mediana", "escala_robusta", "confiavel"]]

    indicadores = status = pd.DataFrame()
    if not flagged.empty:
        indicadores = flagged.assign(
            fonte=source,
            com_tripulacao=flagged["com_tripulacao"].astype(int),
            sinal_atencao=flagged["sinal_atencao"].astype(int),
            sinal_alerta=flagged["sinal_alerta"].astype(int),
        )[["fonte", "astronauta_id", "dia_missao", "metrica", "z_suavizado",
           "z_vs_tripulacao", "com_tripulacao", "sinal_atencao", "sinal_alerta"]]
        status = daily.rename(columns={"sinal_atencao": "metricas_atencao",
                                       "sinal_alerta": "metricas_alerta"}).assign(fonte=source)[
            ["fonte", "astronauta_id", "dia_missao", "status",
             "metricas_atencao", "metricas_alerta"]]

    relatos = pd.concat([load_spoken(source, tipo).assign(tipo=tipo)
                         for tipo in ("sintomas", "medicamentos")], ignore_index=True)
    relatos = relatos.assign(fonte=source)
    relatos = relatos.assign(data=relatos["data"].astype(str))[
        ["fonte", "astronauta_id", "dia_missao", "data", "tipo", "texto"]]

    with connect() as conn:
        for table in SYNCED_TABLES:
            conn.execute(f"DELETE FROM {table} WHERE fonte = ?", (source,))
        return {
            "relatos": _insert(conn, "relatos", relatos),
            "reportes": _insert(conn, "reportes", reportes),
            "padrao_pessoal": _insert(conn, "padrao_pessoal", padrao),
            "indicadores": _insert(conn, "indicadores", indicadores),
            "status_diario": _insert(conn, "status_diario", status),
        }


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Consulta livre ao banco (para gráficos e relatórios)."""
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def metric_history(source: str, astronauta_id: str, metrica: str) -> pd.DataFrame:
    """Série completa de uma métrica: valor diário, padrão da Terra e status."""
    return query(
        """
        SELECT r.dia_missao, r.fase, r.valor,
               p.mediana, p.escala_robusta,
               i.z_suavizado, s.status
        FROM reportes r
        JOIN padrao_pessoal p USING (fonte, astronauta_id, metrica)
        LEFT JOIN indicadores i USING (fonte, astronauta_id, dia_missao, metrica)
        LEFT JOIN status_diario s USING (fonte, astronauta_id, dia_missao)
        WHERE r.fonte = ? AND r.astronauta_id = ? AND r.metrica = ?
        ORDER BY r.dia_missao
        """,
        (source, astronauta_id, metrica),
    )


def save_daily_report(source, astronauta_id, dia, gerado_em, status, changed, texto) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO relatorios_diarios VALUES (?, ?, ?, ?, ?, ?, ?)",
            (source, astronauta_id, int(dia), gerado_em, status, int(changed), texto),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atualiza o histórico no banco SQLite")
    parser.add_argument("--fonte", choices=config.SOURCES, default="demo")
    args = parser.parse_args()

    counts = sync(args.fonte)
    print(f"Banco atualizado: {config.DB_PATH}  (fonte: {args.fonte})")
    for table, n in counts.items():
        print(f"  {table}: {n} linhas")
    if counts["indicadores"] == 0:
        print("  (sem indicadores: padrão pessoal ainda em construção ou sem dados de missão)")