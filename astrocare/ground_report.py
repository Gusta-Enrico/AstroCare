"""Relatório comparativo Terra x missão, para envio à base em Terra (PDF).

Gera, para um astronauta, dois arquivos em data/reports/:
- .pdf : relatório para a equipe de saúde ler.
- .json: os mesmos indicadores em formato estruturado, para o sistema da base.

O que o relatório contém:
1. Situação atual e há quantos dias está fora do padrão.
2. Linha do tempo do status (normal / atenção / alerta) na missão.
3. Gráficos e tabela das métricas COMPORTAMENTAIS (as que entram no alerta).
4. Monitoramento fisiológico e operacional (NÃO entra no alerta): sono e
   recuperação, PVT, cardiovascular (wearable), nutrição e hidratação,
   exercício e dosímetro (dose em valor absoluto: em órbita é estruturalmente
   maior que na Terra).
5. Sintomas relatados no check-in (texto literal, sem interpretação).
6. Medicamentos utilizados, relatados no check-in (texto literal).
7. Qualidade dos dados, método e aviso de que não é diagnóstico.

Privacidade: as transcrições completas NÃO entram; só o trecho de sintomas,
literal, e o de medicamentos. O resumo do Gemini usa apenas indicadores agregados.

Uso:
    python -m astrocare.ground_report AST-01 --fonte demo
    python -m astrocare.ground_report --todos --fonte demo --sem-ia
"""
import argparse
import io
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # gera imagens sem abrir janela
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from astrocare import config
from astrocare.baseline import baseline_progress, compute_baseline
from astrocare.comparison import (
    LEVELS, ROLLING_DAYS, NotEnoughDataError, compare_with_crew, daily_status,
    flag_metrics, score_individual, smooth,
)
from astrocare.ingestion import load_dataset
from astrocare.llm import gemini_summary
from astrocare.pdf_common import FONT, FONT_BOLD, clean
from astrocare.report import build_indicators

# Cores (paleta de referência validada; status sempre com ícone + texto)
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, BAND = "#e6e5e1", "#f0efec"
SERIES = "#2a78d6"
STATUS = {"normal": ("#0ca30c", "●"), "atencao": ("#fab219", "▲"), "alerta": ("#d03b3b", "◆")}
NO_DATA = ("#d9d8d4", "○")   # dias sem reportes suficientes na janela de 7 dias
CHART_METRICS = 6            # quantas métricas comportamentais ganham gráfico

# Seções do monitoramento ampliado: (chave do texto, métricas)
MONITORING_SECTIONS = [
    ("sec_sono", config.METRICS_SELF_REPORT_SLEEP + config.METRICS_SLEEP),
    ("sec_pvt", config.METRICS_PVT),
    ("sec_cardio", ["fc_repouso_bpm", "vfc_rmssd_ms"]),
    ("sec_nutricao", config.METRICS_NUTRITION),
    ("sec_exercicio", config.METRICS_EXERCISE),
]
DOSE = "dose_diaria_uSv"

# Textos do relatório, em português e inglês (mesmas chaves nos dois idiomas).
TXT = {
    "pt": {
        "titulo": "AstroCare — Relatório de acompanhamento do astronauta",
        "subtitulo": "Astronauta {ast} · dia {dia} da missão · gerado em {gerado} · fonte: {fonte}",
        "simulado": "DADOS SIMULADOS — cenário de demonstração. Não representam medições de "
                    "nenhuma pessoa.",
        "status": {"normal": "Normal", "atencao": "Atenção", "alerta": "Alerta"},
        "sem_dados": "Sem dados suficientes",
        "situacao_atual": "Situação atual", "dias_fora": "Dias seguidos fora do padrão",
        "primeiro_sinal": "Primeiro sinal", "atencao_dia": "Atenção: dia {d}",
        "alerta_dia": "Alerta: dia {d}",
        "sem_tripulacao": "Sem dados suficientes da tripulação neste período: comparação feita só "
                          "com o padrão pessoal (eventos coletivos não são descontados).",
        "coletivo": "A tripulação como um todo também está fora do padrão neste período "
                    "(possível evento coletivo).",
        "resumo": "Resumo", "resumo_nota": "Texto redigido por IA ({modelo}) a partir dos "
                                           "indicadores agregados deste relatório.",
        "s1": "1. Status ao longo da missão", "dias": "{n} dia(s)",
        "s2": "2. Terra × missão: métricas comportamentais com maior desvio",
        "legenda": "Faixa cinza: faixa típica da pessoa na Terra (padrão ± 2 desvios). Tracejado: "
                   "padrão pessoal. Pontos: registros diários. Linha azul: média de 7 dias. "
                   "Linha vertical: lançamento.",
        "s3": "3. Comparativo completo — métricas comportamentais (regra de alerta)",
        "col_metrica": "Métrica", "col_padrao": "Padrão Terra", "col_faixa": "Faixa típica",
        "col_7d": "Últimos 7 dias", "col_var": "Variação", "col_desvios": "Desvios",
        "col_situacao": "Situação", "faixa": "{a} a {b}",
        "s4": "4. Monitoramento fisiológico e operacional",
        "s4_nota": "Estas métricas são acompanhadas e comparadas com a referência terrestre, mas "
                   "NÃO entram na regra de alerta comportamental.",
        "sec_sono": "4.1 Sono e recuperação", "sec_pvt": "4.2 Atenção e tempo de reação (PVT)",
        "sec_cardio": "4.3 Cardiovascular (wearable biométrico)",
        "sec_nutricao": "4.4 Nutrição e hidratação", "sec_exercicio": "4.5 Exercício físico",
        "sono_nota": "Qualidade do sono e recuperação ao acordar: na Terra, referência SIMULADA; "
                     "na missão, notas ditas pelo astronauta no check-in. Tempo, eficiência e "
                     "despertares: actígrafo.",
        "sec_tabela": "4.6 Comparativo do monitoramento", "dentro": "dentro da faixa",
        "acima": "acima da faixa", "abaixo": "abaixo da faixa", "insuf": "dados insuficientes",
        "sec_dose": "4.7 Dosímetro pessoal (radiação)",
        "dose_nota": "Valores absolutos (sem comparação estatística com a Terra): em órbita a dose "
                     "é estruturalmente maior. Eixo vertical em escala logarítmica.",
        "dose_terra": "Referência na Terra (mediana diária)",
        "dose_missao": "Missão: média dos últimos 7 dias",
        "dose_acum": "Missão: dose acumulada (dias com registro)",
        "dose_razao": "Razão missão / Terra", "dose_dias": "{n} dias com registro",
        "nao_disp": "Não disponível nesta fonte de dados.",
        "s5": "5. Sintomas físicos relatados no check-in",
        "s5_nota": "Texto exatamente como transcrito da fala do astronauta, sem interpretação.",
        "sem_sintomas": "Nenhum sintoma relatado no período.",
        "col_dia": "Dia", "col_data": "Data", "col_relato": "Relato",
        "s6": "6. Medicamentos utilizados",
        "s6_nota": "Medicamentos relatados pelo astronauta no check-in, exatamente como transcritos "
                   "da fala, sem interpretação.",
        "sem_meds": "Sem medicamentos utilizados.",
        "s7": "7. Qualidade dos dados e método",
        "q_terra": "Padrão pessoal: {n} dias de registro na Terra ({a} de {b} dias do período).",
        "q_missao": "Missão: registro em {a} de {b} dias.",
        "q_priv": "Transcrições completas não são incluídas (privacidade); apenas o trecho de "
                  "sintomas, literal.",
        "metodo": "Método: padrão pessoal por mediana e desvio robusto (dados da Terra); média "
                  "móvel de {n} dias; sinal exige ≥3 métricas comportamentais desviadas por ≥5 "
                  "dias seguidos, na mesma direção em relação ao padrão pessoal e à tripulação.",
        "eixo_dia": "dia da missão", "lancamento": " lançamento", "por_dia": "dia",
        "rodape": "AstroCare · {ast} · dia {dia}{sim} · página {n}", "rodape_sim": " · SIMULADO",
        "aviso": config.DISCLAIMERS["pt"],
    },
    "en": {
        "titulo": "AstroCare — Astronaut Monitoring Report",
        "subtitulo": "Astronaut {ast} · mission day {dia} · generated on {gerado} · source: {fonte}",
        "simulado": "SIMULATED DATA — demonstration scenario. These values do not represent "
                    "measurements of any person.",
        "status": {"normal": "Normal", "atencao": "Caution", "alerta": "Alert"},
        "sem_dados": "Insufficient data",
        "situacao_atual": "Current status", "dias_fora": "Consecutive days outside baseline",
        "primeiro_sinal": "First signal", "atencao_dia": "Caution: day {d}",
        "alerta_dia": "Alert: day {d}",
        "sem_tripulacao": "Insufficient crew data for this period: comparison made against the "
                          "individual baseline only (crew-wide events are not discounted).",
        "coletivo": "The crew as a whole is also outside baseline in this period (possible "
                    "crew-wide event).",
        "resumo": "Summary", "resumo_nota": "Text drafted by AI ({modelo}) from the aggregated "
                                            "indicators in this report.",
        "s1": "1. Status across the mission", "dias": "{n} day(s)",
        "s2": "2. Earth × mission: behavioral metrics with the largest deviation",
        "legenda": "Grey band: the individual's typical range on Earth (baseline ± 2 deviations). "
                   "Dashed line: individual baseline. Dots: daily records. Blue line: 7-day "
                   "moving average. Vertical line: launch.",
        "s3": "3. Full comparison — behavioral metrics (alert rule)",
        "col_metrica": "Metric", "col_padrao": "Earth baseline", "col_faixa": "Typical range",
        "col_7d": "Last 7 days", "col_var": "Change", "col_desvios": "Deviations",
        "col_situacao": "Status", "faixa": "{a} to {b}",
        "s4": "4. Physiological and operational monitoring",
        "s4_nota": "These metrics are tracked and compared with the Earth reference, but they "
                   "are NOT part of the behavioral alert rule.",
        "sec_sono": "4.1 Sleep and recovery", "sec_pvt": "4.2 Attention and reaction time (PVT)",
        "sec_cardio": "4.3 Cardiovascular (biometric wearable)",
        "sec_nutricao": "4.4 Nutrition and hydration", "sec_exercicio": "4.5 Physical exercise",
        "sono_nota": "Sleep quality and perceived recovery on waking: on Earth, SIMULATED "
                     "reference; during the mission, scores stated by the astronaut in the "
                     "check-in. Sleep time, efficiency and awakenings: actigraphy.",
        "sec_tabela": "4.6 Monitoring comparison", "dentro": "within range",
        "acima": "above range", "abaixo": "below range", "insuf": "insufficient data",
        "sec_dose": "4.7 Personal dosimeter (radiation)",
        "dose_nota": "Absolute values (no statistical comparison with Earth): in orbit the dose "
                     "is structurally higher. Vertical axis on a logarithmic scale.",
        "dose_terra": "Earth reference (daily median)",
        "dose_missao": "Mission: 7-day average",
        "dose_acum": "Mission: cumulative dose (days with records)",
        "dose_razao": "Mission / Earth ratio", "dose_dias": "{n} days with records",
        "nao_disp": "Not available in this data source.",
        "s5": "5. Physical symptoms reported in the check-in",
        "s5_nota": "Text exactly as transcribed from the astronaut's speech (original language), "
                   "without interpretation.",
        "sem_sintomas": "No symptoms reported in this period.",
        "col_dia": "Day", "col_data": "Date", "col_relato": "Report",
        "s6": "6. Medications used",
        "s6_nota": "Medications reported by the astronaut in the check-in, exactly as transcribed "
                   "from speech (original language), without interpretation.",
        "sem_meds": "No medications used.",
        "s7": "7. Data quality and method",
        "q_terra": "Individual baseline: {n} days of records on Earth ({a} of {b} days in the "
                   "period).",
        "q_missao": "Mission: records on {a} of {b} days.",
        "q_priv": "Full transcripts are not included (privacy); only the symptom excerpt, "
                  "verbatim.",
        "metodo": "Method: individual baseline from the median and a robust deviation (Earth "
                  "data); {n}-day moving average; a signal requires ≥3 behavioral metrics "
                  "deviating for ≥5 consecutive days, in the same direction relative to the "
                  "individual baseline and to the crew.",
        "eixo_dia": "mission day", "lancamento": " launch", "por_dia": "day",
        "rodape": "AstroCare · {ast} · day {dia}{sim} · page {n}", "rodape_sim": " · SIMULATED",
        "aviso": config.DISCLAIMERS["en"],
    },
}

# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
def _run_pipeline(source: str):
    data, warnings = load_dataset(source)
    baseline = compute_baseline(data)
    scored = score_individual(data, baseline)
    if scored.empty:
        raise NotEnoughDataError(
            f"Sem padrão pessoal pronto ou sem dados da missão (fonte: {source})."
        )
    flagged = flag_metrics(compare_with_crew(smooth(scored)))
    return data, baseline, flagged, daily_status(flagged, source), warnings


def _metric_table(ast, ref_day, baseline, flagged) -> pd.DataFrame:
    """Padrão da Terra x últimos 7 dias, para TODAS as métricas com padrão pronto."""
    today = flagged[(flagged["astronauta_id"] == ast) & (flagged["dia_missao"] == ref_day)]
    ref = baseline[(baseline["astronauta_id"] == ast) & baseline["confiavel"]]
    table = ref.merge(today, on=["astronauta_id", "metrica"], how="left")

    table["recente_7d"] = table["mediana"] + table["z_suavizado"] * table["escala_robusta"]
    # Faixa típica limitada aos valores possíveis da métrica (ex.: lapsos não ficam negativos)
    low = table["metrica"].map(lambda m: config.VALID_RANGES.get(m, (None, None))[0])
    high = table["metrica"].map(lambda m: config.VALID_RANGES.get(m, (None, None))[1])
    table["faixa_min"] = (table["mediana"] - 2 * table["escala_robusta"]).clip(lower=low)
    table["faixa_max"] = (table["mediana"] + 2 * table["escala_robusta"]).clip(upper=high)
    pct = 100 * (table["recente_7d"] - table["mediana"]) / table["mediana"].abs()
    # Sem variação % quando a escala cruza o zero ou quando o padrão é zero (divisão por zero)
    table["variacao_pct"] = pct.where(~table["metrica"].isin(config.NO_PERCENT_METRICS)
                                      & (table["mediana"] != 0))

    table["situacao"] = "normal"
    table.loc[table["sinal_atencao"].fillna(False).astype(bool), "situacao"] = "atencao"
    table.loc[table["sinal_alerta"].fillna(False).astype(bool), "situacao"] = "alerta"
    return table.sort_values("z_suavizado", key=lambda z: z.abs(), ascending=False)


def _band_position(row) -> str:
    """Monitoramento: média de 7 dias dentro, acima ou abaixo da faixa típica da Terra."""
    if pd.isna(row["recente_7d"]):
        return "insuf"
    if row["recente_7d"] > row["faixa_max"]:
        return "acima"
    if row["recente_7d"] < row["faixa_min"]:
        return "abaixo"
    return "dentro"


def _adherence(data, ast, ref_day) -> dict:
    """Quantos dias tiveram registro, na Terra e na missão."""
    days = data[data["astronauta_id"] == ast].groupby("fase")["dia_missao"]
    terra = days.get_group("terra") if "terra" in days.groups else pd.Series(dtype=int)
    missao = days.get_group("missao") if "missao" in days.groups else pd.Series(dtype=int)
    missao = missao[missao <= ref_day]
    terra_span = (terra.max() - terra.min() + 1) if len(terra) else 0
    return {
        "terra_dias_com_reporte": int(terra.nunique()),
        "terra_periodo_dias": int(terra_span),
        "missao_dias_com_reporte": int(missao.nunique()),
        "missao_periodo_dias": int(ref_day + 1),
    }


def _dose_summary(data, ast, ref_day) -> dict | None:
    s = data[(data["astronauta_id"] == ast) & (data["metrica"] == DOSE)
             & (data["dia_missao"] <= ref_day)]
    terra, missao = s[s["fase"] == "terra"]["valor"], s[s["fase"] == "missao"]
    if terra.empty or missao.empty:
        return None
    last7 = missao[missao["dia_missao"] > ref_day - ROLLING_DAYS]["valor"]
    return {
        "terra_mediana_uSv_dia": round(float(terra.median()), 1),
        "missao_media_7d_uSv_dia": round(float(last7.mean()), 1) if len(last7) else None,
        "missao_acumulada_mSv": round(float(missao["valor"].sum()) / 1000, 2),
        "missao_dias_com_registro": int(missao["dia_missao"].nunique()),
        "razao_missao_terra": (round(float(last7.mean() / terra.median()), 0)
                               if len(last7) else None),
    }


def _spoken(ast, source, ref_day, tipo) -> list[dict]:
    """Textos ditos no check-in ("sintomas" ou "medicamentos"), do mais recente ao mais antigo."""
    from astrocare.storage import query
    rows = query("SELECT dia_missao, data, texto FROM relatos WHERE fonte = ? AND astronauta_id = ? "
                 "AND tipo = ? AND dia_missao <= ? ORDER BY dia_missao DESC",
                 (source, ast, tipo, ref_day))
    return [{"dia": int(r.dia_missao), "data": str(r.data), "texto": r.texto}
            for r in rows.itertuples()]

# --------------------------------------------------------------------------
# Gráficos (PNG em memória, embutido no PDF)
# --------------------------------------------------------------------------
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _to_png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def _status_chart(daily, ast, t) -> bytes:
    """Faixa com uma célula por dia de missão, colorida pelo status."""
    days = daily[daily["astronauta_id"] == ast].set_index("dia_missao")["status"]
    fig, ax = plt.subplots(figsize=(9, 1.1))
    for day in range(0, int(days.index.max()) + 1):
        color = STATUS[days[day]][0] if day in days.index else NO_DATA[0]
        ax.barh(0, 0.9, left=day - 0.45, height=0.8, color=color, linewidth=0)
    ax.set_yticks([])
    ax.set_xlabel(t["eixo_dia"], color=INK_2, fontsize=8)
    _style(ax)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    return _to_png(fig)


def _series_charts(data, table, ast, metrics, lang, t) -> bytes | None:
    """Small multiples: Terra + missão, com a faixa típica da pessoa."""
    person = data[data["astronauta_id"] == ast]
    metrics = [m for m in metrics if (person["metrica"] == m).any()]
    if not metrics:
        return None
    ref = table.set_index("metrica")
    cols = 2 if len(metrics) > 1 else 1
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(9, 2.3 * rows), squeeze=False)

    for ax, metric in zip(axes.flat, metrics):
        series = person[person["metrica"] == metric].sort_values("dia_missao")
        if metric in ref.index:          # faixa típica só existe com padrão pessoal pronto
            m = ref.loc[metric]
            ax.axhspan(m["faixa_min"], m["faixa_max"], color=BAND, linewidth=0)
            ax.axhline(m["mediana"], color=MUTED, linewidth=1, linestyle="--")
        ax.scatter(series["dia_missao"], series["valor"], s=6, color=MUTED, linewidths=0)
        rolling = (series.set_index("dia_missao")["valor"]
                   .reindex(range(series["dia_missao"].min(), series["dia_missao"].max() + 1))
                   .rolling(ROLLING_DAYS, min_periods=4).mean())
        ax.plot(rolling.index, rolling.values, color=SERIES, linewidth=2)
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.text(0.5, 1.0, t["lancamento"], transform=ax.get_xaxis_transform(),
                fontsize=7, color=INK_2, va="top")
        unit = config.unit(metric, lang)
        ax.set_title(config.label(metric, lang) + (f" ({unit})" if unit else ""),
                     fontsize=9, color=INK, loc="left")
        _style(ax)

    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    fig.tight_layout(h_pad=1.6)
    return _to_png(fig)


def _dose_chart(data, ast, lang, t) -> bytes | None:
    s = data[(data["astronauta_id"] == ast) & (data["metrica"] == DOSE)].sort_values("dia_missao")
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.scatter(s["dia_missao"], s["valor"], s=6, color=MUTED, linewidths=0)
    rolling = (s.set_index("dia_missao")["valor"]
               .reindex(range(s["dia_missao"].min(), s["dia_missao"].max() + 1))
               .rolling(ROLLING_DAYS, min_periods=4).mean())
    ax.plot(rolling.index, rolling.values, color=SERIES, linewidth=2)
    ax.set_yscale("log")
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.text(0.5, 1.0, t["lancamento"], transform=ax.get_xaxis_transform(),
            fontsize=7, color=INK_2, va="top")
    ax.set_title(f"{config.label(DOSE, lang)} ({config.unit(DOSE, lang)}/{t['por_dia']})",
                 fontsize=9, color=INK, loc="left")
    ax.set_xlabel(t["eixo_dia"], color=INK_2, fontsize=8)
    _style(ax)
    return _to_png(fig)

# --------------------------------------------------------------------------
# Montagem
# --------------------------------------------------------------------------
def _fmt(value, digits=1, suffix="", lang="pt"):
    """Número formatado no padrão do idioma: 89,50 (pt) ou 89.50 (en)."""
    if value is None or pd.isna(value):
        return "–"
    text = f"{value:.{digits}f}{suffix}"
    return text.replace(".", ",") if lang == "pt" else text


def _metric_rows(table) -> list[dict]:
    return [
        {
            "metrica": r["metrica"],
            "padrao_terra": round(r["mediana"], 2),
            "faixa_tipica": [round(r["faixa_min"], 2), round(r["faixa_max"], 2)],
            "media_7_dias": None if pd.isna(r["recente_7d"]) else round(r["recente_7d"], 2),
            "variacao_pct": None if pd.isna(r["variacao_pct"]) else round(r["variacao_pct"], 1),
            "desvios_vs_padrao": None if pd.isna(r["z_suavizado"]) else round(r["z_suavizado"], 1),
            "situacao": r["situacao"],
            "posicao_na_faixa": _band_position(r),
        }
        for _, r in table.iterrows()
    ]


def build_report(ast: str, source: str, use_ai: bool = True, dia: int | None = None,
                 lang: str = "pt") -> dict:
    """Monta o relatório no dia `dia` (padrão: o último dia com dados)."""
    t = TXT[lang]
    data, baseline, flagged, daily, _ = _run_pipeline(source)
    person_days = daily[daily["astronauta_id"] == ast]
    if person_days.empty:
        raise NotEnoughDataError(f"Sem dados de missão para {ast} (fonte: {source}).")
    ref_day = int(person_days["dia_missao"].max()) if dia is None else int(dia)
    person_days = person_days[person_days["dia_missao"] <= ref_day]  # só até o dia do relatório
    daily = daily[daily["dia_missao"] <= ref_day]
    data = data[data["dia_missao"] <= ref_day]

    indicators = build_indicators(ast, ref_day, source, lang)
    table = _metric_table(ast, ref_day, baseline, flagged)
    behavioral = table[table["metrica"].isin(config.ALERT_METRICS[source])]
    monitoring = table[table["metrica"].isin(config.METRICS_MONITORING)
                       & ~table["metrica"].isin(config.NO_BASELINE_METRICS)
                       & ~table["metrica"].isin(config.ALERT_METRICS[source])]
    progress = baseline_progress(data)
    progress = progress[progress["astronauta_id"] == ast].iloc[0]

    ai_text, model = (None, None)
    if use_ai and indicators["status"] != "normal":
        ai_text, model = gemini_summary(indicators, lang)

    status_counts = person_days["status"].value_counts()
    first = {lvl: person_days.loc[person_days["status"] != "normal" if lvl == "atencao"
                                  else person_days["status"] == lvl, "dia_missao"].min()
             for lvl in LEVELS}

    charts = {
        "status": _status_chart(daily, ast, t),
        "comportamentais": _series_charts(data, table, ast,
                                          behavioral["metrica"].head(CHART_METRICS).tolist(), lang, t),
        "dose": _dose_chart(data, ast, lang, t),
    }
    for key, metrics in MONITORING_SECTIONS:
        charts[key] = _series_charts(data, table, ast, metrics, lang, t)

    return {
        "astronauta_id": ast,
        "fonte": source,
        "idioma": lang,
        "simulado": source == "demo",
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "dia_referencia": ref_day,
        "status_atual": indicators["status"],
        "dias_consecutivos_fora_do_padrao": indicators["dias_consecutivos_fora_do_padrao"],
        "primeiro_dia": {k: (None if pd.isna(v) else int(v)) for k, v in first.items()},
        "dias_por_status": {k: int(status_counts.get(k, 0)) for k in STATUS},
        "comparacao_com_tripulacao": indicators["comparacao_com_tripulacao"],
        "desvio_coletivo_da_tripulacao": indicators["desvio_coletivo_da_tripulacao"],
        "padrao_pessoal_dias_terra": int(progress["dias_na_terra"]),
        "adesao": _adherence(data, ast, ref_day),
        "metricas": _metric_rows(behavioral),
        "monitoramento": _metric_rows(monitoring),
        "dosimetro": _dose_summary(data, ast, ref_day),
        "sintomas": _spoken(ast, source, ref_day, "sintomas"),
        "medicamentos": _spoken(ast, source, ref_day, "medicamentos"),
        "resumo_ia": ai_text,
        "modelo_ia": model,
        "aviso": t["aviso"],
        "_charts": charts,  # só para o PDF; não vai para o JSON
    }


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def render_pdf(r: dict, path) -> None:
    lang, t = r["idioma"], TXT[r["idioma"]]
    styles = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=styles["Normal"], fontName=FONT, fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor(INK_2))
    cell = ParagraphStyle("c", parent=body, fontSize=8, leading=10)
    big = ParagraphStyle("big", parent=body, fontName=FONT_BOLD, fontSize=16, leading=20)
    h1 = ParagraphStyle("h1", parent=body, fontName=FONT_BOLD, fontSize=16, leading=20, spaceAfter=2)
    # keepWithNext: um título nunca fica sozinho no fim da página
    h2 = ParagraphStyle("h2", parent=body, fontName=FONT_BOLD, fontSize=11.5, leading=15,
                        spaceBefore=12, spaceAfter=4, keepWithNext=1)
    h3 = ParagraphStyle("h3", parent=body, fontName=FONT_BOLD, fontSize=10, leading=13,
                        spaceBefore=8, spaceAfter=3, keepWithNext=1)
    note = ParagraphStyle("n", parent=small, keepWithNext=1)
    P = lambda text, st=body: Paragraph(clean(text), st)
    F = lambda value, digits=1, suffix="": _fmt(value, digits, suffix, lang)  # número no idioma
    width = 170 * mm

    def img(png):
        if png is None:
            return P(t["nao_disp"], small)
        reader = Image(io.BytesIO(png))
        ratio = reader.imageHeight / reader.imageWidth
        return Image(io.BytesIO(png), width=width, height=width * ratio)

    def grid(header, rows, widths):
        data = [[P(h, cell) for h in header]] + [[P(c, cell) for c in row] for row in rows]
        tbl = Table(data, colWidths=widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(INK_2)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return tbl

    def status_txt(key):
        color, icon = STATUS[key]
        return f'<font color="{color}">{clean(icon)}</font> {clean(t["status"][key])}'

    story = [P(t["titulo"], h1),
             P(t["subtitulo"].format(ast=r["astronauta_id"], dia=r["dia_referencia"],
                                     gerado=r["gerado_em"].replace("T", " "), fonte=r["fonte"]), small)]
    if r["simulado"]:
        banner = Table([[P(t["simulado"], ParagraphStyle("sim", parent=body, fontName=FONT_BOLD))]],
                       colWidths=[width])
        banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4d6")),
                                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#e8c25a"))]))
        story += [Spacer(1, 6), banner]

    # Situação atual
    first = r["primeiro_dia"]
    box = Table([[
        [P(t["situacao_atual"], small), Paragraph(status_txt(r["status_atual"]), big)],
        [P(t["dias_fora"], small), P(str(r["dias_consecutivos_fora_do_padrao"]), big)],
        [P(t["primeiro_sinal"], small),
         P(t["atencao_dia"].format(d=first["atencao"] if first["atencao"] is not None else "–")),
         P(t["alerta_dia"].format(d=first["alerta"] if first["alerta"] is not None else "–"))],
    ]], colWidths=[width / 3] * 3)
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(GRID)),
                             ("VALIGN", (0, 0), (-1, -1), "TOP"),
                             ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [Spacer(1, 8), box]
    if not r["comparacao_com_tripulacao"]:
        story.append(P(t["sem_tripulacao"], small))
    if r["desvio_coletivo_da_tripulacao"]:
        story.append(P(t["coletivo"], small))
    if r["resumo_ia"]:
        story += [P(t["resumo"], h2), P(r["resumo_ia"]),
                  P(t["resumo_nota"].format(modelo=r["modelo_ia"]), small)]

    # 1. Status
    no_data_days = r["dia_referencia"] + 1 - sum(r["dias_por_status"].values())
    legend = "   ".join(f"{status_txt(k)}: {clean(t['dias'].format(n=r['dias_por_status'][k]))}"
                        for k in STATUS)
    if no_data_days:
        legend += (f'   <font color="{NO_DATA[0]}">{clean(NO_DATA[1])}</font> '
                   f"{clean(t['sem_dados'])}: {clean(t['dias'].format(n=no_data_days))}")
    story += [P(t["s1"], h2), Paragraph(legend, small), img(r["_charts"]["status"])]

    # 2 e 3. Comportamentais
    story += [P(t["s2"], h2), P(t["legenda"], note), img(r["_charts"]["comportamentais"])]
    header = [t["col_metrica"], t["col_padrao"], t["col_faixa"], t["col_7d"], t["col_var"],
              t["col_desvios"], t["col_situacao"]]
    rows = [[config.label(m["metrica"], lang), F(m["padrao_terra"], 2),
             t["faixa"].format(a=F(m["faixa_tipica"][0], 2), b=F(m["faixa_tipica"][1], 2)),
             F(m["media_7_dias"], 2), F(m["variacao_pct"], 1, "%"),
             F(m["desvios_vs_padrao"], 1),
             f"{STATUS[m['situacao']][1]} {t['status'][m['situacao']]}"] for m in r["metricas"]]
    widths = [44 * mm, 18 * mm, 30 * mm, 18 * mm, 18 * mm, 16 * mm, 26 * mm]
    story += [P(t["s3"], h2), grid(header, rows, widths)]

    # 4. Monitoramento
    story += [P(t["s4"], h2), P(t["s4_nota"], note)]
    for key, _ in MONITORING_SECTIONS:
        block = [P(t[key], h3)]
        if key == "sec_sono":
            block.append(P(t["sono_nota"], note))
        block.append(img(r["_charts"][key]))
        story += block
    header = [t["col_metrica"], t["col_padrao"], t["col_faixa"], t["col_7d"], t["col_var"],
              t["col_situacao"]]
    rows = [[config.label(m["metrica"], lang) + (f" ({config.unit(m['metrica'], lang)})"
                                                 if config.unit(m["metrica"], lang) else ""),
             F(m["padrao_terra"], 2),
             t["faixa"].format(a=F(m["faixa_tipica"][0], 2), b=F(m["faixa_tipica"][1], 2)),
             F(m["media_7_dias"], 2), F(m["variacao_pct"], 1, "%"),
             t[m["posicao_na_faixa"]]] for m in r["monitoramento"]]
    story += [P(t["sec_tabela"], h3),
              grid(header, rows, [52 * mm, 20 * mm, 32 * mm, 20 * mm, 18 * mm, 28 * mm])
              if rows else P(t["nao_disp"], small)]

    dose = r["dosimetro"]
    block = [P(t["sec_dose"], h3), P(t["dose_nota"], note), img(r["_charts"]["dose"])]
    if dose:
        unit = config.unit(DOSE, lang)
        block.append(grid(["", ""], [
            [t["dose_terra"], f"{F(dose['terra_mediana_uSv_dia'], 1)} {unit}"],
            [t["dose_missao"], f"{F(dose['missao_media_7d_uSv_dia'], 1)} {unit}"],
            [t["dose_acum"], f"{F(dose['missao_acumulada_mSv'], 2)} mSv "
                             f"({t['dose_dias'].format(n=dose['missao_dias_com_registro'])})"],
            [t["dose_razao"], f"{F(dose['razao_missao_terra'], 0)}×"],
        ], [80 * mm, 90 * mm]))
    story.append(KeepTogether(block))

    # 5. Sintomas
    story += [P(t["s5"], h2), P(t["s5_nota"], note)]
    if r["sintomas"]:
        story.append(grid([t["col_dia"], t["col_data"], t["col_relato"]],
                          [[s["dia"], s["data"], f"“{s['texto']}”"] for s in r["sintomas"]],
                          [15 * mm, 25 * mm, 130 * mm]))
    else:
        story.append(P(t["sem_sintomas"]))

    # 6. Medicamentos (relatados no check-in)
    story += [P(t["s6"], h2), P(t["s6_nota"], note)]
    if r["medicamentos"]:
        story.append(grid([t["col_dia"], t["col_data"], t["col_relato"]],
                          [[m["dia"], m["data"], f"“{m['texto']}”"] for m in r["medicamentos"]],
                          [15 * mm, 25 * mm, 130 * mm]))
    else:
        story.append(P(t["sem_meds"]))

    # 7. Qualidade, método e aviso
    ad = r["adesao"]
    story += [P(t["s7"], h2),
              P("• " + t["q_terra"].format(n=r["padrao_pessoal_dias_terra"],
                                           a=ad["terra_dias_com_reporte"], b=ad["terra_periodo_dias"]), small),
              P("• " + t["q_missao"].format(a=ad["missao_dias_com_reporte"],
                                            b=ad["missao_periodo_dias"]), small),
              P("• " + t["q_priv"], small), Spacer(1, 6),
              P(t["metodo"].format(n=ROLLING_DAYS), small), Spacer(1, 6), P(r["aviso"], small)]

    sim = t["rodape_sim"] if r["simulado"] else ""

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 7)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(20 * mm, 10 * mm, t["rodape"].format(
            ast=r["astronauta_id"], dia=r["dia_referencia"], sim=sim, n=doc.page))
        canvas.restoreState()

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=16 * mm, bottomMargin=18 * mm, title=t["titulo"],
                      author="AstroCare").build(story, onFirstPage=footer, onLaterPages=footer)


def generate(ast: str, source: str, use_ai: bool = True, dia: int | None = None,
             lang: str = "pt") -> tuple:
    """Gera o PDF e o JSON. Retorna (caminho do PDF, caminho do JSON, status)."""
    report = build_report(ast, source, use_ai, dia, lang)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_SIMULADO" if report["simulado"] else ""
    stem = f"AstroCare_{ast}_dia{report['dia_referencia']:03d}{tag}_{lang}"

    pdf_path = config.REPORTS_DIR / f"{stem}.pdf"
    json_path = config.REPORTS_DIR / f"{stem}.json"
    render_pdf(report, pdf_path)
    payload = {k: v for k, v in report.items() if not k.startswith("_")}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return pdf_path, json_path, report["status_atual"]

def generate_all(ast: str, source: str, use_ai: bool = True, dia: int | None = None) -> tuple:
    """Gera o relatório nos dois idiomas. Retorna (lista de arquivos, status)."""
    files, status = [], None
    for lang in config.LANGUAGES:
        pdf_path, json_path, status = generate(ast, source, use_ai, dia, lang)
        files += [pdf_path, json_path]
    return files, status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relatório comparativo para a base em Terra (PDF)")
    parser.add_argument("astronauta_id", nargs="?", help="ex.: AST-01")
    parser.add_argument("--todos", action="store_true", help="um relatório por astronauta")
    parser.add_argument("--fonte", choices=config.SOURCES, default="demo")
    parser.add_argument("--sem-ia", action="store_true", help="não chamar o Gemini")
    parser.add_argument("--idioma", choices=sorted(TXT) + ["ambos"], default="ambos")
    args = parser.parse_args()
    if not args.todos and not args.astronauta_id:
        parser.error("informe o astronauta (ex.: AST-01) ou use --todos")

    targets = sorted(config.ASTRONAUTS) if args.todos else [args.astronauta_id]
    langs = config.LANGUAGES if args.idioma == "ambos" else (args.idioma,)
    for ast in targets:
        try:
            for lang in langs:
                pdf_path, json_path, status = generate(ast, args.fonte, use_ai=not args.sem_ia,
                                                       lang=lang)
                print(f"{ast}: {TXT[lang]['status'][status]:8} -> {pdf_path.name} (+ .json)")
        except (NotEnoughDataError, FileNotFoundError) as exc:
            print(f"{ast}: não gerado — {exc}")
    print(f"\nArquivos em {config.REPORTS_DIR}")