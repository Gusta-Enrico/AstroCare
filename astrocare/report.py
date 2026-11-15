"""Relatório de um astronauta em um dia da missão.

Fluxo:
1. O código monta os INDICADORES (números, direções, duração).
2. O código gera um relatório-base, sempre disponível, sem IA.
3. Se o Gemini estiver configurado, ele reescreve os indicadores em
   texto corrido. Se falhar ou violar as regras, vale o relatório-base.
4. O aviso de "não é diagnóstico" é sempre anexado pelo código.

Uso:
    python -m astrocare.report AST-03 110
    python -m astrocare.report AST-03 110 --sem-ia
    python -m astrocare.report AST-01 15 --fonte audio
"""
import argparse

from astrocare import config
from astrocare.baseline import compute_baseline
from astrocare.comparison import (
    NotEnoughDataError, compare_with_crew, daily_status, flag_metrics,
    score_individual, smooth,
)
from astrocare.ingestion import load_dataset
from astrocare.llm import gemini_summary

TOP_METRICS = 6  # quantas métricas mostrar no relatório


def build_indicators(astronauta_id: str, dia: int, source: str = "planilha") -> dict:
    """Monta os indicadores de um astronauta em um dia, só com números do código."""
    data, _ = load_dataset(source)
    baseline = compute_baseline(data)
    scored = score_individual(data, baseline)
    if scored.empty:
        raise NotEnoughDataError(
            "Sem padrão pessoal pronto ou sem reportes da missão para comparar. "
            f"Veja o progresso com: python -m astrocare.baseline --fonte {source}"
        )
    flagged = flag_metrics(compare_with_crew(smooth(scored)))
    daily = daily_status(flagged)

    day_status = daily[(daily["astronauta_id"] == astronauta_id) & (daily["dia_missao"] == dia)]
    if day_status.empty:
        raise ValueError(f"Sem dados para {astronauta_id} no dia {dia}")
    status = day_status["status"].iloc[0]

    # Há quantos dias seguidos o status está fora do normal
    history = daily[(daily["astronauta_id"] == astronauta_id) & (daily["dia_missao"] <= dia)]
    abnormal = (history.sort_values("dia_missao")["status"] != "normal").tolist()
    days_out = 0
    for value in reversed(abnormal):
        if not value:
            break
        days_out += 1

    # Métricas do dia, com valor real reconstruído a partir do z
    rows = flagged[(flagged["astronauta_id"] == astronauta_id) & (flagged["dia_missao"] == dia)]
    rows = rows.merge(baseline[["astronauta_id", "metrica", "mediana", "escala_robusta"]])
    rows = rows[rows["sinal_atencao"]].copy()
    rows["valor_7d"] = rows["mediana"] + rows["z_suavizado"] * rows["escala_robusta"]
    rows = rows.reindex(rows["z_vs_tripulacao"].abs().sort_values(ascending=False).index)

    metrics = []
    for _, r in rows.head(TOP_METRICS).iterrows():
        pct = None
        if r["metrica"] not in config.NO_PERCENT_METRICS and r["mediana"] != 0:
            pct = round(100 * (r["valor_7d"] - r["mediana"]) / abs(r["mediana"]), 1)
        metrics.append({
            "metrica": config.METRIC_LABELS.get(r["metrica"], r["metrica"]),
            "direcao": "acima" if r["z_suavizado"] > 0 else "abaixo",
            "media_7_dias": round(r["valor_7d"], 2),
            "padrao_pessoal": round(r["mediana"], 2),
            "variacao_pct": pct,
            "desvios_vs_padrao_pessoal": round(r["z_suavizado"], 1),
            # Sem tripulação para comparar, o campo fica vazio (não inventa número)
            "desvios_vs_tripulacao": round(r["z_vs_tripulacao"], 1) if r["com_tripulacao"] else None,
        })

    # A tripulação inteira está fora do padrão neste dia? (evento coletivo)
    crew_today = flagged[flagged["dia_missao"] == dia]
    crew_on = bool(crew_today["com_tripulacao"].all())
    collective = crew_on and bool(crew_today["z_tripulacao"].abs().median() >= 1.0)

    return {
        "astronauta_id": astronauta_id,
        "dia_missao": int(dia),
        "status": status,
        "dias_consecutivos_fora_do_padrao": days_out,
        "metricas_desviadas": metrics,
        "comparacao_com_tripulacao": crew_on,
        "desvio_coletivo_da_tripulacao": collective,
    }


def build_base_report(ind: dict) -> str:
    """Relatório gerado só pelo código: sempre disponível, sem IA."""
    lines = [f"Status: {ind['status'].upper()}"]
    if ind["status"] == "normal" and not ind["metricas_desviadas"]:
        lines.append("Nenhuma métrica com desvio persistente em relação ao padrão "
                     "pessoal e à tripulação.")
    else:
        if ind["dias_consecutivos_fora_do_padrao"]:
            lines.append(f"Fora do padrão há {ind['dias_consecutivos_fora_do_padrao']} dia(s) seguidos.")
        lines.append("Métricas com desvio (média dos últimos 7 dias):")
        for m in ind["metricas_desviadas"]:
            pct = f" ({m['variacao_pct']:+.1f}%)" if m["variacao_pct"] is not None else ""
            crew = (f", {m['desvios_vs_tripulacao']:+.1f} vs. tripulação"
                    if m["desvios_vs_tripulacao"] is not None else "")
            lines.append(
                f"  - {m['metrica']}: {m['direcao']} do padrão pessoal{pct}; "
                f"{m['desvios_vs_padrao_pessoal']:+.1f} desvios vs. si mesmo{crew}"
            )
    if not ind["comparacao_com_tripulacao"]:
        lines.append("Obs.: menos de 3 astronautas nesta fonte; comparação feita só "
                     "com o padrão pessoal (sem descontar eventos coletivos).")
    if ind["desvio_coletivo_da_tripulacao"]:
        lines.append("Obs.: a tripulação como um todo também está fora do padrão neste "
                     "período (possível evento coletivo).")
    return "\n".join(lines)


def generate_report(astronauta_id: str, dia: int, use_ai: bool = True,
                    source: str = "planilha") -> str:
    ind = build_indicators(astronauta_id, dia, source)
    base = build_base_report(ind)

    ai_text, model = None, None
    if use_ai and ind["status"] != "normal":
        ai_text, model = gemini_summary(ind)
    text_source = f"Gemini ({model})" if ai_text else "relatório automático (sem IA)"

    parts = [f"=== AstroCare | {astronauta_id} | dia {dia} da missão ===", base]
    if ai_text:
        parts += ["", "Resumo:", ai_text]
    parts += ["", f"Fonte do texto: {text_source}", config.DISCLAIMER]
    return "\n".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relatório AstroCare")
    parser.add_argument("astronauta_id", help="ex.: AST-03")
    parser.add_argument("dia", type=int, help="dia da missão, ex.: 110")
    parser.add_argument("--sem-ia", action="store_true", help="não chamar o Gemini")
    parser.add_argument("--fonte", choices=config.SOURCES, default="planilha")
    args = parser.parse_args()

    try:
        print(generate_report(args.astronauta_id, args.dia,
                              use_ai=not args.sem_ia, source=args.fonte))
    except NotEnoughDataError as exc:
        print(f"Ainda não é possível gerar o relatório: {exc}")