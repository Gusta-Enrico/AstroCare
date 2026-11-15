"""Relatório CURTO do dia: como o astronauta está e se mudou em relação à Terra.

Responde a três perguntas, em poucas linhas:
1. Qual a situação hoje? (tendência dos últimos 7 dias: normal/atenção/alerta)
2. Houve alteração em relação ao padrão da Terra? (sim/não e em quê)
3. O reporte de HOJE, isolado, teve algo fora da faixa típica da Terra?

A diferença entre 2 e 3 é importante: um dia fora da faixa é uma variação
isolada (todo mundo tem dias ruins); "alteração" é quando o desvio persiste.

Lê tudo do banco (storage.py), que é atualizado antes de gerar o relatório.
O texto também fica salvo no banco (tabela relatorios_diarios).

Uso:
    python -m astrocare.daily_report AST-01 45 --fonte demo
    python -m astrocare.daily_report AST-01 --fonte demo        (último dia)
"""
import argparse
from datetime import datetime

from astrocare import config
from astrocare.storage import query, save_daily_report, sync

STATUS_LABEL = {"normal": "● NORMAL", "atencao": "▲ ATENÇÃO", "alerta": "◆ ALERTA"}
MAX_ITEMS = 4  # métricas citadas por linha, para o relatório continuar curto


def _num(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def _label(metric: str) -> str:
    return config.METRIC_LABELS.get(metric, metric)


def _describe(rows, value_col: str) -> str:
    """'humor ↓ (4,8; Terra 8,0), fadiga ↑ (5,7; Terra 3,0)'"""
    parts = []
    for _, r in rows.head(MAX_ITEMS).iterrows():
        arrow = "↑" if r[value_col] > r["mediana"] else "↓"
        parts.append(f"{_label(r['metrica'])} {arrow} ({_num(r[value_col])}; "
                     f"Terra {_num(r['mediana'])})")
    extra = len(rows) - MAX_ITEMS
    return ", ".join(parts) + (f" e mais {extra}" if extra > 0 else "")

def save_text_file(astronauta_id: str, dia: int, source: str, text: str):
    """Grava o relatório curto em data/reports/ e devolve o caminho."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_SIMULADO" if source == "demo" else ""
    path = config.REPORTS_DIR / f"relatorio_diario_{astronauta_id}_dia{dia:03d}{tag}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def build(astronauta_id: str, dia: int | None, source: str) -> tuple[str, str, bool, int]:
    """Monta o texto. Retorna (texto, status, houve_alteracao, dia)."""
    params = (source, astronauta_id)
    status_hist = query(
        "SELECT dia_missao, status FROM status_diario "
        "WHERE fonte = ? AND astronauta_id = ? ORDER BY dia_missao", params)
    if dia is None:
        if status_hist.empty:
            raise ValueError(f"Sem dados de missão para {astronauta_id} (fonte {source}).")
        dia = int(status_hist["dia_missao"].max())

    today_status = status_hist.loc[status_hist["dia_missao"] == dia, "status"]
    status = today_status.iloc[0] if len(today_status) else None

    # Sequência atual fora do normal: quantos dias e em que dia começou
    streak, streak_start = 0, None
    until_today = status_hist[status_hist["dia_missao"] <= dia]
    for day_, s in zip(reversed(until_today["dia_missao"].tolist()),
                       reversed(until_today["status"].tolist())):
        if s == "normal":
            break
        streak, streak_start = streak + 1, int(day_)

    # Evento coletivo: a tripulação inteira está fora do padrão hoje?
    alert_metrics = config.ALERT_METRICS[source]     # o resto é monitoramento ampliado
    crew = query(
        "SELECT metrica, z_suavizado - z_vs_tripulacao AS z_trip, com_tripulacao FROM indicadores "
        "WHERE fonte = ? AND dia_missao = ?", (source, dia))
    crew = crew[crew["metrica"].isin(alert_metrics)]
    collective = (not crew.empty and bool(crew["com_tripulacao"].all())
                  and crew["z_trip"].abs().median() >= 1.0)
    own_crew = query(
        "SELECT MIN(com_tripulacao) AS ok FROM indicadores "
        "WHERE fonte = ? AND astronauta_id = ? AND dia_missao = ?", (*params, dia))
    crew_missing = not own_crew.empty and own_crew["ok"].iloc[0] == 0

    # Tendência (média de 7 dias) das métricas sinalizadas hoje
    trend = query(
        """SELECT i.metrica, i.z_suavizado, p.mediana, p.escala_robusta
           FROM indicadores i JOIN padrao_pessoal p USING (fonte, astronauta_id, metrica)
           WHERE i.fonte = ? AND i.astronauta_id = ? AND i.dia_missao = ?
             AND i.sinal_atencao = 1""",
        (*params, dia))
    trend = trend[trend["metrica"].isin(alert_metrics)].copy()
    trend["media_7d"] = trend["mediana"] + trend["z_suavizado"] * trend["escala_robusta"]
    trend = trend.reindex(trend["z_suavizado"].abs().sort_values(ascending=False).index)

    # Reporte de hoje, isolado, comparado à faixa típica da Terra
    today = query(
        """SELECT r.metrica, r.valor, p.mediana, p.escala_robusta
           FROM reportes r JOIN padrao_pessoal p USING (fonte, astronauta_id, metrica)
           WHERE r.fonte = ? AND r.astronauta_id = ? AND r.dia_missao = ?
             AND p.confiavel = 1""",
        (*params, dia))
    today = today[today["metrica"].isin(alert_metrics)].copy()
    today["z"] = (today["valor"] - today["mediana"]) / today["escala_robusta"]
    outside_today = today[today["z"].abs() >= 2].sort_values("z", key=abs, ascending=False)

    data_row = query("SELECT MIN(data) AS d FROM reportes WHERE fonte = ? AND dia_missao = ?",
                     (source, dia))
    date_txt = data_row["d"].iloc[0] or "sem reporte"

    # ---- Texto ----
    header = f"AstroCare · {astronauta_id} · dia {dia} da missão ({date_txt})"
    if source == "demo":
        header += " · DADOS SIMULADOS"
    lines = [header, ""]

    if status is None:
        lines.append("Situação: sem dados suficientes nos últimos 7 dias para avaliar a tendência.")
        changed = False
    else:
        streak_txt = f" há {streak} dia(s) seguidos" if status != "normal" else ""
        lines.append(f"Situação (tendência de 7 dias): {STATUS_LABEL[status]}{streak_txt}")
        changed = status != "normal"
        if changed:
            lines.append(f"Alteração em relação à Terra: SIM — {len(trend)} métrica(s) "
                         f"persistentemente fora do padrão: {_describe(trend, 'media_7d')}.")
        else:
            lines.append("Alteração em relação à Terra: NÃO — nenhuma métrica com desvio "
                         "persistente.")

    if today.empty:
        lines.append("Reporte de hoje: não recebido.")
    elif outside_today.empty:
        lines.append(f"Reporte de hoje: todas as {len(today)} métricas dentro da faixa "
                     "típica da Terra.")
    else:
        note = "" if changed else " (sem tendência confirmada)"
        lines.append(f"Reporte de hoje: {len(outside_today)} de {len(today)} métricas fora "
                     f"da faixa típica da Terra{note}: {_describe(outside_today, 'valor')}.")

    if changed and streak_start is not None:
        lines.append(f"Início desta sequência fora do padrão: dia {streak_start}.")
    if collective and changed:
        lines.append("Obs.: a tripulação inteira também está fora do padrão neste período; "
                     "o desvio acima é o que vai ALÉM do que os colegas apresentam.")
    elif collective:
        lines.append("Obs.: a tripulação inteira está fora do padrão neste período "
                     "(possível evento coletivo, não individual).")
    if crew_missing:
        lines.append("Obs.: sem dados suficientes da tripulação neste dia; comparação feita "
                     "só com o padrão pessoal.")
    lines += ["", config.DISCLAIMER]
    return "\n".join(lines), status or "sem_dados", changed, dia


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relatório curto do dia")
    parser.add_argument("astronauta_id", help="ex.: AST-01")
    parser.add_argument("dia", type=int, nargs="?", help="dia da missão (padrão: o último)")
    parser.add_argument("--fonte", choices=config.SOURCES, default="demo")
    args = parser.parse_args()

    sync(args.fonte)  # garante que o banco está com os dados mais recentes
    text, status, changed, dia = build(args.astronauta_id, args.dia, args.fonte)
    save_daily_report(args.fonte, args.astronauta_id, dia,
                      datetime.now().isoformat(timespec="seconds"), status, changed, text)
    print(text)

    print(f"\nSalvo em {save_text_file(args.astronauta_id, dia, args.fonte, text)}")