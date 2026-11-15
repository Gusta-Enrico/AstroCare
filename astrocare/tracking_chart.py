"""Gráfico de acompanhamento lido direto do banco (data/astrocare.db).

Mostra, para um astronauta:
- no topo, o status de cada dia da missão (normal / atenção / alerta);
- abaixo, as métricas escolhidas: valores diários, média de 7 dias e a
  faixa típica da Terra (padrão ± 2 desvios).

Serve de exemplo de como qualquer gráfico pode ser montado a partir do
histórico salvo, sem recalcular nada.

Uso:
    python -m astrocare.tracking_chart AST-01 --fonte demo
    python -m astrocare.tracking_chart AST-01 --fonte demo --metricas humor_1a10 pausas_pct_tempo
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astrocare import config
from astrocare.ground_report import BAND, INK, INK_2, MUTED, NO_DATA, SERIES, STATUS, _style
from astrocare.storage import metric_history, query

DEFAULT_COUNT = 4  # sem --metricas: as 4 mais desviadas no último dia
# Usadas quando não dá para escolher pelas mais desviadas
DEFAULT_METRICS = ["humor_1a10", "velocidade_fala_ppm", "pausas_pct_tempo",
                   "variacao_tom_semitons"]


def _most_deviated(source: str, ast: str) -> list[str]:
    last = query(
        """SELECT metrica, ABS(z_suavizado) AS z FROM indicadores
           WHERE fonte = ? AND astronauta_id = ? AND dia_missao =
             (SELECT MAX(dia_missao) FROM indicadores WHERE fonte = ? AND astronauta_id = ?)
           ORDER BY z DESC""", (source, ast, source, ast))
    return last["metrica"].head(DEFAULT_COUNT).tolist() or DEFAULT_METRICS


def draw(source: str, ast: str, metrics: list[str]) -> str:
    status = query("SELECT dia_missao, status FROM status_diario "
                   "WHERE fonte = ? AND astronauta_id = ?", (source, ast)).set_index("dia_missao")["status"]
    if status.empty:
        raise SystemExit(f"Sem histórico de missão para {ast} na fonte '{source}'. "
                         f"Rode antes: python -m astrocare.storage --fonte {source}")

    fig, axes = plt.subplots(len(metrics) + 1, 1, figsize=(9, 1 + 2 * len(metrics)), sharex=True,
                             gridspec_kw={"height_ratios": [0.35] + [1] * len(metrics)})

    # Faixa de status (só dias de missão)
    for day in range(0, int(status.index.max()) + 1):
        color = STATUS[status[day]][0] if day in status.index else NO_DATA[0]
        axes[0].barh(0, 0.9, left=day - 0.45, color=color, linewidth=0)
    axes[0].set_yticks([])
    legend = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _, _ in STATUS.values()]
    axes[0].legend(legend + [plt.Rectangle((0, 0), 1, 1, color=NO_DATA[0])],
                   [lbl for _, _, lbl in STATUS.values()] + ["Sem dados"],
                   loc="upper left", bbox_to_anchor=(0, 1), ncol=4, frameon=False,
                   fontsize=8, labelcolor=INK_2, handlelength=1, handleheight=1)
    axes[0].set_title(f"{ast} — status diário e métricas (Terra × missão)",
                      loc="left", fontsize=10, color=INK)
    _style(axes[0])
    axes[0].grid(False)

    for ax, metric in zip(axes[1:], metrics):
        h = metric_history(source, ast, metric)
        median, scale = h["mediana"].iloc[0], h["escala_robusta"].iloc[0]
        ax.axhspan(median - 2 * scale, median + 2 * scale, color=BAND, linewidth=0)
        ax.axhline(median, color=MUTED, linewidth=1, linestyle="--")
        ax.scatter(h["dia_missao"], h["valor"], s=6, color=MUTED, linewidths=0)
        rolling = (h.set_index("dia_missao")["valor"]
                   .reindex(range(h["dia_missao"].min(), h["dia_missao"].max() + 1))
                   .rolling(7, min_periods=4).mean())
        ax.plot(rolling.index, rolling.values, color=SERIES, linewidth=2)
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_title(config.METRIC_LABELS.get(metric, metric), loc="left", fontsize=9, color=INK)
        _style(ax)
    axes[-1].set_xlabel("dia da missão (negativo = Terra)", color=INK_2, fontsize=8)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_SIMULADO" if source == "demo" else ""
    path = config.REPORTS_DIR / f"acompanhamento_{ast}_{source}{tag}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gráfico de acompanhamento")
    parser.add_argument("astronauta_id")
    parser.add_argument("--fonte", choices=config.SOURCES, default="demo")
    parser.add_argument("--metricas", nargs="+", help="nomes das colunas (ex.: humor_1a10)")
    args = parser.parse_args()

    metrics = args.metricas or _most_deviated(args.fonte, args.astronauta_id)
    print(f"Gráfico salvo em {draw(args.fonte, args.astronauta_id, metrics)}")