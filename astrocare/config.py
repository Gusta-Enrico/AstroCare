"""Configurações centrais do AstroCare.

Todo caminho, nome de aba e lista de métricas fica aqui, para que
nenhum outro módulo tenha esses valores "chumbados" no código.
"""
from pathlib import Path

# --- Caminhos -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
EXCEL_PATH = BASE_DIR / "data" / "raw" / "BioShield_dados_simulados.xlsx"

AUDIO_DIR = BASE_DIR / "data" / "audio"            # gravações dos reportes
PROCESSED_DIR = BASE_DIR / "data" / "processed"    # métricas extraídas
REPORTS_DIR = BASE_DIR / "data" / "reports"  # relatórios para a base em Terra
DB_PATH = BASE_DIR / "data" / "astrocare.db"   # histórico (SQLite) para gráficos
DEMO_METRICS_CSV = PROCESSED_DIR / "reportes_demo_SIMULADO.csv"  # cenário de apresentação

# --- Missão simulada -------------------------------------------------------
# A data do lançamento simulado é definida pelo demo_data e guardada aqui.
# Ver sim_clock.py: liga as datas reais das gravações aos dias de missão.
SIMULATION_FILE = BASE_DIR / "data" / "processed" / "simulacao.json"

# --- Tripulação cadastrada -----------------------------------------------
# IDs aceitos na identificação pela fala. Um ID fora desta lista faz o
# reporte ser recusado (evita "AST-12" por erro de transcrição).
ASTRONAUTS = {"AST-01", "AST-02", "AST-03", "AST-04"}

# --- Fontes de dados -----------------------------------------------------
# Cada fonte tem padrões e comparações PRÓPRIOS. Nunca se misturam.
# - "planilha": base simulada do Excel (BioShield).
# - "demo":     cenário SIMULADO de reportes em áudio (gerado por demo_data.py).
#               As gravações de voz entram aqui, como dias de missão.
SOURCES = ("planilha", "demo")

# --- Chaves que identificam uma medição -----------------------------------
KEY_COLS = ["astronauta_id", "dia_missao"]

# --- Abas da planilha, separadas por função -------------------------------
# Indicadores comportamentais (reporte diário / journal)
SHEETS_BEHAVIORAL = {"terra": "Terra_Journal", "missao": "Missao_Journal"}

# Contexto: de onde vêm as métricas de sono
SHEETS_CONTEXT = {"terra": "Terra_Camiseta", "missao": "Missao_Camiseta"}

# GABARITO: usar SOMENTE para avaliar o detector.
# Nunca usar como entrada de modelo (vazamento de dados).
SHEETS_EVALUATION = {"astronautas": "Astronautas", "eventos": "Eventos"}

# --- Métricas, agrupadas pela origem no futuro reporte em áudio -----------
# Extraídas da acústica da voz
METRICS_VOICE = [
    "duracao_video_s",
    "pausas_pct_tempo",
    "tom_medio_Hz",
    "variacao_tom_semitons",
    "volume_voz_dB",
]

# Extraídas da transcrição (velocidade = palavras / duração)
METRICS_TEXT = [
    "velocidade_fala_ppm",
    "saldo_sentimento_m1a1",
    "palavras_positivas_pct",
    "palavras_negativas_pct",
]

# Informadas pelo próprio astronauta. A IA NÃO deve inferir essas notas.
METRICS_SELF_REPORT = [
    "humor_1a10",
    "fadiga_1a10",
    "estresse_1a10",
    "solidao_1a10",
]

# Sono (vem da camiseta, aba de contexto)
METRICS_SLEEP = [
    "sono_total_h",
    "sono_eficiencia_pct",
    "despertares_noite",
]

# Rosto (vídeo). Fora do escopo enquanto o reporte for só em áudio.
METRICS_VIDEO = [
    "piscadas_por_min",
    "perclos_pct",
    "expressividade_0a1",
    "sorrisos_por_min",
]

# Métricas que entram no detector nesta fase do protótipo
METRICS_JOURNAL = METRICS_VOICE + METRICS_TEXT + METRICS_SELF_REPORT

# --- Monitoramento ampliado (Parte 1, evolução) ----------------------------
# Sono e recuperação: na MISSÃO, ditos no check-in de áudio (escala 0 a 10);
# na fase TERRA, SIMULADOS (30 dias) para servir de referência.
METRICS_SELF_REPORT_SLEEP = ["qualidade_sono_0a10", "recuperacao_0a10"]

# Sensores: SIMULADOS automaticamente (Terra e a cada novo registro na missão).
METRICS_PVT = ["pvt_rt_medio_ms", "pvt_lapsos"]             # atenção / tempo de reação
METRICS_WEARABLE = METRICS_SLEEP + ["fc_repouso_bpm", "vfc_rmssd_ms"]  # actígrafo
METRICS_NUTRITION = ["agua_ml", "calorias_kcal", "refeicoes_dia"]
METRICS_EXERCISE = ["exercicio_min"]
METRICS_DOSIMETER = ["dose_diaria_uSv"]
METRICS_SENSORS = (METRICS_PVT + METRICS_WEARABLE + METRICS_NUTRITION
                   + METRICS_EXERCISE + METRICS_DOSIMETER)

# Tudo o que o monitoramento ampliado mostra no relatório (NÃO entra no alerta)
METRICS_MONITORING = METRICS_SELF_REPORT_SLEEP + METRICS_SENSORS

# Métricas que CONTAM na regra de alerta, por fonte. Mantém o comportamento
# já validado: a planilha conta journal + sono da camiseta; o demo, só o journal.
ALERT_METRICS = {
    "planilha": METRICS_JOURNAL + METRICS_SLEEP,
    "demo": METRICS_JOURNAL,
}

# Dose de radiação: em órbita é estruturalmente maior que na Terra. Não faz
# sentido medir "desvio do padrão pessoal"; o relatório mostra valor absoluto.
NO_BASELINE_METRICS = {"dose_diaria_uSv"}

# --- Faixas fisicamente plausíveis (validação de qualidade) ---------------
# Não são faixas "normais" da pessoa: servem só para pegar erro de dado
# (ex.: tom de voz de 5000 Hz, sono de 30 h).
VALID_RANGES = {
    "duracao_video_s": (0, 3600),
    "pausas_pct_tempo": (0, 100),
    "tom_medio_Hz": (50, 400),
    "variacao_tom_semitons": (0, 24),
    "volume_voz_dB": (20, 120),
    "velocidade_fala_ppm": (30, 300),
    "saldo_sentimento_m1a1": (-1, 1),
    "palavras_positivas_pct": (0, 100),
    "palavras_negativas_pct": (0, 100),
    "humor_1a10": (1, 10),
    "fadiga_1a10": (1, 10),
    "estresse_1a10": (1, 10),
    "solidao_1a10": (1, 10),
    "sono_total_h": (0, 24),
    "sono_eficiencia_pct": (0, 100),
        "despertares_noite": (0, 50),
    "qualidade_sono_0a10": (0, 10),
    "recuperacao_0a10": (0, 10),
    "pvt_rt_medio_ms": (100, 2000),
    "pvt_lapsos": (0, 200),
    "fc_repouso_bpm": (25, 150),
    "vfc_rmssd_ms": (3, 300),
    "agua_ml": (0, 10000),
    "calorias_kcal": (0, 8000),
    "refeicoes_dia": (0, 10),
    "exercicio_min": (0, 600),
    "dose_diaria_uSv": (0, 100000),
}

# --- Relatório ------------------------------------------------------------
# Nomes legíveis para o relatório (o que o profissional vai ler)
METRIC_LABELS = {
    "duracao_video_s": "duração do reporte",
    "pausas_pct_tempo": "tempo em pausas na fala",
    "tom_medio_Hz": "tom médio de voz",
    "variacao_tom_semitons": "variação do tom de voz",
    "volume_voz_dB": "volume de voz",
    "velocidade_fala_ppm": "velocidade da fala",
    "saldo_sentimento_m1a1": "saldo de sentimento do texto",
    "palavras_positivas_pct": "proporção de palavras positivas",
    "palavras_negativas_pct": "proporção de palavras negativas",
    "humor_1a10": "humor autorrelatado",
    "fadiga_1a10": "fadiga autorrelatada",
    "estresse_1a10": "estresse autorrelatado",
    "solidao_1a10": "solidão autorrelatada",
    "sono_total_h": "tempo total de sono",
    "sono_eficiencia_pct": "eficiência do sono",
        "despertares_noite": "despertares por noite",
    "qualidade_sono_0a10": "qualidade do sono (autorrelato)",
    "recuperacao_0a10": "sensação de recuperação ao acordar",
    "pvt_rt_medio_ms": "PVT — tempo médio de reação",
    "pvt_lapsos": "PVT — lapsos de atenção",
    "fc_repouso_bpm": "frequência cardíaca de repouso",
    "vfc_rmssd_ms": "variabilidade da frequência cardíaca (RMSSD)",
    "agua_ml": "consumo de água",
    "calorias_kcal": "ingestão calórica",
    "refeicoes_dia": "refeições no dia",
    "exercicio_min": "exercício físico",
    "dose_diaria_uSv": "dose de radiação diária (dosímetro)",
}

METRIC_LABELS_EN = {
    "duracao_video_s": "report duration",
    "pausas_pct_tempo": "time spent in speech pauses",
    "tom_medio_Hz": "mean voice pitch",
    "variacao_tom_semitons": "voice pitch variation",
    "volume_voz_dB": "voice loudness",
    "velocidade_fala_ppm": "speech rate",
    "saldo_sentimento_m1a1": "text sentiment balance",
    "palavras_positivas_pct": "share of positive words",
    "palavras_negativas_pct": "share of negative words",
    "humor_1a10": "self-reported mood",
    "fadiga_1a10": "self-reported fatigue",
    "estresse_1a10": "self-reported stress",
    "solidao_1a10": "self-reported loneliness",
    "sono_total_h": "total sleep time",
    "sono_eficiencia_pct": "sleep efficiency",
    "despertares_noite": "awakenings per night",
    "qualidade_sono_0a10": "sleep quality (self-reported)",
    "recuperacao_0a10": "perceived recovery on waking",
    "pvt_rt_medio_ms": "PVT — mean reaction time",
    "pvt_lapsos": "PVT — attention lapses",
    "fc_repouso_bpm": "resting heart rate",
    "vfc_rmssd_ms": "heart rate variability (RMSSD)",
    "agua_ml": "water intake",
    "calorias_kcal": "caloric intake",
    "refeicoes_dia": "meals per day",
    "exercicio_min": "physical exercise",
    "dose_diaria_uSv": "daily radiation dose (dosimeter)",
}
LABELS = {"pt": METRIC_LABELS, "en": METRIC_LABELS_EN}

# Unidades exibidas nos gráficos e tabelas, por idioma
_UNITS_COMMON = {
    "duracao_video_s": "s", "pausas_pct_tempo": "%", "tom_medio_Hz": "Hz",
    "variacao_tom_semitons": "st", "volume_voz_dB": "dB", "saldo_sentimento_m1a1": "-1..1",
    "palavras_positivas_pct": "%", "palavras_negativas_pct": "%",
    "humor_1a10": "1-10", "fadiga_1a10": "1-10", "estresse_1a10": "1-10", "solidao_1a10": "1-10",
    "sono_total_h": "h", "sono_eficiencia_pct": "%",
    "qualidade_sono_0a10": "0-10", "recuperacao_0a10": "0-10",
    "pvt_rt_medio_ms": "ms", "fc_repouso_bpm": "bpm", "vfc_rmssd_ms": "ms",
    "agua_ml": "mL", "calorias_kcal": "kcal", "exercicio_min": "min", "dose_diaria_uSv": "µSv",
}
METRIC_UNITS = {
    "pt": {**_UNITS_COMMON, "velocidade_fala_ppm": "palavras/min",
           "despertares_noite": "por noite", "pvt_lapsos": "lapsos", "refeicoes_dia": "refeições"},
    "en": {**_UNITS_COMMON, "velocidade_fala_ppm": "words/min",
           "despertares_noite": "per night", "pvt_lapsos": "lapses", "refeicoes_dia": "meals"},
}


def label(metric: str, lang: str = "pt") -> str:
    """Nome legível da métrica no idioma pedido (cai no código se faltar)."""
    return LABELS[lang].get(metric, metric)


def unit(metric: str, lang: str = "pt") -> str:
    return METRIC_UNITS[lang].get(metric, "")


# Métricas em escala que cruza o zero (-1 a 1): variação % não faz sentido
NO_PERCENT_METRICS = {"saldo_sentimento_m1a1"}

# Termos clínicos/diagnósticos que o texto da IA NÃO pode conter.
# Se aparecer qualquer um, o texto é descartado e usa-se o relatório do código.
FORBIDDEN_TERMS = [
    "diagnóstico", "diagnostico", "depressão", "depressao", "depressivo",
    "ansiedade", "ansioso", "transtorno", "síndrome", "sindrome", "burnout",
    "patologia", "patológico", "doença", "doenca", "psicose", "psicótico",
    "bipolar", "tept", "estresse pós-traumático", "distúrbio", "disturbio",
    "sofre de", "apresenta quadro",
]

# Mesma proteção para textos em inglês (o filtro verifica as duas listas)
FORBIDDEN_TERMS_EN = [
    "diagnosis", "diagnose", "diagnosed", "depression", "depressive", "depressed",
    "anxiety", "anxious", "disorder", "syndrome", "burnout", "pathology", "pathological",
    "disease", "psychosis", "psychotic", "bipolar", "ptsd", "post-traumatic",
    "suffers from", "suffering from", "mental illness",
]

# Aviso fixo, anexado pelo código (nunca depende da IA)
DISCLAIMER = (
    "Este relatório descreve desvios estatísticos em relação ao padrão pessoal "
    "e à tripulação. Não é diagnóstico clínico nem psicológico; a interpretação "
    "cabe a um profissional de saúde qualificado."
)
DISCLAIMER_EN = (
    "This report describes statistical deviations from the individual baseline and "
    "from the crew. It is not a clinical or psychological diagnosis; interpretation "
    "is the responsibility of a qualified health professional."
)
DISCLAIMERS = {"pt": DISCLAIMER, "en": DISCLAIMER_EN}

# --- Etapa 2: módulo de emergência (100% offline) -----------------------
EMERGENCY_DB_PATH = BASE_DIR / "data" / "emergencia.db"
EXPIRY_WARNING_DAYS = 30   # alerta de "vence em breve"
PROTOCOLS_DIR = BASE_DIR / "astrocare" / "emergency" / "protocols"  # protocolos (JSON)
LANGUAGES = ("pt", "en")   # idiomas do atendimento


# --- Áudio ----------------------------------------------------------------
SAMPLE_RATE = 16000       # 16 kHz mono: padrão para análise de fala
PITCH_FLOOR_HZ = 75       # faixa de busca do tom de voz humano
PITCH_CEILING_HZ = 500
SILENCE_DB_BELOW_PEAK = 25  # quanto abaixo do pico conta como silêncio
MIN_PAUSE_S = 0.25          # silêncios menores que isso não são pausa


def check_definitions() -> list[str]:
    """Confere se toda métrica tem faixa válida, rótulo e unidade nos dois idiomas."""
    problems = []
    for metric in dict.fromkeys(METRICS_JOURNAL + METRICS_SLEEP + METRICS_MONITORING):
        if metric not in VALID_RANGES:
            problems.append(f"{metric}: sem faixa válida (VALID_RANGES)")
        for lang in LANGUAGES:
            if metric not in LABELS[lang]:
                problems.append(f"{metric}: sem rótulo em '{lang}'")
            if metric not in METRIC_UNITS[lang]:
                problems.append(f"{metric}: sem unidade em '{lang}'")
    return problems


if __name__ == "__main__":
    print(EXCEL_PATH, EXCEL_PATH.exists())
    groups = {"autorrelato de sono (áudio)": METRICS_SELF_REPORT_SLEEP, "PVT": METRICS_PVT,
              "wearable": METRICS_WEARABLE, "nutrição": METRICS_NUTRITION,
              "exercício": METRICS_EXERCISE, "dosímetro": METRICS_DOSIMETER}
    for name, metrics in groups.items():
        print(f"\n{name}:")
        for m in metrics:
            low, high = VALID_RANGES[m]
            print(f"  {m:<22} {low:>6}–{high:<7} {unit(m):<12} {label(m)}  |  {label(m, 'en')}")
    problems = check_definitions()
    print("\nDefinições OK" if not problems else "\nPROBLEMAS:\n  " + "\n  ".join(problems))