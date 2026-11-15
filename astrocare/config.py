"""Configurações centrais do AstroCare.

Todo caminho, nome de aba e lista de métricas fica aqui, para que
nenhum outro módulo tenha esses valores "chumbados" no código.
"""
from datetime import date
from pathlib import Path

# --- Missão ---------------------------------------------------------------
# Data do lançamento (dia 0), conforme a aba Leia-me da planilha.
# Antes dela, o dia é negativo = fase Terra (construção do padrão pessoal).
LAUNCH_DATE = date(2027, 3, 1)

# --- Fontes de dados -----------------------------------------------------
# Cada fonte tem padrões e comparações PRÓPRIOS. Nunca se misturam:
# o "AST-01" da planilha (simulado) não é a mesma pessoa do "AST-01" do áudio.
SOURCES = ("planilha", "audio")

# --- Caminhos -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
EXCEL_PATH = BASE_DIR / "data" / "raw" / "BioShield_dados_simulados.xlsx"

AUDIO_DIR = BASE_DIR / "data" / "audio"            # gravações dos reportes
PROCESSED_DIR = BASE_DIR / "data" / "processed"    # métricas extraídas
AUDIO_METRICS_CSV = PROCESSED_DIR / "reportes_audio.csv"

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
}

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

# Aviso fixo, anexado pelo código (nunca depende da IA)
DISCLAIMER = (
    "Este relatório descreve desvios estatísticos em relação ao padrão pessoal "
    "e à tripulação. Não é diagnóstico clínico nem psicológico; a interpretação "
    "cabe a um profissional de saúde qualificado."
)

# --- Áudio ----------------------------------------------------------------
SAMPLE_RATE = 16000       # 16 kHz mono: padrão para análise de fala
PITCH_FLOOR_HZ = 75       # faixa de busca do tom de voz humano
PITCH_CEILING_HZ = 500
SILENCE_DB_BELOW_PEAK = 25  # quanto abaixo do pico conta como silêncio
MIN_PAUSE_S = 0.25          # silêncios menores que isso não são pausa

if __name__ == "__main__":
    print(EXCEL_PATH, EXCEL_PATH.exists())