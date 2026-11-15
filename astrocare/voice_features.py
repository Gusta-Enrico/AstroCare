"""Métricas acústicas da voz, extraídas de um arquivo WAV.

Usa o Praat (via parselmouth), ferramenta de referência em pesquisa de voz.
Tudo roda localmente: o áudio não sai da máquina nesta etapa.

Métricas geradas (mesmos nomes das colunas da planilha):
- duracao_video_s        duração da fala, sem silêncio no início e no fim
- pausas_pct_tempo       % do tempo de fala ocupado por pausas
- tom_medio_Hz           tom de voz típico (mediana dos trechos com voz)
- variacao_tom_semitons  quanto o tom varia (desvio, em semitons)
- volume_voz_dB          intensidade média nos trechos com som

ATENÇÃO: volume em dB depende do microfone e da distância até ele.
O padrão pessoal só é comparável se gravado com o mesmo equipamento.
"""
from pathlib import Path

import numpy as np
import parselmouth

from astrocare import config


def _speech_mask(intensity: parselmouth.Intensity) -> tuple[np.ndarray, float]:
    """Marca cada quadro como som (True) ou silêncio (False)."""
    values = intensity.values[0]
    # Percentil 95 em vez do máximo: um estalo isolado não distorce o limiar
    threshold = np.percentile(values, 95) - config.SILENCE_DB_BELOW_PEAK
    return values >= threshold, intensity.time_step


def _pause_stats(is_sound: np.ndarray, step: float) -> tuple[float, float]:
    """Retorna (duração útil em s, segundos em pausa).

    Descarta o silêncio antes da primeira e depois da última fala, e só
    conta como pausa o silêncio com pelo menos MIN_PAUSE_S.
    """
    sound_idx = np.flatnonzero(is_sound)
    if sound_idx.size == 0:
        return 0.0, 0.0
    core = is_sound[sound_idx[0]: sound_idx[-1] + 1]
    duration = core.size * step

    pause_s, run = 0.0, 0
    for frame_is_sound in core:
        if frame_is_sound:
            if run * step >= config.MIN_PAUSE_S:
                pause_s += run * step
            run = 0
        else:
            run += 1
    return duration, pause_s


def extract_voice_features(wav_path: str | Path) -> dict:
    """Extrai as métricas acústicas de um arquivo WAV."""
    snd = parselmouth.Sound(str(wav_path)).convert_to_mono()

    intensity = snd.to_intensity(minimum_pitch=config.PITCH_FLOOR_HZ)
    is_sound, step = _speech_mask(intensity)
    duration, pause_s = _pause_stats(is_sound, step)
    if duration == 0:
        raise ValueError(f"Nenhuma fala detectada em {wav_path}")

    pitch = snd.to_pitch(pitch_floor=config.PITCH_FLOOR_HZ,
                         pitch_ceiling=config.PITCH_CEILING_HZ)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]  # 0 = trecho sem voz (silêncio, consoante surda)
    if f0.size == 0:
        raise ValueError(f"Nenhum trecho com voz detectado em {wav_path}")

    median_f0 = float(np.median(f0))
    semitones = 12 * np.log2(f0 / median_f0)  # semitom: escala musical, igual p/ vozes graves e agudas

    return {
        "duracao_video_s": round(duration, 1),
        "pausas_pct_tempo": round(100 * pause_s / duration, 1),
        "tom_medio_Hz": round(median_f0, 1),
        "variacao_tom_semitons": round(float(np.std(semitones)), 2),
        "volume_voz_dB": round(float(intensity.values[0][is_sound].mean()), 1),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit("Uso: python -m astrocare.voice_features caminho/do/audio.wav")
    for name, value in extract_voice_features(sys.argv[1]).items():
        print(f"{name}: {value}")