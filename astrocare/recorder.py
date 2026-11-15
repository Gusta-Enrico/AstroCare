"""Gravador simples pelo microfone, para testar o pipeline com a sua voz.

No produto final, quem grava é o app; o backend só recebe o arquivo.

Uso:
    python -m astrocare.recorder
Fale o reporte (começando por "Check-in diário, AST-XX") e aperte Enter.
"""
import sys
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd

from astrocare import config


def record_until_enter(prefix: str = "REPORTE") -> str:
    """Grava até o usuário apertar Enter. Retorna o caminho do WAV salvo.

    O arquivo se chama REPORTE_AAAAMMDD_HHMMSS.wav: quem está falando é
    descoberto depois, pelo ID dito no início do reporte.
    """
    chunks = []

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        chunks.append(indata.copy())

    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.AUDIO_DIR / f"{prefix}_{stamp}.wav"

    with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                        dtype="int16", callback=callback):
        input("Gravando... comece com 'Check-in diário, AST-XX' e aperte Enter para parar.")

    audio = np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype="int16")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16 bits
        wav.setframerate(config.SAMPLE_RATE)
        wav.writeframes(audio.tobytes())

    print(f"Salvo em {path} ({len(audio) / config.SAMPLE_RATE:.1f} s)")
    return str(path)


if __name__ == "__main__":
    record_until_enter()