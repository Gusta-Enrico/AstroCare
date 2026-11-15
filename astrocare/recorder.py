"""Gravador simples pelo microfone, para testar o pipeline com a sua voz.

No produto final, quem grava é o app; o backend só recebe o arquivo.

Uso:
    python -m astrocare.recorder AST-01
Fale o reporte e aperte Enter para encerrar.
"""
import sys
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd

from astrocare import config


def record_until_enter(astronauta_id: str) -> str:
    """Grava até o usuário apertar Enter. Retorna o caminho do WAV salvo."""
    chunks = []

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        chunks.append(indata.copy())

    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.AUDIO_DIR / f"{astronauta_id}_{stamp}.wav"

    with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                        dtype="int16", callback=callback):
        input("Gravando... fale o seu reporte e aperte Enter para parar.")

    audio = np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype="int16")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16 bits
        wav.setframerate(config.SAMPLE_RATE)
        wav.writeframes(audio.tobytes())

    print(f"Salvo em {path} ({len(audio) / config.SAMPLE_RATE:.1f} s)")
    return str(path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python -m astrocare.recorder AST-01")
    record_until_enter(sys.argv[1])