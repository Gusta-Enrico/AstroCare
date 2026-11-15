"""Recursos comuns dos PDFs do AstroCare (relatório da Parte 1 e registro de emergência).

Fonte: Arial (Windows) ou DejaVu (Linux), que têm acentos e símbolos como ⚠, ≥ e µ.
Sem nenhuma delas, cai na Helvetica do PDF e troca os símbolos que ela não tem.
"""
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

_CANDIDATES = [
    (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
    (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
     Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
]


def _register_font() -> tuple[str, str]:
    for regular, bold in _CANDIDATES:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("AC", str(regular)))
            pdfmetrics.registerFont(TTFont("AC-Bold", str(bold)))
            return "AC", "AC-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = _register_font()


def clean(text) -> str:
    """Escapa o texto para o PDF; sem fonte TTF, troca símbolos que a Helvetica não tem."""
    text = "" if text is None else str(text)
    if FONT == "Helvetica":
        for a, b in (("⚠", "(!)"), ("≥", ">="), ("≤", "<="), ("–", "-"), ("—", "-"),
                     ("µ", "u"), ("●", "o"), ("▲", "^"), ("◆", "*"), ("○", "o")):
            text = text.replace(a, b)
    return escape(text)