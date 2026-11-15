"""Registro do atendimento (estilo prontuário) em PDF, em português e inglês.

Gerado automaticamente quando um atendimento termina (session_log.finish) e
também sob demanda pela linha de comando. 100% offline (reportlab).

Conteúdo: identificação, protocolo (versão + SHA-256), desfecho, medicamentos
utilizados (lote, validade, local, responsável, etapa) e a linha do tempo
completa. Os textos clínicos vêm do JSON do protocolo no idioma do relatório;
falas do astronauta aparecem como foram transcritas (sem tradução).

O registro DESCREVE o que foi feito; não é diagnóstico.

Uso:
    python -m astrocare.emergency.care_report gerar 5
    python -m astrocare.emergency.care_report gerar 5 --idioma en
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from astrocare import config
from astrocare.emergency import session_log
from astrocare.emergency.db import read
from astrocare.emergency.protocols import load_all, pick

REPORT_DIR = config.REPORTS_DIR / "emergencias"

TXT = {
    "pt": {
        "titulo": "AstroCare — Registro de Atendimento de Emergência",
        "aviso": "Registro gerado automaticamente a partir dos dados do atendimento. Descreve as "
                 "ações executadas segundo o protocolo; não constitui diagnóstico.",
        "aviso_demo": "PROTOCOLO DE DEMONSTRAÇÃO — conteúdo pendente de validação clínica.",
        "s_ident": "1. Identificação", "s_proto": "2. Protocolo", "s_desfecho": "3. Desfecho",
        "s_meds": "4. Medicamentos utilizados", "s_alertas": "5. Alertas e lacunas",
        "s_linha": "6. Linha do tempo",
        "atendimento": "Atendimento nº", "paciente": "Paciente", "relator": "Relatado por",
        "inicio": "Início", "fim": "Término", "duracao": "Duração", "ativacao": "Ativação",
        "idioma": "Idioma do atendimento", "protocolo": "Protocolo", "versao": "Versão",
        "status_proto": "Status do protocolo", "fonte": "Fonte", "status": "Situação",
        "motivo": "Motivo", "nenhum_proto": "Nenhum protocolo foi aberto.",
        "med": "Medicamento", "qtd": "Qtd", "lote": "Lote", "validade": "Validade",
        "local": "Local", "resp": "Responsável", "hora": "Hora", "etapa": "Etapa",
        "sem_meds": "Nenhum medicamento foi baixado do estoque neste atendimento.",
        "nao_usados": "Materiais previstos e informados como NÃO utilizados (0)",
        "alerta": "Alerta obrigatório exibido", "lacuna": "Lacuna do protocolo",
        "lacuna_txt": "O protocolo não define o próximo passo; equipe médica deve ser acionada.",
        "sem_alertas": "Nenhum alerta obrigatório ou lacuna neste atendimento.",
        "evento": "Evento", "detalhe": "Detalhe",
        "gerado": "Gerado em {d} — AstroCare (offline)", "pagina": "Página {n}",
        "voz": "voz", "manual": "teclado (manual)", "nao_ident": "não identificado",
        "status_map": {"concluido": "Concluído (protocolo seguido até o fim)",
                       "lacuna": "Encerrado em lacuna do protocolo (equipe médica acionada)",
                       "interrompido": "Interrompido",
                       "nao_identificada": "Emergência não identificada",
                       "em_andamento": "Em andamento"},
        "ev": {"inicio": "Atendimento aberto", "fala": "Fala transcrita",
               "classificacao": "IA", "confirmacao": "Confirmação", "paciente": "Paciente",
               "protocolo": "Protocolo aberto", "no": "Etapa", "resposta": "Resposta",
               "alerta": "Alerta", "material": "Material", "lacuna": "Lacuna",
               "erro": "Erro/aviso técnico", "fim": "Encerramento"},
        "sugestao": "sugestão {p} ({c:.0%})", "nao_entendeu": "não entendido ({m})",
        "modo_teclado": "modo teclado liberado após {n} tentativa(s)",
        "nao_identificada": "emergência não identificada após {n} tentativa(s)",
        "usado": "{q} unidade(s) — lotes: {l}", "nao_usado": "0 (não utilizado)",
        "sim": "SIM", "nao": "NÃO", "nao_entendida": "não entendida",
    },
    "en": {
        "titulo": "AstroCare — Emergency Care Record",
        "aviso": "Record generated automatically from the care session data. It describes the "
                 "actions performed according to the protocol; it is not a diagnosis.",
        "aviso_demo": "DEMONSTRATION PROTOCOL — content pending clinical validation.",
        "s_ident": "1. Identification", "s_proto": "2. Protocol", "s_desfecho": "3. Outcome",
        "s_meds": "4. Medications used", "s_alertas": "5. Alerts and gaps",
        "s_linha": "6. Timeline",
        "atendimento": "Care session #", "paciente": "Patient", "relator": "Reported by",
        "inicio": "Start", "fim": "End", "duracao": "Duration", "ativacao": "Activation",
        "idioma": "Session language", "protocolo": "Protocol", "versao": "Version",
        "status_proto": "Protocol status", "fonte": "Source", "status": "Outcome",
        "motivo": "Reason", "nenhum_proto": "No protocol was opened.",
        "med": "Medication", "qtd": "Qty", "lote": "Lot", "validade": "Expiry",
        "local": "Location", "resp": "Responsible", "hora": "Time", "etapa": "Step",
        "sem_meds": "No medication was taken from inventory during this session.",
        "nao_usados": "Planned supplies reported as NOT used (0)",
        "alerta": "Mandatory alert shown", "lacuna": "Protocol gap",
        "lacuna_txt": "The protocol does not define the next step; the medical team must be contacted.",
        "sem_alertas": "No mandatory alerts or gaps in this session.",
        "evento": "Event", "detalhe": "Detail",
        "gerado": "Generated on {d} — AstroCare (offline)", "pagina": "Page {n}",
        "voz": "voice", "manual": "keyboard (manual)", "nao_ident": "not identified",
        "status_map": {"concluido": "Completed (protocol followed to the end)",
                       "lacuna": "Ended at a protocol gap (medical team contacted)",
                       "interrompido": "Interrupted",
                       "nao_identificada": "Emergency not identified",
                       "em_andamento": "In progress"},
        "ev": {"inicio": "Session opened", "fala": "Transcribed speech",
               "classificacao": "AI", "confirmacao": "Confirmation", "paciente": "Patient",
               "protocolo": "Protocol opened", "no": "Step", "resposta": "Answer",
               "alerta": "Alert", "material": "Supply", "lacuna": "Gap",
               "erro": "Technical error/notice", "fim": "Closing"},
        "sugestao": "suggested {p} ({c:.0%})", "nao_entendeu": "not understood ({m})",
        "modo_teclado": "keyboard mode enabled after {n} attempt(s)",
        "nao_identificada": "emergency not identified after {n} attempt(s)",
        "usado": "{q} unit(s) — lots: {l}", "nao_usado": "0 (not used)",
        "sim": "YES", "nao": "NO", "nao_entendida": "not understood",
    },
}


# --- Fonte com acentos e símbolos (⚠, ≥, –) -------------------------------------------
def _register_font() -> tuple[str, str]:
    """Usa Arial (Windows) ou DejaVu (Linux). Sem elas, Helvetica (sem alguns símbolos)."""
    candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("AC", str(regular)))
            pdfmetrics.registerFont(TTFont("AC-Bold", str(bold)))
            return "AC", "AC-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = _register_font()


def _clean(text) -> str:
    """Escapa para o PDF. Sem fonte TTF, troca símbolos que a Helvetica não tem."""
    text = "" if text is None else str(text)
    if FONT == "Helvetica":
        for a, b in (("⚠", "(!)"), ("≥", ">="), ("≤", "<="), ("–", "-"), ("—", "-")):
            text = text.replace(a, b)
    return escape(text)


# --- Dados ----------------------------------------------------------------------------
def _consumptions(session_id: int) -> list[dict]:
    return [dict(r) for r in read(
        """SELECT m.nome, mv.quantidade, l.lote, l.validade, l.localizacao, mv.responsavel,
                  mv.motivo, mv.registrado_em
           FROM movimentacoes mv JOIN lotes l ON l.id = mv.lote_id
           JOIN medicamentos m ON m.id = l.medicamento_id
           WHERE mv.atendimento_id = ? AND mv.tipo = 'consumo' ORDER BY mv.id""", (session_id,))]


def _hour(ts: str | None) -> str:
    return ts[11:19] if ts else "-"


def _when(ts: str | None) -> str:
    return ts.replace("T", " ")[:19] if ts else "-"


def _duration(start: str, end: str | None) -> str:
    if not end:
        return "-"
    secs = int((datetime.fromisoformat(end) - datetime.fromisoformat(start[:19])).total_seconds())
    return f"{secs // 60} min {secs % 60:02d} s"


def _event_detail(e: dict, lang: str, t: dict, protocol) -> str:
    tipo = e["tipo"]
    node = protocol.nos.get(e.get("no")) if protocol and e.get("no") else None
    if tipo == "inicio":
        return f"{t['ativacao']}: {t['voz'] if e.get('ativacao') == 'voz' else t['manual']}"
    if tipo == "fala":
        return f"“{e.get('texto') or '—'}”"
    if tipo == "classificacao":
        if e.get("resultado") == "modo_teclado":
            return t["modo_teclado"].format(n=e.get("tentativas"))
        if e.get("resultado") == "nao_identificada":
            return t["nao_identificada"].format(n=e.get("tentativas"))
        if e.get("rotulo"):
            return t["sugestao"].format(p=e["rotulo"], c=e.get("confianca", 0))
        return t["nao_entendeu"].format(m=e.get("motivo"))
    if tipo == "confirmacao":
        resp = t.get(e.get("resposta"), e.get("resposta"))
        alvo = e.get("protocolo") or e.get("etapa") or ""
        return f"{alvo}: {resp}" + (f" ({e['metodo']})" if e.get("metodo") else "")
    if tipo == "paciente":
        return e.get("paciente", "")
    if tipo == "protocolo":
        return f"{e.get('id')} v{e.get('versao')} [{e.get('status')}]"
    if tipo == "no":
        return f"{e.get('no')} — {pick(node.titulo, lang)[0]}" if node else str(e.get("no"))
    if tipo == "resposta":
        cond = e.get("condicao")
        if node and node.transicao.casos:          # condição no idioma do relatório
            for case in node.transicao.casos:
                if pick(case.condicao, "pt")[0] == cond:
                    cond = pick(case.condicao, lang)[0]
        resp = t["sim"] if e.get("resposta") == "SIM" else t["nao"]
        return f"{e.get('no')}: {cond} → {resp}"
    if tipo == "alerta":
        text = e.get("texto")
        if node and node.alerta:
            text = pick(node.alerta, lang)[0]
        return f"{e.get('no')}: ⚠ {text}"
    if tipo == "material":
        if not e.get("quantidade"):
            return f"{e.get('no')}: {e.get('item')} — {t['nao_usado']}"
        lots = ", ".join(f"{l['lote']} ({l['quantidade']})" for l in e.get("lotes", []))
        return f"{e.get('no')}: {e.get('item')} — " + t["usado"].format(q=e["quantidade"], l=lots)
    if tipo == "lacuna":
        return f"{e.get('no')}: {t['lacuna_txt']}"
    if tipo == "erro":
        return " — ".join(str(v) for k, v in e.items() if k not in ("em", "tipo") and v)
    if tipo == "fim":
        status = t["status_map"].get(e.get("status"), e.get("status"))
        return status + (f" — {e['motivo']}" if e.get("motivo") else "")
    return json.dumps({k: v for k, v in e.items() if k not in ("em", "tipo")}, ensure_ascii=False)


# --- PDF --------------------------------------------------------------------------------
def build_pdf(session_id: int, lang: str) -> Path:
    t = TXT[lang]
    att = session_log.get(session_id)
    events = session_log.timeline(session_id)
    protocols, _ = load_all()
    protocol = protocols.get(att["protocolo_id"]) if att["protocolo_id"] else None
    # Se o protocolo mudou depois do atendimento, não usa textos de outra versão.
    if protocol and protocol.sha256 != att["protocolo_sha256"]:
        protocol = None

    styles = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=styles["Normal"], fontName=FONT, fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10)
    h1 = ParagraphStyle("h1", parent=body, fontName=FONT_BOLD, fontSize=15, leading=19,
                        spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=body, fontName=FONT_BOLD, fontSize=11, leading=14,
                        spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#1F3A5F"))
    warn = ParagraphStyle("w", parent=body, fontName=FONT_BOLD, textColor=colors.HexColor("#9A3412"))
    P = lambda text, st=body: Paragraph(_clean(text), st)

    def kv_table(rows):
        tbl = Table([[P(k, small), P(v)] for k, v in rows], colWidths=[45 * mm, 125 * mm])
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), FONT), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
        ]))
        return tbl

    def grid(header, rows, widths):
        data = [[P(h, small) for h in header]] + [[P(c, small) for c in r] for r in rows]
        tbl = Table(data, colWidths=widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BBBBBB")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return tbl

    story = [P(t["titulo"], h1), P(t["aviso"], small)]
    if att["protocolo_status"] and att["protocolo_status"] != "VALIDADO":
        story.append(Spacer(1, 3))
        story.append(P(t["aviso_demo"], warn))

    # 1. Identificação
    story.append(P(t["s_ident"], h2))
    story.append(kv_table([
        (t["atendimento"], att["id"]),
        (t["paciente"], att["paciente"] if att["paciente"] not in (None, "NAO_IDENTIFICADO")
         else t["nao_ident"]),
        (t["relator"], att["relator"] or "-"),
        (t["inicio"], _when(att["iniciado_em"])),
        (t["fim"], _when(att["encerrado_em"])),
        (t["duracao"], _duration(att["iniciado_em"], att["encerrado_em"])),
        (t["ativacao"], t["voz"] if att["ativacao"] == "voz" else t["manual"]),
        (t["idioma"], att["idioma"].upper()),
    ]))

    # 2. Protocolo
    story.append(P(t["s_proto"], h2))
    if att["protocolo_id"]:
        name = protocol.name(lang) if protocol else att["protocolo_id"]
        rows = [(t["protocolo"], f"{att['protocolo_id']} — {name}"),
                (t["versao"], att["protocolo_versao"]),
                (t["status_proto"], att["protocolo_status"]),
                ("SHA-256", att["protocolo_sha256"])]
        if protocol:
            rows.append((t["fonte"], pick(protocol.fonte, lang)[0]))
        story.append(kv_table(rows))
    else:
        story.append(P(t["nenhum_proto"]))

    # 3. Desfecho
    fim = next((e for e in reversed(events) if e["tipo"] == "fim"), {})
    story.append(P(t["s_desfecho"], h2))
    story.append(kv_table([(t["status"], t["status_map"].get(att["status"], att["status"])),
                           (t["motivo"], fim.get("motivo") or "-")]))

    # 4. Medicamentos
    story.append(P(t["s_meds"], h2))
    used = _consumptions(session_id)
    if used:
        story.append(grid(
            [t["hora"], t["med"], t["qtd"], t["lote"], t["validade"], t["local"], t["resp"], t["etapa"]],
            [[_hour(u["registrado_em"]), u["nome"], u["quantidade"], u["lote"], u["validade"],
              u["localizacao"], u["responsavel"], u["motivo"]] for u in used],
            [18 * mm, 28 * mm, 10 * mm, 17 * mm, 19 * mm, 32 * mm, 24 * mm, 22 * mm]))
    else:
        story.append(P(t["sem_meds"]))
    zero = [e for e in events if e["tipo"] == "material" and not e.get("quantidade")]
    if zero:
        story.append(Spacer(1, 4))
        story.append(P(t["nao_usados"] + ": " +
                       "; ".join(f"{e.get('item')} ({e.get('no')})" for e in zero), small))

    # 5. Alertas e lacunas
    story.append(P(t["s_alertas"], h2))
    flagged = [e for e in events if e["tipo"] in ("alerta", "lacuna")]
    if flagged:
        for e in flagged:
            label = t["alerta"] if e["tipo"] == "alerta" else t["lacuna"]
            story.append(P(f"{_hour(e['em'])} — {label}: {_event_detail(e, lang, t, protocol)}"))
    else:
        story.append(P(t["sem_alertas"]))

    # 6. Linha do tempo
    story.append(P(t["s_linha"], h2))
    story.append(grid([t["hora"], t["evento"], t["detalhe"]],
                      [[_hour(e["em"]), t["ev"].get(e["tipo"], e["tipo"]),
                        _event_detail(e, lang, t, protocol)] for e in events],
                      [18 * mm, 32 * mm, 120 * mm]))

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 7)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(20 * mm, 10 * mm, t["gerado"].format(d=generated))
        canvas.drawRightString(190 * mm, 10 * mm, t["pagina"].format(n=doc.page))
        canvas.restoreState()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"atendimento_{session_id:05d}_{lang}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title=f"{t['titulo']} {session_id}", author="AstroCare")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def generate(session_id: int) -> list[Path]:
    """Gera os dois PDFs (PT e EN) de um atendimento."""
    return [build_pdf(session_id, lang) for lang in config.LANGUAGES]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Registro do atendimento em PDF")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gerar", help="gera o PDF de um atendimento")
    g.add_argument("id", type=int)
    g.add_argument("--idioma", choices=config.LANGUAGES, help="só um idioma (padrão: os dois)")
    args = parser.parse_args()
    try:
        paths = [build_pdf(args.id, args.idioma)] if args.idioma else generate(args.id)
    except session_log.SessionError as exc:
        raise SystemExit(f"ERRO: {exc}")
    for path in paths:
        print(f"PDF gerado: {path}")