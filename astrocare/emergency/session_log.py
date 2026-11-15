"""Registro dos atendimentos de emergência e fila de envio para a Terra.

- atendimentos: um registro por emergência (quem relatou, paciente, idioma,
  protocolo usado + versão + status + SHA-256, desfecho).
- eventos: linha do tempo de TUDO o que aconteceu. Só inclusão.
- saida: fila para a base em Terra. Ao encerrar, o resumo completo entra na
  fila NA MESMA TRANSAÇÃO do encerramento: ou os dois acontecem, ou nenhum.
  Sem conexão nada se perde; o item só sai da fila quando o envio é confirmado.

Uso:
    python -m astrocare.emergency.session_log simular
    python -m astrocare.emergency.session_log listar
    python -m astrocare.emergency.session_log ver 1
    python -m astrocare.emergency.session_log fila
"""
import argparse
import json

from astrocare.emergency.db import read, transaction

# Tipos de evento aceitos. Uma lista fechada evita erros de digitação
# ("lacunna") que fariam um evento sumir dos relatórios.
EVENT_TYPES = {
    "inicio",          # atendimento aberto (voz ou manual)
    "fala",            # o que o sistema ouviu (transcrição)
    "classificacao",   # o que a IA entendeu + confiança
    "confirmacao",     # resposta do astronauta ao "Confirma?"
    "paciente",        # quem é o paciente
    "protocolo",       # protocolo aberto (id, versão, sha256)
    "no",              # etapa do protocolo mostrada à tripulação
    "resposta",        # SIM/NÃO da tripulação a uma condição
    "alerta",          # ⚠ alerta obrigatório exibido
    "material",        # material usado (ou 0) e baixa no estoque
    "lacuna",          # o protocolo não define o próximo passo
    "erro",            # falha técnica (microfone, banco, estoque...)
    "fim",             # encerramento
}
FINAL_STATUS = {"concluido", "lacuna", "interrompido", "nao_identificada"}
_NOW = "strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime')"


class SessionError(Exception):
    """Atendimento inexistente, já encerrado ou dado inválido."""


def _insert_event(conn, session_id: int, tipo: str, detalhe: dict) -> None:
    if tipo not in EVENT_TYPES:
        raise SessionError(f"Tipo de evento desconhecido: '{tipo}'")
    conn.execute("INSERT INTO eventos (atendimento_id, tipo, detalhe) VALUES (?, ?, ?)",
                 (session_id, tipo, json.dumps(detalhe, ensure_ascii=False)))


def _require_open(conn, session_id: int) -> None:
    row = conn.execute("SELECT status FROM atendimentos WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        raise SessionError(f"Atendimento {session_id} não existe")
    if row["status"] != "em_andamento":
        raise SessionError(f"Atendimento {session_id} já foi encerrado ({row['status']})")


# --- Escrita --------------------------------------------------------------------
def start(idioma: str, ativacao: str, relator: str | None = None) -> int:
    """Abre um atendimento e devolve o número dele."""
    with transaction() as conn:
        cur = conn.execute("INSERT INTO atendimentos (idioma, ativacao, relator) VALUES (?, ?, ?)",
                           (idioma, ativacao, relator))
        session_id = cur.lastrowid
        _insert_event(conn, session_id, "inicio",
                      {"idioma": idioma, "ativacao": ativacao, "relator": relator})
    return session_id


def log(session_id: int, tipo: str, **detalhe) -> None:
    """Acrescenta um evento à linha do tempo (só em atendimento aberto)."""
    with transaction() as conn:
        _require_open(conn, session_id)
        _insert_event(conn, session_id, tipo, detalhe)


def set_patient(session_id: int, paciente: str) -> None:
    paciente = (paciente or "").strip()
    if not paciente:
        raise SessionError("Paciente não informado")
    with transaction() as conn:
        _require_open(conn, session_id)
        conn.execute("UPDATE atendimentos SET paciente = ? WHERE id = ?", (paciente, session_id))
        _insert_event(conn, session_id, "paciente", {"paciente": paciente})


def set_protocol(session_id: int, protocol) -> None:
    """Grava QUAL protocolo foi aberto, com versão e SHA-256 (rastreabilidade)."""
    with transaction() as conn:
        _require_open(conn, session_id)
        conn.execute("UPDATE atendimentos SET protocolo_id = ?, protocolo_versao = ?, "
                     "protocolo_status = ?, protocolo_sha256 = ? WHERE id = ?",
                     (protocol.id, protocol.versao, protocol.status, protocol.sha256, session_id))
        _insert_event(conn, session_id, "protocolo",
                      {"id": protocol.id, "versao": protocol.versao,
                       "status": protocol.status, "sha256": protocol.sha256})


def finish(session_id: int, status: str, motivo: str | None = None) -> int:
    """Encerra o atendimento e põe o resumo na fila para a Terra. Devolve o id na fila."""
    if status not in FINAL_STATUS:
        raise SessionError(f"Status final deve ser um de {sorted(FINAL_STATUS)}")
    with transaction() as conn:
        _require_open(conn, session_id)
        _insert_event(conn, session_id, "fim", {"status": status, "motivo": motivo})
        conn.execute(f"UPDATE atendimentos SET status = ?, encerrado_em = {_NOW} WHERE id = ?",
                     (status, session_id))
        summary = _summary(conn, session_id)
        cur = conn.execute("INSERT INTO saida (tipo, conteudo) VALUES ('atendimento', ?)",
                           (json.dumps(summary, ensure_ascii=False),))
        item_id = cur.lastrowid
    _write_reports(session_id)        # depois do COMMIT: o encerramento já está salvo
    return item_id


def _write_reports(session_id: int) -> None:
    """Gera os PDFs do registro (PT e EN). Uma falha aqui NUNCA desfaz o encerramento."""
    try:
        from astrocare.emergency.care_report import generate
        for path in generate(session_id):
            print(f"(registro do atendimento: {path})")
    except Exception as exc:
        print(f"(AVISO: não foi possível gerar o PDF do atendimento {session_id}: {exc})")

def close_stale(motivo: str = "sistema reiniciado com atendimento aberto") -> list[int]:
    """Encerra atendimentos que ficaram abertos (programa fechou no meio).

    Chamado na inicialização: eles viram 'interrompido' e o resumo vai para a
    fila da Terra, em vez de ficarem esquecidos como 'em_andamento'.
    """
    ids = [r["id"] for r in read("SELECT id FROM atendimentos WHERE status = 'em_andamento'")]
    for session_id in ids:
        finish(session_id, "interrompido", motivo)
    return ids


def mark_sent(item_id: int) -> None:
    """Chamado pelo módulo de comunicação SÓ depois que a Terra confirmar o recebimento."""
    with transaction() as conn:
        cur = conn.execute(f"UPDATE saida SET enviado_em = {_NOW} "
                           "WHERE id = ? AND enviado_em IS NULL", (item_id,))
        if cur.rowcount == 0:
            raise SessionError(f"Item {item_id} não existe ou já foi enviado")


# --- Leitura --------------------------------------------------------------------
def _summary(conn, session_id: int) -> dict:
    """Resumo completo: dados do atendimento + linha do tempo + materiais baixados."""
    att = dict(conn.execute("SELECT * FROM atendimentos WHERE id = ?", (session_id,)).fetchone())
    att["eventos"] = [
        {"em": e["registrado_em"], "tipo": e["tipo"], **json.loads(e["detalhe"])}
        for e in conn.execute("SELECT * FROM eventos WHERE atendimento_id = ? ORDER BY id",
                              (session_id,))]
    att["consumos"] = [dict(r) for r in conn.execute(
        """SELECT m.nome, l.lote, mv.quantidade, mv.responsavel, mv.registrado_em
           FROM movimentacoes mv JOIN lotes l ON l.id = mv.lote_id
           JOIN medicamentos m ON m.id = l.medicamento_id
           WHERE mv.atendimento_id = ? ORDER BY mv.id""", (session_id,))]
    return att


def get(session_id: int) -> dict:
    rows = read("SELECT * FROM atendimentos WHERE id = ?", (session_id,))
    if not rows:
        raise SessionError(f"Atendimento {session_id} não existe")
    return dict(rows[0])


def timeline(session_id: int) -> list[dict]:
    return [{"em": r["registrado_em"], "tipo": r["tipo"], **json.loads(r["detalhe"])}
            for r in read("SELECT registrado_em, tipo, detalhe FROM eventos "
                          "WHERE atendimento_id = ? ORDER BY id", (session_id,))]


def sessions(limite: int = 20) -> list[dict]:
    return [dict(r) for r in read(
        "SELECT id, iniciado_em, status, protocolo_id, paciente, idioma, ativacao "
        "FROM atendimentos ORDER BY id DESC LIMIT ?", (limite,))]


def pending_outbox() -> list[dict]:
    return [dict(r) for r in read("SELECT id, criado_em, tipo FROM saida "
                                  "WHERE enviado_em IS NULL ORDER BY id")]


# --- Linha de comando -----------------------------------------------------------
def _simulate() -> None:
    """Atendimento FICTÍCIO de ponta a ponta, só para testar o registro."""
    from astrocare.emergency.inventory import consume
    from astrocare.emergency.protocols import load_all

    protocols, _ = load_all()
    pcr = protocols["PCR"]
    sid = start("pt", "manual", relator="AST-01")
    log(sid, "fala", texto="parada cardíaca", origem="SIMULACAO")
    log(sid, "classificacao", protocolo="PCR", confianca=0.93)
    log(sid, "confirmacao", resposta="sim")
    set_patient(sid, "AST-03")
    set_protocol(sid, pcr)
    log(sid, "no", no="PCR-04B")
    consume("Epinefrina 1 mg", 1, responsavel="AST-01", motivo="PCR-04B (SIMULACAO)",
            atendimento_id=sid)
    log(sid, "material", item="Epinefrina 1 mg", quantidade=1, no="PCR-04B")
    item = finish(sid, "concluido")
    print(f"Atendimento {sid} simulado e encerrado. Resumo na fila de saída (item {item}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Registro de atendimentos de emergência")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("simular", help="cria um atendimento FICTÍCIO para teste")
    sub.add_parser("listar", help="últimos atendimentos")
    ver = sub.add_parser("ver", help="linha do tempo de um atendimento")
    ver.add_argument("id", type=int)
    sub.add_parser("fila", help="itens aguardando envio para a Terra")
    args = parser.parse_args()

    try:
        if args.cmd == "simular":
            _simulate()
        elif args.cmd == "listar":
            for s in sessions():
                print(f"  #{s['id']}  {s['iniciado_em']}  {s['status']:<16} "
                      f"protocolo={s['protocolo_id'] or '-'}  paciente={s['paciente'] or '-'}  "
                      f"{s['idioma']}/{s['ativacao']}")
        elif args.cmd == "ver":
            att = get(args.id)
            print(f"Atendimento #{att['id']} — {att['status']}  (protocolo {att['protocolo_id'] or '-'} "
                  f"v{att['protocolo_versao'] or '-'}, sha256={(att['protocolo_sha256'] or '-')[:12]})")
            for e in timeline(args.id):
                em, tipo = e.pop("em"), e.pop("tipo")
                print(f"  {em}  {tipo:<13} {json.dumps(e, ensure_ascii=False)}")
        else:
            pend = pending_outbox()
            print(f"{len(pend)} item(ns) aguardando envio para a Terra")
            for p in pend:
                print(f"  item {p['id']}  {p['criado_em']}  {p['tipo']}")
    except SessionError as exc:
        raise SystemExit(f"ERRO: {exc}")