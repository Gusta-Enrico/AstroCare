"""Controle de estoque de medicamentos da missão (offline).

- medicamento: o item (nome, apresentação, unidade, estoque mínimo).
- lote: quantidade + validade + localização.
- movimentação: toda entrada e saída, com responsável e motivo.
"""
import argparse
from dataclasses import dataclass
from datetime import date, timedelta

from astrocare import config
from astrocare.emergency.db import read, transaction


# --------------------------------------------------------------------------
# Erros: mensagens claras, pensadas para aparecer direto para o astronauta
# --------------------------------------------------------------------------
class InventoryError(Exception):
    """Erro de estoque: a operação foi cancelada e nada foi alterado."""


class MedicationNotFound(InventoryError):
    """O medicamento pedido não está cadastrado."""

class InsufficientStock(InventoryError):
    """Não há estoque VÁLIDO suficiente para o pedido."""


@dataclass
class Withdrawal:
    """Uma retirada de um lote específico (um consumo pode usar vários lotes)."""
    lote: str
    quantidade: int
    localizacao: str
    validade: str


# --------------------------------------------------------------------------
# Validações
# --------------------------------------------------------------------------
def _require_text(value, field: str) -> str:
    """Garante que um texto obrigatório foi preenchido (sem só espaços)."""
    if value is None or not str(value).strip():
        raise InventoryError(f"'{field}' é obrigatório.")
    return str(value).strip()


def _require_positive(value, field: str) -> int:
    """Garante um número inteiro maior que zero."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InventoryError(f"'{field}' deve ser um número inteiro maior que zero.")
    return value


def _medication_id(conn, nome: str) -> int:
    """Busca o id do medicamento pelo nome; erro claro se não existir."""
    row = conn.execute("SELECT id FROM medicamentos WHERE nome = ?", (nome,)).fetchone()
    if row is None:
        raise MedicationNotFound(f"Medicamento '{nome}' não está cadastrado.")
    return row["id"]


# --------------------------------------------------------------------------
# Cadastro de medicamento
# --------------------------------------------------------------------------
def add_medication(nome, apresentacao, unidade, estoque_minimo=0) -> None:
    nome = _require_text(nome, "nome")
    if not isinstance(estoque_minimo, int) or estoque_minimo < 0:
        raise InventoryError("'estoque mínimo' deve ser um número inteiro >= 0.")

    with transaction() as conn:
        exists = conn.execute("SELECT 1 FROM medicamentos WHERE nome = ?", (nome,)).fetchone()
        if exists:
            raise InventoryError(f"Medicamento '{nome}' já está cadastrado.")
        conn.execute(
            "INSERT INTO medicamentos (nome, apresentacao, unidade, estoque_minimo) "
            "VALUES (?, ?, ?, ?)",
            (nome, _require_text(apresentacao, "apresentação"),
             _require_text(unidade, "unidade"), estoque_minimo),
        )


# --------------------------------------------------------------------------
# Entrada de lote (novo ou reposição de um lote existente)
# --------------------------------------------------------------------------
def receive_lot(nome, lote, quantidade, validade, localizacao, responsavel,
                motivo="entrada de estoque") -> None:
    quantidade = _require_positive(quantidade, "quantidade")
    lote = _require_text(lote, "lote")
    responsavel = _require_text(responsavel, "responsável")
    motivo = _require_text(motivo, "motivo")
    try:
        validade = date.fromisoformat(validade).isoformat()   # confere AAAA-MM-DD
    except (TypeError, ValueError):
        raise InventoryError(f"Validade '{validade}' inválida; use o formato AAAA-MM-DD.")

    with transaction() as conn:
        med_id = _medication_id(conn, nome)
        row = conn.execute("SELECT id, quantidade, validade FROM lotes "
                           "WHERE medicamento_id = ? AND lote = ?", (med_id, lote)).fetchone()

        if row is None:
            # Lote novo
            cur = conn.execute(
                "INSERT INTO lotes (medicamento_id, lote, quantidade, validade, localizacao) "
                "VALUES (?, ?, ?, ?, ?)",
                (med_id, lote, quantidade, validade, _require_text(localizacao, "localização")))
            lot_id, balance = cur.lastrowid, quantidade
        else:
            # Reposição de um lote que já existe: a validade tem que bater
            if row["validade"] != validade:
                raise InventoryError(
                    f"Lote '{lote}' já existe com validade {row['validade']}, não {validade}.")
            lot_id, balance = row["id"], row["quantidade"] + quantidade
            conn.execute("UPDATE lotes SET quantidade = ? WHERE id = ?", (balance, lot_id))

        # Toda entrada fica registrada no histórico
        conn.execute(
            "INSERT INTO movimentacoes (lote_id, tipo, quantidade, saldo_apos, responsavel, motivo) "
            "VALUES (?, 'entrada', ?, ?, ?, ?)",
            (lot_id, quantidade, balance, responsavel, motivo))


# --------------------------------------------------------------------------
# Consumo: FEFO (vence primeiro, sai primeiro), nunca vencido, tudo ou nada
# --------------------------------------------------------------------------
def _valid_lots(conn, med_id: int, hoje: str) -> list:
    """Lotes com saldo e dentro da validade, do que vence antes ao que vence depois."""
    return conn.execute(
        "SELECT id, lote, quantidade, validade, localizacao FROM lotes "
        "WHERE medicamento_id = ? AND quantidade > 0 AND validade >= ? "
        "ORDER BY validade, id",
        (med_id, hoje)).fetchall()


def _plan(lots, quantidade: int) -> list[tuple]:
    """Decide quanto tirar de cada lote. Retorna [(lote, quanto_tirar), ...]."""
    plan, remaining = [], quantidade
    for lot in lots:
        if remaining == 0:
            break
        take = min(remaining, lot["quantidade"])
        plan.append((lot, take))
        remaining -= take
    return plan


def _insufficient(conn, nome, med_id, quantidade, available, hoje) -> InsufficientStock:
    """Monta a mensagem de estoque insuficiente, avisando se há unidades vencidas."""
    expired = conn.execute(
        "SELECT COALESCE(SUM(quantidade), 0) FROM lotes "
        "WHERE medicamento_id = ? AND validade < ?", (med_id, hoje)).fetchone()[0]
    note = f" ({expired} unidade(s) vencida(s) não podem ser usadas)" if expired else ""
    return InsufficientStock(f"Estoque insuficiente de '{nome}': pedido {quantidade}, "
                             f"disponível {available}{note}. Nada foi baixado.")


def preview(nome, quantidade, hoje: date | None = None) -> list[Withdrawal]:
    """De onde SERIA retirado, SEM baixar nada (para orientar antes do uso)."""
    quantidade = _require_positive(quantidade, "quantidade")
    hoje = (hoje or date.today()).isoformat()
    with transaction() as conn:   # só leitura; a transação garante uma visão consistente
        med_id = _medication_id(conn, nome)
        lots = _valid_lots(conn, med_id, hoje)
        available = sum(lot["quantidade"] for lot in lots)
        if available < quantidade:
            raise _insufficient(conn, nome, med_id, quantidade, available, hoje)
        return [Withdrawal(lot["lote"], take, lot["localizacao"], lot["validade"])
                for lot, take in _plan(lots, quantidade)]


def consume(nome, quantidade, responsavel, motivo, atendimento_id=None,
            hoje: date | None = None) -> list[Withdrawal]:
    """Baixa a quantidade pedida, lote a lote. Recusa TUDO se não houver estoque válido."""
    quantidade = _require_positive(quantidade, "quantidade")
    responsavel = _require_text(responsavel, "responsável")
    motivo = _require_text(motivo, "motivo")
    hoje = (hoje or date.today()).isoformat()

    with transaction() as conn:
        med_id = _medication_id(conn, nome)
        lots = _valid_lots(conn, med_id, hoje)
        available = sum(lot["quantidade"] for lot in lots)
        if available < quantidade:
            raise _insufficient(conn, nome, med_id, quantidade, available, hoje)

        withdrawals = []
        for lot, take in _plan(lots, quantidade):
            balance = lot["quantidade"] - take
            conn.execute("UPDATE lotes SET quantidade = ? WHERE id = ?", (balance, lot["id"]))
            conn.execute(
                "INSERT INTO movimentacoes (lote_id, tipo, quantidade, saldo_apos, "
                "responsavel, motivo, atendimento_id) VALUES (?, 'consumo', ?, ?, ?, ?, ?)",
                (lot["id"], take, balance, responsavel, motivo, atendimento_id))
            withdrawals.append(Withdrawal(lot["lote"], take, lot["localizacao"], lot["validade"]))
    return withdrawals


# --------------------------------------------------------------------------
# Descarte de vencidos
# --------------------------------------------------------------------------
def discard_expired(responsavel, hoje: date | None = None) -> int:
    """Zera os lotes vencidos e registra o descarte. Retorna as unidades descartadas."""
    responsavel = _require_text(responsavel, "responsável")
    hoje = (hoje or date.today()).isoformat()
    total = 0
    with transaction() as conn:
        expired = conn.execute("SELECT id, quantidade FROM lotes "
                               "WHERE validade < ? AND quantidade > 0", (hoje,)).fetchall()
        for lot in expired:
            conn.execute("UPDATE lotes SET quantidade = 0 WHERE id = ?", (lot["id"],))
            conn.execute(
                "INSERT INTO movimentacoes (lote_id, tipo, quantidade, saldo_apos, responsavel, motivo) "
                "VALUES (?, 'descarte', ?, 0, ?, 'lote vencido')",
                (lot["id"], lot["quantidade"], responsavel))
            total += lot["quantidade"]
    return total


# --------------------------------------------------------------------------
# Consultas: situação do estoque, alertas e histórico
# --------------------------------------------------------------------------
def stock(hoje: date | None = None) -> list[dict]:
    """Situação de cada medicamento: estoque válido, vencido e próximo vencimento."""
    hoje = (hoje or date.today()).isoformat()
    rows = read(
        """SELECT m.nome, m.unidade, m.estoque_minimo,
                  COALESCE(SUM(CASE WHEN l.validade >= ? THEN l.quantidade END), 0) AS valido,
                  COALESCE(SUM(CASE WHEN l.validade <  ? THEN l.quantidade END), 0) AS vencido,
                  MIN(CASE WHEN l.validade >= ? AND l.quantidade > 0 THEN l.validade END)
                      AS proxima_validade,
                  GROUP_CONCAT(DISTINCT CASE WHEN l.quantidade > 0 THEN l.localizacao END)
                      AS locais
           FROM medicamentos m LEFT JOIN lotes l ON l.medicamento_id = m.id
           GROUP BY m.id
           ORDER BY m.nome""",
        (hoje, hoje, hoje))
    return [dict(r) for r in rows]


SEVERITY = {"INDISPONIVEL": 0, "VENCIDO": 1, "ESTOQUE_BAIXO": 2, "VENCE_EM_BREVE": 3}


def alerts(hoje: date | None = None, dias_aviso: int = config.EXPIRY_WARNING_DAYS) -> list[dict]:
    """Alertas do mais grave para o menos grave."""
    hoje = hoje or date.today()
    limite = (hoje + timedelta(days=dias_aviso)).isoformat()
    found = []

    for item in stock(hoje):
        if item["valido"] == 0:
            found.append(("INDISPONIVEL", item["nome"], "sem unidades válidas em estoque"))
        elif item["valido"] <= item["estoque_minimo"]:
            found.append(("ESTOQUE_BAIXO", item["nome"],
                          f"{item['valido']} {item['unidade']}(s); mínimo {item['estoque_minimo']}"))
        if item["vencido"] > 0:
            found.append(("VENCIDO", item["nome"],
                          f"{item['vencido']} {item['unidade']}(s) vencida(s) aguardando descarte"))

    soon = read(
        """SELECT m.nome, l.lote, l.quantidade, l.validade
           FROM lotes l JOIN medicamentos m ON m.id = l.medicamento_id
           WHERE l.quantidade > 0 AND l.validade >= ? AND l.validade <= ?
           ORDER BY l.validade""",
        (hoje.isoformat(), limite))
    for row in soon:
        found.append(("VENCE_EM_BREVE", row["nome"],
                      f"lote {row['lote']} ({row['quantidade']}) vence em {row['validade']}"))

    found.sort(key=lambda a: SEVERITY[a[0]])
    return [{"tipo": t, "medicamento": n, "detalhe": d} for t, n, d in found]


def history(nome: str | None = None, limite: int = 50) -> list[dict]:
    """Últimas movimentações (todas, ou de um medicamento)."""
    sql = """SELECT mv.registrado_em, m.nome, l.lote, mv.tipo, mv.quantidade,
                    mv.saldo_apos, mv.responsavel, mv.motivo, mv.atendimento_id
             FROM movimentacoes mv
             JOIN lotes l ON l.id = mv.lote_id
             JOIN medicamentos m ON m.id = l.medicamento_id"""
    params: tuple = ()
    if nome:
        sql += " WHERE m.nome = ?"
        params = (nome,)
    sql += " ORDER BY mv.id DESC LIMIT ?"
    return [dict(r) for r in read(sql, params + (limite,))]


# --------------------------------------------------------------------------
# Carga de exemplo: estoque FICTÍCIO para testes e apresentação
# --------------------------------------------------------------------------
def load_example(hoje: date | None = None) -> None:
    hoje = hoje or date.today()
    d = lambda dias: (hoje + timedelta(days=dias)).isoformat()   # data relativa a hoje
    items = [
        # nome, apresentação, unidade, mínimo, [(lote, quantidade, validade, local)]
        ("Paracetamol 500 mg", "comprimido 500 mg", "comprimido", 10,
         [("PAR-A", 12, d(20), "Kit médico, gaveta 1"),
          ("PAR-B", 30, d(400), "Kit médico, gaveta 1")]),
        ("Ibuprofeno 400 mg", "comprimido 400 mg", "comprimido", 10,
         [("IBU-A", 8, d(300), "Kit médico, gaveta 1")]),
        ("Prometazina 25 mg", "comprimido 25 mg", "comprimido", 6,
         [("PRO-A", 20, d(500), "Kit médico, gaveta 2")]),
        ("Soro fisiológico 0,9% 250 ml", "bolsa 250 ml", "bolsa", 2,
         [("SF-A", 4, d(700), "Módulo de suprimentos, armário B")]),
        ("Atadura de crepe 10 cm", "rolo", "rolo", 3,
         [("ATA-A", 10, d(1500), "Kit de primeiros socorros")]),
        # --- Itens citados nos protocolos do PDF (PCR, SANS, PSIQ) ---------------
        # Os nomes precisam ser IGUAIS aos do campo "materiais" dos JSON.
        # EPI-A vencido de propósito: testa que lote vencido nunca é usado.
        ("Epinefrina 1 mg", "ampola 1 mg/ml", "ampola", 4,
         [("EPI-A", 2, d(-10), "Kit ACLS, compartimento vermelho"),
          ("EPI-B", 10, d(600), "Kit ACLS, compartimento vermelho")]),
        ("Amiodarona 300 mg", "dose de 300 mg (2 ampolas de 150 mg/3 ml)", "dose", 1,
         [("AMI-A", 3, d(500), "Kit ACLS, compartimento vermelho")]),
        ("Acetazolamida 250 mg", "comprimido 250 mg", "comprimido", 10,
         [("ACZ-A", 40, d(450), "Kit médico, gaveta 3")]),
        ("Lorazepam 1 mg VO", "comprimido 1 mg", "comprimido", 4,
         [("LZP-VO-A", 10, d(400), "Kit médico, cofre de controlados")]),
        ("Lorazepam 2 mg IM", "ampola 2 mg/ml", "ampola", 2,
         [("LZP-IM-A", 4, d(400), "Kit médico, cofre de controlados")]),
        ("Olanzapina 5 mg orodispersível", "comprimido orodispersível 5 mg", "comprimido", 4,
         [("OLZ-A", 10, d(500), "Kit médico, cofre de controlados")]),
        ("Haloperidol 5 mg IM", "ampola 5 mg/ml", "ampola", 2,
         [("HAL-A", 4, d(500), "Kit médico, cofre de controlados")]),
        ("Midazolam 5 mg", "ampola 5 mg/ml (IM ou intranasal)", "ampola", 2,
         [("MDZ-A", 4, d(15), "Kit médico, cofre de controlados")]),
        ("Flumazenil", "ampola 0,5 mg/5 ml", "ampola", 2,
         [("FLU-A", 3, d(500), "Kit ACLS, compartimento vermelho")]),
        ("Sulfato de Magnésio", "ampola 2 g/10 ml", "ampola", 2,
         [("MGS-A", 1, d(500), "Kit ACLS, compartimento vermelho")]),
    ]
    
    for nome, apresentacao, unidade, minimo, lots in items:
        add_medication(nome, apresentacao, unidade, minimo)
        for lote, qtd, validade, local in lots:
            receive_lot(nome, lote, qtd, validade, local, "SISTEMA", "carga inicial (exemplo)")


# --------------------------------------------------------------------------
# Linha de comando
# --------------------------------------------------------------------------
def _print_table(rows: list[dict], cols: list[str]) -> None:
    if not rows:
        print("  (vazio)")
        return
    show = lambda v: "-" if v is None else str(v)
    widths = {c: max(len(c), *(len(show(r[c])) for r in rows)) for c in cols}
    print("  " + "  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  " + "  ".join(show(r[c]).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estoque de medicamentos (offline)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("carga-exemplo", help="cadastra um estoque FICTÍCIO para testes")
    sub.add_parser("estoque", help="situação de cada medicamento")
    sub.add_parser("alertas", help="estoque baixo, vencidos, vencendo, indisponíveis")
    h = sub.add_parser("historico", help="últimas movimentações")
    h.add_argument("--medicamento")
    c = sub.add_parser("consumir", help="registra um consumo")
    c.add_argument("medicamento")
    c.add_argument("quantidade", type=int)
    c.add_argument("--responsavel", required=True)
    c.add_argument("--motivo", required=True)
    dsc = sub.add_parser("descartar-vencidos", help="zera e registra os lotes vencidos")
    dsc.add_argument("--responsavel", required=True)
    args = parser.parse_args()

    try:
        if args.cmd == "carga-exemplo":
            load_example()
            print(f"Estoque de EXEMPLO cadastrado em {config.EMERGENCY_DB_PATH}")
        elif args.cmd == "estoque":
            _print_table(stock(), ["nome", "valido", "vencido", "estoque_minimo",
                                   "proxima_validade", "locais"])
        elif args.cmd == "alertas":
            found = alerts()
            print(f"{len(found)} alerta(s):" if found else "Nenhum alerta.")
            for a in found:
                print(f"  [{a['tipo']}] {a['medicamento']}: {a['detalhe']}")
        elif args.cmd == "historico":
            _print_table(history(args.medicamento),
                         ["registrado_em", "nome", "lote", "tipo", "quantidade",
                          "saldo_apos", "responsavel", "motivo"])
        elif args.cmd == "consumir":
            out = consume(args.medicamento, args.quantidade, args.responsavel, args.motivo)
            print(f"Baixa registrada: {args.quantidade} de {args.medicamento}. Retirar de:")
            for w in out:
                print(f"  {w.quantidade} do lote {w.lote} (validade {w.validade}) — {w.localizacao}")
        elif args.cmd == "descartar-vencidos":
            print(f"{discard_expired(args.responsavel)} unidade(s) vencida(s) descartada(s).")
    except InventoryError as exc:
        raise SystemExit(f"ERRO: {exc}")