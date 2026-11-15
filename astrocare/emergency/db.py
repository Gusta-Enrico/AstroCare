"""Banco do módulo de emergência (data/emergencia.db), 100% offline."""
import sqlite3

from contextlib import contextmanager
from astrocare import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS medicamentos (
    id              INTEGER PRIMARY KEY,
    nome            TEXT NOT NULL UNIQUE COLLATE NOCASE,
    apresentacao    TEXT NOT NULL,
    unidade         TEXT NOT NULL,
    estoque_minimo  INTEGER NOT NULL DEFAULT 0 CHECK (estoque_minimo >= 0),
    criado_em       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS lotes (
    id              INTEGER PRIMARY KEY,
    medicamento_id  INTEGER NOT NULL REFERENCES medicamentos(id),
    lote            TEXT NOT NULL,
    quantidade      INTEGER NOT NULL CHECK (quantidade >= 0),
    validade        TEXT NOT NULL CHECK (validade GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    localizacao     TEXT NOT NULL,
    UNIQUE (medicamento_id, lote)
);

CREATE TABLE IF NOT EXISTS movimentacoes (
    id              INTEGER PRIMARY KEY,
    lote_id         INTEGER NOT NULL REFERENCES lotes(id),
    tipo            TEXT NOT NULL CHECK (tipo IN ('entrada', 'consumo', 'descarte')),
    quantidade      INTEGER NOT NULL CHECK (quantidade > 0),
    saldo_apos      INTEGER NOT NULL CHECK (saldo_apos >= 0),
    responsavel     TEXT NOT NULL CHECK (length(trim(responsavel)) > 0),
    motivo          TEXT NOT NULL CHECK (length(trim(motivo)) > 0),
    atendimento_id  INTEGER,
    registrado_em   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime'))
);

CREATE TRIGGER IF NOT EXISTS movimentacoes_sem_update
BEFORE UPDATE ON movimentacoes
BEGIN SELECT RAISE(ABORT, 'Histórico de movimentações não pode ser alterado'); END;

CREATE TRIGGER IF NOT EXISTS movimentacoes_sem_delete
BEFORE DELETE ON movimentacoes
BEGIN SELECT RAISE(ABORT, 'Histórico de movimentações não pode ser apagado'); END;

-- Parte 9: atendimentos ----------------------------------------------------
-- Um registro por emergência. Enquanto está 'em_andamento' pode receber
-- paciente e protocolo; depois de encerrado fica congelado.
CREATE TABLE IF NOT EXISTS atendimentos (
    id INTEGER PRIMARY KEY,
    iniciado_em TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime')),
    encerrado_em TEXT,
    ativacao TEXT NOT NULL CHECK (ativacao IN ('voz', 'manual')),
    idioma TEXT NOT NULL CHECK (idioma IN ('pt', 'en')),
    relator TEXT,
    paciente TEXT,
    protocolo_id TEXT,
    protocolo_versao TEXT,
    protocolo_status TEXT,
    protocolo_sha256 TEXT,
    status TEXT NOT NULL DEFAULT 'em_andamento' CHECK (status IN
        ('em_andamento', 'concluido', 'lacuna', 'interrompido', 'nao_identificada')),
    CHECK ((status = 'em_andamento') = (encerrado_em IS NULL)));
CREATE TRIGGER IF NOT EXISTS atendimentos_encerrado_congelado BEFORE UPDATE ON atendimentos
WHEN OLD.status <> 'em_andamento'
BEGIN SELECT RAISE(ABORT, 'Atendimento encerrado não pode ser alterado'); END;
CREATE TRIGGER IF NOT EXISTS atendimentos_sem_delete BEFORE DELETE ON atendimentos
BEGIN SELECT RAISE(ABORT, 'Atendimento não pode ser apagado'); END;

-- Linha do tempo: tudo o que aconteceu no atendimento. Só inclusão.
CREATE TABLE IF NOT EXISTS eventos (
    id INTEGER PRIMARY KEY,
    atendimento_id INTEGER NOT NULL REFERENCES atendimentos(id),
    tipo TEXT NOT NULL,
    detalhe TEXT NOT NULL CHECK (json_valid(detalhe)),
    registrado_em TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now', 'localtime')));
CREATE TRIGGER IF NOT EXISTS eventos_sem_update BEFORE UPDATE ON eventos
BEGIN SELECT RAISE(ABORT, 'Linha do tempo não pode ser alterada'); END;
CREATE TRIGGER IF NOT EXISTS eventos_sem_delete BEFORE DELETE ON eventos
BEGIN SELECT RAISE(ABORT, 'Linha do tempo não pode ser apagada'); END;
CREATE INDEX IF NOT EXISTS idx_eventos_atendimento ON eventos(atendimento_id, id);

-- Fila para a Terra: o item só sai da fila quando o envio é CONFIRMADO.
-- A única alteração permitida é preencher enviado_em, uma única vez.
CREATE TABLE IF NOT EXISTS saida (
    id INTEGER PRIMARY KEY,
    tipo TEXT NOT NULL,
    conteudo TEXT NOT NULL CHECK (json_valid(conteudo)),
    criado_em TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime')),
    enviado_em TEXT);
CREATE TRIGGER IF NOT EXISTS saida_so_marca_envio BEFORE UPDATE ON saida
WHEN OLD.enviado_em IS NOT NULL OR NEW.enviado_em IS NULL
  OR NEW.tipo IS NOT OLD.tipo OR NEW.conteudo IS NOT OLD.conteudo OR NEW.criado_em IS NOT OLD.criado_em
BEGIN SELECT RAISE(ABORT, 'Fila de saída: só é permitido marcar o envio, uma vez'); END;
CREATE TRIGGER IF NOT EXISTS saida_sem_delete BEFORE DELETE ON saida
BEGIN SELECT RAISE(ABORT, 'Fila de saída não pode ser apagada'); END;
CREATE INDEX IF NOT EXISTS idx_saida_pendente ON saida(enviado_em);
"""


def _open(path=None) -> sqlite3.Connection:
    # Usa o caminho do config.py quando nenhum outro é informado
    path = path or config.EMERGENCY_DB_PATH
    # Cria a pasta "data" se ela ainda não existir
    path.parent.mkdir(parents=True, exist_ok=True)
    # Abre o banco (ou cria o arquivo, se não existir)
    conn = sqlite3.connect(path, isolation_level=None, timeout=10)
    # Permite acessar as colunas pelo nome: linha["nome"]
    conn.row_factory = sqlite3.Row
    # Gravação mais resistente a travamentos
    conn.execute("PRAGMA journal_mode = WAL")
    # Só confirma depois de gravar no disco (protege contra queda de energia)
    conn.execute("PRAGMA synchronous = FULL")
    # Liga a checagem de relação entre tabelas (vem desligada no SQLite)
    conn.execute("PRAGMA foreign_keys = ON")
    # Cria as tabelas (só as que ainda não existem)
    conn.executescript(SCHEMA)
    return conn

class DatabaseBusy(Exception):
    """Outra operação está usando o banco há mais de 10 segundos."""


@contextmanager
def transaction(path=None):
    """Transação atômica: ou TUDO dentro do bloco é gravado, ou NADA é."""
    conn = _open(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as exc:
        conn.close()
        raise DatabaseBusy("Banco ocupado por outra operação; tente novamente.") from exc
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def read(sql: str, params: tuple = (), path=None) -> list[sqlite3.Row]:
    """Consulta somente leitura: abre, lê e fecha a conexão."""
    conn = _open(path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


if __name__ == "__main__":
    # Prepara um medicamento com dois lotes
    with transaction() as conn:
        conn.execute("INSERT INTO medicamentos (nome, apresentacao, unidade) "
                     "VALUES ('Paracetamol 500 mg', 'comprimido 500 mg', 'comprimido')")
        med_id = conn.execute("SELECT id FROM medicamentos "
                              "WHERE nome = 'Paracetamol 500 mg'").fetchone()["id"]
        conn.execute("INSERT INTO lotes (medicamento_id, lote, quantidade, validade, localizacao) "
                     "VALUES (?, 'PAR-A', 12, '2026-12-04', 'Kit médico, gaveta 1')", (med_id,))
        conn.execute("INSERT INTO lotes (medicamento_id, lote, quantidade, validade, localizacao) "
                     "VALUES (?, 'PAR-B', 30, '2027-12-19', 'Kit médico, gaveta 1')", (med_id,))

    def saldos():
        return {r["lote"]: r["quantidade"] for r in read("SELECT lote, quantidade FROM lotes")}

    print("Antes:              ", saldos())

    # 1) Falha no meio: a 1ª baixa é desfeita
    try:
        with transaction() as conn:
            conn.execute("UPDATE lotes SET quantidade = 0 WHERE lote = 'PAR-A'")
            raise RuntimeError("queda de energia simulada")
    except RuntimeError as e:
        print(f"Falha simulada ({e})")
    print("Depois da falha:    ", saldos(), "<- nada mudou")

    # 2) Tudo certo: as duas baixas são gravadas juntas
    with transaction() as conn:
        conn.execute("UPDATE lotes SET quantidade = 0 WHERE lote = 'PAR-A'")
        conn.execute("UPDATE lotes SET quantidade = 28 WHERE lote = 'PAR-B'")
    print("Depois do sucesso:  ", saldos(), "<- as duas baixas juntas")