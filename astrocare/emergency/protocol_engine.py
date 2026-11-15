"""Motor que conduz um protocolo (árvore de decisão) passo a passo.

Princípios:
- O sistema NUNCA decide nada clínico. Cada condição ("Ritmo = Chocável?")
  é perguntada à TRIPULAÇÃO, que responde SIM ou NÃO.
- Todo texto mostrado vem do JSON do protocolo. As únicas frases do motor
  são as de navegação (MSG abaixo), em PT e EN, sem tradução automática.
- Alerta obrigatório (⚠) é sempre exibido e precisa de "ciente".
- Material: o motor pergunta QUANTO foi usado (0 vale) e dá baixa no estoque,
  ligada ao atendimento. Estoque insuficiente bloqueia a baixa e pergunta de novo.
- Lacuna: o protocolo não define o próximo passo -> avisa, registra e encerra
  com status 'lacuna'. Nada é inventado.
- Tudo vai para a linha do tempo do atendimento (session_log).

O motor não sabe se a interface é tela ou voz: ele devolve um Prompt e
recebe a resposta em answer(). As partes 13 e 14 (diálogo e voz) usam o
mesmo motor.

Uso (teste pelo teclado):
    python -m astrocare.emergency.protocol_engine executar PCR --paciente AST-03 --relator AST-01
"""
import argparse
from dataclasses import dataclass

from astrocare.emergency import session_log
from astrocare.emergency.inventory import InventoryError, consume, stock
from astrocare.emergency.protocols import Branch, Protocol, load_all, pick

MSG = {
    "pt": {
        "demo": "⚠ PROTOCOLO DE DEMONSTRAÇÃO — conteúdo pendente de validação clínica.",
        "regras": "Regras críticas",
        "gatilho": "Gatilho",
        "criterios": "Critérios",
        "senao": "Caminho",
        "condicao": "Condição do protocolo: {c}\nResponda SIM ou NÃO.",
        "sim_nao": "Resposta não entendida. Responda SIM ou NÃO.",
        "material": "Material: {item}\nDisponível (válido): {n} {un}. Local: {local}\n"
                    "Quantas unidades foram usadas? (0 se não usou)",
        "inteiro": "Informe um número inteiro maior ou igual a 0.",
        "nao_pergunta": "Esta etapa não é uma pergunta. Pressione Enter quando estiver ciente/concluída.",
        "sem_cadastro": "Material '{item}' não está cadastrado no estoque. Registre o uso manualmente.",
        "estoque": "Estoque: {erro}",
        "baixa": "Baixa registrada: {qtd} de {item}.",
        "lacuna": "O protocolo não define o próximo passo para esta situação. Acione a equipe médica.",
        "fim": "Protocolo encerrado.",
        "interrompido": "Atendimento interrompido.",
        "demo_fala": "Protocolo de demonstração.",
        "local_fala": "{item}: {local}.",
        "quantas_fala": "Quantas unidades de {item} usou?",
        "baixa_fala": "Registrado.",
    },
    "en": {
        "demo": "⚠ DEMONSTRATION PROTOCOL — content pending clinical validation.",
        "regras": "Critical rules",
        "gatilho": "Trigger",
        "criterios": "Criteria",
        "senao": "Path",
        "condicao": "Protocol condition: {c}\nAnswer YES or NO.",
        "sim_nao": "Answer not understood. Answer YES or NO.",
        "material": "Supply: {item}\nAvailable (valid): {n} {un}. Location: {local}\n"
                    "How many units were used? (0 if none)",
        "inteiro": "Enter a whole number greater than or equal to 0.",
        "nao_pergunta": "This step is not a question. Press Enter when acknowledged/done.",
        "sem_cadastro": "Supply '{item}' is not registered in the inventory. Record its use manually.",
        "estoque": "Inventory: {erro}",
        "baixa": "Recorded: {qtd} of {item}.",
        "lacuna": "The protocol does not define the next step for this situation. Contact the medical team.",
        "fim": "Protocol finished.",
        "interrompido": "Care session interrupted.",
        "demo_fala": "Demonstration protocol.",
        "local_fala": "{item}: {local}.",
        "quantas_fala": "How many units of {item} did you use?",
        "baixa_fala": "Recorded.",
    },
}
YES = {"sim", "s", "yes", "y"}
NO = {"não", "nao", "n", "no"}


class EngineError(Exception):
    """Uso incorreto do motor (ex.: responder depois do fim)."""


@dataclass
class Prompt:
    """O que a interface deve mostrar/falar agora."""
    tipo: str                   # info | instrucao | alerta | pergunta | quantidade | fim
    texto: str
    no: str | None = None       # nó do protocolo (ex.: "PCR-04")
    titulo: str | None = None
    status: str | None = None   # só no "fim": concluido | lacuna | interrompido
    status: str | None = None   # só no "fim": concluido | lacuna | interrompido
    fala: str | None = None     # versão curta para falar (None = falar o texto inteiro)

def short_condition(cond: str, lang: str) -> str:
    """'Ritmo = Chocável [FV ou TVSP]' -> 'Ritmo Chocável, FV ou TVSP?'

    Só tira a notação (= SIM, colchetes, AND/OR); as palavras do protocolo ficam iguais.
    """
    for mark in (" = SIM", " = YES"):
        cond = cond.replace(mark, "")
    conj = {"pt": (" e ", " ou "), "en": (" and ", " or ")}[lang]
    cond = cond.replace(" AND ", conj[0]).replace(" OR ", conj[1])
    cond = cond.replace(" = ", " ").replace(" [", ", ").replace("[", "").replace("]", "")
    return cond.strip().rstrip(".") + "?"


class ProtocolRun:
    """Uma execução de protocolo dentro de um atendimento já aberto."""

    def __init__(self, protocol: Protocol, session_id: int, lang: str, responsavel: str):
        self.protocol, self.sid, self.lang, self.resp = protocol, session_id, lang, responsavel
        self.m = MSG[lang]
        self.finished = False
        session_log.set_protocol(session_id, protocol)   # versão + SHA-256 no registro
        self._flow = self._steps()
        self.prompt: Prompt = next(self._flow)           # primeira tela

    # --- Interface pública ----------------------------------------------------
    def answer(self, raw: str = "") -> str | None:
        """Envia a resposta ao prompt atual. Devolve mensagem de erro, ou None se aceita."""
        if self.finished:
            raise EngineError("O protocolo já foi encerrado")
        value, error = self._parse(raw)
        if error:
            return error
        self.prompt = self._flow.send(value)
        return None

    def abort(self, motivo: str) -> None:
        """Interrompe (ex.: Ctrl+C, paciente transferido, falha técnica)."""
        if not self.finished:
            self.finished = True
            self._flow.close()
            session_log.finish(self.sid, "interrompido", motivo)
            self.prompt = Prompt("fim", self.m["interrompido"], status="interrompido")

    # --- Leitura das respostas ------------------------------------------------
    def _parse(self, raw: str):
        text = (raw or "").strip().lower()
        if self.prompt.tipo == "pergunta":
            if text in YES:
                return True, None
            if text in NO:
                return False, None
            return None, self.m["sim_nao"]
        if self.prompt.tipo == "quantidade":
            if text.isdigit():
                return int(text), None
            return None, self.m["inteiro"]
        # info / instrucao / alerta: Enter (ou qualquer texto) = ciente. Mas SIM/NÃO
        # aqui indica que a pessoa achou que era uma pergunta: melhor avisar.
        if text in YES or text in NO:
            return None, self.m["nao_pergunta"]
        return None, None

    # --- Fluxo (gerador): lê como o próprio protocolo ---------------------------
    def _t(self, texts) -> str | None:
        return pick(texts, self.lang)[0]

    def _steps(self):
        p = self.protocol
        header = [p.name(self.lang)]
        if p.is_demo:
            header.append(self.m["demo"])
        header.append(f"{self.m['regras']}: {self._t(p.regras_criticas)}")
        spoken = p.name(self.lang) + (f". {self.m['demo_fala']}" if p.is_demo else "")
        yield Prompt("info", "\n".join(header), fala=spoken)

        node_id = p.inicio
        while True:
            node = p.nos[node_id]
            titulo = self._t(node.titulo)
            session_log.log(self.sid, "no", no=node.id)

            parts = []
            if node.gatilho:
                parts.append(f"{self.m['gatilho']}: {self._t(node.gatilho)}")
            if node.criterios:
                parts.append(f"{self.m['criterios']}: {self._t(node.criterios)}")
            if node.acao:
                parts.append(self._t(node.acao))
            if parts:
                spoken = self._spoken_action(node.criterios, node.acao, node.materiais)
                yield Prompt("instrucao", "\n".join(parts), no=node.id, titulo=titulo, fala=spoken)
            yield from self._alert(node.alerta, node.id)
            yield from self._materials(node.materiais, node.id)

            t = node.transicao
            if t.tipo == "proximo":
                node_id = t.no
                continue
            if t.tipo == "fim":
                yield from self._end("concluido", self._t(t.texto))
                return

            branch = yield from self._choose(node.id, titulo, t.casos, t.senao)
            if branch.acao:
                label = self._t(branch.rotulo)
                text = self._t(branch.acao)
                if label:
                    text = f"{self.m['senao']}: {label}\n{text}"
                spoken = self._spoken_action(None, branch.acao, branch.materiais)
                yield Prompt("instrucao", text, no=node.id, titulo=titulo, fala=spoken)
            yield from self._alert(branch.alerta, node.id)
            yield from self._materials(branch.materiais, node.id)

            if branch.no:
                node_id = branch.no
            elif branch.fim:
                yield from self._end("concluido")
                return
            else:                                  # lacuna
                session_log.log(self.sid, "lacuna", no=node.id)
                yield from self._end("lacuna", self.m["lacuna"], motivo=f"lacuna em {node.id}")
                return

    def _choose(self, node_id, titulo, casos: list[Branch], senao: Branch):
        """Pergunta cada condição, na ordem do PDF. A 1ª com SIM vence; senão, 'senao'."""
        for case in casos:
            cond = self._t(case.condicao)
            yes = yield Prompt("pergunta", self.m["condicao"].format(c=cond), no=node_id,
                               titulo=titulo, fala=short_condition(cond, self.lang))
            session_log.log(self.sid, "resposta", no=node_id,
                            condicao=pick(case.condicao, "pt")[0], resposta="SIM" if yes else "NÃO")
            if yes:
                return case
        return senao

    def _spoken_action(self, criterios, acao, materiais: list[str]) -> str:
        """Fala curta: critérios (se houver) + ação literal + onde está cada medicamento."""
        parts = [self._t(x) for x in (criterios, acao) if x]
        for item in materiais:
            info = next((s for s in stock() if s["nome"].lower() == item.lower()), None)
            if info and info["locais"]:
                parts.append(self.m["local_fala"].format(item=item, local=info["locais"]))
        return " ".join(parts)

    def _alert(self, alerta, node_id):
        if alerta:
            session_log.log(self.sid, "alerta", no=node_id, texto=pick(alerta, "pt")[0])
            yield Prompt("alerta", f"⚠ {self._t(alerta)}", no=node_id)

    def _materials(self, items: list[str], node_id):
        for item in items:
            info = next((s for s in stock() if s["nome"].lower() == item.lower()), None)
            if info is None:
                session_log.log(self.sid, "erro", no=node_id, item=item, erro="não cadastrado")
                yield Prompt("info", self.m["sem_cadastro"].format(item=item), no=node_id)
                continue
            while True:
                qtd = yield Prompt("quantidade", self.m["material"].format(
                    item=item, n=info["valido"], un=info["unidade"], local=info["locais"] or "-"),
                    no=node_id, fala=self.m["quantas_fala"].format(item=item))
                if qtd == 0:
                    session_log.log(self.sid, "material", no=node_id, item=item, quantidade=0)
                    break
                try:
                    lots = consume(item, qtd, responsavel=self.resp,
                                   motivo=f"{self.protocol.id} {node_id}", atendimento_id=self.sid)
                except InventoryError as exc:       # nada foi baixado; pergunta de novo
                    session_log.log(self.sid, "erro", no=node_id, item=item, erro=str(exc))
                    yield Prompt("info", self.m["estoque"].format(erro=exc), no=node_id)
                    info = next(s for s in stock() if s["nome"].lower() == item.lower())
                    continue
                session_log.log(self.sid, "material", no=node_id, item=item, quantidade=qtd,
                                lotes=[{"lote": w.lote, "quantidade": w.quantidade} for w in lots])
                yield Prompt("info", self.m["baixa"].format(qtd=qtd, item=item), no=node_id,
                             fala=self.m["baixa_fala"])
                break

    def _end(self, status: str, texto: str | None = None, motivo: str | None = None):
        session_log.finish(self.sid, status, motivo)
        self.finished = True
        yield Prompt("fim", texto or self.m["fim"], status=status)


# --- Linha de comando (teste pelo teclado; a voz entra na parte 14) -------------
def _render(pr: Prompt) -> None:
    head = f"[{pr.no}] {pr.titulo}" if pr.no and pr.titulo else (f"[{pr.no}]" if pr.no else "")
    print("\n" + "-" * 70)
    if head:
        print(head)
    print(pr.texto)


def _run_cli(protocol_id: str, paciente: str, relator: str, lang: str) -> None:
    protocols, errors = load_all()
    if protocol_id not in protocols:
        raise SystemExit(f"Protocolo '{protocol_id}' não carregado. {errors}")
    sid = session_log.start(lang, "manual", relator=relator)
    session_log.set_patient(sid, paciente)
    run = ProtocolRun(protocols[protocol_id], sid, lang, responsavel=relator)
    print(f"Atendimento #{sid} aberto (Ctrl+C interrompe).")
    try:
        while True:
            _render(run.prompt)
            if run.finished:
                break
            hint = {"pergunta": "SIM/NÃO > ", "quantidade": "quantidade > "}.get(
                run.prompt.tipo, "[Enter] ciente > ")
            error = run.answer(input(hint))
            while error:
                print(error)
                error = run.answer(input(hint))
    except KeyboardInterrupt:
        run.abort("interrompido pelo operador (Ctrl+C)")
        _render(run.prompt)
    except Exception as exc:                       # falha técnica: registra antes de sair
        run.abort(f"falha técnica: {exc}")
        raise
    print(f"\nLinha do tempo: python -m astrocare.emergency.session_log ver {sid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor de protocolos de emergência")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("executar", help="conduz um protocolo pelo teclado")
    ex.add_argument("protocolo")
    ex.add_argument("--paciente", required=True)
    ex.add_argument("--relator", required=True)
    ex.add_argument("--idioma", choices=["pt", "en"], default="pt")
    args = parser.parse_args()
    _run_cli(args.protocolo, args.paciente, args.relator, args.idioma)