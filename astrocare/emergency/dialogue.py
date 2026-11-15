"""Diálogo de emergência: da pergunta inicial até o protocolo, em PT ou EN.

Fluxo:
  "Qual é a emergência?"  -> o astronauta descreve com as palavras dele
  IA classifica           -> "Entendi: <protocolo>. Confirma?"
      sim                 -> "Quem é o paciente?" -> protocolo passo a passo (motor)
      não / não entendi   -> volta para "Qual é a emergência?"
  3 falhas                -> "Emergência não identificada. Repita novamente qual é a emergência."
  4ª falha                -> modo teclado: a tripulação digita o código do protocolo

Regras:
- A IA NUNCA fala opções: espera o astronauta falar e só confirma.
- Silêncio, "não entendi" e "não" no Confirma? contam como falha.
- O paciente não identificado NÃO trava o atendimento: depois de 2 tentativas
  o protocolo começa assim mesmo (e isso fica registrado).
- Tudo é registrado na linha do tempo (session_log).

O diálogo não sabe se a entrada é voz ou teclado: devolve um Turn e recebe
texto em reply(). A parte 14 liga microfone e alto-falante aqui.

Uso (teste digitando o que seria falado; Enter vazio = silêncio):
    python -m astrocare.emergency.dialogue --idioma pt
"""
import argparse
import re
from dataclasses import dataclass, field

from astrocare import config
from astrocare.emergency import session_log
from astrocare.emergency.classifier import EmergencyClassifier
from astrocare.emergency.protocol_engine import ProtocolRun
from astrocare.emergency.protocols import load_all
from astrocare.emergency.training_data import normalize

MAX_ATTEMPTS = 3            # falhas até "emergência não identificada"
MAX_UNCLEAR_CONFIRM = 3     # respostas confusas ao "Confirma?" = conta como "não"
MAX_PATIENT_ATTEMPTS = 2

MSG = {
    "pt": {
        "qual": "Qual é a emergência?",
        "nao_ouvi": "Não ouvi. Qual é a emergência?",
        "nao_entendi": "Não entendi. Qual é a emergência?",
        "confirma": "Entendi: {nome}. Confirma?",
        "sim_nao": "Responda sim ou não. Entendi: {nome}. Confirma?",
        "nao_identificada": "Emergência não identificada. Repita novamente qual é a emergência.",
        "manual": "Emergência não identificada. Escolha o protocolo pelo teclado: "
                  "digite o código ou CANCELAR.",
        "codigo_invalido": "Código não encontrado.",
        "cancelado": "Atendimento encerrado sem protocolo. Acione a equipe médica.",
        "paciente": "Quem é o paciente?",
        "paciente_rep": "Não entendi o paciente. Diga o código, por exemplo AST zero um.",
        "paciente_nao": "Paciente não identificado. Iniciando o protocolo.",
        "paciente_ok": "Paciente {code}.",
        "paciente": "Quem é o paciente? Diga o número do astronauta.",
        "paciente_rep": "Não entendi. Diga só o número do astronauta, por exemplo: três.",
        "fim_confirma": "Finalizar o atendimento? Confirma?",
        "fim_ok": "Atendimento finalizado.",
        "fim_continua": "Continuando o atendimento.",
    },
    "en": {
        "qual": "What is the emergency?",
        "nao_ouvi": "I didn't hear you. What is the emergency?",
        "nao_entendi": "I didn't understand. What is the emergency?",
        "confirma": "Understood: {nome}. Confirm?",
        "sim_nao": "Answer yes or no. Understood: {nome}. Confirm?",
        "nao_identificada": "Emergency not identified. Please say again what the emergency is.",
        "manual": "Emergency not identified. Choose the protocol on the keyboard: "
                  "type the code or CANCEL.",
        "codigo_invalido": "Code not found.",
        "cancelado": "Care session ended without a protocol. Contact the medical team.",
        "paciente": "Who is the patient?",
        "paciente_rep": "I didn't understand the patient. Say the code, for example AST zero one.",
        "paciente_nao": "Patient not identified. Starting the protocol.",
        "paciente_ok": "Patient {code}.",
        "paciente": "Who is the patient? Say the astronaut number.",
        "paciente_rep": "I didn't understand. Say only the astronaut number, for example: three.",
        "fim_confirma": "End the care session? Confirm?",
        "fim_ok": "Care session ended.",
        "fim_continua": "Continuing the care session.",
    },
}

# Palavras faladas -> resposta que o motor entende. Comparadas já normalizadas.
YES_WORDS = {"sim", "s", "confirmo", "confirma", "confirmado", "correto", "isso", "positivo",
             "afirmativo", "yes", "y", "yeah", "correct", "confirm", "confirmed", "affirmative"}
NO_WORDS = {"nao", "n", "negativo", "errado", "incorreto",
            "no", "nope", "negative", "wrong", "incorrect"}
NUMBER_WORDS = {
    "zero": 0, "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5,
    "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "none": 0,
}
CANCEL_WORDS = {"cancelar", "cancel"}
# "Finalizar atendimento": precisa de UM verbo E UM substantivo, para não disparar
# com um "finalizar" solto no meio de uma frase.
END_VERBS = {"finalizar", "encerrar", "end", "finish", "close"}
END_NOUNS = {"atendimento", "session"}

def is_end_command(text: str) -> bool:
    words = set(normalize(text).split())
    return bool(words & END_VERBS) and bool(words & END_NOUNS)


def is_end_command(text: str) -> bool:
    words = set(normalize(text).split())
    return bool(words & END_VERBS) and bool(words & END_NOUNS)


def yes_no(text: str) -> bool | None:
    """True = sim, False = não, None = não entendido (ou ambíguo, ex.: 'sim, não')."""
    words = set(normalize(text).replace(",", " ").replace(".", " ").split())
    yes, no = bool(words & YES_WORDS), bool(words & NO_WORDS)
    return None if yes == no else yes


def spoken_number(text: str) -> str:
    """'três' -> '3', 'two' -> '2'. Dígitos passam direto."""
    clean = normalize(text)
    if clean.isdigit():
        return clean
    words = clean.split()
    if len(words) == 1 and words[0] in NUMBER_WORDS:
        return str(NUMBER_WORDS[words[0]])
    return clean


def find_astronaut(text: str) -> str | None:
    """Acha o astronauta pelo NÚMERO falado e completa o código: 'três' -> 'AST-03'.

    Aceita só o número ('três', 'zero três', '3') ou o número dentro de uma frase
    ('é o três', 'AST zero três'). O zero à esquerda é ignorado. Se aparecerem
    dois números diferentes ('dois três'), não adivinha: devolve None.
    """
    words = normalize(text).replace("-", " ").split()
    numbers = set()
    for w in words:
        value = NUMBER_WORDS.get(w, w)
        digits = str(value).lstrip("0")          # "03" -> "3"; "zero" -> ""
        if digits.isdigit():
            numbers.add(int(digits))
        elif w.startswith("ast") and w[3:].lstrip("0").isdigit():   # "ast03", "AST-03"
            numbers.add(int(w[3:].lstrip("0")))
    if len(numbers) != 1:
        return None
    code = f"AST-{numbers.pop():02d}"
    return code if code in config.ASTRONAUTS else None


@dataclass
class Turn:
    """O que a interface deve falar/mostrar agora."""
    texto: str
    teclado: bool = False                    # True = modo manual (tela + teclado)
    opcoes: list[tuple[str, str]] = field(default_factory=list)   # só no modo teclado
    fim: bool = False
    fala: str | None = None                  # versão curta para falar (None = falar o texto)
    espera: str | None = None                # o que a resposta deve conter (ex.: "paciente")


class EmergencyDialog:
    def __init__(self, lang: str, ativacao: str, relator: str | None = None,
                 classifier: EmergencyClassifier | None = None):
        self.lang, self.m = lang, MSG[lang]
        self.relator = relator
        self.protocols, _ = load_all()
        self.clf = classifier or EmergencyClassifier()
        self.sid = session_log.start(lang, ativacao, relator=relator)
        self.run: ProtocolRun | None = None
        self.finished = False
        self._notice = ""
        self._paused: Turn | None = None     # turno guardado enquanto confirma o fim                    # aviso a juntar na próxima fala
        self._flow = self._steps()
        self.turn: Turn = next(self._flow)

    # --- Interface pública ----------------------------------------------------
    def reply(self, text: str) -> None:
        if self.finished:
            raise RuntimeError("Diálogo encerrado")
        if self._paused is not None:         # respondendo ao "Finalizar o atendimento?"
            self._answer_end(text)
            return
        if is_end_command(text or ""):       # vale em qualquer etapa
            session_log.log(self.sid, "fala", etapa="finalizar", texto=text)
            self._paused = self.turn
            self.turn = Turn(self.m["fim_confirma"])
            return
        self.turn = self._flow.send(text or "")

    def _answer_end(self, text: str) -> None:
        """Só 'sim' encerra. 'Não' ou resposta confusa: continua de onde parou."""
        yn = yes_no(text)
        session_log.log(self.sid, "confirmacao", etapa="finalizar", texto=text,
                        resposta={True: "sim", False: "nao", None: "nao_entendida"}[yn])
        paused, self._paused = self._paused, None
        if yn:
            self.abort("finalizado pelo astronauta (comando 'finalizar atendimento')")
            self.turn = Turn(self.m["fim_ok"], fim=True)
        else:
            self.turn = Turn(f"{self.m['fim_continua']} {paused.texto}",
                             fala=f"{self.m['fim_continua']} {paused.fala or paused.texto}",
                             teclado=paused.teclado, opcoes=paused.opcoes, espera=paused.espera)

    def abort(self, motivo: str) -> None:
        if self.finished:
            return
        self.finished = True
        self._flow.close()
        if self.run and not self.run.finished:
            self.run.abort(motivo)
        elif not self.run:
            session_log.finish(self.sid, "interrompido", motivo)
        self.turn = Turn("Atendimento interrompido." if self.lang == "pt"
                         else "Care session interrupted.", fim=True)

    # --- Fluxo ----------------------------------------------------------------
    def _say(self, text: str, fala: str | None = None, **kw) -> Turn:
        notice = self._notice
        self._notice = ""
        return Turn(f"{notice} {text}".strip(),
                    fala=f"{notice} {fala}".strip() if fala else None, **kw)

    def _steps(self):
        protocol = yield from self._identify()
        if protocol is None:
            return
        yield from self._ask_patient()
        self.run = ProtocolRun(protocol, self.sid, self.lang,
                               responsavel=self.relator or "TRIPULACAO")
        while True:
            pr = self.run.prompt
            if self.run.finished:
                self.finished = True
                yield self._say(pr.texto, fala=pr.fala, fim=True)
                return
            head = f"[{pr.no}] {pr.titulo}\n" if pr.no and pr.titulo else ""
            answer = yield self._say(head + pr.texto, fala=pr.fala)
            if pr.tipo == "pergunta":
                yn = yes_no(answer)
                answer = "sim" if yn else "não" if yn is False else answer
            elif pr.tipo == "quantidade":
                answer = spoken_number(answer)
            error = self.run.answer(answer)
            if error:
                self._notice = error

    def _identify(self):
        """Pergunta a emergência até a confirmação. Devolve o protocolo, ou None."""
        failures, last_chance, question = 0, False, self.m["qual"]
        while True:
            text = yield self._say(question)
            session_log.log(self.sid, "fala", etapa="emergencia", texto=text)
            c = self.clf.classify(text)
            session_log.log(self.sid, "classificacao", rotulo=c.rotulo, candidato=c.candidato,
                            confianca=round(c.confianca, 3), motivo=c.motivo)
            if c.rotulo in self.protocols:
                protocol = self.protocols[c.rotulo]
                if (yield from self._confirm(protocol)):
                    return protocol
                question = self.m["qual"]
            else:
                question = self.m["nao_ouvi"] if c.motivo == "texto_vazio" else self.m["nao_entendi"]

            failures += 1
            if last_chance:
                break
            if failures >= MAX_ATTEMPTS:
                session_log.log(self.sid, "classificacao", resultado="nao_identificada",
                                tentativas=failures)
                question, last_chance = self.m["nao_identificada"], True

        return (yield from self._manual(failures))

    def _confirm(self, protocol):
        name = protocol.name(self.lang)
        question = self.m["confirma"].format(nome=name)
        for _ in range(MAX_UNCLEAR_CONFIRM):
            text = yield self._say(question)
            yn = yes_no(text)
            session_log.log(self.sid, "confirmacao", protocolo=protocol.id, texto=text,
                            resposta={True: "sim", False: "nao", None: "nao_entendida"}[yn])
            if yn is not None:
                return yn
            question = self.m["sim_nao"].format(nome=name)
        return False

    def _manual(self, failures: int):
        session_log.log(self.sid, "classificacao", resultado="modo_teclado", tentativas=failures)
        options = [(pid, p.name(self.lang)) for pid, p in sorted(self.protocols.items())]
        while True:
            text = yield self._say(self.m["manual"], teclado=True, opcoes=options)
            code = (text or "").strip().upper()
            if code in self.protocols:
                session_log.log(self.sid, "confirmacao", protocolo=code, metodo="teclado",
                                resposta="sim")
                return self.protocols[code]
            if normalize(text or "") in CANCEL_WORDS:
                session_log.finish(self.sid, "nao_identificada", "cancelado no modo teclado")
                self.finished = True
                yield self._say(self.m["cancelado"], fim=True)
                return None
            self._notice = self.m["codigo_invalido"]

    def _ask_patient(self):
        question = self.m["paciente"]
        for _ in range(MAX_PATIENT_ATTEMPTS):
            text = yield self._say(question, espera="paciente")
            session_log.log(self.sid, "fala", etapa="paciente", texto=text)
            code = find_astronaut(text)
            if code:
                session_log.set_patient(self.sid, code)
                self._notice = self.m["paciente_ok"].format(code=code)   # fala quem entendeu
                return
            question = self.m["paciente_rep"]
        session_log.set_patient(self.sid, "NAO_IDENTIFICADO")
        self._notice = self.m["paciente_nao"]


# --- Linha de comando: digite o que seria falado (Enter vazio = silêncio) ---------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diálogo de emergência (teste pelo teclado)")
    parser.add_argument("--idioma", choices=config.LANGUAGES, default="pt")
    parser.add_argument("--relator", default=None)
    args = parser.parse_args()

    clf = EmergencyClassifier()
    for w in clf.warnings:
        print(f"AVISO: {w}")
    dialog = EmergencyDialog(args.idioma, "manual", relator=args.relator, classifier=clf)
    print(f"Atendimento #{dialog.sid} (Ctrl+C interrompe; Enter vazio = silêncio)")
    try:
        while True:
            t = dialog.turn
            print("\n" + "-" * 70 + f"\nAstroCare: {t.texto}")
            for code, name in t.opcoes:
                print(f"   {code:<6} {name}")
            if t.fim:
                break
            dialog.reply(input("teclado > " if t.teclado else "fala > "))
    except KeyboardInterrupt:
        dialog.abort("interrompido pelo operador (Ctrl+C)")
        print(f"\nAstroCare: {dialog.turn.texto}")
    except Exception as exc:
        dialog.abort(f"falha técnica: {exc}")
        raise
    print(f"\nLinha do tempo: python -m astrocare.emergency.session_log ver {dialog.sid}")