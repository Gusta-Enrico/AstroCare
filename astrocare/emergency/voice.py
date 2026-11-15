"""Voz 100% offline: ativação "AstroCare" + "Emergência", fala e escuta em PT e EN.

Peças:
- Vosk (reconhecimento de fala offline): DOIS modelos, PT e EN, ouvindo o
  MESMO áudio ao mesmo tempo. O idioma do atendimento é o do comando dito
  ("Emergência" -> pt, "Emergency" -> en).
- pyttsx3 (fala offline): usa as vozes instaladas no Windows.
- Tudo o que é falado também aparece na tela.

Reservas (o atendimento nunca depende só da voz):
- Microfone ou modelos indisponíveis na inicialização -> modo teclado.
- Microfone falha no meio do atendimento -> continua pelo teclado, registrado.
- Fala do computador falha -> o texto continua na tela.

Segurança no protocolo: silêncio NUNCA avança uma etapa. Na identificação
da emergência, silêncio conta como tentativa (regra da parte 13); depois que
o protocolo começa, o sistema espera a tripulação falar ("feito", "pronto").

Uso:
    python -m astrocare.emergency.voice verificar     # autoteste (rode antes de usar)
    python -m astrocare.emergency.voice ouvir         # modo de espera por voz
    python -m astrocare.emergency.voice ouvir --mostrar   # mostra tudo o que foi ouvido
    python -m astrocare.emergency.voice transcrever --idioma pt   # diagnóstico do modelo
"""
import argparse
import json
import queue
import re
import time

from astrocare import config
from astrocare.emergency import session_log
from astrocare.emergency.classifier import ClassifierError, EmergencyClassifier
from astrocare.emergency.dialogue import EmergencyDialog
from astrocare.emergency.training_data import normalize

VOSK_DIR = config.BASE_DIR / "data" / "models" / "vosk"
SAMPLE_RATE = 16000
ANSWER_TIMEOUT_S = 8        # silêncio na identificação da emergência = "não ouvi"
COMMAND_WINDOW_S = 8        # depois de "AstroCare", quanto tempo espera "Emergência"
MAX_UTTERANCE_S = 20

# Formas de "AstroCare" que cada modelo reconhece (com pronúncia brasileira).
# "AstroCare" não existe no dicionário: é montada com palavras que existem.
# Para acrescentar uma forma nova, veja o que aparece no "ouvir --mostrar".
WAKE_FORMS = {
    "pt": ["astro quer", "astro care", "astro"],
    "en": ["astro care", "as through care", "through care", "astro car", "astro"],
}
# Comandos que ativam JUNTO com AstroCare, na mesma frase (com acento).
WAKE_COMMANDS = {
    "pt": ["emergência", "socorro", "ajuda"],
    "en": ["emergency", "mayday", "help"],
}
# Comandos aceitos SOZINHOS, só na janela de 8 s depois de "AstroCare".
# "ajuda"/"help" ficam de fora: só valem na mesma frase do AstroCare.
SOLO_COMMANDS = {
    "pt": ["emergência", "socorro"],
    "en": ["emergency", "mayday"],
}
# Frases que o reconhecedor PROCURA na espera (modo gramática), geradas a partir
# das listas acima. Frases com palavra fora do dicionário são descartadas.
WAKE_PHRASES = {
    lang: ([f"{form} {cmd}" for form in WAKE_FORMS[lang] for cmd in WAKE_COMMANDS[lang]]
           + WAKE_FORMS[lang] + SOLO_COMMANDS[lang])
    for lang in WAKE_FORMS
}
# Formas de AstroCare para conferir no texto ouvido (sem acento).
WAKE_VARIANTS = {normalize(form) for forms in WAKE_FORMS.values() for form in forms}
# Palavras que contam como comando (sem acento). Também definem o idioma do atendimento.
COMMANDS = {
    "pt": {"emergencia", "emergencias", "socorro", "ajuda"},
    "en": {"emergency", "emergencies", "help", "mayday"},
}
# Resposta à pergunta "Quem é o paciente?": SÓ o número do astronauta.
# O sistema completa o código (fala "três" -> AST-03).
_NUMBERS = {"pt": ["um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove"],
            "en": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]}
PATIENT_PHRASES = {
    lang: [phrase
           for code in sorted(config.ASTRONAUTS)
           for n in [_NUMBERS[lang][int(code[-1]) - 1]]
           for phrase in (n, f"zero {n}")]
    for lang in config.LANGUAGES
}

# "Finalizar atendimento" também precisa ser ouvido na pergunta do paciente (gramática).
END_PHRASES = {"pt": ["finalizar atendimento", "encerrar atendimento"],
               "en": ["end session", "finish session"]}
for _lang in config.LANGUAGES:
    PATIENT_PHRASES[_lang] += END_PHRASES[_lang]

# Pergunta de desempate: palavras com sons bem diferentes entre os idiomas.
LANGUAGE_ANSWERS = {"pt": ["português"], "en": ["english"]}
LANGUAGE_QUESTION = {"pt": "Português ou inglês?", "en": "Portuguese or English?"}
MAX_LANGUAGE_ATTEMPTS = 2


class VoiceUnavailable(Exception):
    """Microfone, modelo ou biblioteca de voz indisponível."""


def has_wake_word(text: str) -> bool:
    """True se o texto tem uma forma de AstroCare como PALAVRAS inteiras."""
    clean = f" {normalize(text)} "
    return any(f" {v} " in clean for v in WAKE_VARIANTS)




# --- Microfone ----------------------------------------------------------------------
class Microphone:
    def __init__(self):
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise VoiceUnavailable(f"sounddevice indisponível: {exc}") from exc
        self.q: queue.Queue[bytes] = queue.Queue()
        try:
            self.stream = sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=4000, dtype="int16",
                                            channels=1, callback=lambda d, *_: self.q.put(bytes(d)))
            self.stream.start()
        except Exception as exc:
            raise VoiceUnavailable(f"microfone indisponível: {exc}") from exc

    def read(self, timeout: float) -> bytes | None:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush(self) -> None:
        """Descarta o áudio acumulado (ex.: o próprio computador falando)."""
        while not self.q.empty():
            self.q.get_nowait()

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


# --- Reconhecimento (Vosk) ------------------------------------------------------------
def _looks_like_model(folder) -> bool:
    """Reconhece os dois formatos de modelo Vosk:
    - novo: subpastas am/ e conf/ (ex.: inglês 0.15)
    - antigo: arquivos soltos final.mdl, mfcc.conf (ex.: português 0.3)
    """
    return (folder / "am" / "final.mdl").exists() or (folder / "final.mdl").exists()


def _model_path(lang: str):
    """Aceita a pasta do modelo direto ou com uma subpasta (como vem no .zip)."""
    base = VOSK_DIR / lang
    if _looks_like_model(base):
        return base
    subdirs = [d for d in base.glob("*") if d.is_dir()] if base.exists() else []
    if len(subdirs) == 1 and _looks_like_model(subdirs[0]):
        return subdirs[0]
    raise VoiceUnavailable(f"Modelo Vosk '{lang}' não encontrado em {base}")


class Recognizer:
    def __init__(self, mic: Microphone):
        try:
            import vosk
        except ImportError as exc:
            raise VoiceUnavailable("Biblioteca vosk não instalada (pip install vosk)") from exc
        vosk.SetLogLevel(-1)
        self._vosk = vosk
        self.mic = mic
        self.models = {lang: vosk.Model(str(_model_path(lang))) for lang in config.LANGUAGES}
        self.grammars = {lang: self._grammar(lang) for lang in config.LANGUAGES}

    def _known(self, lang: str, phrases: list[str]) -> list[str]:
        """Só as frases cujas palavras TODAS existem no dicionário do modelo."""
        model = self.models[lang]
        return [p for p in phrases if all(model.vosk_model_find_word(w) >= 0 for w in p.split())]

    def _grammar(self, lang: str) -> list[str]:
        return self._known(lang, WAKE_PHRASES[lang]) 

    def listen(self, langs, timeout: float | None, grammar: bool = False,
               phrases: dict | None = None) -> dict:
        """Escuta até uma frase terminar. Devolve {idioma: {"text": ..., "words": [...]}}.

        timeout: segundos esperando a pessoa COMEÇAR a falar (None = para sempre).
        grammar: True = só reconhece as frases de ativação (ou "[unk]" para o resto).
        phrases: outra lista de frases por idioma no lugar das de ativação (ex.: idioma).
        Sem fala no prazo, devolve textos vazios.
        """
        recs = {}
        for lang in langs:
            if not grammar:
                allowed = None
            elif phrases:                        # frases próprias (paciente, idioma)
                allowed = self._known(lang, phrases[lang])
            else:                                # frases de ativação (já filtradas)
                allowed = self.grammars[lang]
            if allowed:
                r = self._vosk.KaldiRecognizer(self.models[lang], SAMPLE_RATE,
                                               json.dumps(allowed + ["[unk]"], ensure_ascii=False))
            else:
                r = self._vosk.KaldiRecognizer(self.models[lang], SAMPLE_RATE)
            r.SetWords(True)
            recs[lang] = r
        start, speech_at = time.monotonic(), None
        self.last_audio = bytearray()            # áudio desta frase (para reconferir)
        while True:
            data = self.mic.read(0.5)
            now = time.monotonic()
            if data:
                self.last_audio.extend(data)
                for lang, r in recs.items():
                    if r.AcceptWaveform(data):          # fim de frase detectado
                        first = json.loads(r.Result())
                        if first.get("text"):
                            return self._collect(recs, lang, first)
                    elif json.loads(r.PartialResult()).get("partial"):
                        speech_at = speech_at or now
            if speech_at is None and timeout is not None and now - start > timeout:
                return {lang: {"text": "", "words": []} for lang in langs}
            if speech_at is not None and now - speech_at > MAX_UTTERANCE_S:
                return self._collect(recs, None, None)

    def free_transcripts(self, audio: bytes) -> dict:
        """Retranscreve o MESMO áudio em modo livre, nos dois idiomas."""
        out = {}
        for lang in config.LANGUAGES:
            r = self._vosk.KaldiRecognizer(self.models[lang], SAMPLE_RATE)
            for i in range(0, len(audio), 8000):
                r.AcceptWaveform(bytes(audio[i:i + 8000]))
            out[lang] = json.loads(r.FinalResult()).get("text", "")
        return out        

    @staticmethod
    def _collect(recs, done_lang, done_result) -> dict:
        out = {}
        for lang, r in recs.items():
            res = done_result if lang == done_lang else json.loads(r.FinalResult())
            text = res.get("text", "").replace("[unk]", "").strip()
            out[lang] = {"text": text, "words": [w for w in res.get("result", [])
                                                 if w.get("word") != "[unk]"]}
        return out


# --- Fala (pyttsx3) ---------------------------------------------------------------
VOICE_HINTS = {"pt": ("pt-br", "pt_br", "portugu", "maria", "daniel"),
               "en": ("en-us", "en_us", "english", "zira", "david")}


def _find_voice(engine, lang: str):
    for v in engine.getProperty("voices"):
        desc = f"{v.id} {v.name} {' '.join(map(str, getattr(v, 'languages', []) or []))}".lower()
        if any(h in desc for h in VOICE_HINTS[lang]):
            return v.id
    return None


def _speakable(text: str, lang: str) -> str:
    text = text.replace("⚠", "Atenção." if lang == "pt" else "Warning.")
    return re.sub(r"\[([A-Z]+-[0-9A-Z]+)\]", r"\1.", text)


class Speaker:
    def __init__(self):
        self.ok, self.problem = True, None
        try:
            import pyttsx3
            self._pyttsx3 = pyttsx3
            engine = pyttsx3.init()
            self.voices = {lang: _find_voice(engine, lang) for lang in config.LANGUAGES}
            engine.stop()
        except Exception as exc:
            self.ok, self.problem, self.voices = False, f"fala indisponível: {exc}", {}

    def say(self, text: str, lang: str) -> None:
        if not self.ok:
            return
        try:
            engine = self._pyttsx3.init()       # motor novo a cada fala: evita travar no Windows
            if self.voices.get(lang):
                engine.setProperty("voice", self.voices[lang])
            engine.setProperty("rate", 165)
            engine.say(_speakable(text, lang))
            engine.runAndWait()
            engine.stop()
        except Exception as exc:                # a tela continua mostrando o texto
            self.ok, self.problem = False, f"fala falhou: {exc}"
            print(f"(AVISO: {self.problem}; seguindo só com texto)")


# --- Atendimento por voz -------------------------------------------------------------
def _show(text: str) -> None:
    print("\n" + "-" * 70 + f"\nAstroCare: {text}")


def run_dialog(lang: str, clf: EmergencyClassifier, rec: Recognizer | None,
               speaker: Speaker, mic: Microphone | None) -> int:
    """Conduz um atendimento. Sem rec/mic, tudo pelo teclado."""
    dialog = EmergencyDialog(lang, "voz" if rec else "manual", classifier=clf)
    keyboard = rec is None
    try:
        while True:
            turn = dialog.turn
            _show(turn.texto)
            for code, name in turn.opcoes:
                print(f"   {code:<6} {name}")
            if not turn.teclado:
                speaker.say(turn.fala or turn.texto, lang)   # fala curta; a tela mostra tudo
            if turn.fim:
                break
            if turn.teclado or keyboard:
                dialog.reply(input("teclado > "))
                continue
            try:
                mic.flush()                      # descarta o áudio da própria fala do sistema
                while True:
                    heard = rec.listen([lang], timeout=ANSWER_TIMEOUT_S)[lang]["text"]
                    # No protocolo, silêncio nunca avança uma etapa: continua escutando.
                    if heard or dialog.run is None:
                        break
                print(f"(ouvido: {heard or '[silêncio]'})")
                dialog.reply(heard)
            except VoiceUnavailable:
                raise
            except Exception as exc:             # microfone caiu: segue pelo teclado
                keyboard = True
                session_log.log(dialog.sid, "erro", erro=f"voz falhou, seguindo pelo teclado: {exc}")
                print(f"(AVISO: voz falhou ({exc}); continue pelo teclado)")
    except KeyboardInterrupt:
        dialog.abort("interrompido pelo operador (Ctrl+C)")
        _show(dialog.turn.texto)
    except Exception as exc:
        dialog.abort(f"falha técnica: {exc}")
        raise
    return dialog.sid

def _langs_with_command(texts: dict) -> set[str]:
    """Idiomas em cujo texto aparece um comando DAQUELE idioma."""
    return {lang for lang, text in texts.items()
            if set(normalize(text).split()) & COMMANDS[lang]}


def decide_lang(rec: Recognizer, heard: dict, show_all: bool = False) -> str | None:
    """Decide o idioma do comando. Devolve 'pt', 'en', 'ambiguo' ou None (sem comando).

    1) gramática: só um idioma reconheceu o comando -> esse idioma;
    2) empate: retranscreve o mesmo áudio em modo livre e confere de novo;
    3) ainda empatado -> 'ambiguo' (o sistema pergunta; nunca chuta).
    """
    by_grammar = _langs_with_command({lang: r["text"] for lang, r in heard.items()})
    if not by_grammar:
        return None
    if len(by_grammar) == 1:
        return by_grammar.pop()
    free = rec.free_transcripts(rec.last_audio)
    by_free = _langs_with_command(free)
    if show_all:
        print(f"(empate na gramática; reconferência livre: {free})")
    return by_free.pop() if len(by_free) == 1 else "ambiguo"


def ask_language(rec: Recognizer, speaker: "Speaker", mic: Microphone) -> str | None:
    """Pergunta o idioma ('Português ou inglês?'). Devolve 'pt', 'en' ou None."""
    for _ in range(MAX_LANGUAGE_ATTEMPTS):
        for lang in config.LANGUAGES:
            _show(LANGUAGE_QUESTION[lang])
            speaker.say(LANGUAGE_QUESTION[lang], lang)
        mic.flush()
        heard = rec.listen(list(config.LANGUAGES), timeout=ANSWER_TIMEOUT_S, grammar=True,
                           phrases=LANGUAGE_ANSWERS)
        found = {lang for lang, r in heard.items() if r["text"]}
        if len(found) == 1:
            return found.pop()
    return None

def standby(show_all: bool = False) -> None:
    """Fica escutando 'AstroCare' + 'Emergência' (ou 'Emergency').

    show_all=True mostra as frases de ativação aceitas e tudo o que foi ouvido.
    """
    clf = EmergencyClassifier()
    for w in clf.warnings:
        print(f"AVISO: {w}")
    closed = session_log.close_stale()
    if closed:
        print(f"Atendimentos abertos encerrados como 'interrompido': {closed}")
    speaker = Speaker()
    if not speaker.ok:
        print(f"AVISO: {speaker.problem}. O texto aparece só na tela.")
    try:
        mic = Microphone()
        rec = Recognizer(mic)
    except VoiceUnavailable as exc:
        print(f"VOZ INDISPONÍVEL ({exc}). Atendimento pelo TECLADO.")
        lang = input("Idioma (pt/en) > ").strip().lower()
        sid = run_dialog(lang if lang in config.LANGUAGES else "pt", clf, None, speaker, None)
        print(f"\nLinha do tempo: python -m astrocare.emergency.session_log ver {sid}")
        return

    if show_all:
        for lang, phrases in rec.grammars.items():
            print(f"(frases de ativação {lang}: {phrases or 'NENHUMA no dicionário — modo livre'})")
    print("AstroCare em espera. Diga 'AstroCare, emergência' (ou 'AstroCare, emergency'). "
          "Ctrl+C sai.")
    try:
        while True:
            heard = rec.listen(list(config.LANGUAGES), timeout=None, grammar=True)
            texts = {lang: r["text"] for lang, r in heard.items()}
            if show_all and any(texts.values()):
                print(f"(ouvido: {texts})")
            if not any(has_wake_word(t) for t in texts.values()):
                continue
            print(f"(ativação ouvida: {texts})")
            lang = decide_lang(rec, heard, show_all)
            if lang is None:                     # "AstroCare" sozinho: espera o comando
                mic.flush()
                heard = rec.listen(list(config.LANGUAGES), timeout=COMMAND_WINDOW_S, grammar=True)
                lang = decide_lang(rec, heard, show_all)
            if lang is None:
                print("(comando não ouvido; voltando à espera)")
                continue
            if lang == "ambiguo":                # não chuta: pergunta o idioma
                lang = ask_language(rec, speaker, mic)
                if lang is None:
                    print("(idioma não identificado; escolha pelo TECLADO)")
                    while lang not in config.LANGUAGES:
                        lang = input("Idioma (pt/en) > ").strip().lower()
            if show_all:
                print(f"(idioma do atendimento: {lang})")
            sid = run_dialog(lang, clf, rec, speaker, mic)
            print(f"\nLinha do tempo: python -m astrocare.emergency.session_log ver {sid}")
            print("\nAstroCare em espera.")
            mic.flush()
    except KeyboardInterrupt:
        print("\nEncerrado.")
    finally:
        mic.close()

def transcribe(lang: str) -> None:
    """Diagnóstico: transcrição LIVRE com um só idioma, frase por frase."""
    mic = Microphone()
    try:
        rec = Recognizer(mic)
        print(f"Transcrevendo em '{lang}' (só este modelo). Fale frases curtas. Ctrl+C sai.")
        while True:
            text = rec.listen([lang], timeout=None)[lang]["text"]
            print(f"  {lang}: {text or '[vazio]'}")
    except KeyboardInterrupt:
        print("\nEncerrado.")
    finally:
        mic.close()


# --- Autoteste ----------------------------------------------------------------------
def self_test() -> bool:
    """Confere tudo o que o atendimento precisa. Rode com a internet DESLIGADA."""
    from astrocare.emergency.db import transaction
    from astrocare.emergency.inventory import alerts
    from astrocare.emergency.protocols import check_supplies, load_all

    results = []

    def check(nome, func):
        try:
            status, detalhe = func()
        except Exception as exc:
            status, detalhe = "FALHA", str(exc)
        results.append(status)
        print(f"  [{status:^5}] {nome}: {detalhe}")

    def protocolos():
        p, errors = load_all()
        if errors or not p:
            return "FALHA", f"{len(p)} carregados; erros: {errors}"
        gaps = sum(len(x.gaps()) for x in p.values())
        return "OK", f"{len(p)} carregados ({', '.join(sorted(p))}); {gaps} lacuna(s) conhecidas"

    def estoque():
        missing = check_supplies(load_all()[0])
        if missing:
            return "FALHA", "; ".join(missing)
        n = len(alerts())
        return ("AVISO" if n else "OK"), f"materiais dos protocolos cadastrados; {n} alerta(s) de estoque"

    def banco():
        with transaction() as conn:
            conn.execute("SELECT 1")
        abertos = [s for s in session_log.sessions(1000) if s["status"] == "em_andamento"]
        return ("AVISO" if abertos else "OK"), (f"{len(abertos)} atendimento(s) aberto(s) "
                                                "(serão encerrados ao iniciar)" if abertos else "gravação OK")

    def ia():
        clf = EmergencyClassifier()
        c = clf.classify("parada cardíaca")
        if c.rotulo != "PCR":
            return "FALHA", f"'parada cardíaca' classificada como {c.candidato}"
        return ("AVISO" if clf.warnings else "OK"), ("; ".join(clf.warnings) or
                                                      f"modelo de {clf.meta['treinado_em']}, limiar {clf.threshold}")

    def modelo(lang):
        def f():
            import vosk
            vosk.SetLogLevel(-1)
            path = _model_path(lang)
            vosk.Model(str(path))
            return "OK", str(path)
        return f

    def microfone():
        import sounddevice as sd
        dev = sd.query_devices(kind="input")
        return "OK", dev["name"]

    def fala():
        s = Speaker()
        if not s.ok:
            return "AVISO", f"{s.problem} (texto só na tela)"
        missing = [l for l, v in s.voices.items() if not v]
        if missing:
            return "AVISO", f"sem voz instalada para {missing}: o Windows usará a voz padrão"
        return "OK", ", ".join(f"{l}: {v.split(chr(92))[-1]}" for l, v in s.voices.items())

    print("Autoteste AstroCare — emergência (offline)")
    check("Protocolos", protocolos)
    check("Estoque x protocolos", estoque)
    check("Banco de dados", banco)
    check("IA de reconhecimento", ia)
    for lang in config.LANGUAGES:
        check(f"Modelo de fala {lang}", modelo(lang))
    check("Microfone", microfone)
    check("Fala do computador", fala)
    ok = "FALHA" not in results
    print("\nRESULTADO:", "PRONTO PARA USO" if ok else
          "COM FALHAS — o atendimento pelo teclado continua disponível "
          "(python -m astrocare.emergency.dialogue)")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voz offline do módulo de emergência")
    parser.add_argument("acao", choices=["verificar", "ouvir", "transcrever"])
    parser.add_argument("--idioma", choices=config.LANGUAGES, default="pt",
                        help="idioma do 'transcrever'")
    parser.add_argument("--mostrar", action="store_true",
                        help="mostra as frases de ativação e tudo o que foi ouvido")
    args = parser.parse_args()
    if args.acao == "verificar":
        raise SystemExit(0 if self_test() else 1)
    if args.acao == "transcrever":
        transcribe(args.idioma)
        raise SystemExit(0)
    try:
        standby(args.mostrar)
    except ClassifierError as exc:
        raise SystemExit(f"ERRO: {exc}")