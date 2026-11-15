"""Biblioteca de protocolos de emergência de saúde (árvores de decisão em JSON).

Regra central: o TEXTO clínico vem só dos arquivos JSON, copiado literalmente
do PDF validado pela equipe. Nenhuma IA gera, resume ou traduz instruções.
A IA (partes 11-13) apenas escolhe QUAL protocolo abrir.

Formato de cada arquivo (astrocare/emergency/protocols/*.json):
    id, versao, status (VALIDADO | DEMONSTRACAO), fonte {pt,en}, nome {pt,en},
    referencias {pt,en} (opcional), regras_criticas {pt,en},
    palavras_chave {pt:[...], en:[...]}, inicio (id do 1º nó),
    nos: { "PCR-01": { titulo, gatilho?, criterios?, acao?, alerta?, materiais?, transicao } }

Transição de um nó (campo "transicao"):
    {"tipo": "proximo", "no": "PCR-02"}           -> segue direto
    {"tipo": "fim", "texto": {pt,en}?}             -> encerra o protocolo
    {"tipo": "condicional",                        -> a TRIPULAÇÃO responde
        "casos": [ {condicao, acao?, alerta?, materiais?, DESTINO}, ... ],
        "senao":   {rotulo?,  acao?, alerta?, materiais?, DESTINO} }

DESTINO é exatamente um destes:
    "no": "<id>"     vai para outro nó
    "fim": true      encerra o protocolo
    "lacuna": true   o PDF NÃO define o que fazer -> o motor avisa e registra

Rastreabilidade: o SHA-256 do arquivo é calculado ao carregar e gravado no
atendimento, provando qual texto exato foi seguido.

Uso:
    python -m astrocare.emergency.protocols validar
    python -m astrocare.emergency.protocols mostrar PCR --idioma pt
"""
import argparse
import hashlib
import json
from dataclasses import dataclass, field

from astrocare import config

VALID_STATUS = {"VALIDADO", "DEMONSTRACAO"}
TRANSITION_TYPES = {"proximo", "condicional", "fim"}
DESTINATIONS = ("no", "fim", "lacuna")


class ProtocolError(Exception):
    """Arquivo de protocolo inválido: ele NÃO é carregado."""


def pick(texts: dict | None, lang: str) -> tuple[str | None, str | None]:
    """Texto no idioma pedido. Se faltar, usa o outro idioma e diz qual foi usado.

    Nunca traduz: mostrar a versão existente é mais seguro do que inventar.
    Devolve (None, None) quando o campo não existe (campos opcionais).
    """
    if not texts:
        return None, None
    if texts.get(lang):
        return texts[lang], lang
    for other in config.LANGUAGES:
        if texts.get(other):
            return texts[other], other
    return None, None


# --- Estruturas carregadas ----------------------------------------------------
@dataclass
class Branch:
    """Um caminho de uma transição condicional (um "SE" ou o "SENÃO")."""
    condicao: dict | None        # texto literal da condição (None no "senao")
    rotulo: dict | None          # texto literal do SENÃO, quando o PDF tem
    acao: dict | None
    alerta: dict | None
    materiais: list[str]
    no: str | None               # próximo nó
    fim: bool                    # encerra o protocolo
    lacuna: bool                 # o PDF não define o próximo passo


@dataclass
class Transition:
    tipo: str                    # proximo | condicional | fim
    no: str | None = None        # tipo "proximo"
    texto: dict | None = None    # tipo "fim" (opcional)
    casos: list[Branch] = field(default_factory=list)
    senao: Branch | None = None


@dataclass
class Node:
    id: str
    titulo: dict
    gatilho: dict | None
    criterios: dict | None       # critérios que a tripulação avalia (ex.: SANS-03)
    acao: dict | None
    alerta: dict | None          # ⚠ alerta obrigatório: o motor SEMPRE mostra
    materiais: list[str]
    transicao: Transition


@dataclass
class Protocol:
    id: str
    versao: str
    status: str
    fonte: dict
    nome: dict
    referencias: dict | None
    regras_criticas: dict
    palavras_chave: dict
    inicio: str
    nos: dict[str, Node]
    sha256: str
    avisos: list[str] = field(default_factory=list)

    @property
    def is_demo(self) -> bool:
        return self.status != "VALIDADO"

    def name(self, lang: str) -> str:
        return pick(self.nome, lang)[0]

    def gaps(self) -> list[str]:
        """Lista os pontos em que o documento não define o próximo passo."""
        out = []
        for node in self.nos.values():
            t = node.transicao
            for i, b in enumerate(t.casos, start=1):
                if b.lacuna:
                    out.append(f"{node.id} (caso {i})")
            if t.senao and t.senao.lacuna:
                out.append(f"{node.id} (senão)")
        return out

    def supplies(self) -> set[str]:
        """Todos os itens de estoque citados no protocolo."""
        items = set()
        for node in self.nos.values():
            items.update(node.materiais)
            for b in node.transicao.casos + ([node.transicao.senao] if node.transicao.senao else []):
                items.update(b.materiais)
        return items


# --- Validação ----------------------------------------------------------------
def _bilingual(obj, where: str, warnings: list[str], required: bool = True) -> dict | None:
    """Confere um campo {pt, en}. Opcional pode faltar; se existir, é conferido."""
    if obj is None and not required:
        return None
    if not isinstance(obj, dict) or not any(obj.get(l) for l in config.LANGUAGES):
        raise ProtocolError(f"{where}: precisa de texto em pelo menos um idioma {config.LANGUAGES}")
    for lang in config.LANGUAGES:
        if not obj.get(lang):
            warnings.append(f"{where}: falta a versão em '{lang}'")
    return obj


def _materials(obj, where: str) -> list[str]:
    if obj is None:
        return []
    if not isinstance(obj, list) or not all(isinstance(m, str) and m.strip() for m in obj):
        raise ProtocolError(f"{where}: 'materiais' deve ser uma lista de nomes do estoque")
    return obj


def _branch(raw, where: str, warnings: list[str], is_else: bool) -> Branch:
    if not isinstance(raw, dict):
        raise ProtocolError(f"{where}: caminho inválido")
    destinos = [d for d in DESTINATIONS if raw.get(d)]
    if len(destinos) != 1:
        raise ProtocolError(f"{where}: precisa de exatamente um destino entre {DESTINATIONS}")
    return Branch(
        condicao=None if is_else else _bilingual(raw.get("condicao"), f"{where} / condicao", warnings),
        rotulo=_bilingual(raw.get("rotulo"), f"{where} / rotulo", warnings, required=False),
        acao=_bilingual(raw.get("acao"), f"{where} / acao", warnings, required=False),
        alerta=_bilingual(raw.get("alerta"), f"{where} / alerta", warnings, required=False),
        materiais=_materials(raw.get("materiais"), where),
        no=raw.get("no"), fim=bool(raw.get("fim")), lacuna=bool(raw.get("lacuna")),
    )


def _transition(raw, where: str, warnings: list[str]) -> Transition:
    if not isinstance(raw, dict) or raw.get("tipo") not in TRANSITION_TYPES:
        raise ProtocolError(f"{where}: 'tipo' deve ser um de {sorted(TRANSITION_TYPES)}")
    tipo = raw["tipo"]
    if tipo == "proximo":
        if not raw.get("no"):
            raise ProtocolError(f"{where}: transição 'proximo' precisa de 'no'")
        return Transition(tipo, no=raw["no"])
    if tipo == "fim":
        return Transition(tipo, texto=_bilingual(raw.get("texto"), f"{where} / texto",
                                                 warnings, required=False))
    # condicional
    casos = raw.get("casos")
    if not isinstance(casos, list) or not casos:
        raise ProtocolError(f"{where}: transição 'condicional' precisa de 'casos'")
    if "senao" not in raw:
        raise ProtocolError(f"{where}: transição 'condicional' precisa de 'senao' "
                            "(use {\"lacuna\": true} se o documento não define)")
    return Transition(
        tipo,
        casos=[_branch(c, f"{where} / caso {i}", warnings, is_else=False)
               for i, c in enumerate(casos, start=1)],
        senao=_branch(raw["senao"], f"{where} / senao", warnings, is_else=True),
    )


def _check_graph(nodes: dict[str, Node], inicio: str, name: str) -> None:
    """Todo destino precisa existir e todo nó precisa ser alcançável a partir do início."""
    if inicio not in nodes:
        raise ProtocolError(f"{name}: nó inicial '{inicio}' não existe")

    def targets(node: Node) -> list[str]:
        t = node.transicao
        if t.tipo == "proximo":
            return [t.no]
        return [b.no for b in t.casos + ([t.senao] if t.senao else []) if b.no]

    for node in nodes.values():
        for target in targets(node):
            if target not in nodes:
                raise ProtocolError(f"{name} / {node.id}: aponta para nó inexistente '{target}'")

    seen, pending = set(), [inicio]
    while pending:                       # percorre a árvore a partir do início
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(targets(nodes[current]))
    unreachable = sorted(set(nodes) - seen)
    if unreachable:
        raise ProtocolError(f"{name}: nós que nunca são alcançados: {unreachable} "
                            "(provável erro de extração)")


def load_file(path) -> Protocol:
    raw = path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{path.name}: JSON inválido ({exc})") from exc

    for key in ("id", "versao", "status", "fonte", "nome", "regras_criticas",
                "palavras_chave", "inicio", "nos"):
        if key not in data:
            raise ProtocolError(f"{path.name}: campo obrigatório ausente: '{key}'")
    if data["status"] not in VALID_STATUS:
        raise ProtocolError(f"{path.name}: status deve ser um de {sorted(VALID_STATUS)}")
    if not isinstance(data["nos"], dict) or not data["nos"]:
        raise ProtocolError(f"{path.name}: protocolo sem nós")

    w: list[str] = []
    name = path.name
    nodes = {}
    for node_id, rn in data["nos"].items():
        where = f"{name} / {node_id}"
        if not isinstance(rn, dict) or "transicao" not in rn:
            raise ProtocolError(f"{where}: nó precisa de 'transicao'")
        nodes[node_id] = Node(
            id=node_id,
            titulo=_bilingual(rn.get("titulo"), f"{where} / titulo", w),
            gatilho=_bilingual(rn.get("gatilho"), f"{where} / gatilho", w, required=False),
            criterios=_bilingual(rn.get("criterios"), f"{where} / criterios", w, required=False),
            acao=_bilingual(rn.get("acao"), f"{where} / acao", w, required=False),
            alerta=_bilingual(rn.get("alerta"), f"{where} / alerta", w, required=False),
            materiais=_materials(rn.get("materiais"), where),
            transicao=_transition(rn["transicao"], f"{where} / transicao", w),
        )
    _check_graph(nodes, data["inicio"], name)

    return Protocol(
        id=data["id"], versao=str(data["versao"]), status=data["status"],
        fonte=_bilingual(data["fonte"], f"{name} / fonte", w),
        nome=_bilingual(data["nome"], f"{name} / nome", w),
        referencias=_bilingual(data.get("referencias"), f"{name} / referencias", w, required=False),
        regras_criticas=_bilingual(data["regras_criticas"], f"{name} / regras_criticas", w),
        palavras_chave=data["palavras_chave"], inicio=data["inicio"], nos=nodes,
        sha256=hashlib.sha256(raw).hexdigest(), avisos=w,
    )


def load_all() -> tuple[dict[str, Protocol], list[str]]:
    """Carrega todos os protocolos válidos. Retorna (protocolos por id, erros).

    Um arquivo com erro é IGNORADO (e reportado), mas não derruba os outros:
    numa emergência, os protocolos bons precisam continuar disponíveis.
    """
    protocols, errors = {}, []
    for path in sorted(config.PROTOCOLS_DIR.glob("*.json")):
        try:
            p = load_file(path)
        except ProtocolError as exc:
            errors.append(str(exc))
            continue
        if p.id in protocols:
            errors.append(f"{path.name}: id '{p.id}' repetido; arquivo ignorado")
            continue
        protocols[p.id] = p
    return protocols, errors


def check_supplies(protocols: dict[str, Protocol]) -> list[str]:
    """Confere se os itens citados nos protocolos existem no estoque cadastrado."""
    from astrocare.emergency.inventory import stock
    registered = {item["nome"].lower() for item in stock()}
    return [f"{p.id}: '{item}' não está cadastrado no estoque"
            for p in protocols.values() for item in sorted(p.supplies())
            if item.lower() not in registered]


# --- Linha de comando ---------------------------------------------------------
def _show(p: Protocol, lang: str) -> None:
    """Imprime a árvore inteira, para a equipe conferir a extração com o PDF."""
    print(f"\n{p.id} v{p.versao} [{p.status}] — {p.name(lang)}")
    print(f"Fonte: {pick(p.fonte, lang)[0]}")
    print(f"Regras críticas: {pick(p.regras_criticas, lang)[0]}")
    for node in p.nos.values():
        print(f"\n[{node.id}] {pick(node.titulo, lang)[0]}")
        for label, value in (("Gatilho", node.gatilho), ("Critérios", node.criterios),
                             ("Ação", node.acao), ("⚠ ALERTA", node.alerta)):
            if value:
                print(f"   {label}: {pick(value, lang)[0]}")
        if node.materiais:
            print(f"   Materiais: {', '.join(node.materiais)}")
        t = node.transicao
        if t.tipo == "proximo":
            print(f"   -> {t.no}")
        elif t.tipo == "fim":
            print(f"   -> FIM {pick(t.texto, lang)[0] or ''}")
        else:
            for b in t.casos:
                _show_branch(f"{'SE' if lang == 'pt' else 'IF'} ({pick(b.condicao, lang)[0]})", b, lang)
            label = pick(t.senao.rotulo, lang)[0]
            other = "SENÃO" if lang == "pt" else "ELSE"
            _show_branch(f"{other}{f' ({label})' if label else ''}", t.senao, lang)


def _show_branch(head: str, b: Branch, lang: str) -> None:
    dest = (f"-> {b.no}" if b.no else "-> FIM" if b.fim
            else "-> ⚠ LACUNA: o documento não define o próximo passo")
    print(f"   • {head}")
    if b.acao:
        print(f"       Ação: {pick(b.acao, lang)[0]}")
    if b.alerta:
        print(f"       ⚠ ALERTA: {pick(b.alerta, lang)[0]}")
    if b.materiais:
        print(f"       Materiais: {', '.join(b.materiais)}")
    print(f"       {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Biblioteca de protocolos")
    sub = parser.add_subparsers(dest="acao", required=True)
    sub.add_parser("validar", help="carrega e confere todos os protocolos")
    show = sub.add_parser("mostrar", help="imprime a árvore de um protocolo")
    show.add_argument("id")
    show.add_argument("--idioma", choices=config.LANGUAGES, default="pt")
    args = parser.parse_args()

    protocols, errors = load_all()
    if args.acao == "validar":
        print(f"{len(protocols)} protocolo(s) carregado(s) de {config.PROTOCOLS_DIR}")
        for p in protocols.values():
            tag = "DEMONSTRAÇÃO" if p.is_demo else "VALIDADO"
            print(f"  [{tag}] {p.id} v{p.versao} — {p.name('pt')} / {p.name('en')} "
                  f"({len(p.nos)} nós)  sha256={p.sha256[:12]}…")
            for gap in p.gaps():
                print(f"      lacuna: {gap} — o documento não define o próximo passo")
            for w in p.avisos:
                print(f"      aviso: {w}")
        for e in errors + check_supplies(protocols):
            print(f"  ERRO: {e}")
    else:
        if args.id not in protocols:
            raise SystemExit(f"Protocolo '{args.id}' não encontrado. Erros: {errors}")
        _show(protocols[args.id], args.idioma)