"""IA offline que reconhece QUAL emergência o astronauta descreveu (PT e EN).

Modelo: TF-IDF (pedaços de 2 a 5 letras + palavras) -> Regressão Logística.
- Pequeno, rápido e 100% offline: roda em qualquer computador da nave.
- Pedaços de letras toleram erros de transcrição ("fibrilacao", "desmaio").

Regras de segurança:
- A IA só SUGERE. A decisão é sempre do astronauta ("Entendi: X. Confirma?").
- Se a classe mais provável for FORA_DO_ESCOPO, ou a confiança ficar abaixo
  do limiar, a resposta é "não entendi" e o sistema pergunta de novo.
- O modelo guarda o SHA-256 das frases usadas no treino. Se o CSV mudar
  depois, o sistema avisa que o modelo está desatualizado.

Uso:
    python -m astrocare.emergency.classifier treinar
    python -m astrocare.emergency.classifier classificar "ele desmaiou e não responde"
"""
import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from astrocare import config
from astrocare.emergency.training_data import OUT_OF_SCOPE, TRAINING_FILE, load, normalize

MODEL_DIR = config.BASE_DIR / "data" / "models"
MODEL_FILE = MODEL_DIR / "classificador_emergencia.joblib"
META_FILE = MODEL_DIR / "classificador_emergencia.json"
DEFAULT_THRESHOLD = 0.50
SEED = 2026                     # mesma semente = mesmo resultado em qualquer máquina


class ClassifierError(Exception):
    """Modelo ausente ou inválido."""


@dataclass
class Classification:
    rotulo: str | None          # protocolo sugerido, ou None ("não entendi")
    confianca: float            # probabilidade da classe mais provável
    candidato: str              # classe mais provável (mesmo se rejeitada)
    motivo: str                 # aceito | fora_do_escopo | baixa_confianca | texto_vazio


def _build() -> Pipeline:
    features = FeatureUnion([
        ("letras", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ("palavras", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
    ])
    model = LogisticRegression(C=10, max_iter=2000, class_weight="balanced", random_state=SEED)
    return Pipeline([("features", features), ("modelo", model)])


def _file_hash() -> str:
    return hashlib.sha256(TRAINING_FILE.read_bytes()).hexdigest()


# --- Treino e avaliação ----------------------------------------------------------
def evaluate(rows: list[dict], folds: int = 5) -> dict:
    """Validação cruzada: cada frase é prevista por um modelo que NUNCA a viu.

    Divide as frases em 5 partes; treina com 4 e testa na 5ª, cinco vezes.
    Mede como a IA se sai com frases novas, não com as que decorou.
    """
    X = [normalize(r["texto"]) for r in rows]
    y = np.array([r["rotulo"] for r in rows])
    strata = [f"{r['rotulo']}|{r['idioma']}" for r in rows]   # equilibra rótulo E idioma
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    proba = cross_val_predict(_build(), X, y, cv=cv.split(X, strata), method="predict_proba")
    labels = np.array(sorted(set(y)))          # mesma ordem usada pelo modelo
    pred, conf = labels[proba.argmax(1)], proba.max(1)

    by_threshold = {}
    for t in (0.40, 0.50, 0.60, 0.70):
        accepted = (conf >= t) & (pred != OUT_OF_SCOPE)
        wrong_accept = accepted & (pred != y)          # o perigo: abrir o protocolo errado
        missed = (y != OUT_OF_SCOPE) & ~accepted        # emergência real -> "não entendi"
        n_proto = int((y != OUT_OF_SCOPE).sum())
        by_threshold[t] = {
            "protocolo_errado": int(wrong_accept.sum()),
            "fora_aceito_como_protocolo": int((accepted & (y == OUT_OF_SCOPE)).sum()),
            "emergencia_nao_entendida": int(missed.sum()),
            "emergencias": n_proto,
        }
    return {
        "frases": len(rows),
        "acuracia": float((pred == y).mean()),
        "rotulos": labels.tolist(),
        "matriz": confusion_matrix(y, pred, labels=labels).tolist(),
        "por_limiar": {str(k): v for k, v in by_threshold.items()},
        "erros": [{"texto": rows[i]["texto"], "esperado": y[i], "previsto": pred[i],
                   "confianca": round(float(conf[i]), 2)}
                  for i in np.where(pred != y)[0]],
    }


def train(threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Valida o CSV, avalia, treina com TODAS as frases e salva modelo + metadados."""
    rows = load()                                   # TrainingDataError se o CSV tiver problema
    report = evaluate(rows)
    pipe = _build().fit([normalize(r["texto"]) for r in rows], [r["rotulo"] for r in rows])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_FILE)
    meta = {
        "treinado_em": datetime.now().isoformat(timespec="seconds"),
        "csv_sha256": _file_hash(),
        "limiar": threshold,
        "classes": pipe.classes_.tolist(),
        "sklearn": sklearn.__version__,
        "avaliacao": {k: v for k, v in report.items() if k != "erros"},
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# --- Uso em produção ---------------------------------------------------------------
class EmergencyClassifier:
    """Carregado UMA vez na inicialização; depois classifica em milissegundos."""

    def __init__(self):
        if not MODEL_FILE.exists() or not META_FILE.exists():
            raise ClassifierError("Modelo não treinado. Rode: "
                                  "python -m astrocare.emergency.classifier treinar")
        self.meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        self.pipe = joblib.load(MODEL_FILE)
        self.threshold = float(self.meta["limiar"])
        self.warnings = []
        if TRAINING_FILE.exists() and _file_hash() != self.meta["csv_sha256"]:
            self.warnings.append("As frases de treino mudaram depois do último treino: treine de novo.")
        if self.meta["sklearn"] != sklearn.__version__:
            self.warnings.append(f"Modelo treinado com scikit-learn {self.meta['sklearn']}, "
                                 f"instalado {sklearn.__version__}: treine de novo.")

    def classify(self, text: str) -> Classification:
        clean = normalize(text or "")
        if not clean:
            return Classification(None, 0.0, OUT_OF_SCOPE, "texto_vazio")
        proba = self.pipe.predict_proba([clean])[0]
        best = int(proba.argmax())
        label, conf = self.pipe.classes_[best], float(proba[best])
        if label == OUT_OF_SCOPE:
            return Classification(None, conf, label, "fora_do_escopo")
        if conf < self.threshold:
            return Classification(None, conf, label, "baixa_confianca")
        return Classification(label, conf, label, "aceito")


# --- Linha de comando ------------------------------------------------------------
def _print_report(r: dict) -> None:
    labels = r["rotulos"]
    print(f"\nValidação cruzada (5 partes) com {r['frases']} frases — acurácia {r['acuracia']:.1%}")
    print("\nMatriz de confusão (linha = correto, coluna = previsto):")
    print(" " * 16 + "".join(f"{l[:14]:>15}" for l in labels))
    for label, line in zip(labels, r["matriz"]):
        print(f"{label[:15]:<16}" + "".join(f"{v:>15}" for v in line))
    print("\nSegurança por limiar de confiança:")
    print(f"  {'limiar':>6}  {'protocolo errado':>16}  {'fora->protocolo':>15}  {'emergência não entendida':>25}")
    for t, v in r["por_limiar"].items():
        print(f"  {float(t):>6.2f}  {v['protocolo_errado']:>16}  {v['fora_aceito_como_protocolo']:>15}  "
              f"{v['emergencia_nao_entendida']:>14} de {v['emergencias']}")
    if r["erros"]:
        print("\nFrases erradas (para a equipe revisar):")
        for e in r["erros"]:
            print(f"  [{e['esperado']} -> {e['previsto']} {e['confianca']:.2f}] {e['texto']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IA de reconhecimento de emergências (offline)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("treinar")
    tr.add_argument("--limiar", type=float, default=DEFAULT_THRESHOLD)
    cl = sub.add_parser("classificar")
    cl.add_argument("texto")
    args = parser.parse_args()

    if args.cmd == "treinar":
        _print_report(train(args.limiar))
        print(f"\nModelo salvo em {MODEL_FILE} (limiar {args.limiar:.2f})")
    else:
        clf = EmergencyClassifier()
        for w in clf.warnings:
            print(f"AVISO: {w}")
        c = clf.classify(args.texto)
        if c.rotulo:
            print(f"Sugestão: {c.rotulo} (confiança {c.confianca:.0%}) -> pedir confirmação")
        else:
            print(f"Não entendi ({c.motivo}; mais provável: {c.candidato} {c.confianca:.0%}) "
                  "-> perguntar de novo")