"""Motor de simulação fisiológica/operacional — dados SIMULADOS, nunca medições reais.

Gera, para cada astronauta e dia, as métricas de sensores (PVT, wearable,
nutrição, exercício, dosímetro) e, na fase Terra, também o autorrelato de
sono e recuperação (na missão eles vêm do check-in de áudio).

Como os valores ficam realistas (e não "sorteados soltos"):
1. PERFIL PESSOAL fixo por astronauta: médias individuais sorteadas UMA vez a
   partir do código do astronauta (AST-01 sempre tem o mesmo perfil). Isso
   garante consistência estatística entre simulações.
2. CONTINUIDADE: um fator latente de "sono/recuperação" segue um processo
   AR(1): o dia de hoje se parece com o de ontem, sem ser igual.
3. CORRELAÇÕES plausíveis: noite pior -> menos sono, menor eficiência, mais
   despertares, PVT mais lento e com mais lapsos, FC de repouso maior, VFC
   menor, pior qualidade e recuperação autorrelatadas. Mais exercício ->
   mais calorias e mais água.
4. ROTINA: dias de descanso na semana; na missão, exercício diário planejado.
5. SEMENTE: cada simulação usa uma semente nova (registrada). A mesma semente
   reproduz exatamente os mesmos dados, inclusive para um dia novo da missão.

Referências usadas para as faixas (valores típicos, não limites clínicos):
- Dose: ISS ~0,2–0,5 mSv/dia (dose efetiva); fundo natural na Terra ~2–3 mSv/ano
  (~6–10 µSv/dia). PMC10919966; ESA/STCE.
- PVT de 10 min: lapso = resposta > 500 ms; lapsos e velocidade são os mais
  sensíveis à falta de sono. Basner & Dinges, Sleep 2011;34(5):581-591.
- Exercício na ISS: ~2,5 h/dia reservadas (inclui preparo). J Appl Physiol 2012.
- Água: NASA-STD-3001, mínimo 2,5 L/dia; ISS observado ~2,0–2,3 L/dia. OCHMO-TB.
- Calorias em missão: ~2.500–3.500 kcal/dia com 2 h de exercício.
- RMSSD: adultos saudáveis ~20–80 ms, muito individual.
As médias individuais, variações diárias e tamanhos de efeito são SUPOSIÇÕES
razoáveis do protótipo, não parâmetros validados.

Uso:
    python -m astrocare.physio_sim validar
    python -m astrocare.physio_sim exemplo AST-01
"""
import argparse
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from astrocare import config

AR_PHI = 0.55            # continuidade do fator de sono (0 = sem memória, 1 = constante)
HORIZON = (-60, 400)     # dias pré-calculados do fator latente (Terra até 400 dias de missão)
REST_WEEKDAYS_EARTH = 2  # dias de descanso por semana na Terra
REST_WEEKDAYS_MISSION = 1


@dataclass(frozen=True)
class Profile:
    """Médias pessoais (fixas por astronauta). Unidades como em config.METRIC_UNITS."""
    sono_h: float
    eficiencia: float
    despertares: float
    qualidade: float
    recuperacao: float
    rt_ms: float
    lapsos: float
    fc: float
    rmssd: float
    exercicio_terra: float
    kcal_base: float
    agua_base: float
    dose_terra: float
    dose_missao: float
    sono_missao_delta: float     # adaptação individual do sono em órbita (h)


def profile_for(ast: str) -> Profile:
    """Perfil determinístico: o mesmo astronauta tem sempre o mesmo perfil."""
    rng = np.random.default_rng(zlib.crc32(ast.encode("utf-8")))
    u = rng.uniform
    return Profile(
        sono_h=u(6.8, 7.8), eficiencia=u(86, 93), despertares=u(1.0, 2.5),
        qualidade=u(6.5, 8.5), recuperacao=u(6.0, 8.5),
        rt_ms=u(240, 290), lapsos=u(0.5, 2.0),
        fc=u(52, 66), rmssd=u(35, 75),
        exercicio_terra=u(45, 75), kcal_base=u(2100, 2600), agua_base=u(1900, 2400),
        dose_terra=u(5, 9), dose_missao=u(250, 350), sono_missao_delta=rng.normal(0, 0.25),
    )


def _astro_number(ast: str) -> int:
    return zlib.crc32(ast.encode("utf-8")) % 100_000


def _latent(seed: int, ast: str) -> dict[int, float]:
    """Fator de sono (z) dia a dia, AR(1) estacionário. Positivo = noite pior."""
    rng = np.random.default_rng([seed, _astro_number(ast), 0])
    start, end = HORIZON
    values, x = {}, rng.normal()
    for day in range(start, end + 1):
        x = AR_PHI * x + np.sqrt(1 - AR_PHI ** 2) * rng.normal()
        values[day] = x
    return values


class PhysioSimulator:
    """Simula um astronauta. Use a MESMA semente para continuar uma simulação."""

    def __init__(self, ast: str, seed: int):
        self.ast, self.seed = ast, int(seed)
        self.p = profile_for(ast)
        self._latent = _latent(self.seed, ast)

    def day(self, dia: int, stress: float = 0.0, self_report_sleep: bool | None = None) -> dict:
        """Métricas de um dia. dia < 0 = Terra; dia >= 0 = missão.

        stress: piora adicional do fator de sono (cenário de declínio do demo_data).
        self_report_sleep: simular qualidade/recuperação? Padrão: só na Terra.
        """
        if not HORIZON[0] <= dia <= HORIZON[1]:
            raise ValueError(f"dia {dia} fora do horizonte simulado {HORIZON}")
        p, mission = self.p, dia >= 0
        rng = np.random.default_rng([self.seed, _astro_number(self.ast), dia + 10_000])
        f = self._latent[dia] + stress            # fator de sono do dia (z)
        n = rng.normal

        # --- Sono (actígrafo) e autorrelato --------------------------------------
        sono_mu = p.sono_h + (p.sono_missao_delta if mission else 0.0)
        row = {
            "sono_total_h": sono_mu - 0.45 * f + n(0, 0.20),
            "sono_eficiencia_pct": p.eficiencia - 2.5 * f + n(0, 1.0),
            "despertares_noite": rng.poisson(max(0.2, p.despertares + 0.6 * f)),
        }
        if self_report_sleep if self_report_sleep is not None else not mission:
            row["qualidade_sono_0a10"] = round(p.qualidade - 1.0 * f + n(0, 0.5))
            row["recuperacao_0a10"] = round(p.recuperacao - 1.0 * f + n(0, 0.6))

        # --- Atenção e cardiovascular ----------------------------------------------
        row["pvt_rt_medio_ms"] = p.rt_ms + 12 * f + n(0, 8)
        row["pvt_lapsos"] = rng.poisson(max(0.05, p.lapsos * np.exp(0.5 * f)))
        row["fc_repouso_bpm"] = p.fc + 1.5 * f + n(0, 1.5)
        row["vfc_rmssd_ms"] = p.rmssd * np.exp(-0.08 * f + n(0, 0.08))

        # --- Exercício (rotina semanal), nutrição e água -------------------------------
        weekday = (dia % 7 + 7) % 7
        rest = weekday < (REST_WEEKDAYS_MISSION if mission else REST_WEEKDAYS_EARTH)
        if mission:          # exercício planejado em órbita (~2 h efetivas)
            exercise = n(30, 10) if rest else n(120, 12)
        else:
            exercise = abs(n(10, 8)) if rest else n(p.exercicio_terra, 12)
        exercise = max(0.0, exercise)
        row["exercicio_min"] = exercise
        row["calorias_kcal"] = p.kcal_base + 7 * exercise + n(0, 150)
        row["refeicoes_dia"] = int(rng.choice([2, 3, 4], p=[0.1, 0.8, 0.1]))
        row["agua_ml"] = p.agua_base + 4 * exercise + n(0, 200)

        # --- Dosímetro ---------------------------------------------------------------------
        if mission:
            spike = rng.choice([1.0, 2.5], p=[0.98, 0.02])   # passagens/eventos solares (raros)
            dose = p.dose_missao * spike * np.exp(n(0, 0.10))
        else:
            dose = p.dose_terra * np.exp(n(0, 0.08))
        row["dose_diaria_uSv"] = dose

        return _finish(row)


def _finish(row: dict) -> dict:
    """Arredonda e mantém dentro de limites físicos (config.VALID_RANGES)."""
    digits = {"sono_total_h": 2, "sono_eficiencia_pct": 1, "pvt_rt_medio_ms": 1,
              "fc_repouso_bpm": 1, "vfc_rmssd_ms": 1, "dose_diaria_uSv": 1}
    tens = {"calorias_kcal", "agua_ml"}                   # arredonda de 10 em 10
    out = {}
    for metric, value in row.items():
        low, high = config.VALID_RANGES[metric]
        value = float(np.clip(value, low, high))
        if metric == "sono_eficiencia_pct":
            value = min(value, 99.0)
        if metric in digits:
            out[metric] = round(value, digits[metric])
        elif metric in tens:
            out[metric] = int(round(value / 10) * 10)
        else:
            out[metric] = int(round(value))
    return out


def new_seed() -> int:
    """Semente nova para uma simulação diferente das anteriores."""
    return int(np.random.default_rng().integers(1, 2**31 - 1))


def simulate(ast: str, days, seed: int, stress_fn=None) -> pd.DataFrame:
    """Tabela com um registro por dia. stress_fn(ast, dia) -> piora do sono (opcional)."""
    sim = PhysioSimulator(ast, seed)
    rows = []
    for dia in days:
        stress = stress_fn(ast, dia) if stress_fn else 0.0
        rows.append({"astronauta_id": ast, "dia_missao": dia, **sim.day(dia, stress)})
    return pd.DataFrame(rows)


# --- Validação -----------------------------------------------------------------------
def validate() -> list[str]:
    """Confere limites, correlações, continuidade, individualidade e reprodutibilidade."""
    problems = []
    crew = sorted(config.ASTRONAUTS)
    earth, mission = range(-30, 0), range(0, 60)

    def check(ok, msg):
        print(f"  [{'OK' if ok else 'FALHA'}] {msg}")
        if not ok:
            problems.append(msg)

    seeds = [new_seed() for _ in range(3)]
    frames = {s: pd.concat([simulate(a, list(earth) + list(mission), s) for a in crew])
              for s in seeds}
    df = pd.concat(frames.values())
    terra, missao = df[df.dia_missao < 0], df[df.dia_missao >= 0]

    print("Limites físicos")
    bad = [m for m in config.METRICS_SENSORS + config.METRICS_SELF_REPORT_SLEEP
           if m in df and not df[m].dropna().between(*config.VALID_RANGES[m]).all()]
    check(not bad, f"todas as métricas dentro de config.VALID_RANGES {bad or ''}")
    check(terra[config.METRICS_SELF_REPORT_SLEEP].notna().all().all()
          and missao[config.METRICS_SELF_REPORT_SLEEP].isna().all().all(),
          "sono/recuperação autorrelatados: simulados na Terra, vazios na missão (vêm do áudio)")

    print("Correlações (sentido esperado)")
    corr = terra.corr(numeric_only=True)
    for a, b, sign in [("sono_total_h", "pvt_rt_medio_ms", -1),
                       ("sono_eficiencia_pct", "qualidade_sono_0a10", +1),
                       ("qualidade_sono_0a10", "recuperacao_0a10", +1),
                       ("sono_total_h", "fc_repouso_bpm", -1),
                       ("exercicio_min", "calorias_kcal", +1),
                       ("exercicio_min", "agua_ml", +1)]:
        r = corr.loc[a, b]
        check(np.sign(r) == sign, f"{a} x {b}: r = {r:+.2f}")

    print("Continuidade dia a dia")
    one = frames[seeds[0]]                       # UMA simulação (sem misturar sementes)
    lag = (one[one.dia_missao < 0].sort_values(["astronauta_id", "dia_missao"])
           .groupby("astronauta_id")["sono_total_h"].apply(lambda s: s.autocorr(1)).mean())
    check(lag > 0.15, f"autocorrelação do sono entre dias seguidos: {lag:.2f}")

    print("Individualidade e consistência")
    means = one[one.dia_missao < 0].groupby("astronauta_id")["pvt_rt_medio_ms"].mean()
    check(means.max() - means.min() > 10, f"perfis diferentes entre astronautas (PVT médio: "
          + ", ".join(f"{a} {v:.0f} ms" for a, v in means.items()) + ")")
    a, b = frames[seeds[0]], frames[seeds[1]]
    check(not a["sono_total_h"].equals(b["sono_total_h"]), "sementes diferentes geram registros diferentes")
    diff = (a.groupby("astronauta_id")["fc_repouso_bpm"].mean()
            - b.groupby("astronauta_id")["fc_repouso_bpm"].mean()).abs().max()
    check(diff < 2.0, f"mesma distribuição entre simulações (FC média varia {diff:.2f} bpm)")
    again = simulate(crew[0], [5], seeds[0]).iloc[0]
    first = one[(one.astronauta_id == crew[0]) & (one.dia_missao == 5)].iloc[0]
    check(again.equals(first[again.index]),        # mesmas colunas (dia de missão)
          "mesma semente reproduz o mesmo dia (registro novo da missão é consistente)")

    print("Terra x missão")
    ratio = missao["dose_diaria_uSv"].median() / terra["dose_diaria_uSv"].median()
    check(ratio > 20, f"dose em órbita {ratio:.0f}x a da Terra")
    check(missao["exercicio_min"].mean() > terra["exercicio_min"].mean() + 40,
          f"exercício: Terra {terra['exercicio_min'].mean():.0f} min, missão "
          f"{missao['exercicio_min'].mean():.0f} min")

    print("Efeito do cenário de declínio")
    base = simulate(crew[0], range(-30, 0), seeds[0])["pvt_rt_medio_ms"].mean()
    worse = simulate(crew[0], range(-30, 0), seeds[0], lambda a, d: 1.0)["pvt_rt_medio_ms"].mean()
    check(worse > base + 8, f"declínio deixa o PVT mais lento ({base:.0f} -> {worse:.0f} ms)")
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor de simulação (dados SIMULADOS)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validar", help="roda os testes de coerência do simulador")
    ex = sub.add_parser("exemplo", help="mostra alguns dias simulados")
    ex.add_argument("astronauta", choices=sorted(config.ASTRONAUTS))
    ex.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.cmd == "validar":
        problems = validate()
        print("\nSimulador OK" if not problems else f"\n{len(problems)} problema(s)")
        raise SystemExit(1 if problems else 0)

    seed = args.seed or new_seed()
    print(f"SIMULADO — {args.astronauta}, semente {seed}")
    table = simulate(args.astronauta, [-3, -2, -1, 0, 1, 2], seed)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.drop(columns="astronauta_id").set_index("dia_missao").T)