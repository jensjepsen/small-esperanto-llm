"""Generate Esperanto word problems via Gemini Flash Lite.

One driver, many types. Each type's specifics (solver strategies, question
framings, item pool, number ranges, problem domain prose) live in the TYPES
dict at the top. Adding a new type = appending one TypeConfig entry.

Pipeline (shared, type-agnostic):
  build prompt -> Gemini call -> JSON parse -> verify arithmetic ->
  diversity gate (skeleton hash) -> append JSONL

Verifier (shared):
  1. Every `LHS = RHS` line in chain_eo evaluates correctly under sandboxed eval
  2. Final number in chain matches JSON `answer`

Usage:
  GOOGLE_API_KEY=... uv run --extra gemini python scripts/generate_word_problems.py \\
    --type ratio --n 100 --out data/word_problems/ratio.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_NAMES_FILE = PROJECT_ROOT / "src/esperanto_lm/ontology/sampler.py"
_CONCEPTS = PROJECT_ROOT / "src/esperanto_lm/ontology/data/concepts.jsonl"

# ── Pool loaders ──────────────────────────────────────────────────────────

_BAD_OBJECTS = {
    # body parts
    "brako", "dento", "dorso", "fingro", "kapo", "kolo", "korpo", "mano",
    "okulo", "orelo", "piedo", "ventro", "vosto", "ŝultro", "buŝo", "haŭto",
    "nazo", "lipo", "lango", "frunto", "mentono", "trunko", "kruro",
    "genuo", "kubuto", "ostoj", "sango", "haro", "ungo", "muskolo",
    "cerbo", "koro", "pulmo", "stomako", "rumpa", "hepato",
    # nature / non-divisible
    "ĉielo", "ĉielarko", "aŭroro", "fajro", "flako", "vento", "vojo",
    "duno", "vulkano", "ŝtuparo",
}


def load_names() -> list[str]:
    src = _NAMES_FILE.read_text()
    m = re.search(r"PERSON_NAMES\s*=\s*\[(.*?)\]", src, re.DOTALL)
    names = re.findall(r'"([a-zćĉĝĥĵŝŭ]+)"', m.group(1)) if m else []
    return [n.capitalize() for n in names]


def load_objects() -> list[str]:
    out = []
    for line in _CONCEPTS.open():
        r = json.loads(line)
        et = r.get("entity_type", "")
        if isinstance(et, list):
            et = et[0] if et else ""
        lem = r.get("lemma", "")
        if (et in ("artifact", "natural_object", "inanimate")
                and lem.endswith("o")
                and 3 <= len(lem) <= 12
                and "-" not in lem and " " not in lem
                and lem not in _BAD_OBJECTS):
            out.append(lem)
    return sorted(set(out))


PERSON_NAMES = load_names()
OBJECT_POOL = load_objects()


# ── Type configs ──────────────────────────────────────────────────────────

@dataclass
class TypeConfig:
    """Everything that makes a word-problem type distinct.

    `prompt_template` gets `.format(n=, strategy_block=, framing=, names=,
    items=, extras=)` — `extras` is a free-form string built from the type's
    extra_keys for per-type variables (ratios, percents, totals, etc).
    """
    description: str          # one-line summary, shown in --help
    strategies: dict[str, str]  # strategy_name -> example block
    framings: dict[str, str]    # framing_name -> hint string
    item_pool: list[str]        # objects to inject in prompt
    name_pool: list[str]        # names to inject (almost always PERSON_NAMES)
    extras_fn: "callable"       # (rng, strategy) -> str of extra constraints
    prompt_template: str        # uses {n}, {strategy_block}, {framing},
                                # {names}, {items}, {extras}
    require_integer: bool = False  # reject if final answer isn't a whole number


# ── RATIO ─────────────────────────────────────────────────────────────────
_RATIO_RATIOS = ["1:2", "2:3", "3:4", "1:3", "2:5", "3:5", "4:5", "1:4",
                 "3:7", "1:2:3", "2:3:5", "1:1:2", "1:3:4"]

RATIO = TypeConfig(
    description="divide a quantity in a:b ratio; ask for part / larger / diff / etc.",
    strategies={
        "parts": """STRATEGIO: dividu la totalon en partojn.
Ekzemplo:
  "ni dividu en 5 partojn (2+3=5).
  unu parto = 30 / 5 = 6.
  bert ricevas 3 partojn: 3 * 6 = 18.
  #### 18"
""",
        "algebra": """STRATEGIO: starigu algebran ekvacion kun variabla parto.
Ekzemplo:
  "estu x la valoro de unu parto.
  do anna ricevas 2x kaj bert ricevas 3x.
  la totalo: 2x + 3x = 30.
  do 5x = 30.
  x = 30 / 5 = 6.
  bert ricevas: 3x = 3 * 6 = 18.
  #### 18"
""",
        "fraction": """STRATEGIO: esprimu kiel frakcion de la tuto.
Ekzemplo:
  "la totalo de la proporciopartoj: 2 + 3 = 5.
  bert ricevas frakcion 3/5 de la totalo.
  do bert ricevas: 3 / 5 * 30 = 18.
  #### 18"
""",
        "diff": """STRATEGIO: kalkulu unue ambaŭ partojn, poste la DIFERENCON.
Ekzemplo (demando: kiom pli ricevas bert ol anna?):
  "ni dividu en 5 partojn (2+3=5).
  unu parto = 30 / 5 = 6.
  anna ricevas: 2 * 6 = 12.
  bert ricevas: 3 * 6 = 18.
  diferenco: 18 - 12 = 6.
  #### 6"
""",
    },
    framings={
        "direct": "Demandu kiom da aĵoj ricevas unu specifa persono.",
        "larger": "Demandu kiom da aĵoj ricevas la persono kun la PLI GRANDA parto.",
        "smaller": "Demandu kiom da aĵoj ricevas la persono kun la PLI MALGRANDA parto.",
        "diff": "Demandu kiom PLI da aĵoj havas unu ol la alia (diferenco).",
        "given-one": ("Sciigu ke unu persono ricevis X aĵojn, kaj demandu la "
                      "TOTALON aŭ kiom ricevis la alia."),
        "context": ("Vortumu kiel rakonton: lernejo, restoracio, festo, vendejo, "
                    "familio, klubo. Pasinta tempo. Konkretaj detaloj."),
    },
    item_pool=OBJECT_POOL,
    name_pool=PERSON_NAMES,
    extras_fn=lambda rng: f"Proporcioj el: {', '.join(rng.sample(_RATIO_RATIOS, min(7, len(_RATIO_RATIOS))))}\n"
                          f"Totaloj: entjeroj 10–300, divideblaj de la sumo de la proporcio.",
    prompt_template="""Generu {n} esperantajn matematikajn problemojn pri proporcio (ratio).
Ĉiu problemo dividas entjeran kvanton laŭ donita proporcio.

DEVIGE uzu personojn EL: {names}
DEVIGE uzu aĵojn EL: {items}

{extras}

KADRO POR LA DEMANDOJ: {framing}

{strategy_block}
Por ĉiu problemo, redonu JSON-objekton:
- "type": "ratio"
- "question_eo": la problemo (1–3 frazoj, ĝusta esperanta gramatiko)
- "chain_eo": solvo laŭ la STRATEGIO. ĈIU aritmetika paŝo sur PROPRA LINIO kun `=`. Fina linio: "#### N"
- "answer": entjero

Respondu NUR JSON-listo de {n} objektoj, sen ```markdown, sen alia teksto.
""",
)


# ── PERCENT (placeholder — to be expanded next) ───────────────────────────

# Common discount/markup/tax/tip percentages; integer-friendly bases keep
# math whole when paired with bases divisible by 100/percent.
_PCT_PERCENTS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80]

PERCENT = TypeConfig(
    description="percent of a quantity: discount, markup, tax, increase",
    strategies={
        "direct": """STRATEGIO: kalkulu rekte (P% de B = P/100 * B).
Ekzemplo (rabato 25% de 80):
  "rabato = 25 / 100 * 80 = 20.
  nova prezo = 80 - 20 = 60.
  #### 60"
""",
        "decimal": """STRATEGIO: konvertu procenton al decimalo unue.
Ekzemplo (25% de 80 kun decimalo):
  "25% kiel decimalo = 25 / 100 = 0.25.
  25% de 80 = 0.25 * 80 = 20.
  rezulto = 80 - 20 = 60.
  #### 60"
""",
        "multiplier": """STRATEGIO: uzu unu-multobligilon (1 - p/100 por rabato, 1 + p/100 por kresko).
Ekzemplo (kresko de 80 je 25%):
  "multobligilo = 1 + 25 / 100 = 1.25.
  rezulto = 80 * 1.25 = 100.
  #### 100"
""",
    },
    framings={
        "discount": ("Rabato: aĵo kostis B eŭrojn, rabatas P%. Demandu la NOVAN prezon."),
        "markup": ("Kresko: prezo komencas je B, pliiĝas je P%. Demandu la novan prezon."),
        "tax": ("Imposto/Pourboire: aĵo kostas B, aldoniĝas P% imposto. Demandu la totalon."),
        "of-amount": ("Kiom estas P% de B? (rekta procento de kvanto)"),
        "saving": ("Kiom da mono ŝparas oni per P% rabato sur B?"),
    },
    item_pool=OBJECT_POOL,
    name_pool=PERSON_NAMES,
    extras_fn=lambda rng: (
        f"Procentoj el: {', '.join(str(p) for p in rng.sample(_PCT_PERCENTS, min(6, len(_PCT_PERCENTS))))}\n"
        f"Bazaj kvantoj: entjeroj 20–500, elektitaj tiel ke la rezulto estu plejofte entjero "
        f"(do se procento estas 25%, bazo estu multoblo de 4; se 20%, multoblo de 5; ktp)."
    ),
    prompt_template="""Generu {n} esperantajn matematikajn problemojn pri procento.

DEVIGE uzu personojn EL: {names}
DEVIGE uzu aĵojn EL: {items}

{extras}

KADRO POR LA DEMANDOJ: {framing}

{strategy_block}
Por ĉiu problemo, redonu JSON-objekton:
- "type": "percent"
- "question_eo": problemo (1–3 frazoj, kunkrete-vortumita kaj kun valutaj/aĵaj detaloj)
- "chain_eo": solvo laŭ la STRATEGIO. ĈIU aritmetika paŝo sur PROPRA LINIO kun `=`. Fina linio: "#### N"
- "answer": entjero (aŭ decimalo se nemovita)

Respondu NUR JSON-listo, sen ```, sen alia teksto.
""",
)


# ── INVERSE-RATE ──────────────────────────────────────────────────────────

# Agent ↔ task pairs. Each entry is (singular nominative, plural-nominative
# task, typical "unit of work" like horoj/tagoj/minutoj). Keeps the prompt
# grounded so Gemini doesn't drift into incoherent "5 telephones paint walls"
# combinations.
_INV_SCENARIOS = [
    "laboristoj farbas muron",
    "pumpiloj plenigas naĝejon",
    "tubo plenigas akvujon (uzu pluralan: tuboj plenigas akvujon)",
    "maŝinoj presas libron",
    "rikoltistoj rikoltas kampon",
    "fosistoj fosas tranĉeon",
    "kuiristoj preparas manĝon por festo",
    "tajpistoj tajpas manuskripton",
    "robotoj kunmetas aŭton",
    "ĝardenistoj plantas arbojn en parko",
]

_INV_BASES = [
    # (n_workers, time_units) such that product is "nice"
    (2, 12), (3, 12), (4, 6), (3, 8), (5, 6), (6, 10), (2, 18),
    (4, 9), (3, 10), (5, 12), (8, 15), (6, 8),
]

INVERSE_RATE = TypeConfig(
    description="inverse proportion: more workers → less time; product is constant",
    strategies={
        "constant-product": """STRATEGIO: produkto laboristoj * tempo estas KONSTANTA.
ATENTU: pli da laboristoj signifas MALPLI da tempo (inversa rilato).
Ekzemplo (3 laboristoj farbas muron en 6 horoj; kiom da horoj por 6 laboristoj?):
  "konstanta produkto: 3 * 6 = 18 person-horoj.
  por 6 laboristoj: 6 * t = 18.
  t = 18 / 6 = 3.
  #### 3"
""",
        "per-unit": """STRATEGIO: kalkulu totalan laboron (person-horoj), poste dividu.
Ekzemplo (3 laboristoj * 6 horoj = 18 person-horoj de laboro):
  "totala laboro = 3 * 6 = 18 person-horoj.
  6 laboristoj bezonas: 18 / 6 = 3 horojn.
  #### 3"
""",
        "inverse-proportion": """STRATEGIO: skribu inversan proporcion w1/w2 = t2/t1.
Ekzemplo (3 laboristoj → 6 horoj; 6 laboristoj → ?):
  "inversa proporcio: w1 / w2 = t2 / t1.
  3 / 6 = t2 / 6.
  t2 = 6 * 3 / 6 = 3.
  #### 3"
""",
    },
    framings={
        "fewer-more": ("Donu komencan situacion (N1 laboristoj, T1 tempo). "
                       "Demandu: kiom da TEMPO bezonas pli/malpli da laboristoj?"),
        "find-workers": ("Donu komencan situacion (N1 laboristoj, T1 tempo). "
                         "Demandu: kiom da LABORISTOJ bezonas por fini en T2 tempo?"),
        "halving": ("Eksplicite duobligu aŭ duonigu la nombron de laboristoj; "
                    "demandu pri la nova tempo."),
        "context": ("Vortumu kiel rakonton: konstrulaboro, kuirejo, fabrikejo, "
                    "ĝardeno. Konkretaj detaloj. Pasinta tempo."),
    },
    item_pool=_INV_SCENARIOS,
    name_pool=PERSON_NAMES,
    extras_fn=lambda rng: (
        "Komencaj situacioj (W1 laboristoj × T1 tempo-unuoj) elektu el:\n"
        + "\n".join(f"  - {w} laboristoj × {t} tempo-unuoj = {w*t} person-unuoj"
                    for w, t in rng.sample(_INV_BASES, min(6, len(_INV_BASES))))
        + "\nLa NOVA nombro de laboristoj (W2) DEVAS esti tia ke "
          "W1*T1 / W2 estu entjero — elektu W2 kiel divizoro de W1*T1."
    ),
    prompt_template="""Generu {n} esperantajn matematikajn problemojn pri INVERSA proporcio (laboro-tempo).

KRITIKA: PLI da laboristoj signifas MALPLI da tempo. NE konfuzu kun rekta proporcio.

DEVIGE uzu nomojn EL: {names}
DEVIGE uzu scenarojn EL: {items}

{extras}

KADRO POR LA DEMANDOJ: {framing}

{strategy_block}
Por ĉiu problemo, redonu JSON-objekton:
- "type": "inverse-rate"
- "question_eo": problemo (1–3 frazoj, konkrete-vortumita kun la elektita scenaro)
- "chain_eo": solvo laŭ la STRATEGIO. ĈIU aritmetika paŝo sur PROPRA LINIO kun `=`. Fina linio: "#### N"
- "answer": entjero (la nova tempo aŭ nova laboristoj-nombro)

Respondu NUR JSON-listo, sen ```, sen alia teksto.
""",
)


# ── CONSECUTIVE INTEGERS ──────────────────────────────────────────────────

# Real-world hooks where consecutive integers make sense — keeps Gemini
# from inventing nonsensical scenarios.
_CONSEC_SCENARIOS = [
    "tri sinsekvaj paĝnumeroj en libro",
    "kvar sinsekvaj jaroj de medaloj",
    "tri sinsekvaj numeroj sur domoj en strato",
    "tri sinsekvaj numeroj de buslinioj",
    "kvar sinsekvaj numeroj de loĝejoj",
    "sinsekvaj aĝoj de gefratoj",
    "sinsekvaj numeroj de seĝoj en vico",
    "sinsekvaj tagoj de monato",
    "sinsekvaj kapsuloj de medikamento",
    "sinsekvaj numeroj de biletoj en loterio",
    "sinsekvaj rondoj en turniro",
]

# Sum-triples / quadruples / pairs that decompose cleanly (sum divisible
# by count, plus an arithmetic-progression offset).
_CONSEC_PARAMS = [
    # (count, sum, parity) — sum chosen so middle * count = sum exactly
    (3, 36, "any"),  (3, 51, "any"),  (3, 72, "any"),
    (4, 30, "any"),  (4, 50, "any"),  (4, 90, "any"),
    (5, 75, "any"),  (5, 100, "any"),
    (3, 60, "even"), (3, 75, "odd"),
    (4, 48, "even"), (4, 80, "even"),
]

CONSECUTIVE = TypeConfig(
    description="consecutive integers (any/even/odd) summing to a given total",
    strategies={
        "first-as-x": """STRATEGIO: la unua entjero estas x; la sekvaj estas x+1, x+2, x+3...
KRITIKA: sinsekvaj entjeroj diferenciĝas per EKZAKTE 1 (ne 2). Por sinsekvaj PARAJ aŭ NEPARAJ, diferenco estas 2.
Ekzemplo (sumo de tri sinsekvaj entjeroj estas 36; trovu la plej grandan):
  "estu x la unua entjero.
  la tri entjeroj: x, x + 1, x + 2.
  sumo: x + (x+1) + (x+2) = 36.
  3x + 3 = 36.
  3x = 36 - 3 = 33.
  x = 33 / 3 = 11.
  la plej granda estas x + 2 = 11 + 2 = 13.
  #### 13"
""",
        "middle-as-x": """STRATEGIO: por NEPARA nombro da sinsekvaj, nomu la MEZAN x.
La aliaj estas x-1, x+1 (por 3); aŭ x-2, x-1, x+1, x+2 (por 5).
Ekzemplo (sumo de tri sinsekvaj entjeroj estas 36; trovu ĉiujn):
  "estu x la meza entjero.
  la tri entjeroj: x - 1, x, x + 1.
  sumo: (x-1) + x + (x+1) = 36.
  3x = 36.
  x = 36 / 3 = 12.
  do la entjeroj: 11, 12, 13.
  la plej granda: 13.
  #### 13"
""",
        "average": """STRATEGIO: por sinsekvaj entjeroj, la meznombro = sumo / nombro.
Ekzemplo (sumo de tri sinsekvaj entjeroj = 36):
  "meznombro = 36 / 3 = 12.
  por tri sinsekvaj, la meza estas 12.
  do entjeroj: 11, 12, 13.
  la plej granda: 13.
  #### 13"
""",
    },
    framings={
        "find-smallest": "Demandu pri la PLEJ MALGRANDA entjero.",
        "find-largest": "Demandu pri la PLEJ GRANDA entjero.",
        "find-middle": "Demandu pri la MEZA entjero (uzu nur por NEPARA nombro da entjeroj).",
        "find-all": "Demandu liston de ĉiuj entjeroj (respondo = la plej granda, aŭ unu specifa).",
        "consecutive-even": ("Bazi sur sinsekvaj PARAJ entjeroj (ekz. 8, 10, 12). "
                             "Diferenco inter sinsekvaj estas 2."),
        "consecutive-odd": ("Bazi sur sinsekvaj NEPARAJ entjeroj (ekz. 7, 9, 11). "
                            "Diferenco inter sinsekvaj estas 2."),
    },
    item_pool=_CONSEC_SCENARIOS,
    name_pool=PERSON_NAMES,
    extras_fn=lambda rng: (
        "Parametroj (count, sum, parity) elektu el:\n"
        + "\n".join(f"  - {c} entjeroj sumantaj al {s} ({p})"
                    for c, s, p in rng.sample(_CONSEC_PARAMS, min(5, len(_CONSEC_PARAMS))))
    ),
    prompt_template="""Generu {n} esperantajn matematikajn problemojn pri SINSEKVAJ ENTJEROJ.

KRITIKAJ REGULOJ:
- Sinsekvaj entjeroj = x, x+1, x+2, x+3... (diferenco = 1)
- Sinsekvaj PARAJ entjeroj = x, x+2, x+4 (kie x estas para; diferenco = 2)
- Sinsekvaj NEPARAJ entjeroj = x, x+2, x+4 (kie x estas nepara; diferenco = 2)
- La FINA respondo DEVAS esti entjero. NE generu problemojn kies solvo donas decimalon (ekz. 18,5).
- Uzu NUR (count, sum) kombinaĵojn ĉi-suben, kiuj garantias entjerajn solvojn.

DEVIGE uzu nomojn EL: {names}
DEVIGE uzu scenarojn EL: {items}

{extras}

KADRO POR LA DEMANDOJ: {framing}

{strategy_block}
Por ĉiu problemo, redonu JSON-objekton:
- "type": "consecutive"
- "question_eo": problemo (1–3 frazoj, kun scenaro). Se kadro estas "consecutive-even/odd", la nombroj DEVAS esti paraj/neparaj.
- "chain_eo": solvo laŭ la STRATEGIO. ĈIU aritmetika paŝo sur PROPRA LINIO kun `=`. Fina linio: "#### N"
- "answer": ENTJERO (NE decimalo)

Respondu NUR JSON-listo, sen ```, sen alia teksto.
""",
    require_integer=True,
)


# ── COIN / TWO-COUNT LINEAR SYSTEM ────────────────────────────────────────

# (small_unit_value, big_unit_value, unit_label_singular, unit_label_plural,
#  currency_label). Drives both denomination and scenario noun.
_COIN_PAIRS = [
    (5, 10, "pencaĵo", "pencaĵoj", "pencoj"),
    (10, 25, "cendaĵo", "cendaĵoj", "cendoj"),
    (1, 2, "monero", "moneroj de 1 kaj 2 eŭroj", "eŭroj"),
    (5, 10, "bileto", "biletoj", "eŭroj"),
    (20, 50, "monero", "moneroj de 20 kaj 50 cendoj", "cendoj"),
    (2, 5, "poŝtmarko", "poŝtmarkoj", "eŭroj"),
    (10, 50, "bileto", "biletoj de 10 kaj 50 eŭroj", "eŭroj"),
]

# (count, total_value) tuples that decompose into integer counts of small/big.
# At generation time we sample a pair, then a (count, value) that yields ints.
# Picked so that x ∈ [1, count-1] for both possibilities (avoids degenerate
# "all one type" answers).
_COIN_INSTANCES = [
    # (small_val, big_val, total_count, total_value, count_of_big)
    (5, 10, 12, 95, 7),    # 5 small × 5 + 7 big × 10 = 25 + 70 = 95
    (5, 10, 20, 150, 10),  # 10 × 5 + 10 × 10 = 50 + 100 = 150
    (10, 25, 8, 125, 3),   # 5 × 10 + 3 × 25 = 50 + 75 = 125
    (10, 25, 15, 270, 8),  # 7 × 10 + 8 × 25 = 70 + 200 = 270
    (1, 2, 10, 16, 6),     # 4 × 1 + 6 × 2 = 4 + 12 = 16
    (1, 2, 25, 35, 10),    # 15 × 1 + 10 × 2 = 15 + 20 = 35
    (5, 10, 30, 220, 14),  # 16 × 5 + 14 × 10 = 80 + 140 = 220
    (20, 50, 12, 360, 4),  # 8 × 20 + 4 × 50 = 160 + 200 = 360
    (2, 5, 20, 76, 12),    # 8 × 2 + 12 × 5 = 16 + 60 = 76
]

COIN = TypeConfig(
    description="two-denomination count problem (coins/bills/stamps); solve for one count given total count + total value",
    strategies={
        "substitution": """STRATEGIO: estu x la nombro de UNU tipo. La alia tipo: (totalo - x). Skribu valoran ekvacion.
Ekzemplo (12 moneroj, 5 kaj 10 pencoj, totala valoro 95 pencoj. Kiom da 10-pencaj?):
  "estu x la nombro de 10-pencaj moneroj.
  do (12 - x) estas la nombro de 5-pencaj moneroj.
  totala valoro: 10 * x + 5 * (12 - x) = 95.
  10x + 60 - 5x = 95.
  5x = 95 - 60 = 35.
  x = 35 / 5 = 7.
  #### 7"
""",
        "system": """STRATEGIO: starigu DU ekvaciojn: nombra + valora. Solvu per anstataŭigo.
Ekzemplo (12 moneroj, 5 kaj 10 pencoj, totala valoro 95 pencoj):
  "estu a = nombro de 5-pencaj, b = nombro de 10-pencaj.
  ekvacio 1 (kvanto): a + b = 12.
  ekvacio 2 (valoro): 5a + 10b = 95.
  el ekv 1: a = 12 - b.
  anstataŭigu en ekv 2: 5 * (12 - b) + 10b = 95.
  60 - 5b + 10b = 95.
  5b = 95 - 60 = 35.
  b = 35 / 5 = 7.
  #### 7"
""",
        "assume-then-correct": """STRATEGIO: supozu unue ke ĈIUJ moneroj estas de la malpli granda valoro, kalkulu mankon, dividu per diferenco.
Ekzemplo (12 moneroj, 5 kaj 10 pencoj, totala valoro 95):
  "se ĉiuj 12 moneroj estus 5-pencaj, totalo estus: 5 * 12 = 60.
  manko: 95 - 60 = 35.
  ĉiu 10-penca anstataŭ 5-penca aldonas: 10 - 5 = 5.
  do nombro de 10-pencaj: 35 / 5 = 7.
  #### 7"
""",
    },
    framings={
        "find-big": "Donu totalan kvanton + totalan valoron. Demandu kiom da MALI-grandvaloraj moneroj.",
        "find-small": "Donu totalan kvanton + totalan valoron. Demandu kiom da PLI-malgrandvaloraj moneroj.",
        "scenario-coins": "Vortumu kun moneroj en monujo.",
        "scenario-tickets": "Vortumu kun teatrobiletoj de du prezoj.",
        "scenario-stamps": "Vortumu kun poŝtmarkoj de du valoroj.",
    },
    item_pool=[f"{p[3]} ({p[0]} kaj {p[1]} {p[4]})" for p in _COIN_PAIRS],
    name_pool=PERSON_NAMES,
    extras_fn=lambda rng: (
        "Por ĉiu problemo, uzu UNU el ĉi tiuj antaŭ-validigitaj instancoj:\n"
        + "\n".join(
            f"  - {tc} moneroj de {sv} kaj {bv} unuoj, totala valoro {tv} → "
            f"{tc-cb} de {sv}, {cb} de {bv}"
            for sv, bv, tc, tv, cb in rng.sample(_COIN_INSTANCES, min(5, len(_COIN_INSTANCES)))
        )
        + "\nLa demando NUR petu la nombron de UNU tipo. Verkila valoro de la respondo estas entjero inter 1 kaj (kvanto-1)."
    ),
    prompt_template="""Generu {n} esperantajn problemojn pri DU SPECOJ DE MONEROJ (aŭ biletoj/poŝtmarkoj) — du-variabla lineara sistemo.

Donata: totala KVANTO da moneroj + totala VALORO. Petata: kiom da unu tipo.

KRITIKA:
- La valoroj de la du tipoj DEVAS esti EKZAKTE tiuj specifitaj sub.
- La fina respondo DEVAS esti POZITIVA ENTJERO inter 1 kaj (totalo - 1).
- NE inventu novajn monerajn valorojn aŭ valutojn ekster la specifitaj.

DEVIGE uzu nomojn EL: {names}

{extras}

KADRO POR LA DEMANDOJ: {framing}

{strategy_block}
Por ĉiu problemo, redonu JSON-objekton:
- "type": "coin"
- "question_eo": problemo (1–3 frazoj, kun konkretaj valoroj kaj kvantoj)
- "chain_eo": solvo laŭ la STRATEGIO. ĈIU aritmetika paŝo sur PROPRA LINIO kun `=`. Fina linio: "#### N"
- "answer": entjero (kvanto de petata tipo)

Respondu NUR JSON-listo, sen ```, sen alia teksto.
""",
    require_integer=True,
)


# ── AGE ───────────────────────────────────────────────────────────────────

# Relation pairs (older_role, younger_role) — drives problem framing.
_AGE_RELATIONS = [
    "patrino kaj filino", "patro kaj filo", "avo kaj nepo", "avino kaj nepino",
    "instruisto kaj studento", "frato kaj fratino (pliaĝa kaj malpliaĝa)",
    "onklo kaj nevo", "onklino kaj nevino", "estro kaj asistanto",
    "mentoro kaj lernanto",
]

# Pre-validated instances guaranteed to yield integer ages.
# (young_age, old_age, multiplier, sum_now) — basic "now" problems.
_AGE_SIMPLE_NOW = [
    # (young, old, mul, sum)
    (8, 16, 2, 24),
    (9, 27, 3, 36),
    (5, 25, 5, 30),
    (9, 36, 4, 45),
    (8, 40, 5, 48),
    (12, 36, 3, 48),
    (7, 21, 3, 28),
    (10, 30, 3, 40),
    (6, 30, 5, 36),
    (15, 45, 3, 60),
]

# Time-shift instances:
# (young_now, old_now, ratio_now, t, ratio_later) — old = ratio_now * young
# AND (old + t) = ratio_later * (young + t)
# Computed: D = t*(ratio_later-1)/(ratio_now-ratio_later); must be int.
_AGE_TIME_SHIFT = [
    # (young, old, r_now, t, r_later)
    (10, 30, 3, 10, 2),   # in 10y: 40 = 2*20 ✓
    (4, 20, 5, 4, 3),     # in 4y: 24 = 3*8 ✓
    (20, 80, 4, 10, 3),   # in 10y: 90 = 3*30 ✓
    (6, 42, 7, 6, 4),     # in 6y: 48 = 4*12 ✓
    (3, 12, 4, 6, 2),     # in 6y: 18 = 2*9 ✓
    (15, 90, 6, 10, 4),   # in 10y: 100 = 4*25 ✓
    (8, 40, 5, 4, 4),     # in 4y: 44 = 4*11... 11≠12 — recompute
]
# Re-verify _AGE_TIME_SHIFT entries on load and drop any inconsistent ones.
_AGE_TIME_SHIFT = [
    inst for inst in _AGE_TIME_SHIFT
    if (inst[1] == inst[2] * inst[0]
        and inst[1] + inst[3] == inst[4] * (inst[0] + inst[3]))
]

AGE = TypeConfig(
    description="age relations (now / future / past); two-variable systems with multiplicative ratios",
    strategies={
        "simple-now": """STRATEGIO: NUN-aĝoj. Estu x la juna aĝo. La aĝo de la maljuna estas n*x.
KRITIKA: "trifoje pli aĝa" signifas EKZAKTE 3*x, NE x+3.
Ekzemplo (patrino 3x pli aĝa ol filino; sumo 36):
  "estu x la aĝo de la filino.
  patrino: 3 * x.
  sumo: x + 3 * x = 36.
  4x = 36.
  x = 36 / 4 = 9.
  do filino havas 9 jarojn.
  #### 9"
""",
        "time-shift": """STRATEGIO: NUNAJ aĝoj plus tempa ŝovo (post t jaroj).
KRITIKA: "post t jaroj" ambaŭ aĝoj kreskas per t.
Ekzemplo (patrino nun 3x filino; post 10 jaroj patrino 2x filino):
  "estu d la aĝo de la filino nun. patrino: 3 * d.
  post 10 jaroj: filino = d + 10, patrino = 3d + 10.
  patrino estos 2x filino: 3d + 10 = 2 * (d + 10).
  3d + 10 = 2d + 20.
  3d - 2d = 20 - 10.
  d = 10.
  #### 10"
""",
        "past-shift": """STRATEGIO: NUNAJ aĝoj plus retro-ŝovo (antaŭ t jaroj).
Ekzemplo (nun patrino estas 4x filino; antaŭ 5 jaroj estis 9x):
  "estu d aĝo de filino nun, patrino = 4d.
  antaŭ 5 jaroj: filino = d - 5, patrino = 4d - 5.
  tiama: 4d - 5 = 9 * (d - 5).
  4d - 5 = 9d - 45.
  -5d = -40.
  d = 8.
  #### 8"
""",
    },
    framings={
        "find-younger": "Demandu la aĝon de la pli juna persono.",
        "find-older": "Demandu la aĝon de la pli aĝa persono.",
        "find-future": "Demandu kiom aĝa estos unu el ili post N jaroj.",
        "natural-scene": ("Vortumu kun realisma kunteksto (familio, lernejo, "
                          "naskiĝtago, foto). Konkretaj detaloj."),
    },
    item_pool=_AGE_RELATIONS,
    name_pool=PERSON_NAMES,
    extras_fn=lambda rng: (
        "Por SIMPLA-NUN strategio, uzu UNU el ĉi tiuj instancoj:\n"
        + "\n".join(
            f"  - juna={y}, maljuna={o}, multobligilo={m}, sumo={s}"
            for y, o, m, s in rng.sample(_AGE_SIMPLE_NOW, min(4, len(_AGE_SIMPLE_NOW)))
        )
        + "\n\nPor TEMPA-ŜOVO strategio, uzu UNU el ĉi tiuj instancoj:\n"
        + "\n".join(
            f"  - juna_nun={y}, maljuna_nun={o}, ratio_nun={rn}, post={t} jaroj → ratio_post={rl}"
            for y, o, rn, t, rl in rng.sample(_AGE_TIME_SHIFT, min(4, len(_AGE_TIME_SHIFT)))
        )
        + "\n\nĈiuj aĝoj DEVAS esti pozitivaj entjeroj."
    ),
    prompt_template="""Generu {n} esperantajn aĝo-problemojn.

KRITIKA LINGVO-NOTO:
- "N-foje pli aĝa" = N * aĝo (multiplika; ekz. "trifoje pli aĝa" = 3x).
- "N jarojn pli aĝa" = aĝo + N (aldona; ekz. "10 jarojn pli aĝa" = a + 10).
- "post N jaroj" = ambaŭ personoj havas +N jarojn.
- "antaŭ N jaroj" = ambaŭ personoj havis -N jarojn.

DEVIGE uzu nomojn EL: {names}
DEVIGE uzu rolojn EL: {items}

{extras}

KADRO POR LA DEMANDOJ: {framing}

{strategy_block}
Por ĉiu problemo, redonu JSON-objekton:
- "type": "age"
- "question_eo": problemo (2–4 frazoj, klara kaj realisma; uzu nomon kaj/aŭ rolon)
- "chain_eo": solvo laŭ la STRATEGIO. ĈIU aritmetika paŝo sur PROPRA LINIO kun `=`. Fina linio: "#### N"
- "answer": entjero (pozitiva)

Respondu NUR JSON-listo, sen ```, sen alia teksto.
""",
    require_integer=True,
)


# ── MIXTURE ───────────────────────────────────────────────────────────────

# Solvent/solute pairs to keep the prose grounded.
_MIX_SCENARIOS = [
    "salakvo (akvo + salo)",
    "sukerakvo (akvo + sukero)",
    "acida solvaĵo (akvo + acido)",
    "alkohola solvaĵo (akvo + alkoholo)",
    "kafosolvaĵo (akvo + kafopulvoro)",
    "lakta solvaĵo (akvo + lakto)",
    "frostiga solvaĵo (akvo + antifrosto)",
    "fertilizilo (akvo + nutraĵo)",
]

# Pre-validated DILUTION instances (add pure solvent to lower concentration).
# (V1, P1, P2, X_add) where X_add = V1 * (P1/P2 - 1).
_MIX_DILUTE = [
    (200, 10, 5, 200),      # classic from SFT-16k failure
    (150, 20, 15, 50),
    (300, 15, 10, 150),
    (400, 25, 10, 600),
    (100, 40, 20, 100),
    (250, 12, 6, 250),
    (180, 30, 20, 90),
    (500, 8, 4, 500),
]

# Pre-validated MIXING-TWO-SOLUTIONS instances.
# (V1, P1, V2, P2, V_total, P_total) — V1 of P1% + V2 of P2% = V_total of P_total%.
_MIX_BLEND = [
    (100, 20, 100, 10, 200, 15),
    (200, 30, 200, 10, 400, 20),
    (100, 40, 300, 20, 400, 25),
    (50, 50, 150, 10, 200, 20),
    (300, 25, 200, 15, 500, 21),  # 75+30=105, 105/500=21
    (200, 35, 200, 15, 400, 25),
]

# Pre-validated CONCENTRATION-UP instances (add pure solute).
# Have V of P% solution, add X g of pure solute to get P2%.
# Amount of solute now: V*P/100 + X. New total: V + X. Set = P2/100.
# Solve for X: X = V*(P2-P)/(100-P2).
_MIX_CONCENTRATE = [
    (200, 10, 20, 22),   # 200ml of 10%: 20g salt; add X salt for 20%: (20+X)/(200+X) = 0.20 → X = 25? Recompute
]
# Recompute all _MIX_CONCENTRATE on load:
_MIX_CONCENTRATE = []
for V, P1, P2_target in [(200, 10, 20), (300, 15, 30), (100, 5, 25), (400, 10, 20),
                          (250, 20, 50), (150, 8, 20)]:
    # solute_now = V * P1 / 100; let X = added pure solute (g, equivalent units to V)
    # (solute_now + X) / (V + X) = P2_target / 100
    # 100 * (V*P1/100 + X) = P2_target * (V + X)
    # V*P1 + 100*X = P2*V + P2*X
    # X * (100 - P2) = V * (P2 - P1)
    # X = V * (P2 - P1) / (100 - P2)
    num = V * (P2_target - P1)
    den = 100 - P2_target
    if den != 0 and num % den == 0 and num // den > 0:
        _MIX_CONCENTRATE.append((V, P1, P2_target, num // den))

MIXTURE = TypeConfig(
    description="solution mixture: dilute by adding solvent, concentrate by adding solute, or blend two solutions",
    strategies={
        "conservation": """STRATEGIO: la KVANTO de solvato (salt/sukero/acido) restas konstanta dum DILUTO.
KRITIKA: aldoni pura akvo NE ŝanĝas la salokvanton, sed nur la solvaĵan volumon.
Ekzemplo (200 ml de 10% solvaĵo → diluti al 5%; kiom da pura akvo aldoni?):
  "kvanto de salo: 200 * 10 / 100 = 20 g.
  fina koncentriĝo: 5%, do fina volumo V_f.
  20 / V_f = 5 / 100 → V_f = 20 * 100 / 5 = 400 ml.
  aldoni akvo: 400 - 200 = 200 ml.
  #### 200"
""",
        "concentration-equation": """STRATEGIO: skribu rektan koncentriĝan ekvacion kaj solvu.
Ekzemplo (V1=300, P1=15%, P2=10%; aldoni akvo X):
  "salo konstanta = 300 * 15 / 100 = 45 g.
  ekvacio: 45 / (300 + X) = 10 / 100.
  45 * 100 = 10 * (300 + X).
  4500 = 3000 + 10 * X.
  10 * X = 4500 - 3000 = 1500.
  X = 1500 / 10 = 150.
  #### 150"
""",
        "weighted-blend": """STRATEGIO: por MIKSI du solvaĵojn, uzu pezitan averaĝon.
Ekzemplo (100 ml de 20% + 100 ml de 10% → kia koncentriĝo de finita 200 ml?):
  "totala salo: 100 * 20 / 100 + 100 * 10 / 100 = 20 + 10 = 30 g.
  totala volumo: 100 + 100 = 200 ml.
  finita koncentriĝo: 30 / 200 * 100 = 15.
  #### 15"
""",
    },
    framings={
        "dilute-find-water": ("DILUTO: donu V1 + P1 + cela P2. Demandu kiom da pura akvo "
                              "aldoni por atingi P2."),
        "concentrate-find-solute": ("KONCENTRADO: donu V + P1 + cela P2. Demandu kiom da "
                                    "pura solvato aldoni."),
        "blend-find-concentration": ("MIKSADO: donu V1+P1 + V2+P2. Demandu finitan procenton."),
        "blend-find-volume": ("MIKSADO: donu P1, P2, cela P_finita kaj unu volumon. "
                              "Demandu la alian volumon."),
    },
    item_pool=_MIX_SCENARIOS,
    name_pool=PERSON_NAMES,
    extras_fn=lambda rng: (
        "Por DILUTO uzu UNU instancon:\n"
        + "\n".join(f"  - V1={v}, P1={p1}%, cela P2={p2}% → aldoni {x} ml puran akvon"
                    for v, p1, p2, x in rng.sample(_MIX_DILUTE, min(4, len(_MIX_DILUTE))))
        + ("\n\nPor MIKSADO uzu UNU instancon:\n"
           + "\n".join(f"  - {v1}ml de {p1}% + {v2}ml de {p2}% → {vt}ml de {pt}%"
                       for v1, p1, v2, p2, vt, pt in rng.sample(_MIX_BLEND, min(4, len(_MIX_BLEND))))
           if _MIX_BLEND else "")
        + ("\n\nPor KONCENTRADO uzu UNU instancon:\n"
           + "\n".join(f"  - V={v}, P1={p1}%, cela P2={p2}% → aldoni {x} unuojn puran solvaton"
                       for v, p1, p2, x in rng.sample(_MIX_CONCENTRATE, min(3, len(_MIX_CONCENTRATE))))
           if _MIX_CONCENTRATE else "")
        + "\n\nUzu NUR la donitajn instancojn. Ne inventu novajn nombrojn."
    ),
    prompt_template="""Generu {n} esperantajn problemojn pri MIKSAĴOJ kaj SOLVAĴOJ.

KRITIKAJ KONCEPTOJ:
- DILUTO: aldoni puran SOLVANTON (akvon) reduktas la koncentriĝon SED ne ŝanĝas la kvanton de solvato.
- KONCENTRADO: aldoni puran SOLVATON kreskigas ambaŭ — la solvatkvanton kaj la totalan volumon.
- MIKSADO: kombini du solvaĵojn → nova procento = totala solvato / totala volumo.

DEVIGE uzu nomojn EL: {names}
DEVIGE uzu scenarojn EL: {items}

{extras}

KADRO POR LA DEMANDOJ: {framing}

{strategy_block}
Por ĉiu problemo, redonu JSON-objekton:
- "type": "mixture"
- "question_eo": problemo (2–4 frazoj, kun konkretaj volumoj kaj procentoj)
- "chain_eo": solvo laŭ la STRATEGIO. ĈIU aritmetika paŝo sur PROPRA LINIO kun `=`. Fina linio: "#### N"
- "answer": entjero

Respondu NUR JSON-listo, sen ```, sen alia teksto.
""",
    require_integer=True,
)


# ── DISTANCE / RATE / TIME ────────────────────────────────────────────────

_DIST_VEHICLES = [
    "aŭto", "biciklo", "trajno", "motorciklo", "kamiono", "buso",
    "ŝipo", "aviadilo", "kuristo", "boato", "skooter",
]

# Simple D=R*T instances (D in km, R in km/h, T in h). Integer triple.
_DIST_SIMPLE = [
    (180, 60, 3),
    (240, 80, 3),
    (300, 75, 4),
    (120, 60, 2),
    (320, 80, 4),
    (450, 90, 5),
    (200, 50, 4),
    (360, 60, 6),
    (480, 120, 4),
    (210, 70, 3),
]

# Catch-up: A starts first at rate ra for h hours, then B starts at rate rb (rb>ra).
# B catches A after t hours, where t = ra*h/(rb-ra). Picked so t is integer.
_DIST_CATCHUP = [
    # (ra, h, rb, t)
    (80, 2, 120, 4),     # the SFT-16k probe failure
    (50, 4, 70, 10),
    (40, 2, 60, 4),
    (60, 2, 100, 3),
    (75, 2, 125, 3),
    (40, 3, 80, 3),
    (50, 6, 80, 10),
    (60, 3, 90, 6),
    (40, 4, 60, 8),
    (45, 4, 75, 6),
]

# Meeting (opposite directions): start D apart, speeds r1, r2 toward each other.
# Meet at t = D/(r1+r2). Picked so t integer.
_DIST_MEET = [
    # (r1, r2, D, t)
    (40, 60, 200, 2),
    (50, 70, 240, 2),
    (80, 120, 400, 2),
    (30, 50, 240, 3),
    (60, 90, 300, 2),
    (45, 75, 360, 3),
    (70, 50, 360, 3),
    (40, 80, 360, 3),
]

# Round-trip average speed: out at r1, back at r2; avg = 2*r1*r2/(r1+r2). Picked integer.
_DIST_AVG = [
    # (r1, r2, avg)
    (60, 40, 48),
    (80, 120, 96),
    (60, 30, 40),
    (40, 60, 48),
    (50, 75, 60),
    (90, 60, 72),
]

DISTANCE = TypeConfig(
    description="distance/rate/time: simple D=RT, catch-up, meeting head-on, round-trip average",
    strategies={
        "direct-formula": """STRATEGIO: D = R * T. Anstataŭigu kaj solvu.
Ekzemplo (aŭto je 60 km/h dum 3 horoj):
  "distanco = rapideco * tempo.
  D = 60 * 3 = 180.
  #### 180"
""",
        "catch-up": """STRATEGIO: PERSEKUTO en sama direkto. Tempo t kiam B atingas A:
  rb * t = ra * (t + h), kie h estas tempo-avanco de A.
  Ekvacio: rb*t - ra*t = ra*h → t = ra*h / (rb-ra).
KRITIKA: bezonata rb > ra; alie B neniam atingas A.
Ekzemplo (A: 80 km/h, 2h pli frue; B: 120 km/h; t = ?):
  "distanco de A post t horoj de B-foriro: 80 * (t + 2).
  distanco de B post t horoj: 120 * t.
  egalu: 120 * t = 80 * (t + 2).
  120t = 80t + 160.
  120t - 80t = 160.
  40t = 160.
  t = 160 / 40 = 4.
  #### 4"
""",
        "meeting": """STRATEGIO: RENKONTIĜO en kontraŭa direkto. Sumigo de rapidecoj:
  (r1 + r2) * t = D.
Ekzemplo (du objektoj 200 km dise, je 40 kaj 60 km/h):
  "sumo de rapidecoj: 40 + 60 = 100 km/h.
  tempo: D / (r1+r2) = 200 / 100.
  t = 200 / 100 = 2.
  #### 2"
""",
        "average-speed": """STRATEGIO: MEZUMA RAPIDECO por rondiro (aller-retour): NE simpla averaĝo!
  formulo: avg = 2 * r1 * r2 / (r1 + r2).
KRITIKA: ne uzu (r1+r2)/2 — tio estas erara por sama distanco kun malsamaj rapidecoj.
Ekzemplo (60 km/h iri, 40 km/h reveni):
  "averaĝa rapideco = 2 * r1 * r2 / (r1 + r2).
  = 2 * 60 * 40 / (60 + 40).
  = 4800 / 100.
  = 48.
  #### 48"
""",
    },
    framings={
        "simple-distance": "DIREKTA: donu R, T → demandu D. Aŭ donu D, R → T. Aŭ D, T → R.",
        "catch-up": "PERSEKUTO: du veturiloj samdirektaj, ekfaras je diversaj tempoj.",
        "meeting": "RENKONTIĜO: du veturiloj kontraŭ-direktaj, ekfaras samtempe de du punktoj.",
        "round-trip": "RONDIRO: iri kun unu rapideco, reveni kun alia → demandu mezuman rapidecon.",
    },
    item_pool=_DIST_VEHICLES,
    name_pool=PERSON_NAMES,
    extras_fn=lambda rng, strategy: {
        "direct-formula": (
            "Uzu UNU el ĉi tiuj D=R×T-triopoj:\n"
            + "\n".join(f"  - D={d} km, R={r} km/h, T={t} h"
                        for d, r, t in rng.sample(_DIST_SIMPLE, min(4, len(_DIST_SIMPLE))))
            + "\nProblemo: donu DU el la tri valoroj, demandu la trian."
        ),
        "catch-up": (
            "Ĉiu problemo DEVAS havi DU veturilojn samdirekte, kun UNU "
            "ekironta PLI FRUE.\nUzu UNU el ĉi tiuj instancoj:\n"
            + "\n".join(f"  - veturilo A: rapideco {ra} km/h, ekiris {h} horojn pli frue. "
                        f"Veturilo B: rapideco {rb} km/h ekiras. Atingo post t={t} horoj."
                        for ra, h, rb, t in rng.sample(_DIST_CATCHUP, min(4, len(_DIST_CATCHUP))))
        ),
        "meeting": (
            "Ĉiu problemo DEVAS havi DU veturilojn ekirantajn SAMTEMPE el du "
            "punktoj distancaj je D km, moviĝantajn UNU AL LA ALIA.\n"
            "Uzu UNU el ĉi tiuj instancoj:\n"
            + "\n".join(f"  - rapideco1={r1} km/h, rapideco2={r2} km/h, distanco={d} km → "
                        f"renkontiĝo post t={t} horoj"
                        for r1, r2, d, t in rng.sample(_DIST_MEET, min(4, len(_DIST_MEET))))
        ),
        "average-speed": (
            "Ĉiu problemo DEVAS havi RONDIRON: persono iras de A al B kun unu "
            "rapideco kaj revenas kun ALIA rapideco. Demandu la MEZUMAN rapidecon "
            "por la TUTA vojaĝo. La distanco A-B estas la sama por ambaŭ direktoj.\n"
            "Uzu UNU el ĉi tiuj instancoj:\n"
            + "\n".join(f"  - iri je {r1} km/h, reveni je {r2} km/h → mezuma {a} km/h"
                        for r1, r2, a in rng.sample(_DIST_AVG, min(4, len(_DIST_AVG))))
            + "\nNE uzu simplan averaĝon (r1+r2)/2; uzu 2*r1*r2/(r1+r2)."
        ),
    }[strategy] + "\n\nLa fina respondo DEVAS esti pozitiva entjero. Uzu NUR la donitajn instancojn.",
    prompt_template="""Generu {n} esperantajn problemojn pri DISTANCO / RAPIDECO / TEMPO.

KRITIKA:
- D = R * T (sed atentu pri unuoj — uzu km/h kun horoj).
- PERSEKUTO: pli rapida atingas pli malrapidan; bezonata pli granda rapideco.
- RENKONTIĜO: rapidecoj sumiĝas, NE multobligas.
- MEZUMA RAPIDECO de rondiro NE estas (r1+r2)/2; uzu 2*r1*r2/(r1+r2).
- La fina respondo DEVAS esti pozitiva entjero.

DEVIGE uzu nomojn EL: {names}
DEVIGE uzu veturilojn EL: {items}

{extras}

KADRO POR LA DEMANDOJ: {framing}

{strategy_block}
Por ĉiu problemo, redonu JSON-objekton:
- "type": "distance"
- "question_eo": problemo (2–4 frazoj, konkrete-vortumita)
- "chain_eo": solvo laŭ la STRATEGIO. ĈIU aritmetika paŝo sur PROPRA LINIO kun `=`. Fina linio: "#### N"
- "answer": entjero (km, h, aŭ km/h depende de demando)

Respondu NUR JSON-listo, sen ```, sen alia teksto.
""",
    require_integer=True,
)


TYPES = {
    "ratio": RATIO,
    "percent": PERCENT,
    "inverse-rate": INVERSE_RATE,
    "consecutive": CONSECUTIVE,
    "coin": COIN,
    "age": AGE,
    "mixture": MIXTURE,
    "distance": DISTANCE,
}


# ── Shared verifier / parser / driver ─────────────────────────────────────

_SAFE_EXPR = re.compile(r"^[\d\s+\-*/().]+$")
# Require non-alphabetic neighbors on both sides so partial fragments of
# mixed-symbol-and-number lines (e.g. "4d + 10 = 3d + 30") don't yield
# the misleading slice "+ 10 = 3" with mismatched truth value.
# Use a broad alphabetic class covering ASCII + EO diacritics + uppercase.
_ALPHA = "A-Za-zĉĝĥĵŝŭĈĜĤĴŜŬ"
_EQ_LINE = re.compile(
    rf"(?<![{_ALPHA}])([\d\s+\-*/().]+(?:\s*=\s*[\d\s+\-*/().]+)+)(?![{_ALPHA}])"
)
_FINAL_HASH = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
_TRAILING_NUM = re.compile(r"(-?\d+(?:\.\d+)?)[^\d]*$")


def safe_eval(expr: str) -> float | None:
    expr = expr.strip()
    if not _SAFE_EXPR.match(expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def verify_question_numbers_in_chain(question: str, chain: str) -> tuple[bool, str]:
    """Every multi-digit non-percent number in the question must appear in the chain.
    Catches the failure mode where Gemini silently changes problem parameters
    (e.g. question says sum=80, chain solves for sum=84).

    Skips numbers followed by `%` in the question — these legitimately get
    transformed (20% → 0.20 → 20/100 → 0,20 etc.) and matching all forms is
    brittle. Multi-digit non-percent numbers (counts, volumes, totals) are
    what we actually care about preserving."""
    # strip out percent occurrences before extraction
    q_stripped = re.sub(r"\d+\s*%", " ", question)
    q_nums = set(re.findall(r"\b\d{2,}\b", q_stripped))
    chain_norm = re.sub(r"(?<=\d),(?=\d)", ".", chain)
    chain_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", chain_norm))
    missing = [n for n in q_nums if n not in chain_nums]
    if missing:
        return False, f"question-num-missing: {missing}"
    return True, ""


def verify_chain(chain: str, answer) -> tuple[bool, str]:
    """All `LHS = RHS` lines must hold; final number must match `answer`."""
    # Normalize EO decimal comma (0,4 → 0.4). Only between digits, so we don't
    # corrupt list separators like "Anna, Bert".
    chain = re.sub(r"(?<=\d),(?=\d)", ".", chain)
    for match in _EQ_LINE.finditer(chain):
        parts = [p.strip() for p in match.group(1).split("=")]
        if not re.search(r"[+\-*/]", parts[0]):
            continue
        for i in range(len(parts) - 1):
            lhs = safe_eval(parts[i])
            rhs = safe_eval(parts[i + 1])
            if lhs is None or rhs is None:
                continue
            if abs(lhs - rhs) > 1e-6:
                return False, f"arith-mismatch: {parts[i]} != {parts[i+1]}"
    m = _FINAL_HASH.search(chain) or _TRAILING_NUM.search(chain.strip())
    if not m:
        return False, "no-final-number"
    chain_final = float(m.group(1))
    try:
        ans = float(answer)
    except (TypeError, ValueError):
        return False, f"non-numeric-answer: {answer!r}"
    if abs(chain_final - ans) > 1e-6:
        return False, f"chain-vs-json-mismatch: {chain_final} != {ans}"
    return True, ""


def skeleton(question: str, name_pool: list[str], item_pool: list[str]) -> str:
    s = question.lower()
    for n in name_pool:
        s = s.replace(n.lower(), "<X>")
    for it in item_pool:
        s = re.sub(rf"\b{re.escape(it)}[a-zĉĝĥĵŝŭ]*\b", "<O>", s)
    s = re.sub(r"\d+(?:\.\d+)?", "<N>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_response(text: str) -> list[dict]:
    text = text.strip()
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip()
            if chunk.startswith("json"):
                text = chunk[4:].strip()
                break
            if chunk.startswith("["):
                text = chunk
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("["), text.rfind("]")
        if i >= 0 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                pass
    return []


def build_prompt(cfg: TypeConfig, n: int, strategy: str, framing: str,
                 rng: random.Random) -> str:
    names = rng.sample(cfg.name_pool, min(2 * n + 2, len(cfg.name_pool)))
    items = rng.sample(cfg.item_pool, min(n + 4, len(cfg.item_pool)))
    # Try new (rng, strategy) signature; fall back to old (rng,) for types not yet migrated.
    try:
        extras = cfg.extras_fn(rng, strategy)
    except TypeError:
        extras = cfg.extras_fn(rng)
    return cfg.prompt_template.format(
        n=n,
        strategy_block=cfg.strategies[strategy],
        framing=cfg.framings[framing],
        names=", ".join(names),
        items=", ".join(items),
        extras=extras,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", required=True, choices=list(TYPES),
                    help="problem type: " + " | ".join(
                        f"{k} ({v.description})" for k, v in TYPES.items()))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    ap.add_argument("--max-dup", type=int, default=3)
    ap.add_argument("--max-calls", type=int, default=0)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--strategies", default=None,
                    help="comma-list (default: all for the type)")
    ap.add_argument("--framings", default=None,
                    help="comma-list (default: all for the type)")
    args = ap.parse_args()

    cfg = TYPES[args.type]
    strategies = (args.strategies or ",".join(cfg.strategies)).split(",")
    framings = (args.framings or ",".join(cfg.framings)).split(",")
    bad = [s for s in strategies if s not in cfg.strategies]
    if bad:
        print(f"unknown strategies: {bad}; valid: {list(cfg.strategies)}", file=sys.stderr)
        sys.exit(2)
    bad = [f for f in framings if f not in cfg.framings]
    if bad:
        print(f"unknown framings: {bad}; valid: {list(cfg.framings)}", file=sys.stderr)
        sys.exit(2)

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY or pass --api-key", file=sys.stderr)
        sys.exit(2)

    from google import genai
    client = genai.Client(api_key=api_key)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    skel_counts = Counter()
    existing = 0
    if args.out.exists():
        with args.out.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                    skel_counts[skeleton(row["question_eo"], cfg.name_pool, cfg.item_pool)] += 1
                    existing += 1
                except Exception:
                    continue
        print(f"resume: {existing} already in {args.out}", flush=True)

    stats = Counter()
    accepted = existing
    calls = 0
    t0 = time.time()
    out_f = args.out.open("a")

    while accepted < args.n:
        if args.max_calls and calls >= args.max_calls:
            print(f"hit --max-calls={args.max_calls}; stopping", flush=True)
            break
        strategy = strategies[calls % len(strategies)]
        framing = framings[calls % len(framings)]
        rng = random.Random(calls * 1009 + accepted)
        calls += 1
        prompt = build_prompt(cfg, args.batch_size, strategy, framing, rng)
        try:
            resp = client.models.generate_content(model=args.model, contents=prompt)
            text = resp.text or ""
        except Exception as e:
            stats["api-error"] += 1
            print(f"  [call {calls}] API error: {e}", flush=True)
            time.sleep(2)
            continue

        items = parse_response(text)
        if not items:
            stats["parse-fail"] += 1
            print(f"  [call {calls}] parse fail; first 200: {text[:200]!r}", flush=True)
            continue

        for it in items:
            stats["total"] += 1
            q = it.get("question_eo", "").strip()
            chain = it.get("chain_eo", "").strip()
            ans = it.get("answer")
            if not q or not chain or ans is None:
                stats["missing-field"] += 1
                continue
            ok, why = verify_chain(chain, ans)
            if not ok:
                stats[f"verify:{why.split(':')[0]}"] += 1
                continue
            ok, why = verify_question_numbers_in_chain(q, chain)
            if not ok:
                stats[f"verify:{why.split(':')[0]}"] += 1
                continue
            if cfg.require_integer:
                try:
                    af = float(ans)
                    if abs(af - round(af)) > 1e-6:
                        stats["verify:non-integer"] += 1
                        continue
                except (TypeError, ValueError):
                    stats["verify:non-numeric"] += 1
                    continue
            sk = skeleton(q, cfg.name_pool, cfg.item_pool)
            if skel_counts[sk] >= args.max_dup:
                stats["dup-skeleton"] += 1
                continue
            skel_counts[sk] += 1
            row = {
                "type": args.type, "question_eo": q, "chain_eo": chain,
                "answer": float(ans) if "." in str(ans) else int(ans),
                "strategy": strategy, "framing": framing,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            accepted += 1
            stats["accepted"] += 1

        rate = (accepted - existing) / max(1, time.time() - t0) * 60
        print(f"  [call {calls}] accepted={accepted}/{args.n}  "
              f"({rate:.1f}/min)  stats={dict(stats)}", flush=True)

    out_f.close()
    print(f"\ndone: {accepted}/{args.n} accepted, {calls} calls, {time.time()-t0:.0f}s")
    print(f"  stats: {dict(stats)}")
    print(f"  unique skeletons: {len(skel_counts)}")
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()
