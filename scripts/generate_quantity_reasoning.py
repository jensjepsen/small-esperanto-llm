"""Generate quantity-reasoning word problems with digit-level CoT answers.

Each example pairs a natural-Esperanto word problem with a step-by-step answer
that (a) restates the given quantities, (b) identifies the operation, and
(c) shows the digit-by-digit arithmetic decomposition (reusing the existing
arithmetic_cot decompose_* functions). This trains both word-problem parsing
and verifiable arithmetic in one shot.

Output: data/sft/sft_quantity_reasoning.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Reuse the digit-by-digit decomposition from the arithmetic CoT generator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_arithmetic_cot import decompose_add, decompose_sub, decompose_mul, decompose_div


# ---- Vocabulary ---------------------------------------------------------

MALE_NAMES = ["Petro", "Ivan", "Karlo", "Andreo", "Pablo", "Marko", "Tomas"]
FEMALE_NAMES = ["Maria", "Anna", "Olga", "Ester", "Sofia", "Helena", "Lara", "Klara"]
NAMES = MALE_NAMES + FEMALE_NAMES


def _pron(name: str) -> tuple[str, str, str]:
    """Return (subject, object, possessive) for a name."""
    if name in FEMALE_NAMES:
        return ("ŝi", "ŝin", "ŝia")
    return ("li", "lin", "lia")

# Each entry: (sg_nom, sg_acc, pl_nom, pl_acc)
# pl_nom is used after "kiom da", "estas N", "X estas..."
# pl_acc is used after transitive verbs (havas, tenas, aĉetas, ricevas)
ITEMS = [
    ("pomo",      "pomon",      "pomoj",      "pomojn"),
    ("libro",     "libron",     "libroj",     "librojn"),
    ("krajono",   "krajonon",   "krajonoj",   "krajonojn"),
    ("moneto",    "moneton",    "monetoj",    "monetojn"),
    ("floro",     "floron",     "floroj",     "florojn"),
    ("ovo",       "ovon",       "ovoj",       "ovojn"),
    ("ŝtono",     "ŝtonon",     "ŝtonoj",     "ŝtonojn"),
    ("karto",     "karton",     "kartoj",     "kartojn"),
    ("biskvito",  "biskviton",  "biskvitoj",  "biskvitojn"),
    ("kuko",      "kukon",      "kukoj",      "kukojn"),
    ("birdo",     "birdon",     "birdoj",     "birdojn"),
    ("fiŝo",      "fiŝon",      "fiŝoj",      "fiŝojn"),
    ("hundeto",   "hundeton",   "hundetoj",   "hundetojn"),
    ("baloneto",  "baloneton",  "balonetoj",  "balonetojn"),
    ("rozo",      "rozon",      "rozoj",      "rozojn"),
    ("dolĉaĵo",   "dolĉaĵon",   "dolĉaĵoj",   "dolĉaĵojn"),
    ("legombrelo","legombrelon","legombreloj","legombrelojn"),
]

# (sg_nom, sg_acc, pl_nom, pl_acc, "in/on each")
CONTAINERS = [
    ("skatolo", "skatolon", "skatoloj", "skatolojn", "en ĉiu"),
    ("ĉambro",  "ĉambron",  "ĉambroj",  "ĉambrojn",  "en ĉiu"),
    ("korbo",   "korbon",   "korboj",   "korbojn",   "en ĉiu"),
    ("sako",    "sakon",    "sakoj",    "sakojn",    "en ĉiu"),
    ("paketo",  "paketon",  "paketoj",  "paketojn",  "en ĉiu"),
    ("telero",  "teleron",  "teleroj",  "telerojn",  "sur ĉiu"),
    ("ĝardeno", "ĝardenon", "ĝardenoj", "ĝardenojn", "en ĉiu"),
]

# (plural nominative, plural accusative)
# kosti is transitive ("kostas 5 eŭrojn") but copular "estas 5 eŭroj" is nominative
CURRENCIES = [("eŭroj", "eŭrojn"), ("dolaroj", "dolarojn"), ("monetoj", "monetojn")]

# ---- Helpers ------------------------------------------------------------

def _render_calc(expr: str, steps: list[str]) -> str:
    """Format the digit-by-digit decomposition as one line: 'expr: step1, step2'."""
    return f"{expr}: {', '.join(steps)}"


def _opening() -> str:
    return random.choice([
        "Pripensu paŝon post paŝo.",
        "Ni kalkulu paŝon post paŝo.",
        "Solvu paŝon post paŝo.",
        "Analizu la problemon paŝon post paŝo.",
    ])


# ---- Problem generators -------------------------------------------------

def gen_distribution(rng):
    """N people each have K items → total = N × K."""
    n = rng.randint(2, 9)
    k = rng.randint(2, 12)
    person = rng.choice(NAMES)
    sg, sg_a, pl_n, pl_a = rng.choice(ITEMS)

    q_templates = [
        f"{person} havas {n} fratojn. Ĉiu frato havas po {k} {pl_a}. Kiom da {pl_n} estas entute?",
        f"{person} havas {n} amikojn. Ĉiu amiko donacis al ŝi {k} {pl_a}. Kiom da {pl_n} ŝi nun havas?",
        f"En klaso estas {n} infanoj. Ĉiu tenas {k} {pl_a}. Kiom da {pl_n} entute?",
    ]
    q = rng.choice(q_templates)

    expr, steps, total = decompose_mul(n, k)
    a = (f"{_opening()} "
         f"Estas {n} grupoj, kaj ĉiu grupo havas {k} {pl_a}. "
         f"Por trovi la totalon, ni multipliki: {n} × {k}. "
         f"{_render_calc(expr, steps)} → {total}. "
         f"La respondo estas {total}. #### {total}")
    return q, a


def gen_sharing(rng):
    """X items shared among Y people → each gets X÷Y."""
    n_people = rng.choice([2, 3, 4, 5, 6, 8, 10])
    each = rng.randint(2, 12)
    total = n_people * each
    person = rng.choice(NAMES)
    sg, sg_a, pl_n, pl_a = rng.choice(ITEMS)

    q = rng.choice([
        f"{person} havas {total} {pl_a} kaj volas dividi ilin egale inter {n_people} amikoj. Kiom da {pl_n} ricevas ĉiu?",
        f"{total} {pl_n} estas dividitaj egale inter {n_people} infanoj. Kiom ricevas ĉiu?",
        f"Estas {total} {pl_n} kaj {n_people} personoj. Se ili dividas egale, kiom ricevas ĉiu?",
    ])

    expr, steps, result = decompose_div(total, n_people)
    a = (f"{_opening()} "
         f"Estas {total} {pl_n} en totalo, dividitaj inter {n_people} personoj. "
         f"Por trovi kiom ĉiu ricevas, ni dividi: {total} ÷ {n_people}. "
         f"{_render_calc(expr, steps)} → {result}. "
         f"La respondo estas {result}. #### {result}")
    return q, a


def gen_sum(rng):
    """A has X, B has Y → together = X+Y."""
    a_amount = rng.randint(5, 99)
    b_amount = rng.randint(5, 99)
    n1, n2 = rng.sample(NAMES, 2)
    sg, sg_a, pl_n, pl_a = rng.choice(ITEMS)

    q = rng.choice([
        f"{n1} havas {a_amount} {pl_a} kaj {n2} havas {b_amount} {pl_a}. Kiom da {pl_n} ili havas kune?",
        f"{n1} kolektis {a_amount} {pl_a}, kaj {n2} kolektis {b_amount}. Kiom entute?",
    ])

    expr, steps, total = decompose_add(a_amount, b_amount)
    a = (f"{_opening()} "
         f"{n1} havas {a_amount} {pl_a}. "
         f"{n2} havas {b_amount} {pl_a}. "
         f"Por trovi kiom ili kune havas, ni adicii: {a_amount} + {b_amount}. "
         f"{_render_calc(expr, steps)} → {total}. "
         f"La respondo estas {total}. #### {total}")
    return q, a


def gen_difference(rng):
    """A has X more than B → A−B."""
    smaller = rng.randint(5, 80)
    diff = rng.randint(2, 30)
    larger = smaller + diff
    n1, n2 = rng.sample(NAMES, 2)
    sg, sg_a, pl_n, pl_a = rng.choice(ITEMS)

    q = rng.choice([
        f"{n1} havas {larger} {pl_a} kaj {n2} havas {smaller} {pl_a}. Kiom pli havas {n1}?",
        f"{n1} havas {larger} {pl_a}. {n2} havas {smaller}. Kiom estas la diferenco?",
    ])

    expr, steps, result = decompose_sub(larger, smaller)
    a = (f"{_opening()} "
         f"{n1} havas {larger} {pl_a}. "
         f"{n2} havas {smaller} {pl_a}. "
         f"Por trovi la diferencon, ni subtrahi: {larger} − {smaller}. "
         f"{_render_calc(expr, steps)} → {result}. "
         f"La respondo estas {result}. #### {result}")
    return q, a


def gen_sequential(rng):
    """Started with X, gained Y, lost Z → X+Y−Z."""
    start = rng.randint(20, 200)
    gain = rng.randint(5, 50)
    loss = rng.randint(5, min(start + gain - 1, 50))
    person = rng.choice(NAMES)
    sg, sg_a, pl_n, pl_a = rng.choice(ITEMS)

    q = rng.choice([
        f"{person} havis {start} {pl_a}. Li ricevis {gain} pliajn, sed perdis {loss}. Kiom li nun havas?",
        f"{person} komencis kun {start} {pl_n}, gajnis {gain}, kaj poste perdis {loss}. Kiom restas?",
    ])

    expr1, steps1, after_gain = decompose_add(start, gain)
    expr2, steps2, final = decompose_sub(after_gain, loss)

    a = (f"{_opening()} "
         f"{person} komencis kun {start} {pl_n}. "
         f"Unue, li gajnis {gain}: {_render_calc(expr1, steps1)} → {after_gain}. "
         f"Poste, li perdis {loss}: {_render_calc(expr2, steps2)} → {final}. "
         f"La respondo estas {final}. #### {final}")
    return q, a


def gen_cost(rng):
    """1 item costs X, buy N → total = N×X."""
    price = rng.randint(2, 50)
    quantity = rng.randint(2, 12)
    person = rng.choice(NAMES)
    sg, sg_a, pl_n, pl_a = rng.choice(ITEMS)
    cur_nom, cur_acc = rng.choice(CURRENCIES)

    q = rng.choice([
        f"Unu {sg} kostas {price} {cur_acc}. {person} aĉetas {quantity} {pl_a}. Kiom kostas entute?",
        f"{person} aĉetas {quantity} {pl_a} po {price} {cur_acc}. Kiom li pagas?",
    ])

    expr, steps, total = decompose_mul(quantity, price)
    a = (f"{_opening()} "
         f"Unu {sg} kostas {price} {cur_acc}. "
         f"{person} aĉetas {quantity} {pl_a}. "
         f"Por trovi la totalan koston, ni multipliki: {quantity} × {price}. "
         f"{_render_calc(expr, steps)} → {total}. "
         f"La respondo estas {total} {cur_nom}. #### {total}")
    return q, a


def gen_age_diff(rng):
    """A is X years old, B is Y years older → B's age."""
    a_age = rng.randint(5, 60)
    diff = rng.randint(2, 40)
    n1, n2 = rng.sample(NAMES, 2)

    q = rng.choice([
        f"{n1} estas {a_age}-jara. {n2} estas {diff} jarojn pli aĝa ol {n1}. Kiom aĝa estas {n2}?",
        f"{n1} havas {a_age} jarojn. {n2} estas {diff} jarojn pli juna ol {n1}. Kiom aĝa estas {n2}?",
    ])
    is_older = "pli aĝa" in q

    if is_older:
        expr, steps, result = decompose_add(a_age, diff)
        operation = "adicii"
        op_sym = "+"
        rhs = f"{a_age} + {diff}"
    else:
        expr, steps, result = decompose_sub(a_age, diff)
        operation = "subtrahi"
        op_sym = "−"
        rhs = f"{a_age} − {diff}"

    a = (f"{_opening()} "
         f"{n1} estas {a_age}-jara. "
         f"{n2} estas {diff} jarojn {'pli aĝa' if is_older else 'pli juna'}. "
         f"Por trovi la aĝon de {n2}, ni {operation}: {rhs}. "
         f"{_render_calc(expr, steps)} → {result}. "
         f"La respondo estas {result}. #### {result}")
    return q, a


def gen_capacity(rng):
    """Container holds K items, total T items → boxes = T÷K."""
    per_box = rng.choice([2, 3, 4, 5, 6, 8, 10, 12])
    n_boxes = rng.randint(2, 12)
    total = per_box * n_boxes
    sg, sg_a, pl_n, pl_a = rng.choice(ITEMS)
    box_sg, box_sg_a, box_pl_n, box_pl_a, _ = rng.choice(CONTAINERS)

    q = rng.choice([
        f"Unu {box_sg} povas enhavi {per_box} {pl_a}. Kiom da {box_pl_n} bezonatas por {total} {pl_n}?",
        f"Estas {total} {pl_n}. Ĉiu {box_sg} enhavas {per_box}. Kiom da {box_pl_n} estos plenaj?",
    ])

    expr, steps, result = decompose_div(total, per_box)
    a = (f"{_opening()} "
         f"Estas {total} {pl_n} en totalo. "
         f"Ĉiu {box_sg} enhavas {per_box} {pl_a}. "
         f"Por trovi kiom da {box_pl_n}, ni dividi: {total} ÷ {per_box}. "
         f"{_render_calc(expr, steps)} → {result}. "
         f"La respondo estas {result}. #### {result}")
    return q, a


# ---- Multi-step problems ------------------------------------------------

def gen_multistep_give_then_share(rng):
    """A has X items, gives Y to one person, shares rest among Z others.
    Two-step: (X-Y) then ÷Z.  Chosen so results divide cleanly."""
    per_sister = rng.randint(2, 15)
    n_sisters = rng.choice([2, 3, 4, 5, 6])
    given_away = rng.randint(3, 20)
    total = given_away + per_sister * n_sisters
    n1, n2 = rng.sample(NAMES, 2)
    sg, sg_a, pl_n, pl_a = rng.choice(ITEMS)
    sub, _, _ = _pron(n1)  # use correct pronoun for n1

    q = rng.choice([
        f"{n1} havas {total} {pl_a}. {sub.capitalize()} donas {given_away} al {n2} kaj la reston egale al {n_sisters} geamikoj. Kiom da {pl_n} ricevas ĉiu?",
        f"{n1} havas {total} {pl_a}. {sub.capitalize()} donacas {given_away} al {n2}, poste dividas la reston egale inter {n_sisters} geamikoj. Kiom ricevas ĉiu?",
    ])

    expr1, steps1, after_giving = decompose_sub(total, given_away)
    expr2, steps2, final = decompose_div(after_giving, n_sisters)

    a = (f"{_opening()} "
         f"Komence {n1} havas {total} {pl_a}. "
         f"Unue, {sub} donas {given_away}: {_render_calc(expr1, steps1)} → {after_giving}. "
         f"Poste, la reston ({after_giving}) {sub} dividas inter {n_sisters}: "
         f"{_render_calc(expr2, steps2)} → {final}. "
         f"Ĉiu ricevas {final} {pl_a}. La respondo estas {final}. #### {final}")
    return q, a


def gen_multistep_earn_save_spend(rng):
    """Earn X, save P%, spend Y, remainder.
    Choose X and P such that X * P / 100 is integer."""
    # p in divisors of 100 so n*p/100 is integer for n divisible by 100/gcd(p,100)
    p = rng.choice([10, 20, 25, 50])
    multiple_of = {10: 10, 20: 5, 25: 4, 50: 2}[p]
    units = rng.randint(5, 40)
    earned = units * multiple_of  # guaranteed clean for this p
    saved = earned * p // 100
    if earned - saved < 100:
        return gen_multistep_earn_save_spend(rng)  # retry if too tight
    spent = rng.randint(50, earned - saved - 50)
    person = rng.choice(NAMES)
    cur_nom, cur_acc = rng.choice(CURRENCIES)
    pron, _, _ = _pron(person)

    q = rng.choice([
        f"{person} gajnas {earned} {cur_acc} monate. {pron.capitalize()} ŝparas {p}% kaj elspezas {spent} {cur_acc} por luprezo. Kiom {pron} havas restanta?",
        f"Monate {person} gajnas {earned} {cur_acc}. {pron.capitalize()} metas {p}% en ŝparkonton, poste elspezas {spent} {cur_acc}. Kiom restas al {pron}?",
    ])

    # Step 1: compute savings = earned × p / 100
    # Since p*earned/100 is clean, use mul then div
    expr_m, steps_m, saved_total = decompose_mul(earned, p)
    expr_d, steps_d, saved_check = decompose_div(saved_total, 100)
    # Step 2: earned - saved - spent
    expr_s1, steps_s1, after_save = decompose_sub(earned, saved_check)
    expr_s2, steps_s2, final = decompose_sub(after_save, spent)

    a = (f"{_opening()} "
         f"{person} gajnis {earned} {cur_acc}. "
         f"Unue, kalkulu la ŝparaĵon ({p}% de {earned}): "
         f"{_render_calc(expr_m, steps_m)} → {saved_total}; {_render_calc(expr_d, steps_d)} → {saved_check}. "
         f"Do {pron} ŝparis {saved_check} {cur_acc}. "
         f"Due, post la ŝparo restas: {_render_calc(expr_s1, steps_s1)} → {after_save}. "
         f"Trie, post la elspezo: {_render_calc(expr_s2, steps_s2)} → {final}. "
         f"La respondo estas {final} {cur_nom}. #### {final}")
    return q, a


# ---- Percentages --------------------------------------------------------

def gen_percentage(rng):
    """P% of N, where (P × N) is divisible by 100 so the answer is integer.
    Decomposed as (N × P) ÷ 100."""
    p = rng.choice([5, 10, 20, 25, 30, 40, 50, 60, 75, 80, 90])
    # Pick n so p*n is divisible by 100
    gcd_p = 100 // (100 // __import__("math").gcd(p, 100))
    step = 100 // __import__("math").gcd(p, 100)  # smallest n multiple
    units = rng.randint(3, 80)
    n = units * step
    if n > 9999:
        n = rng.choice([100, 200, 500, 1000])

    expected = n * p // 100
    sg, sg_a, pl_n, pl_a = rng.choice(ITEMS)
    context = rng.choice([
        ("biblioteko", pl_n, pl_a),   # books
        ("ĝardeno", pl_n, pl_a),      # flowers
        ("klaso", "studentoj", "studentojn"),
        ("vendejo", pl_n, pl_a),
    ])
    loc, subj_n, subj_a = context
    attr = rng.choice(["infanaj", "malnovaj", "novaj", "grandaj", "verdaj", "specialaj"])

    q = rng.choice([
        f"En {loc} estas {n} {subj_n}. {p}% estas {attr}. Kiom da {subj_n} estas {attr}?",
        f"El {n} {subj_n} en {loc}, {p}% estas {attr}. Kiom da {subj_n} tio signifas?",
        f"Kiom estas {p}% de {n}?",
    ])

    # Step 1: n × p
    expr_m, steps_m, prod = decompose_mul(n, p)
    # Step 2: prod ÷ 100
    expr_d, steps_d, result = decompose_div(prod, 100)

    a = (f"{_opening()} "
         f"Por trovi {p}% de {n}, ni kalkulas ({n} × {p}) ÷ 100. "
         f"Unue, {_render_calc(expr_m, steps_m)} → {prod}. "
         f"Poste, {_render_calc(expr_d, steps_d)} → {result}. "
         f"La respondo estas {result}. #### {result}")
    return q, a


def gen_percentage_complement(rng):
    """P% are X; how many are NOT X?  Answer = (100-P)% of N."""
    p = rng.choice([10, 20, 25, 30, 40, 50, 60, 70, 75, 80])
    step = 100 // __import__("math").gcd(p, 100)
    units = rng.randint(3, 80)
    n = units * step
    if n > 9999:
        n = rng.choice([100, 200, 500, 1000])

    part_p = n * p // 100
    rest = n - part_p
    attr_yes = rng.choice(["infanaj", "malnovaj", "novaj", "kopiitaj", "verdaj"])

    # Use the same attribute in question and answer — 'havas X' for nouns,
    # 'estas X' for adjectives.
    use_generic = rng.random() < 0.5
    if use_generic:
        q = f"El {n} aĵoj, {p}% havas specialan econ. Kiom NE havas ĝin?"
        opening_clause = f"El {n} aferoj, {p}% havas specialan econ"
    else:
        q = f"En biblioteko estas {n} libroj. {p}% estas {attr_yes}. Kiom da libroj NE estas {attr_yes}?"
        opening_clause = f"El {n} libroj, {p}% estas {attr_yes}"

    expr_m, steps_m, prod = decompose_mul(n, p)
    expr_d, steps_d, part = decompose_div(prod, 100)
    expr_s, steps_s, result = decompose_sub(n, part)

    a = (f"{_opening()} "
         f"{opening_clause}; ni volas trovi la reston. "
         f"Unue, {p}% de {n}: {_render_calc(expr_m, steps_m)} → {prod}; "
         f"{_render_calc(expr_d, steps_d)} → {part}. "
         f"Poste, la resto estas {n} − {part}: {_render_calc(expr_s, steps_s)} → {result}. "
         f"La respondo estas {result}. #### {result}")
    return q, a


# ---- Time arithmetic ----------------------------------------------------

def _fmt_time(h: int, m: int) -> str:
    return f"{h:02d}:{m:02d}"


def gen_time_add(rng):
    """HH:MM + Xh Ym = ?  Handle minute overflow into hours."""
    h = rng.randint(0, 22)
    m = rng.randint(0, 59)
    add_h = rng.randint(1, 5)
    add_m = rng.randint(0, 59)

    total_min = m + add_m
    carry_h, new_m = divmod(total_min, 60)
    new_h = h + add_h + carry_h
    if new_h >= 24:
        new_h -= 24

    start = _fmt_time(h, m)
    end = _fmt_time(new_h, new_m)

    q = rng.choice([
        f"Trajno forveturas je {start} kaj vojaĝas {add_h} horojn {add_m} minutojn. Je kioma horo ĝi alvenas?",
        f"Renkonto komenciĝas je {start} kaj daŭras {add_h} horojn {add_m} minutojn. Kiam ĝi finiĝas?",
        f"Nun estas {start}. Post {add_h} horoj kaj {add_m} minutoj, kioma horo estos?",
    ])

    # Minutes step
    min_expr, min_steps, min_sum = decompose_add(m, add_m)
    # Hours step (no carry yet)
    h_with_carry = h + add_h + carry_h
    if carry_h:
        h_expr, h_steps, h_sum = decompose_add(h + add_h, carry_h)
        hours_line = (f"Ĉar {min_sum} ≥ 60, ni portas 1 horon kaj la minutoj fariĝas {new_m}. "
                      f"La horoj: {h} + {add_h} + 1 = {_render_calc(h_expr, h_steps)} → {h_sum}.")
    else:
        h_expr, h_steps, h_sum = decompose_add(h, add_h)
        hours_line = (f"Ĉar {min_sum} < 60, ne estas porto. "
                      f"La horoj: {_render_calc(h_expr, h_steps)} → {h_sum}.")

    a = (f"{_opening()} "
         f"Komence {start}, ni aldonas {add_h}h {add_m}m. "
         f"Unue, la minutoj: {_render_calc(min_expr, min_steps)} → {min_sum}. "
         f"{hours_line} "
         f"La respondo estas {end}. #### {end}")
    return q, a


def gen_time_subtract(rng):
    """HH:MM - Xh Ym = ?  Handle minute borrow."""
    # Ensure result is ≥ 00:00 for simplicity
    end_h = rng.randint(3, 23)
    end_m = rng.randint(0, 59)
    sub_h = rng.randint(1, min(end_h, 5))
    sub_m = rng.randint(0, 59)

    # Compute start = end - (sub_h, sub_m)
    total_m = end_m - sub_m
    borrow_h = 0
    if total_m < 0:
        borrow_h = 1
        start_m = total_m + 60
    else:
        start_m = total_m
    start_h = end_h - sub_h - borrow_h
    if start_h < 0:
        # skip impossible cases by offsetting
        start_h += 24
    start = _fmt_time(start_h, start_m)
    end_s = _fmt_time(end_h, end_m)

    q = rng.choice([
        f"Kunveno finiĝis je {end_s} post daŭro de {sub_h} horoj {sub_m} minutoj. Kiam ĝi komenciĝis?",
        f"Filmo finiĝis je {end_s} kaj ĝi daŭris {sub_h} horojn {sub_m} minutojn. Kiam ĝi komenciĝis?",
    ])

    if borrow_h:
        # minutes: (end_m + 60) - sub_m = start_m
        min_expr, min_steps, min_res = decompose_sub(end_m + 60, sub_m)
        h_expr, h_steps, h_res = decompose_sub(end_h - 1, sub_h)
        minutes_line = (f"Ĉar {end_m} < {sub_m}, ni prunteprenas 1 horon: la minutoj estas "
                        f"{end_m} + 60 − {sub_m} = {_render_calc(min_expr, min_steps)} → {min_res}. ")
    else:
        min_expr, min_steps, min_res = decompose_sub(end_m, sub_m)
        h_expr, h_steps, h_res = decompose_sub(end_h, sub_h)
        minutes_line = (f"La minutoj: {_render_calc(min_expr, min_steps)} → {min_res}. ")

    a = (f"{_opening()} "
         f"Fino je {end_s}, ni subtrahi {sub_h}h {sub_m}m. "
         f"Unue, {minutes_line}"
         f"Poste, la horoj: {_render_calc(h_expr, h_steps)} → {h_res}. "
         f"La respondo estas {start}. #### {start}")
    return q, a


GENERATORS = [
    gen_distribution,
    gen_sharing,
    gen_sum,
    gen_difference,
    gen_sequential,
    gen_cost,
    gen_age_diff,
    gen_capacity,
    gen_multistep_give_then_share,
    gen_multistep_earn_save_spend,
    gen_percentage,
    gen_percentage_complement,
    gen_time_add,
    gen_time_subtract,
]


# ---- Main ---------------------------------------------------------------

def make_example(rng) -> dict:
    gen = rng.choice(GENERATORS)
    q, a = gen(rng)
    return {"messages": [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/sft/sft_quantity_reasoning.jsonl"))
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.dry_run:
        print("--- 6 sample word problems ---\n")
        for i in range(6):
            ex = make_example(rng)
            print(f"=== {i+1} ===")
            print(f"Q: {ex['messages'][0]['content']}")
            print(f"A: {ex['messages'][1]['content']}")
            print()
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "w") as f:
        while written < args.n:
            ex = make_example(rng)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1
            if written % 2500 == 0:
                print(f"  written {written:,}/{args.n:,}")
    print(f"\nWrote {written:,} quantity-reasoning examples → {args.out}")


if __name__ == "__main__":
    main()
