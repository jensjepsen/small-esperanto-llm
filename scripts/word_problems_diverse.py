"""2-layer diverse word-problem generator for all 8 procedural types.

Layer 1 (per-type, hand-authored):
  - math language fragments: 2-5 ways to state the same math
  - field-builder: how to fill placeholder vars from the problem instance

Layer 2 (shared across types):
  - narrative wrapper pool (currently hand-authored 12; LLM-augmented later)
  - question-form templates (direct / nominative / imperative / passive / completion)

Driver: sample type → sample problem (via word_problems_procedural) → pick
language/wrapper/qform → compose. Chain stays canonical procedural so the
verifier still guards math integrity.

Usage:
  uv run python scripts/word_problems_diverse.py \\
    --types ratio,percent,age --n 1000 --out data/wp_diverse.jsonl
"""
import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path


# ── Esperanto grammar normalizer ──────────────────────────────────────────
# Templates use {a} / {b} placeholders alongside plural-form nouns (e.g.
# "{a} partojn"). When a == 1 the EO grammar requires singular ("1 parton",
# "1 parto"), not plural. Same for "{a} jarojn" etc. This pass also handles
# double spaces, sentence-internal capitalization after a period, and the
# leading capital of the whole rendered text.

# Pattern: number "1" followed by a noun ending in oj/ojn — strip the "j".
# Also handles 1 with optional space and any number ending in 1 (11, 21,
# 101…) since those also take plural in EO. So we restrict to literal "1".
_ONE_PLURAL_ACC = re.compile(r"\b1\s+([a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+)ojn\b")
_ONE_PLURAL_NOM = re.compile(r"\b1\s+([a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+)oj\b")
# Sentence boundary lowercase → uppercase: ". word" or "! word"
_SENT_BOUNDARY = re.compile(r"([.!?])\s+([a-zĉĝĥĵŝŭ])")
# Double whitespace
_DOUBLE_WS = re.compile(r"\s{2,}")


def normalize_eo(text: str) -> str:
    """Light grammar+formatting cleanup for diverse-form output."""
    # "1 Xojn" → "1 Xon"; "1 Xoj" → "1 Xo"
    text = _ONE_PLURAL_ACC.sub(r"1 \1on", text)
    text = _ONE_PLURAL_NOM.sub(r"1 \1o", text)
    # Capitalize after sentence boundary
    text = _SENT_BOUNDARY.sub(lambda m: m.group(1) + " " + m.group(2).upper(), text)
    # Collapse runs of whitespace
    text = _DOUBLE_WS.sub(" ", text).strip()
    # Capitalize first character of the entire text if it's lowercase
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    # Final-sentence terminator: most templates leave the question without
    # punctuation. Add a `?` unless the text already ends with `.`, `?`, `!`,
    # or `___` (completion form).
    if text and not text.endswith((".", "?", "!", "___", "___.", "___?")):
        text = text.rstrip(",;: ") + "?"
    return text

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from esperanto_lm.data import _morpheme_preprocess  # noqa
from word_problems_procedural import (
    Ratio, Percent, InverseRate, Consec, Coin, Age, Mixture, Distance,
    sample_ratio, sample_percent, sample_inverse_rate, sample_consec,
    sample_coin, sample_age, sample_mixture, sample_distance,
    _RATIO_CHAINS, _PCT_CHAINS, _INV_CHAINS, _CONSEC_CHAINS,
    _COIN_CHAINS, _AGE_CHAINS, _MIX_CHAINS, _DIST_CHAINS,
)


# ── Shared layer 2: wrapper pool ──────────────────────────────────────────
# {MATH_STATEMENT} is the type-specific math fragment.
# {QUESTION} is the type-specific question form.

_WRAPPER_FILE = Path(__file__).resolve().parent.parent / "data" / "wp_wrappers.json"
_HANDCRAFTED_WRAPPERS = [
    # Always-available fallback bare templates if the file isn't present
    {"id": "bare-1",  "tone": "bare",   "template": "{MATH_STATEMENT}. {QUESTION}"},
    {"id": "bare-2",  "tone": "terse",  "template": "{MATH_STATEMENT}. {QUESTION}"},
    {"id": "qfirst",  "tone": "question-first",
     "template": "{QUESTION}? Konsideru ke {MATH_STATEMENT}."},
]


# Filter LLM-generated wrappers that produce double-question or noun-slot
# artifacts when {QUESTION} (a clause) is plugged in.
_BAD_OWN_QUESTION_BEFORE = re.compile(r"\?[^?]*\{QUESTION\}")
_BAD_NOUN_SLOT = re.compile(
    r"(kalkul[uia]\s+la\s+rezulto[nm]?\s+de|respondu\s+al|demandante|"
    r"trovu\s+la\s+rezulto|kalkuli\s+la\s+respondon)\s*\{QUESTION\}",
    re.I,
)


def _wrapper_ok(w: dict) -> bool:
    t = w.get("template", "")
    if _BAD_OWN_QUESTION_BEFORE.search(t):
        return False
    if _BAD_NOUN_SLOT.search(t):
        return False
    return True


def load_wrappers() -> list[dict]:
    if _WRAPPER_FILE.exists():
        loaded = json.loads(_WRAPPER_FILE.read_text())
        loaded = [w for w in loaded if _wrapper_ok(w)]
        # always mix in a handful of bare-form ones so we keep some signal
        # close to the trained distribution
        return loaded + _HANDCRAFTED_WRAPPERS
    return _HANDCRAFTED_WRAPPERS


WRAPPERS = load_wrappers()


# ── Question forms (per-type) ─────────────────────────────────────────────
# Each is a template with type-specific placeholders.
# Forms used across types:

QFORMS_DIRECT = ["direct", "nominative", "imperative", "passive", "completion"]


# ────────────────────────────────────────────────────────────────────────
# Per-type configurations
# ────────────────────────────────────────────────────────────────────────

# Each entry returns:
#   math_languages: dict[lang_name] -> list of templates
#   question_forms: dict[form_name] -> template
#   build_fields: (instance, rng) -> dict of values for both templates
#   valid_lang: (instance, lang_name) -> bool (some langs only fit certain instances)

# ── RATIO ─────────────────────────────────────────────────────────────────

def _ratio_fields(p: Ratio, rng):
    a, b = (p.ratio + (0,))[:2]
    rsum = sum(p.ratio)
    return {
        "a": a, "b": b, "sum": rsum, "total": p.total,
        "name_a": p.names[0], "name_b": p.names[1] if len(p.names) > 1 else "Anna",
        "target": p.names[p.ask_idx],
        "item_npl": p.item + "j",
        "item_acc_pl": p.item + "jn",
        "pct_a": (a * 100) // rsum if rsum and (100 % rsum == 0) else None,
        "pct_b": (b * 100) // rsum if rsum and (100 % rsum == 0) else None,
    }

def _ratio_valid_lang(p: Ratio, lang: str) -> bool:
    if len(p.ratio) != 2 or p.ask != "direct":
        return False
    if lang == "procento":
        return (sum(p.ratio) > 0) and (100 % sum(p.ratio) == 0)
    return True


RATIO_DIVERSE = {
    "math_languages": {
        "proporcio": [
            "{name_a} kaj {name_b} dividis {total} {item_acc_pl} en proporcio {a}:{b}",
            "{total} {item_npl} estas dividitaj inter {name_a} kaj {name_b} en proporcio {a}:{b}",
            "{name_a} kaj {name_b} dividis {total} {item_acc_pl} laŭ rilato {a}-al-{b}",
        ],
        "frakcio": [
            "{name_a} ricevis {a}/{sum} de {total} {item_npl} kaj {name_b} la ceteran {b}/{sum}",
            "el la totalo de {total} {item_npl}, {name_a} prenis {a}-onon (de {sum}) kaj {name_b} {b}-onon",
        ],
        "procento": [
            "{name_a} ricevis {pct_a}% el {total} {item_npl}, kaj {name_b} la reston",
            "el {total} {item_npl}, {pct_a}% iris al {name_a} kaj {pct_b}% al {name_b}",
        ],
        "implicit-multiplicative": [
            "{name_a} kaj {name_b} dividis {total} {item_acc_pl} tiel ke por ĉiu {a} {item_npl} ricevitaj de {name_a}, {name_b} ricevis {b}",
            "el {total} {item_acc_pl}, {name_a} ricevis {a} {item_npl} por ĉiu {b} de {name_b}",
        ],
        "verbose": [
            "{name_a} kaj {name_b} kunhavigis {total} {item_acc_pl}; ili konsentis ke {name_a} ricevu {a} partojn por ĉiu {b} partoj ricevitaj de {name_b}",
        ],
    },
    "question_forms": {
        "direct": "kiom da {item_npl} ricevis {target}",
        "nominative": "kio estas la parto de {target}",
        "imperative": "kalkulu la kvanton de {item_npl} kiun ricevis {target}",
        "passive": "kiom da {item_npl} estis donitaj al {target}",
        "completion": "la kvanto de {item_npl} ricevita de {target} egalas al ___",
    },
}


# ── PERCENT ──────────────────────────────────────────────────────────────

def _pct_fields(p: Percent, rng):
    return {
        "name": p.name, "item": p.item, "item_acc": p.item + "n",
        "base": p.base, "pct": p.pct, "amount": p.amount,
        "result": p.answer, "pct_dec": f"{p.pct/100:.2f}".rstrip("0").rstrip("."),
        "complement_pct": 100 - p.pct,
    }

def _pct_valid_lang(p: Percent, lang: str) -> bool:
    if lang == "fraction" and p.pct not in (10, 20, 25, 50, 75):
        return False  # only "nice" % express as small fractions
    return True


PERCENT_DIVERSE = {
    "math_languages": {
        "percent-of": {
            "discount":  ["{name} aĉetis {item_acc} kiu kostis {base} eŭrojn, kun rabato de {pct}%",
                          "la {item} kostis {base} eŭrojn; {name} ricevis {pct}% rabaton"],
            "markup":    ["la prezo de {item} estis {base} eŭroj, sed pliiĝis je {pct}%",
                          "{name} aĉetis {item_acc} por {base} eŭroj; la prezo poste pliiĝis je {pct}%"],
            "tax":       ["{item} kostis {base} eŭrojn; impostoj aldonas {pct}%",
                          "antaŭ imposto, {item} kostis {base} eŭroj; la imposto estas {pct}%"],
            "of-amount": ["{pct}% el {base} egalas",
                          "{name} kalkulis {pct}% el {base}"],
            "saving":    ["{name} aĉetis {item_acc} kun {pct}% rabato sur {base} eŭroj"],
        },
        "decimal": {
            "discount":  ["{name} pagis multobligante la originan prezon {base} per {pct_dec}-faktoron"],
            "markup":    ["la nova prezo estas la origina {base} multobligita per (1 + {pct_dec})"],
        },
        "fraction": {  # only for 10/20/25/50/75
            "discount":  ["{name} ricevis 1/4 rabaton sur la prezo {base} eŭroj" if False else
                          "{name} ricevis rabaton egala al la frakcio {pct}/100 de {base}"],
        },
    },
    "question_forms": {
        "discount":  {"direct": "kiom kostas la {item} nun"},
        "markup":    {"direct": "kiom kostas la {item} nun"},
        "tax":       {"direct": "kiom estas la totala kosto"},
        "of-amount": {"direct": "kiu estas la rezulto"},
        "saving":    {"direct": "kiom da eŭroj ŝparis {name}"},
    },
}


# ── INVERSE-RATE ──────────────────────────────────────────────────────────

def _inv_fields(p: InverseRate, rng):
    workers, verb, task, unit = p.scenario
    return {
        "w1": p.w1, "t1": p.t1, "w2": p.w2, "t2": p.t2,
        "workers": workers, "verb": verb, "task": task, "unit": unit,
        "const": p.const, "ask": p.ask,
    }

def _inv_valid_lang(p: InverseRate, lang: str) -> bool:
    return True


INVERSE_RATE_DIVERSE = {
    "math_languages": {
        "constant-product": [
            "{w1} {workers} {verb} {task} en {t1} {unit}",
            "ĉar {w1} {workers} bezonas {t1} {unit} por {verb} {task}",
        ],
        "per-unit": [
            "la tasko postulas {w1} {workers} dum {t1} {unit} (mezuru en person-{unit})",
            "{w1} {workers} kompletigas la taskon en {t1} {unit}; pensu pri tio kiel person-{unit} laboro",
        ],
        "implicit-doubling": [
            "se {w1} {workers} {verb} {task} en {t1} {unit}, kaj nun ekzistas {w2} {workers}",
            "duobligi la nombron de {workers} (kompare al {w1}, do {w2}) ŝanĝos la tempon",
        ],
    },
    "question_forms": {
        "find-time": "kiom da {unit} ili bezonos por la sama tasko",
        "find-workers": "kiom da {workers} estus necesa por fini en {t2} {unit}",
    },
}


# ── CONSECUTIVE ──────────────────────────────────────────────────────────

def _consec_fields(p: Consec, rng):
    return {
        "count": p.count, "total": p.total, "start": p.start, "step": p.step,
        "name": p.name, "scenario": p.scenario,
        "par_adj": {"any": "", "even": "parajn ", "odd": "neparajn "}[p.parity],
        "smallest": p.values[0], "largest": p.values[-1],
        "ask": p.ask,
    }

def _consec_valid_lang(p: Consec, lang: str) -> bool:
    return True


CONSEC_DIVERSE = {
    "math_languages": {
        "explicit-sum": [
            "la sumo de {count} sinsekvaj {par_adj}entjeroj estas {total}",
            "{name} rimarkis {count} sinsekvajn {par_adj}{scenario} kies sumo egalas al {total}",
        ],
        "implicit": [
            "{name} havas {count} sinsekvajn {par_adj}{scenario} kiuj kunsumiĝas al {total}",
        ],
        "average-stated": [
            "la meznombro de {count} sinsekvaj {par_adj}entjeroj estas {total} dividita per {count}",
        ],
    },
    "question_forms": {
        "smallest": "kio estas la plej malgranda el ili",
        "largest": "kio estas la plej granda el ili",
        "middle": "kio estas la meza valoro",
    },
}


# ── COIN ──────────────────────────────────────────────────────────────────

def _coin_fields(p: Coin, rng):
    return {
        "small_val": p.small_val, "big_val": p.big_val,
        "currency": p.currency, "currency_sg": p.currency.rstrip("oj") or p.currency,
        "item": p.item, "item_acc_pl": p.item + "jn", "item_npl": p.item + "j",
        "total_count": p.total_count, "total_value": p.total_value,
        "name": p.name, "ask": p.ask,
        "target_val": p.big_val if p.ask == "find-big" else p.small_val,
    }

def _coin_valid_lang(p: Coin, lang: str) -> bool:
    return True


COIN_DIVERSE = {
    "math_languages": {
        "explicit": [
            "{name} havas {total_count} {item_acc_pl} de du valoroj: {small_val} kaj {big_val} {currency}, kun totala valoro {total_value} {currency}",
            "en sia poŝo, {name} havas {total_count} {item_acc_pl} valorantajn aŭ {small_val} aŭ {big_val} {currency}; la sumo estas {total_value} {currency}",
        ],
        "implicit-mix": [
            "{name} kunmetis {total_count} {item_acc_pl} (iuj po {small_val}, iuj po {big_val} {currency}) kies tuta valoro estas {total_value} {currency}",
        ],
    },
    "question_forms": {
        "find-big": "kiom da {target_val}-{currency_sg}-aj {item_npl} {name} havas",
        "find-small": "kiom da {target_val}-{currency_sg}-aj {item_npl} {name} havas",
    },
}


# ── AGE ───────────────────────────────────────────────────────────────────

def _age_fields(p: Age, rng):
    old_role, young_role = p.relation
    mul_word_now = {2:"dufoje",3:"trifoje",4:"kvarfoje",5:"kvinfoje",6:"sesfoje",7:"sepfoje"}.get(p.ratio_now, f"{p.ratio_now}-foje")
    mul_word_later = {2:"dufoje",3:"trifoje",4:"kvarfoje",5:"kvinfoje"}.get(p.ratio_later, f"{p.ratio_later}-foje")
    return {
        "ny": p.name_young, "no": p.name_old,
        "old_role": old_role, "young_role": young_role,
        "ratio_now": p.ratio_now, "sum_now": p.sum_now,
        "mul_word_now": mul_word_now, "mul_word_later": mul_word_later,
        "t": p.t, "ratio_later": p.ratio_later, "ask": p.ask,
        "ask_t": p.ask_t,
    }

def _age_valid_lang(p: Age, lang: str) -> bool:
    if lang == "time-shift-explicit" and p.kind != "time-shift":
        return False
    if lang == "simple-multiplicative" and p.kind != "simple-now":
        return False
    return True


AGE_DIVERSE = {
    "math_languages": {
        "simple-multiplicative": [
            "{no} estas {mul_word_now} pli aĝa ol {ny}, kaj iliaj aĝoj sumiĝas al {sum_now}",
            "{ny} kaj {no} estas {young_role} kaj {old_role}; {no} estas {ratio_now}-foje la aĝo de {ny}, kaj kune ili havas {sum_now} jarojn",
        ],
        "ratio-fraction": [
            "{ny}-a aĝo estas 1/{ratio_now} de la aĝo de {no}; kune ili havas {sum_now} jarojn",
        ],
        "time-shift-explicit": [
            "nun {no} estas {mul_word_now} pli aĝa ol {ny}; post {t} jaroj {no} estos {mul_word_later} pli aĝa",
            "hodiaŭ la aĝo de {no} estas {ratio_now}-foje la aĝo de {ny}, sed post {t} jaroj nur {ratio_later}-foje",
        ],
    },
    "question_forms": {
        "young": "kiom da jaroj havas {ny} nun",
        "old": "kiom da jaroj havas {no} nun",
        "future": "kiom aĝa estos {ny} post {ask_t} jaroj",
    },
}


# ── MIXTURE ───────────────────────────────────────────────────────────────

def _mix_fields(p: Mixture, rng):
    return {
        "name": p.name, "sol_name": p.sol_name, "solute": p.solute,
        "v1": p.v1, "p1": p.p1, "p2": p.p2, "add": p.add,
        "v2": p.v2, "p2_blend": p.p2_blend, "p_avg": p.p_avg,
        "kind": p.kind,
    }

def _mix_valid_lang(p: Mixture, lang: str) -> bool:
    if lang == "implicit-conservation":
        return p.kind == "dilute"  # only dilute conserves solute
    return lang == f"{p.kind}-explicit"  # explicit langs must match kind


MIXTURE_DIVERSE = {
    "math_languages": {
        "dilute-explicit": [
            "{name} havas {v1} ml da {sol_name} kun koncentriĝo de {p1}%; ŝi celas {p2}% per aldono de pura akvo",
        ],
        "concentrate-explicit": [
            "{name} havas {v1} ml da {sol_name} kun koncentriĝo de {p1}%; ŝi celas {p2}% per aldono de pura {solute}",
        ],
        "blend-explicit": [
            "{name} miksas {v1} ml de {p1}-procenta {sol_name} kun {v2} ml de {p2_blend}-procenta {sol_name}",
        ],
        "implicit-conservation": [
            "{v1} ml de {p1}-procenta {sol_name} estas modifita; la kvanto de {solute} restas konstanta dum nur akvo aldoniĝas",
        ],
    },
    "question_forms": {
        "dilute": "kiom da pura akvo {name} devas aldoni por atingi {p2}%",
        "concentrate": "kiom da pura {solute} {name} devas aldoni por atingi {p2}%",
        "blend": "kio estos la fina procento de la miksaĵo",
    },
}


# ── DISTANCE ──────────────────────────────────────────────────────────────

def _dist_fields(p: Distance, rng):
    return {
        "name": p.name, "vehicle": p.vehicle,
        "d": p.d, "r": p.r, "t": p.t, "ask_dir": p.ask,
        "ra": p.ra, "rb": p.rb, "h": p.h, "catch_t": p.catch_t,
        "r1": p.r1, "r2": p.r2, "meet_d": p.meet_d, "meet_t": p.meet_t,
        "rout": p.rout, "rback": p.rback, "ravg": p.ravg,
        "name2": p.name2, "kind": p.kind,
    }

def _dist_valid_lang(p: Distance, lang: str) -> bool:
    if lang == "catch-up-explicit" and p.kind != "catch-up": return False
    if lang == "meeting-explicit" and p.kind != "meeting": return False
    if lang == "average-explicit" and p.kind != "average": return False
    if lang == "direct-explicit" and p.kind != "direct": return False
    return True


DISTANCE_DIVERSE = {
    "math_languages": {
        "direct-explicit": [
            "{name} veturas per {vehicle} kun rapideco de {r} km/h dum {t} horoj",
            "kun {vehicle} kovrante {d} km en {t} horoj",
        ],
        "catch-up-explicit": [
            "{name} ekveturis je {ra} km/h. Post {h} horoj, {name2} ekiris en la sama direkto je {rb} km/h",
            "{name2} provas atingi {name}n: {name} havas {h} horan avancon je {ra} km/h, dum {name2} veturas je {rb} km/h",
        ],
        "meeting-explicit": [
            "{name} stiras per {vehicle} de urbo A al B, kaj {name2} de B al A; la distanco estas {meet_d} km. Ili ekiras samtempe je {r1} kaj {r2} km/h",
        ],
        "average-explicit": [
            "{name} iras per {vehicle} je {rout} km/h kaj revenas je {rback} km/h",
        ],
    },
    "question_forms": {
        "direct":   "kiu estas la distanco kovrata" if False else "kiom da kilometroj",  # generic
        "catch-up": "post kiom da horoj de la ekveturo de {name2} li atingos {name}n",
        "meeting":  "post kiom da horoj ili renkontiĝos",
        "average":  "kio estas la mezuma rapideco de la tuta rondiro",
    },
}


# ── Type registry ─────────────────────────────────────────────────────────

DIVERSE = {
    "ratio":        {**RATIO_DIVERSE,        "sample": sample_ratio,        "build_fields": _ratio_fields,  "valid_lang": _ratio_valid_lang,  "chains": _RATIO_CHAINS,   "ask_attr": "ask",  "chain_attr": None},
    "percent":      {**PERCENT_DIVERSE,      "sample": sample_percent,      "build_fields": _pct_fields,    "valid_lang": _pct_valid_lang,    "chains": _PCT_CHAINS,     "ask_attr": "op",   "chain_attr": None},
    "inverse-rate": {**INVERSE_RATE_DIVERSE, "sample": sample_inverse_rate, "build_fields": _inv_fields,    "valid_lang": _inv_valid_lang,    "chains": _INV_CHAINS,     "ask_attr": "ask",  "chain_attr": None},
    "consecutive":  {**CONSEC_DIVERSE,       "sample": sample_consec,       "build_fields": _consec_fields, "valid_lang": _consec_valid_lang, "chains": _CONSEC_CHAINS,  "ask_attr": "ask",  "chain_attr": None},
    "coin":         {**COIN_DIVERSE,         "sample": sample_coin,         "build_fields": _coin_fields,   "valid_lang": _coin_valid_lang,   "chains": _COIN_CHAINS,    "ask_attr": "ask",  "chain_attr": None},
    "age":          {**AGE_DIVERSE,          "sample": sample_age,          "build_fields": _age_fields,    "valid_lang": _age_valid_lang,    "chains": _AGE_CHAINS,     "ask_attr": "ask",  "chain_attr": "kind"},
    "mixture":      {**MIXTURE_DIVERSE,      "sample": sample_mixture,      "build_fields": _mix_fields,    "valid_lang": _mix_valid_lang,    "chains": _MIX_CHAINS,     "ask_attr": "kind", "chain_attr": "kind"},
    "distance":     {**DISTANCE_DIVERSE,     "sample": sample_distance,     "build_fields": _dist_fields,   "valid_lang": _dist_valid_lang,   "chains": _DIST_CHAINS,    "ask_attr": "kind", "chain_attr": "kind"},
}


def render_diverse(type_name: str, rng: random.Random) -> dict | None:
    cfg = DIVERSE[type_name]
    p = cfg["sample"](rng)
    # math_languages may be flat dict[lang -> [tmpls]] OR nested dict[lang -> dict[op -> [tmpls]]]
    langs = list(cfg["math_languages"].keys())
    rng.shuffle(langs)
    lang = None
    for cand in langs:
        if cfg["valid_lang"](p, cand):
            lang = cand
            break
    if lang is None:
        return None
    raw = cfg["math_languages"][lang]
    # If raw is dict (op-specific), select the matching op's templates
    if isinstance(raw, dict):
        ask_key = getattr(p, cfg["ask_attr"], None)
        if ask_key not in raw:
            return None
        templates = raw[ask_key]
    else:
        templates = raw
    math_tmpl = rng.choice(templates)
    fields = cfg["build_fields"](p, rng)
    # question form: 3 cases
    #  a) nested dict (per-op {form: tmpl}) — used by percent
    #  b) flat dict whose keys ARE the per-kind/op names — use direct lookup
    #  c) flat dict of free phrasings — random choice (used by ratio)
    qforms = cfg["question_forms"]
    ask_key = getattr(p, cfg["ask_attr"], None)
    if qforms and isinstance(next(iter(qforms.values())), dict):
        # (a) per-op nested
        qf_dict = qforms.get(ask_key, {})
        if not qf_dict:
            return None
        qform_name = rng.choice(list(qf_dict.keys()))
        qtmpl = qf_dict[qform_name]
    elif ask_key in qforms:
        # (b) flat dict keyed by kind/op
        qform_name = ask_key
        qtmpl = qforms[ask_key]
    else:
        # (c) free choice
        qform_name = rng.choice(list(qforms.keys()))
        qtmpl = qforms[qform_name]

    try:
        math_stmt = math_tmpl.format(**fields)
        question = qtmpl.format(**fields)
    except (KeyError, ValueError):
        return None

    wrapper = rng.choice(WRAPPERS)
    text = wrapper["template"].format(MATH_STATEMENT=math_stmt, QUESTION=question)
    text = normalize_eo(text)
    # chain selection: use chain_attr if specified (chain is kind-determined),
    # otherwise the chain dict is strategy-flexible → random choice.
    chain_attr = cfg.get("chain_attr")
    if chain_attr:
        strat = getattr(p, chain_attr)
        if strat not in cfg["chains"]:
            return None
    else:
        strat = rng.choice(list(cfg["chains"]))
    chain = cfg["chains"][strat](p)
    return {
        "type": f"{type_name}-diverse",
        "question_eo": text,
        "chain_eo": chain,
        "answer": p.answer,
        "math_language": lang,
        "wrapper_tone": wrapper["tone"],
        "question_form": qform_name,
        "strategy": strat,
        # In-memory only — callers that JSON-serialize must `pop`.
        "_problem": p,
        "_base_type": type_name,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="ratio,percent,inverse-rate,consecutive,coin,age,mixture,distance")
    ap.add_argument("--n", type=int, default=100, help="problems per type")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    types = args.types.split(",")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    n_written = 0
    skipped = 0
    with args.out.open("w") as f:
        for t in types:
            if t not in DIVERSE:
                print(f"skip unknown type {t}")
                continue
            type_written = 0
            attempts = 0
            while type_written < args.n and attempts < args.n * 5:
                attempts += 1
                row = render_diverse(t, rng)
                if row is None:
                    skipped += 1
                    continue
                row.pop("_problem", None)
                row.pop("_base_type", None)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                type_written += 1
                n_written += 1
            print(f"  {t}: wrote {type_written}/{args.n}")
    print(f"\ntotal: {n_written} problems → {args.out}  (skipped {skipped} invalid combos)")


if __name__ == "__main__":
    main()
