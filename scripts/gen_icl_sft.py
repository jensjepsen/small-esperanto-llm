"""Generate an in-context-learning SFT dataset with a sampled output format.

Motivation, from probes on the v31 SFT base (fp32, k=8/16, temp 1.0, 20
dane_plus test sentences):

  * 0-shot, untrained formats are unreachable: "type: enhed" lines produced 0
    valid outputs in 60 samples; inline <person>..</person> spans produced 13
    tag emissions in 60 samples of which 0 wrapped any text from the passage.
  * 4-shot demonstrations did NOT teach the format. Format emission rose from
    1% to 23% while LEAK -- emitting an entity that appears in an EXEMPLAR but
    not in the target -- rose from 0% to 19%, and grounding reached 3%.
    Correctness was 0/20 in every cell. The model copies demonstrations, it
    does not induce from them.

So the capability to train is: read an output format off the demonstrations
and apply it to new content. The construction principle that follows is that
THE FORMAT IS SAMPLED PER ROW. Train on one fixed format and the model
memorises the format and learns to ignore the exemplars, which is the
behaviour we already have. Vary the format every row and the only way to
produce the answer is to attend to the demonstrations.

Everything is rendered deterministically from gold annotations, so there is
no LLM in the loop and no cost. Correctness is by construction, and gated:
every row is round-tripped through a parser built from the same format
descriptor, and a one-edit perturbation must fail to parse back (see
_gate_format).

Anti-leak construction, aimed at the 19% LEAK measured above:
  * exemplars share no entity surface with the target
  * hard negatives: an exemplar carries a type the target lacks, so copying
    produces a false positive and is penalised rather than rewarded
  * shot count varies 1-5 so nothing keys on a fixed count

Row shape. Exemplars are packed INSIDE the single user turn rather than as
separate chat turns, because train_sft.py's collator masks only up to the
first <|assistant|> token; with multi-turn rows the exemplar answers would
themselves become training targets, teaching format production without a
demonstration -- exactly the memorisation this is meant to avoid. Packing
them into the user turn makes the existing masking correct with no trainer
change.

Held-out formats: the (shape, key-lexicon) space is partitioned, not the
rows. Evaluating on shuffled rows of seen formats measures memorisation of
the formats we happened to generate. The eval split uses combinations that
never appear in training.

Usage:
  python scripts/gen_icl_sft.py --n 200 --out scratch/icl_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import itertools
import random
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- format grammar

BUCKETS = ("person", "org", "sted", "dato")

# Buckets actually present in the loaded source, set by main(). dane_plus
# train/dev are CoNLL-style (PER/MISC/LOC/ORG, NO date) while test is
# OntoNotes-style and does contain 67 DATE spans. Asking for a key the source
# can never populate trains the model that the key is always empty, and is
# also just an unanswerable slot -- so the schema is derived from the data
# rather than hardcoded to the four we happen to have names for.
ACTIVE = list(BUCKETS)

# Key lexicons. The `arb*` ones carry no semantic hint, so the mapping from
# label to entity type CANNOT be guessed and must be induced from the
# demonstrations -- the purest ICL supervision in the set.
KEY_LEXICONS = {
    "da_full":  {"person": "person", "org": "organisation", "sted": "sted", "dato": "dato"},
    "da_plural": {"person": "personer", "org": "organisationer", "sted": "steder", "dato": "datoer"},
    "da_alt":   {"person": "navn", "org": "firma", "sted": "by", "dato": "tidspunkt"},
    "abbrev":   {"person": "PER", "org": "ORG", "sted": "LOC", "dato": "DAT"},
    "single":   {"person": "P", "org": "O", "sted": "S", "dato": "D"},
    "en":       {"person": "person", "org": "organization", "sted": "location", "dato": "date"},
    "arb_kat":  {"person": "kat_a", "org": "kat_b", "sted": "kat_c", "dato": "kat_d"},
    "arb_num":  {"person": "type1", "org": "type2", "sted": "type3", "dato": "type4"},
    "arb_greek": {"person": "alfa", "org": "beta", "sted": "gamma", "dato": "delta"},
}
ARBITRARY = {"arb_kat", "arb_num", "arb_greek"}

SHAPES = ("kv_colon", "kv_paren", "kv_bracket", "kv_eq", "kv_arrow",
          "numbered", "json")
ORDERS = ("occurrence", "alpha", "grouped")
SEPS = ("\n", "; ", " | ", ", ")
CASES = ("lower", "title", "upper")
EMPTIES = ("ingen", "-", "[]", "(tom)", "ingen enheder")


def _cased(k: str, case: str) -> str:
    if case == "upper":
        return k.upper()
    if case == "title":
        return k[:1].upper() + k[1:]
    return k


def sample_format(rng: random.Random) -> dict:
    return {
        "shape": rng.choice(SHAPES),
        "lex": rng.choice(list(KEY_LEXICONS)),
        "order": rng.choice(ORDERS),
        "sep": rng.choice(SEPS),
        "case": rng.choice(CASES),
        "empty": rng.choice(EMPTIES),
    }


def fmt_id(f: dict) -> str:
    return f"{f['shape']}|{f['lex']}"


def is_heldout(f: dict, frac: float = 0.2) -> bool:
    """Partition the (shape, lexicon) space deterministically.

    Hash-based so train and eval generation agree without sharing state, and
    so re-running with a different --n cannot leak a held-out combination
    into training.
    """
    h = hashlib.sha1(fmt_id(f).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < frac


# ---------------------------------------------------------------- render / parse

def _order(ents, order: str):
    if order == "alpha":
        return sorted(ents, key=lambda e: e[0].lower())
    if order == "grouped":
        return sorted(ents, key=lambda e: (ACTIVE.index(e[1]), ents.index(e)))
    return list(ents)


def _sep_for(ents, f: dict) -> str:
    """Fall back to newline when the separator occurs inside a value.

    "Halifax, Nova Scotia" under a ", " separator would render an output that
    cannot be parsed back to the entity list, which would silently corrupt
    the target. Detected here rather than discovered downstream.
    """
    sep = f["sep"]
    if sep.strip() and any(sep.strip() in v for v, _ in ents):
        return "\n"
    return sep


def render(ents, f: dict) -> str:
    keys = KEY_LEXICONS[f["lex"]]
    ents = _order(ents, f["order"])
    if f["shape"] == "json":
        out = {}
        for b in ACTIVE:
            vs = [v for v, bb in ents if bb == b]
            if vs or not ents:
                out[_cased(keys[b], f["case"])] = vs
        if not ents:
            out = {_cased(keys[b], f["case"]): [] for b in ACTIVE}
        return json.dumps(out, ensure_ascii=False)
    if not ents:
        return f["empty"]
    sep = _sep_for(ents, f)
    items = []
    for i, (v, b) in enumerate(ents, 1):
        k = _cased(keys[b], f["case"])
        if f["shape"] == "kv_colon":
            items.append(f"{k}: {v}")
        elif f["shape"] == "kv_paren":
            items.append(f"{v} ({k})")
        elif f["shape"] == "kv_bracket":
            items.append(f"[{k}] {v}")
        elif f["shape"] == "kv_eq":
            items.append(f"{k}={v}")
        elif f["shape"] == "kv_arrow":
            items.append(f"{v} -> {k}")
        elif f["shape"] == "numbered":
            items.append(f"{i}. {k}: {v}")
    return sep.join(items)


def parse(text: str, f: dict):
    """Inverse of render, built from the same descriptor. None = unparseable.

    Used only to gate generation -- it is the assertion that a rendered
    target actually encodes the entity list it claims to.
    """
    keys = KEY_LEXICONS[f["lex"]]
    inv = {_cased(v, f["case"]): k for k, v in keys.items()}
    t = (text or "").strip()
    if f["shape"] == "json":
        try:
            d = json.loads(t)
        except Exception:
            return None
        out = []
        for k, vs in d.items():
            if k not in inv or not isinstance(vs, list):
                return None
            out += [(str(v), inv[k]) for v in vs]
        return out
    if t == f["empty"]:
        return []
    kpat = "|".join(re.escape(k) for k in inv)
    pats = {
        "kv_colon":   rf"^({kpat})\s*:\s*(.+)$",
        "kv_eq":      rf"^({kpat})\s*=\s*(.+)$",
        "kv_bracket": rf"^\[({kpat})\]\s*(.+)$",
        "numbered":   rf"^\d+\.\s*({kpat})\s*:\s*(.+)$",
        "kv_paren":   rf"^(.+?)\s*\(({kpat})\)$",
        "kv_arrow":   rf"^(.+?)\s*->\s*({kpat})$",
    }
    rx = re.compile(pats[f["shape"]])
    val_first = f["shape"] in ("kv_paren", "kv_arrow")
    # split on the separator actually used, which _sep_for may have overridden
    parts = re.split(r"\n", t) if "\n" in t else t.split(f["sep"])
    out = []
    for p in parts:
        m = rx.match(p.strip())
        if not m:
            return None
        a, b = m.group(1), m.group(2)
        v, k = (a, b) if val_first else (b, a)
        out.append((v.strip(), inv[k]))
    return out


def _gate_format(f: dict) -> None:
    """Constructive control: a compliant render must parse back, and a
    one-edit break must not. Runs on every distinct format drawn."""
    probe = [("Knud Vilby", ACTIVE[0]), ("Aarhus", ACTIVE[-1])]
    r = render(probe, f)
    got = parse(r, f)
    assert got is not None and sorted(got) == sorted(probe), \
        f"round-trip failed for {f}: {r!r} -> {got!r}"
    assert parse(render([], f), f) == [], f"empty case failed for {f}"
    broken = r.replace(":", "").replace("=", "").replace("->", "") \
              .replace("(", "").replace("[", "") + "@@"
    if f["shape"] != "json" and broken != r:
        assert parse(broken, f) != sorted(probe), f"break not detected for {f}"


# ---------------------------------------------------------------- instructions

INSTR_OPEN = [
    "Find alle {types} i teksten",
    "Udtræk {types} fra teksten",
    "Identificér {types} i følgende tekst",
    "Angiv de {types} der optræder i teksten",
    "Hvilke {types} nævnes i teksten?",
]
GLOSS = {"person": "personer", "org": "organisationer", "sted": "steder",
         "dato": "datoer"}


def _da_list(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " og " + xs[-1]


def describe(f: dict) -> str:
    """Spell the format out in words -- for instruction-only rows, where
    there are no demonstrations to induce it from."""
    keys = KEY_LEXICONS[f["lex"]]
    ex = {b: _cased(keys[b], f["case"]) for b in ACTIVE}
    if f["shape"] == "json":
        shape = ("JSON på formen {"
                 + ", ".join(f'"{ex[b]}": []' for b in ACTIVE) + "}")
    else:
        tmpl = {"kv_colon": f"{ex['person']}: enhed",
                "kv_eq": f"{ex['person']}=enhed",
                "kv_bracket": f"[{ex['person']}] enhed",
                "kv_paren": f"enhed ({ex['person']})",
                "kv_arrow": f"enhed -> {ex['person']}",
                "numbered": f"1. {ex['person']}: enhed"}[f["shape"]]
        joiner = {"\n": "én per linje", "; ": "adskilt af semikolon",
                  " | ": "adskilt af lodret streg",
                  ", ": "adskilt af komma"}[f["sep"]]
        shape = f'"{tmpl}", {joiner}'
    # skip the gloss where the key IS the word ("Brug personer for personer")
    labs = ", ".join(f"{ex[b]} for {GLOSS[b]}" for b in ACTIVE
                     if ex[b].lower() != GLOSS[b])
    labs = labs or "de viste nøgler"
    return (f"Svar i formatet {shape}. Brug {labs}. "
            f'Er der ingen enheder, så skriv "{f["empty"]}". '
            f"Skriv enhederne præcis som de står i teksten.")


# ---------------------------------------------------------------- data source

CANON = {"PERSON": "person", "PER": "person", "ORGANIZATION": "org",
         "ORG": "org", "GPE": "sted", "LOCATION": "sted", "LOC": "sted",
         "FACILITY": "sted", "DATE": "dato"}


def load_rows(split: str, maxlen=240):
    from datasets import load_dataset
    rows = []
    for r in load_dataset("KennethEnevoldsen/dane_plus", split=split):
        t = (r.get("text") or "").strip()
        if not t or len(t) > maxlen:
            continue
        ents = []
        for e in sorted(r["ents"] or [], key=lambda e: e["start"]):
            lab = CANON.get(str(e.get("label", "")).upper())
            s = t[e["start"]:e["end"]].strip()
            if lab and s:
                ents.append((s, lab))
        # dedupe preserving occurrence order
        seen, uniq = set(), []
        for v, b in ents:
            if (v.lower(), b) not in seen:
                seen.add((v.lower(), b))
                uniq.append((v, b))
        rows.append({"text": t, "ents": uniq,
                     "surf": {v.lower() for v, _ in uniq},
                     "types": {b for _, b in uniq}})
    return rows


# ---------------------------------------------------------------- row building

def build_row(rng, target, pool, f, mode: str, shots: int):
    """One training row. mode: examples | both | instruction."""
    # anti-leak: no shared entity surface, and no exemplar surface occurring
    # anywhere in the target text (a leaked value must be traceable to the
    # prompt, and must be wrong)
    tl = target["text"].lower()
    cands = [r for r in pool
             if r is not target
             and not (r["surf"] & target["surf"])
             and not any(s in tl for s in r["surf"])]
    if len(cands) < shots:
        return None
    full = [r for r in cands if r["ents"]]
    void = [r for r in cands if not r["ents"]]

    # WELL-POSEDNESS: every type in the target's answer must be demonstrated
    # by some exemplar. Without this the row can be unanswerable -- under an
    # arbitrary lexicon (Alfa/Beta/Gamma/Delta) a label carries no semantic
    # hint, so a target place with no exemplar showing Gamma asks the model
    # to produce a mapping it was never given. Training on those rewards
    # guessing a plausible label without evidence, the exact behaviour the
    # 19% LEAK measurement flagged.
    picks, covered = [], set()
    need_types = set(target["types"])
    pool_c = list(full)
    rng.shuffle(pool_c)
    while need_types - covered and len(picks) < shots:
        # greedy set cover: take the exemplar contributing most missing types
        best = max(pool_c, key=lambda r: len(r["types"] & (need_types - covered)),
                   default=None)
        if best is None or not (best["types"] & (need_types - covered)):
            break
        picks.append(best)
        covered |= best["types"]
        pool_c.remove(best)
    if need_types - covered:
        return None          # not coverable within this shot budget -- redraw

    # hard negative: an exemplar carrying a type the target lacks, so copying
    # it yields a false positive. Secondary to coverage.
    # When the ANSWER is the empty marker, the marker IS the format being
    # tested, and it is drawn from five options (ingen / - / [] / (tom) /
    # ingen enheder). A row answering "(tom)" with no exemplar showing
    # "(tom)" is unanswerable for the same reason a Gamma answer with no
    # Gamma demonstration is: part of the format is required but never
    # shown. Reserve a slot for it before spending shots on anything else.
    must_void = 1 if (not target["ents"] and void) else 0
    if not target["ents"] and not void:
        return None

    missing = set(ACTIVE) - target["types"]
    hard = [r for r in pool_c if r["types"] & missing] if missing else []
    if hard and len(picks) < shots - must_void:
        picks.append(rng.choice(hard))
        pool_c.remove(picks[-1])
    # At most one empty exemplar. dane_plus is ~2/3 entity-free sentences, so
    # unbiased sampling gives rows whose demonstrations are mostly the empty
    # marker -- those show the format's fallback but never the format itself,
    # which is the thing to be induced. One empty exemplar still teaches the
    # fallback.
    need = shots - len(picks)
    n_void = min(len(void), need,
                 max(must_void, 1 if (void and rng.random() < 0.35) else 0))
    if must_void and n_void < 1:
        return None       # no slot left to demonstrate the marker -- redraw
    if len(pool_c) < need - n_void:
        return None
    picks += rng.sample(void, n_void) + rng.sample(pool_c, need - n_void)
    rng.shuffle(picks)

    types = _da_list([GLOSS[b] for b in ACTIVE])
    instr = rng.choice(INSTR_OPEN).format(types=types)

    parts = []
    if mode in ("both", "instruction"):
        # several openings are questions; appending "." unconditionally
        # produced "... nævnes i teksten?."
        parts.append(instr if instr.endswith("?") else instr + ".")
    if mode == "instruction":
        parts.append(describe(f))
    if mode in ("examples", "both") and picks:
        parts.append("Eksempler:")
        for e in picks:
            parts.append(f'Tekst: {e["text"]}\nSvar: {render(e["ents"], f)}')
    # The passage is NOT wrapped in quotes: dane_plus sentences frequently
    # contain a double quote of their own, which produced sequences like
    # ...igen."" and made the delimiter ambiguous. "Svar:" is a sufficient
    # terminator for single-line passages.
    parts.append(f'Tekst: {target["text"]}\nSvar:')
    user = "\n\n".join(parts)
    answer = render(target["ents"], f)

    # gate: the target must parse back to exactly the gold entity list
    got = parse(answer, f)
    if got is None or sorted((v.lower(), b) for v, b in got) != \
            sorted((v.lower(), b) for v, b in target["ents"]):
        return None
    return {
        "messages": [{"role": "user", "content": user},
                     {"role": "assistant", "content": answer}],
        "meta": {"task": "ner", "mode": mode, "shots": len(picks) if mode != "instruction" else 0,
                 "fmt": fmt_id(f), "shape": f["shape"], "lex": f["lex"],
                 "order": f["order"], "case": f["case"],
                 "arbitrary_keys": f["lex"] in ARBITRARY,
                 "n_ents": len(target["ents"]),
                 # types the answer uses, and the types the exemplars showed --
                 # the first must be a subset of the second for the row to be
                 # answerable from context
                 "answer_types": sorted(target["types"]),
                 "demoed_types": sorted({b for e in picks for b in e["types"]}),
                 "heldout_fmt": is_heldout(f)},
    }


# ---------------------------------------------------------------- pattern tasks

def _words(t):
    return [w for w in re.findall(r"[^\W\d_]+", t, re.UNICODE) if w]




# Character-level families (rot / reverse-each-word / strip-vowels /
# first-letter / word-length / longer-than-N / sort-by-length / contains-æøå)
# were removed: they require the model to see inside a token, which a subword
# tokenizer actively obstructs -- the output shares no subwords with the
# input. They are answerable in principle but would likely be the families
# that never converge. Everything kept here treats words as atoms.
#
# A family is (input kind, hypothesis params, apply). apply(param, inp) ->
# list[str]. `params` enumerates EVERY hypothesis the family admits; an
# unparameterised family carries [None]. Keeping hypotheses explicit and
# enumerable is what makes the ambiguity gate below possible -- a row is only
# usable if the demonstrations pin down a single answer for the target, and
# that cannot be checked without knowing what else the learner might infer.
FAMILIES = {
    # ---- word lists
    "store":         ("words", [None], lambda p, w: [x.upper() for x in w]),
    "omvendt":       ("words", [None], lambda p, w: list(reversed(w))),
    "sorteret":      ("words", [None], lambda p, w: sorted(w, key=str.lower)),
    "unik":          ("words", [None], lambda p, w: list(dict.fromkeys(w))),
    "hvert_n":       ("words", [2, 3], lambda p, w: w[::p]),
    "første_k":      ("words", [1, 2, 3], lambda p, w: w[:p]),
    "sidste_k":      ("words", [1, 2, 3], lambda p, w: w[-p:]),
    "affin":         ("nums", [(a, b) for a in (2, 3, 4) for b in (0, 1, -1, 5)],
                      lambda p, x: [str(p[0] * v + p[1]) for v in x]),
    "modulo":        ("nums", [3, 4, 5, 7, 9],
                      lambda p, x: [str(v % p) for v in x]),
    "kvadrat":       ("nums", [None], lambda p, x: [str(v * v) for v in x]),
    "cifersum":      ("nums", [None],
                      lambda p, x: [str(sum(int(c) for c in str(v))) for v in x]),
    "over_t":        ("nums", [20, 30, 40, 50, 60],
                      lambda p, x: [str(v) for v in x if v > p]),
    "voksende":      ("nums", [None], lambda p, x: [str(v) for v in sorted(x)]),
    "faldende":      ("nums", [None],
                      lambda p, x: [str(v) for v in sorted(x, reverse=True)]),
    "løbende_sum":   ("nums", [None],
                      lambda p, x: [str(v) for v in itertools.accumulate(x)]),
    # ---- "navn: tal" pairs
    "kun_navne":     ("pairs", [None], lambda p, z: [n for n, _ in z]),
    "kun_værdier":   ("pairs", [None], lambda p, z: [str(v) for _, v in z]),
    "byt_om":        ("pairs", [None], lambda p, z: [f"{v}: {n}" for n, v in z]),
    "værdi_over":    ("pairs", [20, 30, 40, 50, 60],
                      lambda p, z: [n for n, v in z if v > p]),
    "sorter_værdi":  ("pairs", [None],
                      lambda p, z: [n for n, v in sorted(z, key=lambda y: y[1])]),
}


def _gen_input(kind, rng, texts, persons):
    if kind == "words":
        for t in rng.sample(texts, min(len(texts), 40)):
            ws = _words(t)[:8]
            if len(ws) >= 3:
                return ws
        return None
    if kind == "nums":
        return [rng.randint(1, 99) for _ in range(rng.randint(3, 6))]
    # names are drawn from the corpus's own PER surfaces rather than a
    # hardcoded list, so the pair inputs carry the same name distribution as
    # everything else in the set
    if len(persons) < 5:
        return None
    return [(n, rng.randint(5, 95))
            for n in rng.sample(persons, rng.randint(3, 5))]


def _consistent(kind, demos):
    """Every (family, param) hypothesis reproducing all demonstrations."""
    out = []
    for nm, (k, params, fn) in FAMILIES.items():
        if k != kind:
            continue
        for q in params:
            try:
                if all(fn(q, i) == o for i, o in demos):
                    out.append((nm, q, fn))
            except Exception:
                pass
    return out


def build_pattern_row(rng, texts, persons, shots: int):
    """A sampled mapping shown ONLY by example, never described.

    These are the rows where in-context induction is the sole route to the
    answer: there is no instruction, and both the operation and its parameter
    are drawn fresh, so neither a task prior nor a memorised format can
    substitute for reading the demonstrations.

    Parameterised families make well-posedness non-trivial. "keep words
    longer than 5" and "keep words longer than 6" agree on plenty of inputs,
    so demonstrations can leave the target genuinely undetermined -- the same
    defect as an answer key that no exemplar shows. The gate: collect every
    hypothesis in the family space consistent with all demonstrations, and
    keep the row only if they all yield the same target output.
    """
    name = rng.choice(list(FAMILIES))
    kind, params, fn = FAMILIES[name]
    param = rng.choice(params)

    picked, seen = [], set()
    for _ in range(80):
        inp = _gen_input(kind, rng, texts, persons)
        if inp is None:
            continue
        key = tuple(map(str, inp))
        if key in seen:
            continue
        try:
            out = fn(param, inp)
        except Exception:
            continue
        if not out:
            continue
        seen.add(key)
        picked.append((inp, out))
        if len(picked) == shots + 1:
            break
    if len(picked) < shots + 1:
        return None

    demos, (t_in, t_out) = picked[:-1], picked[-1]
    cons = _consistent(kind, demos)
    if len({tuple(f(q, t_in)) for _, q, f in cons}) != 1:
        return None          # demonstrations do not determine the target
    # Degenerate target: the transform happens to be the identity here (a
    # 3-word input under "first 3", an already-uppercase input under
    # "uppercase", a duplicate-free list under "dedupe"). Answerable, but it
    # supervises copying rather than the rule, which is the one habit this
    # model already has too much of.
    if [str(x) for x in t_in] == list(t_out):
        return None

    sep = rng.choice([" ", ", ", " | ", "\n"])

    def show(i):
        if kind == "pairs":
            return ", ".join(f"{n}: {v}" for n, v in i)
        return ", ".join(map(str, i)) if kind == "nums" else " ".join(i)

    lines = [f"{show(i)}\n-> {sep.join(o)}" for i, o in demos]
    user = "\n\n".join(lines + [f"{show(t_in)}\n->"])
    return {
        "messages": [{"role": "user", "content": user},
                     {"role": "assistant", "content": sep.join(t_out)}],
        "meta": {"task": "pattern", "mode": "examples", "shots": shots,
                 "fmt": f"pattern|{name}", "shape": f"pattern_{kind}",
                 "lex": name, "order": "occurrence", "case": "lower",
                 "arbitrary_keys": True, "n_ents": len(t_out),
                 "kind": kind, "param": str(param),
                 # how many distinct hypotheses survived the demonstrations;
                 # 1 means the demos identify the rule outright, >1 means
                 # several rules fit but agree on this target
                 "n_hypotheses": len(cons),
                 "heldout_fmt": False},
    }


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="scratch/icl_smoke")
    ap.add_argument("--pattern-frac", type=float, default=0.25)
    ap.add_argument("--instruction-frac", type=float, default=0.0,
                    help="Share of rows that SPELL OUT the format in words "
                         "instead of demonstrating it. Those rows train "
                         "described-format following, not in-context "
                         "learning, so the default is 0 for a pure ICL set. "
                         "Note 'both' rows are still ICL: their instruction "
                         "names the task but never the format.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show", type=int, default=4)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train_pool = load_rows("train")
    dev_pool = load_rows("dev")
    global ACTIVE
    present = {b for r in train_pool + dev_pool for _, b in r["ents"]}
    dropped = [b for b in BUCKETS if b not in present]
    ACTIVE = [b for b in BUCKETS if b in present]
    print(f"source: dane_plus train={len(train_pool)} dev={len(dev_pool)}",
          flush=True)
    print(f"active buckets: {ACTIVE}"
          + (f"   DROPPED (absent from source): {dropped}" if dropped else ""),
          flush=True)

    # exemplars are drawn from the same split as the target, so a row never
    # mixes splits; held-out FORMATS are the eval axis, not held-out text
    texts = [r["text"] for r in train_pool]
    persons = sorted({v for r in train_pool for v, b in r["ents"]
                      if b == "person" and " " not in v and len(v) > 2})
    fi = args.instruction_frac
    MODES = ["examples", "both", "instruction"]
    MW = [(1 - fi) / 2, (1 - fi) / 2, fi]
    print(f"modes: examples={MW[0]:.2f} both={MW[1]:.2f} "
          f"instruction={MW[2]:.2f}"
          + ("   (pure ICL: format is never described, only demonstrated)"
             if fi == 0 else ""), flush=True)

    rows, gated, tried = [], set(), 0
    n_pat = int(args.n * args.pattern_frac)
    while len(rows) < args.n and tried < args.n * 60:
        tried += 1
        if sum(1 for r in rows if r["meta"]["task"] == "pattern") < n_pat \
                and rng.random() < 0.35:
            r = build_pattern_row(rng, texts, persons,
                                  rng.randint(2, 5))
            if r:
                rows.append(r)
            continue
        f = sample_format(rng)
        if fmt_id(f) not in gated:
            _gate_format(f)          # raises on a bad format descriptor
            gated.add(fmt_id(f))
        split_pool = dev_pool if is_heldout(f) else train_pool
        # Cap empty targets. Two thirds of dane_plus sentences carry no
        # entity, so an unbiased draw made 67% of targets the bare empty
        # marker -- rows that supervise the fallback and nothing about
        # applying the sampled format. Keep enough to teach the fallback.
        n_ner = sum(1 for r in rows if r["meta"]["task"] == "ner")
        n_empty = sum(1 for r in rows
                      if r["meta"]["task"] == "ner" and r["meta"]["n_ents"] == 0)
        want_empty = n_ner == 0 or n_empty / max(1, n_ner) < 0.25
        cand = [r for r in split_pool if bool(r["ents"]) != want_empty]
        target = rng.choice(cand or split_pool)
        mode = rng.choices(MODES, weights=MW)[0]
        shots = rng.randint(1, 5)
        r = build_row(rng, target, split_pool, f, mode, shots)
        if r:
            rows.append(r)

    rng.shuffle(rows)
    tr = [r for r in rows if not r["meta"]["heldout_fmt"]]
    ev = [r for r in rows if r["meta"]["heldout_fmt"]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for nm, rs in (("train", tr), ("eval_heldout_fmt", ev)):
        p = out / f"{nm}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                             for r in rs))
        print(f"-> {p}  ({len(rs)} rows)")

    print(f"\n{len(rows)} rows from {tried} draws; "
          f"{len(gated)} distinct formats gated (round-trip + break)")
    print(f"train={len(tr)}  heldout-format eval={len(ev)}")
    seen_f = {r["meta"]["fmt"] for r in tr}
    ev_f = {r["meta"]["fmt"] for r in ev}
    print(f"format overlap train/eval: {sorted(seen_f & ev_f) or 'none'}")
    for k in ("task", "mode", "shots", "shape", "lex", "order", "case"):
        c = Counter(r["meta"][k] for r in rows)
        print(f"  {k:<7} " + "  ".join(f"{a}={b}" for a, b in
                                       sorted(c.items(), key=lambda x: -x[1])))
    arb = sum(r["meta"]["arbitrary_keys"] for r in rows)
    print(f"  arbitrary-key rows: {arb}/{len(rows)} ({100*arb/len(rows):.0f}%)")

    for i, r in enumerate(rows[:args.show], 1):
        m = r["meta"]
        print("\n" + "=" * 78)
        print(f"[{i}] task={m['task']} mode={m['mode']} shots={m['shots']} "
              f"fmt={m['fmt']} order={m['order']} case={m['case']} "
              f"heldout={m['heldout_fmt']}")
        print("-" * 78)
        print(r["messages"][0]["content"])
        print("--- ANSWER (only tokens with loss) ---")
        print(r["messages"][1]["content"])


if __name__ == "__main__":
    main()
