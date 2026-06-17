"""Generate morphology-pattern ICL SFT data.

Teaches the model to apply Esperanto morphological transformations from
few-shot demonstrations. Covers:

- Plural (-oj):       kato → katoj
- Accusative (-on):   pomo → pomon
- Mal- antonyms:      granda → malgranda
- Diminutive (-et):   kato → kateto
- Augmentative (-eg): domo → domego
- Profession (-ist):  instrui → instruisto
- Feminine (-in):     patro → patrino
- Place (-ej):        lerni → lernejo
- Verb tenses:        kuri → kuras / kuris / kuros / kurus
- Gerund (-ad):       kuri → kurado

Output: data/sft/sft_morphology_icl.jsonl
"""

import argparse
import json
import random
from pathlib import Path


# ---- Curated transformation tables ------------------------------------

# Mal- antonyms (must be meaningful pairs)
MAL_ANTONYMS = [
    ("granda", "malgranda"), ("alta", "malalta"), ("longa", "mallonga"),
    ("varma", "malvarma"), ("bona", "malbona"), ("bela", "malbela"),
    ("rapida", "malrapida"), ("facila", "malfacila"), ("juna", "maljuna"),
    ("nova", "malnova"), ("pura", "malpura"), ("riĉa", "malriĉa"),
    ("forta", "malforta"), ("plena", "malplena"), ("alta", "malalta"),
    ("ĝoja", "malĝoja"), ("vera", "malvera"), ("amiko", "malamiko"),
    ("fermi", "malfermi"), ("ami", "malami"), ("vasta", "malvasta"),
    ("dika", "maldika"), ("simpla", "malsimpla"), ("klara", "malklara"),
    ("multa", "malmulta"), ("frua", "malfrua"), ("kara", "malkara"),
    ("luma", "mallumo"), ("granda", "malgranda"), ("danĝera", "maldanĝera"),
    ("interesa", "malinteresa"), ("dolĉa", "maldolĉa"), ("kruda", "malkruda"),
    ("trankvila", "maltrankvila"), ("supre", "malsupre"), ("antaŭ", "malantaŭ"),
    ("dekstra", "maldekstra"), ("vere", "malvere"), ("paco", "malpaco"),
    ("sukceso", "malsukceso"), ("aperti", "malaperti"), ("amiko", "malamiko"),
]

# Profession nouns from verbs (-isto)
PROFESSIONS = [
    ("instrui", "instruisto"), ("kanti", "kantisto"), ("verki", "verkisto"),
    ("kuiri", "kuiristo"), ("danci", "dancisto"), ("ludi", "ludisto"),
    ("legi", "leganto"),  # actually -anto here; included for variety
    ("piano", "pianisto"), ("violono", "violonisto"), ("ĵurnal", "ĵurnalisto"),
    ("artikolo", "artikolisto"), ("scienco", "sciencisto"),
    ("naturo", "naturisto"), ("biologio", "biologisto"),
    ("ekonomio", "ekonomiisto"), ("medicino", "mediciinisto"),
    ("histori", "historiisto"), ("politiko", "politikisto"),
    ("art", "artisto"), ("muziko", "muzikisto"),
    ("ŝak", "ŝakisto"), ("futbal", "futbalisto"), ("tennis", "tenisisto"),
    ("optik", "okulisto"),
]

# Feminine pairs (curated — most family / role pairs)
FEMININE = [
    ("patro", "patrino"), ("frato", "fratino"), ("filo", "filino"),
    ("avo", "avino"), ("onklo", "onklino"), ("kuzo", "kuzino"),
    ("knabo", "knabino"), ("viro", "virino"), ("reĝo", "reĝino"),
    ("aktoro", "aktorino"), ("amiko", "amikino"), ("instruisto", "instruistino"),
    ("studento", "studentino"), ("najbaro", "najbarino"),
    ("kato", "katino"), ("hundo", "hundino"), ("ĉevalo", "ĉevalino"),
    ("leono", "leonino"), ("ŝafo", "ŝafino"), ("hano", "hanino"),
    ("kelnero", "kelnerino"), ("doktoro", "doktorino"),
]

# Common nouns for plural / accusative / diminutive / augmentative
COMMON_NOUNS = [
    "kato", "hundo", "ĉevalo", "birdo", "fiŝo", "leono", "muso", "elefanto",
    "domo", "ĉambro", "lernejo", "vojo", "urbo", "lando", "muro", "pordo",
    "libro", "skribilo", "krajono", "ĉapelo", "tablo", "seĝo", "fenestro",
    "pomo", "pano", "akvo", "lakto", "kafo", "teo", "supo", "fromaĝo",
    "viro", "virino", "knabo", "knabino", "homo", "infano", "amiko",
    "tago", "nokto", "horo", "minuto", "jaro", "monato", "semajno",
    "stelo", "luno", "suno", "nubo", "monto", "rivero", "lago", "maro",
    "arbo", "floro", "herbo", "folio", "branĉo",
    "manĝaĵo", "trinkaĵo", "donaco", "kanto", "rakonto",
]

# Common verbs (infinitives)
COMMON_VERBS = [
    "kuri", "iri", "veni", "fari", "diri", "vidi", "aŭdi", "scii", "havi",
    "esti", "manĝi", "trinki", "dormi", "vivi", "labori", "ludi", "lerni",
    "skribi", "legi", "instrui", "danci", "kanti", "kuiri", "forĝi",
    "konstrui", "rompi", "fermi", "malfermi", "porti", "preni", "doni",
    "aĉeti", "vendi", "pagi", "atendi", "rideti", "ridi", "plori", "krii",
    "pensi", "kompreni", "amari", "ami", "ĝui", "esperi", "memori",
]


# ---- Programmatic transformations -------------------------------------

def add_n(word: str) -> str:
    if word.endswith(("o", "a", "oj", "aj")):
        return word + "n"
    return word

def plural(word: str) -> str:
    if word.endswith("o"):
        return word + "j"
    return word

def diminutive(word: str) -> str:
    if word.endswith("o"):
        return word[:-1] + "eto"
    return word

def augmentative(word: str) -> str:
    if word.endswith("o"):
        return word[:-1] + "ego"
    return word

def gerund(verb: str) -> str:
    if verb.endswith("i"):
        return verb[:-1] + "ado"
    return verb

def present(verb: str) -> str:
    if verb.endswith("i"):
        return verb[:-1] + "as"
    return verb

def past(verb: str) -> str:
    if verb.endswith("i"):
        return verb[:-1] + "is"
    return verb

def future(verb: str) -> str:
    if verb.endswith("i"):
        return verb[:-1] + "os"
    return verb

def conditional(verb: str) -> str:
    if verb.endswith("i"):
        return verb[:-1] + "us"
    return verb


# Transformations as named pairs (label, list of (input, output))
TRANSFORMATIONS = {
    "mal_antonym": MAL_ANTONYMS,
    "feminine":    FEMININE,
    "profession":  PROFESSIONS,
    # generated programmatically below
}

def _build_programmatic():
    TRANSFORMATIONS["plural"]      = [(n, plural(n))      for n in COMMON_NOUNS]
    TRANSFORMATIONS["accusative"]  = [(n, add_n(n))       for n in COMMON_NOUNS]
    TRANSFORMATIONS["diminutive"]  = [(n, diminutive(n))  for n in COMMON_NOUNS]
    TRANSFORMATIONS["augmentative"]= [(n, augmentative(n))for n in COMMON_NOUNS]
    TRANSFORMATIONS["gerund"]      = [(v, gerund(v))      for v in COMMON_VERBS]
    TRANSFORMATIONS["present"]     = [(v, present(v))     for v in COMMON_VERBS]
    TRANSFORMATIONS["past"]        = [(v, past(v))        for v in COMMON_VERBS]
    TRANSFORMATIONS["future"]      = [(v, future(v))      for v in COMMON_VERBS]
    TRANSFORMATIONS["conditional"] = [(v, conditional(v)) for v in COMMON_VERBS]

_build_programmatic()


# ---- Renderers (subset / variants from atomic ICL) --------------------

def r_arrow(shots, query):
    lines = [f"{x} → {y}" for x, y in shots] + [f"{query} →"]
    return "\n".join(lines), ""

def r_eq(shots, query):
    lines = [f"{x} = {y}" for x, y in shots] + [f"{query} ="]
    return "\n".join(lines), ""

def r_colon(shots, query):
    lines = [f"{x}: {y}" for x, y in shots] + [f"{query}:"]
    return "\n".join(lines), ""

def r_dash(shots, query):
    lines = [f"{x} — {y}" for x, y in shots] + [f"{query} —"]
    return "\n".join(lines), ""

def r_doublearr(shots, query):
    lines = [f"{x} => {y}" for x, y in shots] + [f"{query} =>"]
    return "\n".join(lines), ""

def r_qa_tag(shots, query):
    lines = [f"<Q>{x}</Q> <A>{y}</A>" for x, y in shots] + [f"<Q>{query}</Q>"]
    return "\n".join(lines), "<A>{ans}</A>"

def r_bracket(shots, query):
    lines = [f"[A] {x} [B] {y}" for x, y in shots] + [f"[A] {query}"]
    return "\n".join(lines), "[B] {ans}"

def r_numbered(shots, query):
    lines = [f"{i+1}. {x} → {y}" for i, (x, y) in enumerate(shots)]
    lines.append(f"{len(shots)+1}. {query} →")
    return "\n".join(lines), ""

def r_inline(shots, query):
    chunks = [f"{x} → {y}" for x, y in shots] + [f"{query} →"]
    return " ; ".join(chunks), ""

RENDERERS = [r_arrow, r_eq, r_colon, r_dash, r_doublearr, r_qa_tag, r_bracket, r_numbered, r_inline]


PREAMBLES = [
    "", "", "", "", "",   # often no preamble
    "Plenigu la mankon:\n",
    "Daŭrigu la ŝablonon:\n",
    "Sekvante la ekzemplojn:\n",
    "Identigu la regulon kaj plenumu:\n",
    "Apliku la saman transformon:\n",
    "Komplete la lastan linion:\n",
    "Studu kaj kompletigu:\n",
]


def make_example(rng):
    """Sample one morphology ICL example."""
    label = rng.choice(list(TRANSFORMATIONS.keys()))
    pairs = TRANSFORMATIONS[label]
    K = rng.choice([2, 3, 3, 3, 4, 4, 5])
    if len(pairs) < K + 1:
        K = len(pairs) - 1
    sample = rng.sample(pairs, K + 1)
    shots = sample[:-1]
    query, answer = sample[-1]

    renderer = rng.choice(RENDERERS)
    user_body, ans_template = renderer(shots, query)
    preamble = rng.choice(PREAMBLES)
    user_msg = preamble + user_body
    assistant_msg = ans_template.format(ans=answer) if ans_template else answer
    return {
        "messages": [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/sft/sft_morphology_icl.jsonl"))
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Transformation types: {len(TRANSFORMATIONS)}")
    for label, pairs in TRANSFORMATIONS.items():
        print(f"  {label:15s} {len(pairs):>4} pairs")

    rng = random.Random(args.seed)

    if args.dry_run:
        print("\n--- 8 sample examples ---\n")
        for i in range(8):
            ex = make_example(rng)
            print(f"=== Example {i+1} ===")
            print("USER:")
            print(ex["messages"][0]["content"])
            print("\nASSISTANT:", ex["messages"][1]["content"])
            print()
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "w") as f:
        while written < args.n:
            f.write(json.dumps(make_example(rng), ensure_ascii=False) + "\n")
            written += 1
            if written % 2500 == 0:
                print(f"  written {written:,}/{args.n:,}")
    print(f"\nWrote {written:,} morphology ICL examples → {args.out}")


if __name__ == "__main__":
    main()
