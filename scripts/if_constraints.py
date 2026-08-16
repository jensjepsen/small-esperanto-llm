"""Atomic instruction-following constraints for the DA IF-data build.

Each constraint has:
  - name              : short slug for logging / provenance
  - params            : concrete parameters for one instance
  - render_variants   : LIST of Danish natural-language phrasings for the rule.
                        One is picked randomly per row so the model learns
                        the constraint semantics, not one specific phrasing.
  - check(text,params)-> bool : programmatic verifier
  - tags              : used by the combo picker to avoid conflicts

Scope: length + lexical + format + case + content-structure (no code, no LaTeX).
"""
from __future__ import annotations

import random
import re
import json
import unicodedata
from dataclasses import dataclass, field
from typing import Callable


# ────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────

_SENT_END_RE = re.compile(r"[.!?]+(?:\s+|$)")


def sentences(text: str) -> list[str]:
    parts = _SENT_END_RE.split(text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def words(text: str) -> list[str]:
    return re.findall(r"\S+", text.strip())


def norm(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def plur(n: int, sg: str, pl: str) -> str:
    return f"{n} {sg if n == 1 else pl}"


# ────────────────────────────────────────────────────────────────────────────
# Constraint dataclass
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Constraint:
    name: str
    render_variants: list[Callable[[dict], str]]
    check: Callable[[str, dict], bool]
    tags: frozenset[str]
    sample: Callable[[random.Random, dict | None], dict]
    solo: bool = False
    applicable: Callable[[dict | None], bool] = field(default_factory=lambda: (lambda ctx: True))


# ────────────────────────────────────────────────────────────────────────────
# Content pools
# ────────────────────────────────────────────────────────────────────────────

_KEYWORD_POOL = [
    # Geography / place
    "Danmark", "København", "Aarhus", "Odense", "Aalborg", "Jylland", "Fyn",
    "Sjælland", "Bornholm", "Norden", "Europa", "havet", "søen", "skoven",
    "byen", "landet", "landsby", "hovedstad", "region", "kommune", "Grønland",
    "Færøerne", "fjord", "strand", "kyst", "øen", "bjerg", "dal", "eng",
    # Nature / weather
    "sommer", "vinter", "forår", "efterår", "solen", "månen", "regn", "sne",
    "vind", "storm", "blomster", "træer", "fugle", "bølge", "sky", "tåge",
    "torden", "lyn", "regnbue", "is", "frost", "dug", "vandet", "ilden",
    "jord", "sand", "sten", "klippe", "græs", "løv", "blad", "rod",
    # Animals
    "hund", "kat", "hest", "ko", "får", "gris", "kylling", "ørn", "ulv",
    "bjørn", "ræv", "hare", "sæl", "hval", "ørred", "laks", "bi", "myre",
    # Body / health
    "hånd", "fod", "hjerte", "øje", "ansigt", "hår", "hoved", "krop", "hud",
    "blod", "søvn", "smerte", "helbred", "puls",
    # Colors
    "rød", "blå", "grøn", "gul", "hvid", "sort", "brun", "grå", "lyseblå",
    # Everyday food + drink
    "kaffe", "brød", "vin", "smør", "morgenmad", "middag", "aftensmad",
    "æble", "pære", "banan", "ost", "fisk", "kød", "sukker", "salt",
    "honning", "kage", "chokolade", "te", "juice", "øl", "mælk",
    # Transport / travel
    "cykel", "tog", "bus", "bil", "gåtur", "fly", "skib", "båd", "metro",
    "sporvogn", "havn", "lufthavn", "station",
    # Society / abstract
    "regering", "kunst", "musik", "litteratur", "historie", "videnskab",
    "teknologi", "sundhed", "uddannelse", "arbejde", "familie", "venner",
    "kærlighed", "glæde", "håb", "drøm", "tanke", "idé", "angst", "sorg",
    "vrede", "mod", "frygt", "latter", "sang", "dans",
    # Objects / household
    "bog", "avis", "brev", "telefon", "computer", "lampe", "vindue", "dør",
    "bord", "stol", "sæbe", "vand", "seng", "tæppe", "kop", "gaffel", "kniv",
    "ur", "nøgle", "spejl", "billede",
    # Buildings
    "hus", "slot", "kirke", "tårn", "bro", "borg", "bibliotek", "museum",
    "hospital", "rådhus", "butik", "café",
    # People roles
    "lærer", "læge", "kok", "bonde", "kunstner", "forfatter", "politiker",
    "barn", "voksen", "ven", "nabo", "chef", "kollega", "student", "pensionist",
    "arkitekt", "ingeniør", "musiker", "atlet",
    # School / life
    "skole", "eksamen", "ferie", "weekend", "fest", "møde", "rejse",
    "universitet", "kursus", "opgave", "projekt", "bryllup", "fødselsdag",
    # Time
    "morgen", "aften", "nat", "dag", "uge", "måned", "år", "time", "minut",
    "sekund", "dato", "årstid",
    # Danish culture
    "hygge", "kolonihave", "smørrebrød", "Dannebrog", "gadekær", "flag",
    "julegaven", "sommerhus",
    # Numbers / abstract quantities
    "tal", "nul", "en", "to", "tre", "hundred", "tusind",
]

_COMMON_CRUTCHES = [
    # Filler adverbs / discourse
    "meget", "faktisk", "altså", "sådan", "godt", "man", "jo", "vist",
    "bare", "lige", "simpelthen", "egentlig", "typisk", "generelt",
    "grundlæggende", "især", "nemlig", "således",
    # Mid-difficulty connectives — harder to avoid without breaking flow,
    # but not impossible (unlike `og` which is Danish's primary conjunction).
    "men", "eller", "hvis", "fordi", "derfor", "også", "når", "mens",
]

_STARTS_POOL = [
    "For det første,", "Til at begynde med,", "Først og fremmest,",
    "Indledningsvis,", "Som udgangspunkt,", "Faktisk", "Kort sagt,",
    "Ifølge min viden,", "Sagt på en anden måde,", "Ærligt talt,",
    "Det er værd at nævne, at", "Det interessante er, at",
    "Det korte svar er, at", "Uden tvivl,", "Historisk set,",
    "I moderne tid,", "Fra begyndelsen,", "Med tiden,",
    "Ved nærmere overvejelse,", "I det følgende,",
    "Med henblik på dette spørgsmål,", "Godt spørgsmål —",
    "Jeg vil gerne fremhæve, at",
    "Der findes flere måder at forklare dette på, men",
]

_ENDS_POOL = [
    "Det er det vigtigste at huske", "Kort sagt er det interessant",
    "Så meget for det", "Og sådan er det", "Det giver stof til eftertanke",
    "Alt taget i betragtning er det klart",
    "Der er meget mere at sige om emnet", "Sagen taler for sig selv",
    "Held og lykke med det", "God fornøjelse", "Prøv det selv",
    "Ses vi snart", "Tak for din tid", "Tak for at læse",
    "Overvej det grundigt", "Læs mere om emnet",
    "Husk at gøre det med omhu", "Del gerne dine tanker",
    "Hvad synes du selv", "Hvad er din holdning",
    "P.S. Husk at drikke vand", "P.S. God fornøjelse",
    "P.P.S. Tak for læsningen", "P.S. Del gerne dine tanker",
    "P.S. Skriv gerne igen",
    "Og det er ikke det værste af det", "Sådan er livet",
    "Vi ses på den anden side",
]

_HEADER_POOL = [
    "Baggrund", "Formål", "Metode", "Resultat", "Konklusion",
    "Analyse", "Diskussion", "Vurdering", "Refleksion",
    "Fordele", "Ulemper", "Argumenter", "Modargumenter",
    "Positive sider", "Negative sider",
    "Historie", "Baggrundshistorie", "Beskrivelse", "Definition",
    "Oprindelse", "Udvikling", "Kontekst",
    "Eksempler", "Anbefalinger", "Fremgangsmåde", "Trin",
    "Tips", "Alternativer", "Overvejelser",
    "Introduktion", "Sammenfatning", "Perspektiv", "Fremtiden",
    "Effekter", "Konsekvenser", "Årsager",
]

_TABLE_COLS_POOL = [
    "Navn", "Titel", "Beskrivelse", "Kategori", "Type",
    "År", "Dato", "Periode", "Sted", "By", "Land", "Region",
    "Antal", "Pris", "Vægt", "Længde", "Højde", "Størrelse",
    "Farve", "Materiale", "Producent", "Stil", "Genre",
    "Sværhedsgrad", "Prioritet", "Status",
    "Forfatter", "Instruktør", "Ingredienser", "Metode",
    "Fordel", "Ulempe", "Anbefaling", "Bemærkning",
]

_NTH_WORDS_POOL = [
    "Derefter", "Endelig", "Faktisk", "Herudover", "Konklusion",
    "Først", "Dernæst", "Yderligere", "Ligeledes", "Desuden",
    "Til gengæld", "Modsat", "Alligevel", "Ikke desto mindre",
    "Sammenfattende", "Kort sagt", "Endvidere", "Samtidig",
    "På den anden side", "Endelig kan man sige,",
]


# ────────────────────────────────────────────────────────────────────────────
# LENGTH constraints
# ────────────────────────────────────────────────────────────────────────────

exactly_n_sentences = Constraint(
    name="exactly_n_sentences",
    render_variants=[
        lambda p: f"Svaret skal være på præcis {plur(p['n'], 'sætning', 'sætninger')}.",
        lambda p: f"Skriv præcis {plur(p['n'], 'sætning', 'sætninger')} — hverken flere eller færre.",
        lambda p: f"Hold svaret til nøjagtig {plur(p['n'], 'sætning', 'sætninger')}.",
        lambda p: f"{plur(p['n'], 'sætning', 'sætninger')}, tak.",
        lambda p: f"Formulér svaret som præcis {plur(p['n'], 'sætning', 'sætninger')}.",
        lambda p: f"Længde: nøjagtig {plur(p['n'], 'sætning', 'sætninger')}.",
        lambda p: f"Kan du give det som præcis {plur(p['n'], 'sætning', 'sætninger')}?",
    ],
    check=lambda t, p: len(sentences(t)) == p["n"],
    tags=frozenset({"length:sentences"}),
    sample=lambda rng, ctx: {"n": rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15])},
)

at_most_n_sentences = Constraint(
    name="at_most_n_sentences",
    render_variants=[
        lambda p: f"Svaret må højst være på {plur(p['n'], 'sætning', 'sætninger')}.",
        lambda p: f"Skriv højst {plur(p['n'], 'sætning', 'sætninger')}.",
        lambda p: f"Maks. {plur(p['n'], 'sætning', 'sætninger')} — kort og præcist.",
        lambda p: f"Kan du holde det under {p['n']+1} sætninger?",
        lambda p: f"Ikke mere end {plur(p['n'], 'sætning', 'sætninger')}, tak.",
        lambda p: f"Hold det kort: op til {plur(p['n'], 'sætning', 'sætninger')}.",
    ],
    check=lambda t, p: len(sentences(t)) <= p["n"],
    tags=frozenset({"length:sentences"}),
    sample=lambda rng, ctx: {"n": rng.choice([2, 3, 5, 8, 10])},
)

at_least_n_sentences = Constraint(
    name="at_least_n_sentences",
    render_variants=[
        lambda p: f"Svaret skal være på mindst {plur(p['n'], 'sætning', 'sætninger')}.",
        lambda p: f"Skriv mindst {plur(p['n'], 'sætning', 'sætninger')}.",
        lambda p: f"Uddyb — minimum {plur(p['n'], 'sætning', 'sætninger')}.",
        lambda p: f"Giv et grundigt svar (mindst {plur(p['n'], 'sætning', 'sætninger')}).",
        lambda p: f"Ikke færre end {plur(p['n'], 'sætning', 'sætninger')}.",
    ],
    check=lambda t, p: len(sentences(t)) >= p["n"],
    tags=frozenset({"length:sentences"}),
    sample=lambda rng, ctx: {"n": rng.choice([2, 3, 4, 5, 6, 8])},
)

at_most_n_words = Constraint(
    name="at_most_n_words",
    render_variants=[
        lambda p: f"Svaret må højst være på {p['n']} ord.",
        lambda p: f"Hold svaret på under {p['n']+1} ord.",
        lambda p: f"Maks. {p['n']} ord.",
        lambda p: f"Kort format: under {p['n']+1} ord.",
        lambda p: f"Skriv højst {p['n']} ord — vær koncis.",
        lambda p: f"Ordgrænse: {p['n']}.",
        lambda p: f"Kan du klare det på {p['n']} ord eller derunder?",
    ],
    check=lambda t, p: len(words(t)) <= p["n"],
    tags=frozenset({"length:words"}),
    sample=lambda rng, ctx: {"n": rng.choice([20, 40, 60, 100])},
)

at_least_n_words = Constraint(
    name="at_least_n_words",
    render_variants=[
        lambda p: f"Svaret skal indeholde mindst {p['n']} ord.",
        lambda p: f"Minimum {p['n']} ord — skriv fyldigt.",
        lambda p: f"Ikke under {p['n']} ord.",
        lambda p: f"Vær udførlig: mindst {p['n']} ord.",
    ],
    check=lambda t, p: len(words(t)) >= p["n"],
    tags=frozenset({"length:words"}),
    sample=lambda rng, ctx: {"n": rng.choice([30, 50, 75, 100, 150])},
)


first_sentence_max_words = Constraint(
    name="first_sentence_max_words",
    render_variants=[
        lambda p: f"Første sætning må højst indeholde {p['n']} ord.",
        lambda p: f"Åbningssætningen skal være kort — maks. {p['n']} ord.",
        lambda p: f"Start med en kort sætning på højst {p['n']} ord.",
        lambda p: f"Første sætning: op til {p['n']} ord.",
        lambda p: f"Din første sætning må ikke være længere end {p['n']} ord.",
    ],
    check=lambda t, p: (
        (sents := sentences(t)) != [] and len(words(sents[0])) <= p["n"]
    ),
    tags=frozenset({"length:first_sentence"}),
    sample=lambda rng, ctx: {"n": rng.choice([8, 10, 12, 15])},
)


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


n_paragraphs = Constraint(
    name="n_paragraphs",
    render_variants=[
        lambda p: f"Svaret skal opdeles i præcis {plur(p['n'], 'afsnit', 'afsnit')}, adskilt af en tom linje mellem hvert.",
        lambda p: f"Skriv præcis {plur(p['n'], 'afsnit', 'afsnit')} med tom linje imellem.",
        lambda p: f"{plur(p['n'], 'afsnit', 'afsnit')}, adskilt med blank linje.",
        lambda p: f"Struktur: nøjagtig {plur(p['n'], 'afsnit', 'afsnit')}.",
        lambda p: f"Del svaret op i {plur(p['n'], 'afsnit', 'afsnit')} (blank linje mellem).",
    ],
    check=lambda t, p: len(_paragraphs(t)) == p["n"],
    tags=frozenset({"length:paragraphs", "structure:paragraphs"}),
    sample=lambda rng, ctx: {"n": rng.choice([1, 2, 3, 4, 5, 6, 7])},
)

nth_paragraph_first_word = Constraint(
    name="nth_paragraph_first_word",
    render_variants=[
        lambda p: f'Afsnit nummer {p["n"]} i svaret skal begynde med ordet "{p["w"]}".',
        lambda p: f'{p["n"]}. afsnit skal starte med "{p["w"]}".',
        lambda p: f'Lad det {p["n"]}. afsnit begynde med ordet "{p["w"]}".',
        lambda p: f'Start afsnit {p["n"]} med "{p["w"]}".',
    ],
    check=lambda t, p: (
        (paras := _paragraphs(t)) and len(paras) >= p["n"]
        and paras[p["n"] - 1].split()[0].strip('.,!?"\'').lower() == p["w"].lower()
    ),
    tags=frozenset({"length:paragraph_word", "structure:paragraphs"}),
    sample=lambda rng, ctx: {
        "n": rng.choice([2, 3, 4]),
        "w": rng.choice(_NTH_WORDS_POOL),
    },
)


# ────────────────────────────────────────────────────────────────────────────
# LEXICAL constraints
# ────────────────────────────────────────────────────────────────────────────

def _in_text(text: str, w: str) -> bool:
    return re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) is not None


def _count_in_text(text: str, w: str) -> int:
    return len(re.findall(rf"\b{re.escape(w)}\b", text, re.IGNORECASE))


include_keyword = Constraint(
    name="include_keyword",
    render_variants=[
        lambda p: f'Ordet "{p["w"]}" skal forekomme mindst én gang i svaret.',
        lambda p: f'Brug ordet "{p["w"]}" mindst én gang.',
        lambda p: f'Sørg for at nævne "{p["w"]}".',
        lambda p: f'Inkludér "{p["w"]}" et sted i teksten.',
        lambda p: f'"{p["w"]}" skal indgå i svaret.',
    ],
    check=lambda t, p: _in_text(t, p["w"]),
    tags=frozenset({"lexical:include"}),
    sample=lambda rng, ctx: {"w": rng.choice(_KEYWORD_POOL)},
)

include_all_keywords = Constraint(
    name="include_all_keywords",
    render_variants=[
        lambda p: "Alle disse ord skal forekomme i svaret: " + ", ".join(f'"{w}"' for w in p["ws"]) + ".",
        lambda p: "Brug samtlige af disse ord: " + ", ".join(f'"{w}"' for w in p["ws"]) + ".",
        lambda p: "Inkludér hvert af følgende ord: " + ", ".join(f'"{w}"' for w in p["ws"]) + ".",
        lambda p: "Alle ordene " + " og ".join(f'"{w}"' for w in p["ws"]) + " skal indgå.",
    ],
    check=lambda t, p: all(_in_text(t, w) for w in p["ws"]),
    tags=frozenset({"lexical:include"}),
    sample=lambda rng, ctx: {"ws": rng.sample(_KEYWORD_POOL, k=rng.choice([2, 3]))},
)

exclude_word = Constraint(
    name="exclude_word",
    render_variants=[
        lambda p: f'Ordet "{p["w"]}" må IKKE forekomme i svaret.',
        lambda p: f'Undgå ordet "{p["w"]}" helt.',
        lambda p: f'Brug ikke ordet "{p["w"]}".',
        lambda p: f'Skriv svaret uden at bruge "{p["w"]}".',
        lambda p: f'"{p["w"]}" må ikke stå nogen steder i svaret.',
    ],
    check=lambda t, p: not _in_text(t, p["w"]),
    tags=frozenset({"lexical:exclude"}),
    sample=lambda rng, ctx: {"w": rng.choice(_COMMON_CRUTCHES)},
)

starts_with_phrase = Constraint(
    name="starts_with_phrase",
    render_variants=[
        lambda p: f'Svaret skal begynde direkte med ordene "{p["s"]}" — ikke med markdown, overskrifter, bullets eller nummerering foran.',
        lambda p: f'Start svaret med "{p["s"]}".',
        lambda p: f'Begynd med sætningen "{p["s"]}" og skriv derefter dit svar.',
        lambda p: f'De første ord i svaret skal være "{p["s"]}".',
        lambda p: f'Åbn med "{p["s"]}" — intet markdown foran.',
    ],
    check=lambda t, p: _strip_leading_noise(t).lower().startswith(p["s"].lower()),
    tags=frozenset({"lexical:starts_with", "structure:opening"}),
    sample=lambda rng, ctx: {"s": rng.choice(_STARTS_POOL)},
)


_LEADING_NOISE_RE = re.compile(r"^\s*(?:[-*>]\s+|\d+[.)]\s+|#{1,6}\s+|\*\*|__)+")


def _strip_leading_noise(t: str) -> str:
    prev = None
    while t != prev:
        prev = t
        t = _LEADING_NOISE_RE.sub("", t.lstrip()).lstrip()
    return t


ends_with_phrase = Constraint(
    name="ends_with_phrase",
    render_variants=[
        lambda p: f'Svaret skal slutte med sætningen "{p["s"]}".',
        lambda p: f'Afslut med "{p["s"]}".',
        lambda p: f'Din sidste sætning skal være "{p["s"]}".',
        lambda p: f'Lad svaret ende på "{p["s"]}".',
        lambda p: f'Rund af med "{p["s"]}" til allersidst.',
    ],
    check=lambda t, p: t.strip().rstrip(".!?").lower().endswith(
        p["s"].lower().rstrip(".!?")),
    tags=frozenset({"lexical:ends_with", "structure:closing"}),
    sample=lambda rng, ctx: {"s": rng.choice(_ENDS_POOL)},
)

keyword_exactly_n_times = Constraint(
    name="keyword_exactly_n_times",
    render_variants=[
        lambda p: f'Ordet "{p["w"]}" skal forekomme præcis {plur(p["n"], "gang", "gange")} i svaret.',
        lambda p: f'Brug ordet "{p["w"]}" nøjagtig {plur(p["n"], "gang", "gange")}.',
        lambda p: f'"{p["w"]}" skal optræde {plur(p["n"], "gang", "gange")} — hverken flere eller færre.',
        lambda p: f'Antal forekomster af "{p["w"]}": præcis {p["n"]}.',
    ],
    check=lambda t, p: _count_in_text(t, p["w"]) == p["n"],
    tags=frozenset({"lexical:count"}),
    sample=lambda rng, ctx: {"w": rng.choice(_KEYWORD_POOL), "n": rng.choice([1, 2, 3, 4, 5, 6])},
)

uppercase_keyword = Constraint(
    name="uppercase_keyword",
    render_variants=[
        lambda p: f'Ordet "{p["w"]}" skal skrives med STORE bogstaver hver gang det forekommer.',
        lambda p: f'Skriv "{p["w"]}" med versaler hver gang.',
        lambda p: f'Vis "{p["w"]}" i STORE BOGSTAVER (fremhævet).',
        lambda p: f'Hver forekomst af "{p["w"]}" skal være STORT SKREVET.',
    ],
    check=lambda t, p: (
        all(m.group() == p["w"].upper()
            for m in re.finditer(rf"\b{re.escape(p['w'])}\b", t, re.IGNORECASE))
        and _in_text(t, p["w"])
    ),
    tags=frozenset({"lexical:casing"}),
    sample=lambda rng, ctx: {"w": rng.choice(_KEYWORD_POOL)},
)

letter_frequency = Constraint(
    name="letter_frequency",
    render_variants=[
        lambda p: f'Bogstavet "{p["letter"]}" skal forekomme mindst {p["n"]} gange (store og små ens).',
        lambda p: f'Brug bogstavet "{p["letter"]}" mindst {p["n"]} gange i svaret.',
        lambda p: f'Antal "{p["letter"]}"-bogstaver: minimum {p["n"]}.',
        lambda p: f'Sørg for at "{p["letter"]}" optræder ≥ {p["n"]} gange samlet.',
    ],
    check=lambda t, p: t.lower().count(p["letter"].lower()) >= p["n"],
    tags=frozenset({"lexical:letter_freq"}),
    sample=lambda rng, ctx: {
        "letter": rng.choice(["a", "e", "i", "o", "s", "t", "r", "n"]),
        "n": rng.choice([5, 8, 12, 20]),
    },
)


letter_exactly_n_times = Constraint(
    name="letter_exactly_n_times",
    render_variants=[
        lambda p: f'Bogstavet "{p["letter"]}" skal forekomme præcis {p["n"]} gange (store og små ens).',
        lambda p: f'Brug bogstavet "{p["letter"]}" nøjagtig {p["n"]} gange — hverken flere eller færre.',
        lambda p: f'Antal "{p["letter"]}"-bogstaver: præcis {p["n"]}.',
        lambda p: f'"{p["letter"]}" skal optræde nøjagtigt {p["n"]} gange samlet i svaret.',
    ],
    check=lambda t, p: t.lower().count(p["letter"].lower()) == p["n"],
    # Shares tag with letter_frequency so combos don't pick both at once.
    tags=frozenset({"lexical:letter_freq"}),
    sample=lambda rng, ctx: {
        # Full Danish alphabet (29 letters). Common vowels like 'e' are
        # genuinely hard to hit exactly in natural prose, but that's the
        # point — forces targeted composition.
        "letter": rng.choice(list("abcdefghijklmnopqrstuvwxyzæøå")),
        "n": rng.choice([1, 2, 3, 4, 5, 6, 7, 8]),
    },
)


# ────────────────────────────────────────────────────────────────────────────
# FORMAT constraints
# ────────────────────────────────────────────────────────────────────────────

_NUM_LIST_LINE_RE = re.compile(r"^\s*(\d+)[.)]\s+\S", re.MULTILINE)
_BULLET_LINE_RE = re.compile(r"^\s*[-*]\s+\S", re.MULTILINE)
_HEADER_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)")
_TITLE_RE = re.compile(r"<<\s*([^<>\n]+?)\s*>>")
_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")
# Match ANY list line start: bulleted (- * • –) or numbered (1. 1) (1))
_LIST_LINE_RE = re.compile(r"(?m)^\s*(?:[-*•–]|\(?\d+[.)])\s+\S")
_PLACEHOLDER_RE = re.compile(r"\[[^\[\]\n]+\]")

numbered_list_n_items = Constraint(
    name="numbered_list_n_items",
    render_variants=[
        lambda p: f"Svaret skal være en nummereret liste med præcis {plur(p['n'], 'punkt', 'punkter')}, hvor hvert punkt starter med '{{i}}.' (fx '1.', '2.').",
        lambda p: f"Formatér som nummereret liste — nøjagtig {plur(p['n'], 'punkt', 'punkter')} (1., 2., …).",
        lambda p: f"Giv præcis {plur(p['n'], 'punkt', 'punkter')} som en nummereret liste.",
        lambda p: f"Skriv {p['n']} punkter nummereret 1 til {p['n']}.",
    ],
    check=lambda t, p: len(_NUM_LIST_LINE_RE.findall(t)) == p["n"],
    tags=frozenset({"format:list", "structure:list"}),
    sample=lambda rng, ctx: {"n": rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 10])},
)

bullet_list_n_items = Constraint(
    name="bullet_list_n_items",
    render_variants=[
        lambda p: f"Svaret skal være en punktopstilling med præcis {plur(p['n'], 'punkt', 'punkter')}, hvor hvert punkt begynder med en bindestreg ('- ').",
        lambda p: f"Formatér som bullet-liste — {plur(p['n'], 'punkt', 'punkter')} (hver linje starter med '- ').",
        lambda p: f"Giv {p['n']} punkter som en dash-liste ('- xxx').",
        lambda p: f"Punktopstilling med præcis {p['n']} bullets.",
    ],
    check=lambda t, p: len(_BULLET_LINE_RE.findall(t)) == p["n"],
    tags=frozenset({"format:list", "structure:list"}),
    sample=lambda rng, ctx: {"n": rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 10])},
)


def _check_table(text: str, cols: list[str], rows: int) -> bool:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2 + rows:
        return False
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    if [norm(c) for c in header] != [norm(c) for c in cols]:
        return False
    if not re.match(r"^\|[\s:|-]+\|$", lines[1]):
        return False
    return len(lines) - 2 == rows


markdown_table = Constraint(
    name="markdown_table",
    render_variants=[
        lambda p: ("Svaret skal indeholde en markdown-tabel med kolonnerne: "
                  + ", ".join(f'"{c}"' for c in p["cols"])
                  + f", og præcis {plur(p['rows'], 'datarække', 'datarækker')}."),
        lambda p: ("Lav en markdown-tabel med kolonner "
                  + ", ".join(p["cols"])
                  + f" og {p['rows']} rækker data."),
        lambda p: (f"Tabelformat: kolonner ({', '.join(p['cols'])}), "
                  f"{p['rows']} datarækker."),
        lambda p: ("Formatér som markdown-tabel; kolonner: "
                  + ", ".join(p["cols"]) + f"; {p['rows']} rækker."),
    ],
    check=lambda t, p: _check_table(t, p["cols"], p["rows"]),
    tags=frozenset({"format:table"}),
    sample=lambda rng, ctx: {
        "cols": rng.sample(_TABLE_COLS_POOL, k=rng.choice([2, 3, 4])),
        "rows": rng.choice([2, 3, 4, 5, 6]),
    },
)

section_headers = Constraint(
    name="section_headers",
    render_variants=[
        lambda p: ("Svaret skal opdeles med markdown-overskrifter (## ...), nemlig præcis disse: "
                  + ", ".join(f'"{h}"' for h in p["headers"]) + ", i denne rækkefølge."),
        lambda p: ("Brug markdown-overskrifter (## ) med disse titler i rækkefølge: "
                  + ", ".join(p["headers"]) + "."),
        lambda p: ("Del svaret op i afsnit med ## overskrifter: "
                  + " → ".join(p["headers"]) + "."),
        lambda p: ("Strukturér med ## overskrifter i denne rækkefølge: "
                  + ", ".join(p["headers"]) + "."),
    ],
    check=lambda t, p: (
        [norm(h) for h in _HEADER_RE.findall(t)] == [norm(h) for h in p["headers"]]),
    tags=frozenset({"format:headers", "structure:sections"}),
    sample=lambda rng, ctx: {
        "headers": rng.sample(_HEADER_POOL, k=rng.choice([2, 3, 4])),
    },
)

n_italic_sections = Constraint(
    name="n_italic_sections",
    render_variants=[
        lambda p: f"Svaret skal indeholde præcis {plur(p['n'], 'kursiv sektion', 'kursive sektioner')} i markdown (*sådan her*).",
        lambda p: f"Brug præcis {p['n']} *kursiv-markeringer* (enkelt stjerne).",
        lambda p: f"{p['n']} steder skal være i kursiv (markdown *…*).",
        lambda p: f"Fremhæv nøjagtig {p['n']} passager med *kursiv*.",
    ],
    check=lambda t, p: len(_ITALIC_RE.findall(t)) == p["n"],
    tags=frozenset({"format:italic"}),
    sample=lambda rng, ctx: {"n": rng.choice([1, 2, 3, 4, 5, 6])},
)

title_wrapped = Constraint(
    name="title_wrapped",
    render_variants=[
        lambda p: "Svaret skal indeholde en titel omgivet af dobbelte vinkelparenteser, fx <<Min titel>>.",
        lambda p: "Giv svaret en titel i formatet <<Titel her>>.",
        lambda p: "Inkludér en titel skrevet som <<…>>.",
        lambda p: "Start eller inkludér en titel i vinkelparenteser (<<Titel>>).",
    ],
    check=lambda t, p: len(_TITLE_RE.findall(t)) >= 1,
    tags=frozenset({"format:title"}),
    sample=lambda rng, ctx: {},
)

n_bold_sections = Constraint(
    name="n_bold_sections",
    render_variants=[
        lambda p: f"Svaret skal indeholde præcis {plur(p['n'], 'fed sektion', 'fede sektioner')} i markdown (**sådan her**).",
        lambda p: f"Brug præcis {p['n']} **fede fremhævninger** (dobbelt-stjerne).",
        lambda p: f"{p['n']} steder skal være i fed skrift (markdown **…**).",
        lambda p: f"Fremhæv nøjagtig {p['n']} passager med **fed skrift**.",
    ],
    check=lambda t, p: len(_BOLD_RE.findall(t)) == p["n"],
    tags=frozenset({"format:bold"}),
    sample=lambda rng, ctx: {"n": rng.choice([1, 2, 3, 4, 5, 6])},
)


no_lists = Constraint(
    name="no_lists",
    render_variants=[
        lambda p: "Svaret skal være i sammenhængende prosa — ingen punkter, bullets eller nummererede lister.",
        lambda p: "Ingen lister (hverken bullet-punkter eller nummererede). Kun prosa.",
        lambda p: "Skriv som løbende tekst uden lister af nogen art.",
        lambda p: "Undgå lister — svaret skal være hele sætninger som sammenhængende tekst.",
        lambda p: "Ingen punktopstilling. Kun almindelige afsnit.",
    ],
    check=lambda t, p: not _LIST_LINE_RE.search(t),
    tags=frozenset({"format:no_lists"}),
    sample=lambda rng, ctx: {},
)


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


exactly_n_words = Constraint(
    name="exactly_n_words",
    render_variants=[
        lambda p: f"Svaret skal indeholde omkring {p['n']} ord (±5).",
        lambda p: f"Sigt efter {p['n']} ord i alt — små afvigelser (±5) er OK.",
        lambda p: f"Cirka {p['n']} ord (plus/minus 5).",
        lambda p: f"Ordantal: {p['n']} (plus/minus 5).",
    ],
    # LLMs are notoriously bad at exact word counts. ±5 keeps the signal
    # meaningful (targeting the range) without demanding impossible precision.
    check=lambda t, p: abs(_count_words(t) - p["n"]) <= 5,
    tags=frozenset({"length:words"}),
    sample=lambda rng, ctx: {"n": rng.choice([15, 20, 25, 30, 40, 50])},
)


def _prompt_prefix(ctx) -> str | None:
    if not ctx: return None
    t = ctx.get("task_text")
    if not t: return None
    # Use first ~80 chars (roughly first sentence) as the required prefix.
    # Trim to word boundary for cleaner matching.
    trimmed = t.strip()[:80]
    if len(trimmed) < len(t.strip()):
        # cut at last word boundary
        sp = trimmed.rfind(" ")
        if sp > 40: trimmed = trimmed[:sp]
    return trimmed


repeat_prompt_prefix = Constraint(
    name="repeat_prompt_prefix",
    render_variants=[
        lambda p: f'Begynd svaret med at gentage denne tekst ord for ord: "{p["prefix"]}", derefter dit svar.',
        lambda p: f'Først skal du gengive dette ordret: "{p["prefix"]}". Så kommer selve svaret.',
        lambda p: f'Indled med denne præcise tekst: "{p["prefix"]}", og derefter dit egentlige svar.',
        lambda p: f'Gentag først (uden ændring): "{p["prefix"]}". Så: dit svar.',
    ],
    check=lambda t, p: t.strip().startswith(p["prefix"]),
    # Only "format:echo" — NOT structure:opening. structure:opening on
    # itself would create a self-conflict (same bug as entire_in_quotes had).
    # The conflict pair (format:echo, structure:opening) below blocks co-
    # occurrence with starts_with_phrase etc.
    tags=frozenset({"format:echo"}),
    sample=lambda rng, ctx: {"prefix": _prompt_prefix(ctx)},
    applicable=lambda ctx: _prompt_prefix(ctx) is not None,
)


n_placeholders = Constraint(
    name="n_placeholders",
    render_variants=[
        lambda p: f"Svaret skal indeholde præcis {plur(p['n'], 'pladsholder', 'pladsholdere')} i firkantede parenteser, fx [dit navn], [adresse].",
        lambda p: f"Brug præcis {p['n']} pladsholdere i formen [xxx].",
        lambda p: f"{p['n']} steder skal være erstattet af pladsholdere [som denne].",
        lambda p: f"Marker {p['n']} felter som [pladsholder] i firkantede parenteser.",
    ],
    check=lambda t, p: len(_PLACEHOLDER_RE.findall(t)) == p["n"],
    tags=frozenset({"format:placeholders"}),
    sample=lambda rng, ctx: {"n": rng.choice([1, 2, 3, 4, 5, 6])},
)


# ────────────────────────────────────────────────────────────────────────────
# CASE (whole-response)
# ────────────────────────────────────────────────────────────────────────────

all_lowercase = Constraint(
    name="all_lowercase",
    render_variants=[
        lambda p: "Hele svaret skal skrives med kun små bogstaver — ingen store bogstaver nogen steder.",
        lambda p: "Brug udelukkende små bogstaver i svaret.",
        lambda p: "kun små bogstaver, tak.",
        lambda p: "Ingen versaler — hele svaret i lowercase.",
        lambda p: "Skriv alt med små bogstaver.",
    ],
    check=lambda t, p: all(not c.isupper() for c in t),
    tags=frozenset({"case:whole_response"}),
    sample=lambda rng, ctx: {},
)

all_uppercase = Constraint(
    name="all_uppercase",
    render_variants=[
        lambda p: "Hele svaret skal skrives MED STORE BOGSTAVER — ingen små bogstaver nogen steder.",
        lambda p: "SKRIV ALT MED VERSALER.",
        lambda p: "Kun store bogstaver i hele svaret.",
        lambda p: "Skriv svaret i STORE BOGSTAVER.",
        lambda p: "Ingen minuskler — hele svaret er UPPERCASE.",
    ],
    check=lambda t, p: all(not c.islower() for c in t),
    tags=frozenset({"case:whole_response"}),
    sample=lambda rng, ctx: {},
)

capital_word_frequency = Constraint(
    name="capital_word_frequency",
    render_variants=[
        lambda p: f"Svaret skal indeholde præcis {p['n']} ord skrevet HELT MED STORE BOGSTAVER (mindst 2 bogstaver).",
        lambda p: f"Præcis {p['n']} ord skal være i VERSALER (fx SÅDAN).",
        lambda p: f"Brug {p['n']} ord skrevet i CAPS et sted i svaret.",
        lambda p: f"{p['n']} ALL-CAPS-ord skal indgå (fx VIGTIGT, HUSK).",
    ],
    check=lambda t, p: (
        len([w for w in re.findall(r"\b[A-ZÆØÅ]{2,}\b", t) if not w.isdigit()]) == p["n"]
    ),
    tags=frozenset({"case:word_freq"}),
    sample=lambda rng, ctx: {"n": rng.choice([1, 2, 3, 4, 5, 6, 7, 8])},
)


# ────────────────────────────────────────────────────────────────────────────
# CONTENT-STRUCTURE
# ────────────────────────────────────────────────────────────────────────────

# Valid open/close quote pairs the check accepts. Model can use any of them
# — that mirrors how a Danish writer might wrap text and matches the quote-
# style variation applied in if_generate._maybe_swap_quote_style.
_QUOTE_PAIRS = [
    ('"',  '"'), ('“',  '”'), ('»',  '«'), ('«',  '»'),
    ("'",  "'"), ('‘',  '’'),
]


def _wrapped_in_quotes(text: str) -> bool:
    t = text.strip()
    if len(t) < 2:
        return False
    for opener, closer in _QUOTE_PAIRS:
        if t.startswith(opener) and t.endswith(closer):
            return True
    return False


entire_in_quotes = Constraint(
    name="entire_in_quotes",
    render_variants=[
        lambda p: 'Hele svaret skal være omgivet af dobbelte anførselstegn — svaret begynder og slutter med ".',
        lambda p: 'Pak hele svaret ind i dobbelte anførselstegn: "svaret her".',
        lambda p: 'Skriv svaret inde i "…" — første og sidste tegn er ".',
        lambda p: 'Omgiv hele svaret med citationstegn.',
        lambda p: 'Sæt hele svaret i anførselstegn (fx ", ", eller »«).',
    ],
    check=lambda t, p: _wrapped_in_quotes(t),
    tags=frozenset({"format:quote_wrap"}),
    sample=lambda rng, ctx: {},
)

no_commas = Constraint(
    name="no_commas",
    render_variants=[
        lambda p: "Svaret må IKKE indeholde nogen kommaer.",
        lambda p: "Brug ingen kommaer i svaret.",
        lambda p: "Undgå kommaer helt.",
        lambda p: "Ingen kommaer, tak.",
        lambda p: "Skriv svaret uden at bruge kommaer.",
        lambda p: "Komma-frit svar — ingen kommaer nogen steder.",
    ],
    check=lambda t, p: "," not in t,
    tags=frozenset({"punctuation:no_comma"}),
    sample=lambda rng, ctx: {},
)

two_responses_split = Constraint(
    name="two_responses_split",
    render_variants=[
        lambda p: 'Giv præcis to versioner af svaret, adskilt af markørlinjen "***" på egen linje.',
        lambda p: 'Skriv to alternative svar med "***" mellem dem på en tom linje.',
        lambda p: 'Lav to udkast, adskilt af *** (på sin egen linje).',
        lambda p: 'Præsentér to versioner — mellem dem: en linje med kun ***.',
    ],
    check=lambda t, p: len(re.findall(r"(?m)^\s*\*{3}\s*$", t)) == 1,
    tags=frozenset({"format:split_responses", "structure:sections"}),
    sample=lambda rng, ctx: {},
)


# ────────────────────────────────────────────────────────────────────────────
# MC-ONLY (solo)
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# PUNCTUATION / REGISTER / MARKDOWN patterns
# ────────────────────────────────────────────────────────────────────────────

_PUNCT_LABEL = {".": "punktum", "!": "udråbstegn", "?": "spørgsmålstegn"}

ends_with_punctuation = Constraint(
    name="ends_with_punctuation",
    render_variants=[
        lambda p: f"Svaret skal slutte med et {_PUNCT_LABEL[p['ch']]} ({p['ch']}).",
        lambda p: f"Sidste tegn i svaret: {p['ch']}.",
        lambda p: f"Afslut med {_PUNCT_LABEL[p['ch']]}.",
        lambda p: f"Svarets allersidste tegn skal være {p['ch']}.",
    ],
    check=lambda t, p: t.strip().endswith(p["ch"]),
    tags=frozenset({"structure:closing"}),
    sample=lambda rng, ctx: {"ch": rng.choice([".", "!", "?"])},
)


_FIRST_PERSON_RE = re.compile(
    r"\b(jeg|mig|min|mit|mine|vi|os|vores|vor|vort|vore)\b", re.I
)

no_first_person = Constraint(
    name="no_first_person",
    render_variants=[
        lambda p: "Skriv i tredje person — ingen brug af 'jeg', 'vi', 'mig', 'os', 'min', 'vores' eller lignende.",
        lambda p: "Undgå første person (ingen 'jeg' / 'vi' / 'mig' / 'os' / 'min' / 'vores').",
        lambda p: "Objektiv stil — første-persons pronominer må ikke forekomme.",
        lambda p: "Ingen 'jeg' eller 'vi'-formuleringer.",
    ],
    check=lambda t, p: _FIRST_PERSON_RE.search(t) is None,
    tags=frozenset({"register:no_first_person"}),
    sample=lambda rng, ctx: {},
)


_SECOND_PERSON_RE = re.compile(r"\b(du|dig|din|dit|dine)\b", re.I)

in_second_person = Constraint(
    name="in_second_person",
    render_variants=[
        lambda p: "Henvend dig til læseren som 'du' — mindst én forekomst af 'du' / 'dig' / 'din' i svaret.",
        lambda p: "Skriv i anden person: brug 'du' / 'dig' i din tiltale.",
        lambda p: "Adressér læseren direkte (du/dig/din).",
        lambda p: "Tal til læseren som 'du' mindst én gang.",
    ],
    check=lambda t, p: _SECOND_PERSON_RE.search(t) is not None,
    tags=frozenset({"register:second_person"}),
    sample=lambda rng, ctx: {},
)


single_paragraph = Constraint(
    name="single_paragraph",
    render_variants=[
        lambda p: "Svaret skal være ét sammenhængende afsnit — ingen tomme linjer.",
        lambda p: "Skriv i ét afsnit (ingen linjeskift til nyt afsnit).",
        lambda p: "Kun ét afsnit. Ingen blanke linjer.",
        lambda p: "Hold svaret som en enkelt blok tekst uden afsnitsopdeling.",
    ],
    check=lambda t, p: "\n\n" not in t.strip(),
    tags=frozenset({"length:paragraphs"}),
    sample=lambda rng, ctx: {},
)


_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b")

contains_year = Constraint(
    name="contains_year",
    render_variants=[
        lambda p: "Svaret skal indeholde mindst ét årstal (fx 1789, 1953, 2024).",
        lambda p: "Nævn et årstal (4-cifret) i svaret.",
        lambda p: "Inkludér et konkret år i svaret.",
        lambda p: "Der skal forekomme mindst ét år (fx 1848) i svaret.",
    ],
    check=lambda t, p: _YEAR_RE.search(t) is not None,
    tags=frozenset({"content:year"}),
    sample=lambda rng, ctx: {},
)


_PERCENT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*%")

contains_percentage = Constraint(
    name="contains_percentage",
    render_variants=[
        lambda p: "Svaret skal indeholde mindst én procentangivelse (fx 25%).",
        lambda p: "Nævn en procentsats i svaret (formatet N%).",
        lambda p: "Inkludér mindst ét tal med %-tegn.",
        lambda p: "Der skal være mindst én procent (fx 42%) i svaret.",
    ],
    check=lambda t, p: _PERCENT_RE.search(t) is not None,
    tags=frozenset({"content:percentage"}),
    sample=lambda rng, ctx: {},
)


_HYPERLINK_RE = re.compile(r"\[[^\]\n]+\]\([^)\n]+\)")

include_hyperlink = Constraint(
    name="include_hyperlink",
    render_variants=[
        lambda p: "Svaret skal indeholde mindst ét markdown-hyperlink i formatet [tekst](url).",
        lambda p: "Inkludér et hyperlink som markdown: [tekst](url).",
        lambda p: "Tilføj et markdown-link — fx [Wikipedia](https://da.wikipedia.org).",
        lambda p: "Brug markdown-linkformat [...](...) mindst én gang.",
    ],
    check=lambda t, p: _HYPERLINK_RE.search(t) is not None,
    tags=frozenset({"format:hyperlink"}),
    sample=lambda rng, ctx: {},
)


answer_only_letter = Constraint(
    name="answer_only_letter",
    render_variants=[
        lambda p: f"Svar med kun ét bogstav — {' / '.join(p['choices'])} — intet andet. Ingen forklaring, ingen prosa, kun bogstavet.",
        lambda p: f"Kun ét bogstav i svaret: {'/'.join(p['choices'])}. Intet andet.",
        lambda p: f"Angiv kun bogstavet ({', '.join(p['choices'])}). Ingen forklaring.",
        lambda p: f"Ét bogstav er nok: {' eller '.join(p['choices'])}.",
    ],
    check=lambda t, p: t.strip().rstrip(".").upper() == p["gold_letter"].upper(),
    tags=frozenset({"output:letter_only"}),
    sample=lambda rng, ctx: {
        "choices": (ctx or {}).get("mc_choices", ["A", "B", "C", "D"]),
        "gold_letter": (ctx or {}).get("gold_letter", "A"),
    },
    solo=True,
    applicable=lambda ctx: bool(ctx) and "mc_choices" in ctx and "gold_letter" in ctx,
)


# ────────────────────────────────────────────────────────────────────────────
# Registry + combo picker
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# Google IFEval-aligned constraints. These match Google's verifiers in
# scripts/ifeval_google/instructions.py exactly, unlike our older
# {n_bold_sections, n_italic_sections, repeat_prompt_prefix, two_responses_split}
# which have subtly-different semantics. Kept in parallel so v3 back-compat
# holds; use these in v4+ to teach behaviour that transfers to ifeval-da.
# ────────────────────────────────────────────────────────────────────────────

# Combined italic (*x*) + bold (**x**) matcher; matches google's regex.
_HIGHLIGHT_STAR_RE = re.compile(r"\*[^\n\*]*\*")
_HIGHLIGHT_DOUBLE_RE = re.compile(r"\*\*[^\n\*]*\*\*")


def _count_highlighted(text: str) -> int:
    n = 0
    for m in _HIGHLIGHT_STAR_RE.findall(text):
        if m.strip("*").strip():
            n += 1
    for m in _HIGHLIGHT_DOUBLE_RE.findall(text):
        if m.removeprefix("**").removesuffix("**").strip():
            n += 1
    return n


ifeval_highlighted_min_n = Constraint(
    name="ifeval_highlighted_min_n",
    render_variants=[
        lambda p: f"Fremhæv mindst {plur(p['n'], 'sektion', 'sektioner')} i dit svar med markdown, dvs. *fremhævet sektion*.",
        lambda p: f"Brug markdown til at fremhæve mindst {p['n']} steder i svaret (fx *sådan* eller **sådan**).",
        lambda p: f"Dit svar skal indeholde mindst {p['n']} markdown-fremhævede afsnit (*…* eller **…**).",
        lambda p: f"Marker mindst {plur(p['n'], 'passage', 'passager')} i svaret med markdown-fremhævning (*…*).",
    ],
    check=lambda t, p: _count_highlighted(t) >= p["n"],
    tags=frozenset({"format:highlight"}),
    sample=lambda rng, ctx: {"n": rng.choice([1, 2, 3, 4])},
)


ifeval_repeat_prompt = Constraint(
    name="ifeval_repeat_prompt",
    render_variants=[
        lambda p: f'Gentag først anmodningen ord for ord uden ændring, og giv derefter dit svar. Anmodningen der skal gentages: "{p["prompt_to_repeat"]}"',
        lambda p: f'Skriv først dette ordret (uden ændringer), og derefter dit egentlige svar: "{p["prompt_to_repeat"]}"',
        lambda p: f'Din besvarelse skal begynde med at gengive dette præcis ord for ord, og derefter komme med svaret: "{p["prompt_to_repeat"]}"',
        lambda p: f'Først: gentag denne tekst uden ét ord ændret. Derefter: dit svar. Tekst: "{p["prompt_to_repeat"]}"',
    ],
    check=lambda t, p: (bool(p["prompt_to_repeat"])
                        and t.strip().lower().startswith(
                            p["prompt_to_repeat"].strip().lower())),
    # Uses the seed task as the "prompt to repeat" — pulled from ctx by sample.
    # Applicable only when ctx has a non-empty task_text.
    sample=lambda rng, ctx: {"prompt_to_repeat": (ctx or {}).get("task_text", "")},
    applicable=lambda ctx: bool((ctx or {}).get("task_text", "").strip()),
    tags=frozenset({"format:echo"}),
)


def _check_two_responses_6star(text: str, _p) -> bool:
    parts = text.split("******")
    valid = []
    for i, r in enumerate(parts):
        if not r.strip():
            if i not in (0, len(parts) - 1):
                return False
        else:
            valid.append(r)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


ifeval_two_responses_6star = Constraint(
    name="ifeval_two_responses_6star",
    render_variants=[
        lambda p: 'Giv to forskellige svar. Kun svarene skal være adskilt med præcis seks stjerner: ******.',
        lambda p: 'Lever to alternative besvarelser. Adskil dem med "******" (seks stjerner). Selve svarene skal være forskellige.',
        lambda p: 'Skriv to versioner af svaret — adskilt af separatoren ****** (seks stjerner). Ingen ekstra tekst omkring separatoren.',
        lambda p: 'Præsentér to indbyrdes forskellige svar, adskilt af nøjagtig seks asterisker: ******.',
    ],
    check=_check_two_responses_6star,
    tags=frozenset({"format:split_responses", "structure:sections"}),
    sample=lambda rng, ctx: {},
)


def _check_json_format(text: str, _p) -> bool:
    t = (text.strip()
             .removeprefix("```json")
             .removeprefix("```Json")
             .removeprefix("```JSON")
             .removeprefix("```")
             .removesuffix("```")
             .strip())
    try:
        json.loads(t)
    except (ValueError, json.JSONDecodeError):
        return False
    return True


ifeval_json_format = Constraint(
    name="ifeval_json_format",
    render_variants=[
        lambda p: "Hele svaret skal være i JSON-format. Du må gerne indpakke det i markdown-kodeblokke (```json … ```).",
        lambda p: "Formatér hele svaret som gyldigt JSON (evt. inde i ``` blokke).",
        lambda p: "Svaret skal være ét stykke gyldigt JSON — ingen prosa udenfor. Markdown-kodefencer er tilladt.",
        lambda p: "Returner svaret som JSON og kun JSON.",
    ],
    check=_check_json_format,
    tags=frozenset({"format:json"}),
    sample=lambda rng, ctx: {},
)


_CONSTRAINED_OPTIONS_DA = ("Mit svar er ja.", "Mit svar er nej.", "Mit svar er måske.")

ifeval_constrained_response = Constraint(
    name="ifeval_constrained_response",
    render_variants=[
        lambda p: f'Svar med én af følgende muligheder: {", ".join(repr(o) for o in _CONSTRAINED_OPTIONS_DA)}',
        lambda p: f'Dit svar skal indeholde nøjagtig én af disse tre sætninger: {", ".join(_CONSTRAINED_OPTIONS_DA)}',
        lambda p: f'Vælg et af følgende svar: {" / ".join(_CONSTRAINED_OPTIONS_DA)}',
        lambda p: f'Besvar spørgsmålet med præcis én af: {_CONSTRAINED_OPTIONS_DA[0]!r} / {_CONSTRAINED_OPTIONS_DA[1]!r} / {_CONSTRAINED_OPTIONS_DA[2]!r}',
    ],
    check=lambda t, p: any(opt in t.strip() for opt in _CONSTRAINED_OPTIONS_DA),
    tags=frozenset({"format:constrained_choice"}),
    sample=lambda rng, ctx: {},
    # "Answer with ja/nej/måske" only makes semantic sense on question-shaped
    # seeds — MC rows or short factual questions. Long wiki-summary seeds
    # give the model a hopeless task (override the natural answer with a
    # 3-option override), leading to lots of wasted retries.
    applicable=lambda ctx: (
        bool((ctx or {}).get("mc_choices"))
        or len(((ctx or {}).get("task_text") or "")) < 240
    ),
)


_POSTSCRIPT_MARKERS_DA = ("P.S.", "P.P.S.")


def _check_postscript(text: str, params: dict) -> bool:
    marker = params["marker"]
    tl = text.lower()
    if marker == "P.P.S.":
        pat = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pat = r"\s*p\.\s?s\..*$"
    else:
        pat = r"\s*" + re.escape(marker.lower()) + r".*$"
    return bool(re.findall(pat, tl, flags=re.MULTILINE))


ifeval_postscript = Constraint(
    name="ifeval_postscript",
    render_variants=[
        lambda p: f"Tilføj eksplicit et postscript i slutningen af svaret der starter med {p['marker']}",
        lambda p: f"Afslut svaret med et postskriptum der begynder med {p['marker']}",
        lambda p: f"Efter selve svaret, tilføj et P.S. — det skal begynde med {p['marker']}",
        lambda p: f"I slutningen af dit svar: tilføj en linje der starter med {p['marker']} og indeholder en efterskrift.",
    ],
    check=_check_postscript,
    tags=frozenset({"format:postscript", "structure:closing"}),
    sample=lambda rng, ctx: {"marker": rng.choice(_POSTSCRIPT_MARKERS_DA)},
)


ALL: list[Constraint] = [
    # Length
    exactly_n_sentences, at_most_n_sentences, at_least_n_sentences,
    at_most_n_words, at_least_n_words, exactly_n_words, first_sentence_max_words,
    n_paragraphs, nth_paragraph_first_word,
    # Lexical
    include_keyword, include_all_keywords, exclude_word,
    starts_with_phrase, ends_with_phrase,
    keyword_exactly_n_times, uppercase_keyword, letter_frequency, letter_exactly_n_times,
    # Format
    numbered_list_n_items, bullet_list_n_items, markdown_table, section_headers,
    n_italic_sections, n_bold_sections, title_wrapped, n_placeholders, no_lists,
    include_hyperlink, single_paragraph,
    # Content
    contains_year, contains_percentage,
    # Opening / echo / closing
    repeat_prompt_prefix, ends_with_punctuation,
    # Register
    no_first_person, in_second_person,
    # Case (whole response)
    all_lowercase, all_uppercase, capital_word_frequency,
    # Content structure
    entire_in_quotes, no_commas, two_responses_split,
    # MC-only (solo)
    answer_only_letter,
    # Google IFEval-aligned (v4+; parallel with older ~mismatched variants)
    ifeval_highlighted_min_n, ifeval_repeat_prompt, ifeval_two_responses_6star,
    # Google IFEval families we didn't have before
    ifeval_json_format, ifeval_constrained_response, ifeval_postscript,
]

SOLO_PROBABILITY = 0.5

DEFAULT_SIZE_WEIGHTS = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}


_CONFLICTING_TAG_PAIRS = [
    # length axis
    ("length:sentences",  "length:words"),
    ("length:sentences",  "length:paragraphs"),
    ("length:words",      "length:paragraphs"),
    # format shape conflicts
    ("format:list",       "length:sentences"),
    ("format:list",       "length:first_sentence"),
    ("format:list",       "length:paragraphs"),
    ("format:table",      "length:sentences"),
    ("format:table",      "length:first_sentence"),
    ("format:table",      "length:paragraphs"),
    ("format:table",      "length:words"),
    ("format:headers",    "length:first_sentence"),
    ("format:headers",    "length:paragraphs"),
    ("format:list",       "format:table"),
    ("format:list",       "format:headers"),
    ("format:table",      "format:headers"),
    ("format:list",       "format:split_responses"),
    ("format:table",      "format:split_responses"),
    ("format:headers",    "format:split_responses"),
    # lexical
    ("lexical:include",   "lexical:exclude"),
    ("structure:opening", "length:first_sentence"),
    # case
    ("case:whole_response", "case:word_freq"),
    ("case:whole_response", "lexical:casing"),
    ("case:whole_response", "format:italic"),
    ("case:whole_response", "format:title"),
    # quote wrap
    ("format:quote_wrap", "structure:opening"),
    ("format:quote_wrap", "structure:closing"),
    ("format:quote_wrap", "format:split_responses"),
    # placeholders
    ("format:placeholders", "case:whole_response"),
    # two-responses split with paragraph counting
    ("format:split_responses", "length:paragraphs"),
    ("format:split_responses", "structure:paragraphs"),
    # nth_paragraph_first_word (length:paragraph_word) needs the same
    # incompatibilities as n_paragraphs — otherwise it gets over-sampled
    # by being compatible with everything.
    ("length:paragraph_word", "length:sentences"),
    ("length:paragraph_word", "length:words"),
    ("length:paragraph_word", "format:list"),
    ("length:paragraph_word", "format:table"),
    ("length:paragraph_word", "format:headers"),
    ("length:paragraph_word", "format:split_responses"),
    # no_lists is the direct negation of a list-format requirement — mutually
    # exclusive with numbered_list / bullet_list / markdown_table.
    ("format:no_lists", "format:list"),
    ("format:no_lists", "format:table"),
    # bold interacts with case:whole_response (all-caps / all-lower) — the
    # bold markers survive but the content-in-caps distinguishes poorly.
    ("format:bold", "case:whole_response"),
    # repeat_prompt_prefix uses opening slot; can't co-exist with other
    # opening-slot constraints (starts_with_phrase, entire_in_quotes)
    ("format:echo", "structure:opening"),
    ("format:echo", "format:quote_wrap"),
    # (`structure:closing` vs `format:quote_wrap` already listed above.)
    # ends_with_punctuation and ends_with_phrase share `structure:closing`
    # so they auto-exclude via tag-overlap in _compatible().
    #
    # register:no_first_person is compatible with second-person but
    # conflicts with repeat_prompt_prefix (which echoes user text that
    # may itself contain first-person pronouns).
    ("register:no_first_person", "format:echo"),
]


def _compatible(existing_tags: set[str], new_tags: frozenset[str]) -> bool:
    if existing_tags & new_tags:
        return False
    combined = existing_tags | set(new_tags)
    for a, b in _CONFLICTING_TAG_PAIRS:
        if a in combined and b in combined:
            return False
    return True


def sample_combo(rng: random.Random, min_size: int = 1, max_size: int = 5,
                 ctx: dict | None = None,
                 size_weights: dict[int, int] | None = None) -> list[dict]:
    """Return list of {name, params, render, _check}. Each `render` is one
    randomly-chosen phrasing from that constraint's render_variants list."""
    applicable = [c for c in ALL if c.applicable(ctx)]
    solos = [c for c in applicable if c.solo]

    if solos and rng.random() < SOLO_PROBABILITY:
        c = rng.choice(solos)
        params = c.sample(rng, ctx)
        return [{"name": c.name, "params": params,
                 "render": rng.choice(c.render_variants)(params),
                 "_check": c.check}]

    weights = dict(size_weights or DEFAULT_SIZE_WEIGHTS)
    filtered = {k: v for k, v in weights.items() if min_size <= k <= max_size}
    if not filtered:
        filtered = {min_size: 1}
    sizes = list(filtered.keys())
    ws = list(filtered.values())
    target = rng.choices(sizes, weights=ws, k=1)[0]

    pool = [c for c in applicable if not c.solo]
    chosen: list[Constraint] = []
    chosen_tags: set[str] = set()
    for c in rng.sample(pool, k=len(pool)):
        if len(chosen) >= target:
            break
        if _compatible(chosen_tags, c.tags):
            chosen.append(c)
            chosen_tags |= c.tags
    out = []
    for c in chosen:
        params = c.sample(rng, ctx)
        out.append({
            "name": c.name, "params": params,
            "render": rng.choice(c.render_variants)(params),
            "_check": c.check,
        })
    return out


def render_rules(combo: list[dict]) -> str:
    return "\n".join(f"- {r['render']}" for r in combo)


def verify_all(text: str, combo: list[dict]) -> tuple[bool, list[str]]:
    failures = []
    for r in combo:
        try:
            if not r["_check"](text, r["params"]):
                failures.append(r["name"])
        except Exception as e:
            failures.append(f"{r['name']}:err({type(e).__name__})")
    return (len(failures) == 0), failures


# ────────────────────────────────────────────────────────────────────────────
# Demo
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = random.Random(42)
    print("=== 8 sampled combos (with random variant per constraint) ===\n")
    for i in range(8):
        combo = sample_combo(rng, 1, 5)
        print(f"combo {i + 1}: {[r['name'] for r in combo]}")
        for r in combo:
            print(f"    - {r['render']}")
        print()

    print("=== variant coverage — 20 samples of same constraint ===")
    rng2 = random.Random(0)
    for _ in range(20):
        params = no_commas.sample(rng2, None)
        print(f"  {rng2.choice(no_commas.render_variants)(params)}")

    print("\n=== self-check ===")
    combo = [
        {"name": "at_most_n_sentences", "params": {"n": 3},
         "render": at_most_n_sentences.render_variants[0]({"n": 3}),
         "_check": at_most_n_sentences.check},
        {"name": "include_keyword", "params": {"w": "Danmark"},
         "render": include_keyword.render_variants[0]({"w": "Danmark"}),
         "_check": include_keyword.check},
    ]
    good = "Danmark er et fladt land. Højeste punkt er 171 meter."
    bad = "Sverige har mange bjerge. Vandet er koldt. Der er meget skov. Og lange vintre."
    print(f"  good: {verify_all(good, combo)}")
    print(f"  bad:  {verify_all(bad, combo)}")
