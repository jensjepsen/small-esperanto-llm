"""Esperanto syntax + morphology verifier.

Composable pipeline of checks. Each check is a class implementing
    check(tokens: list[Token], nps: list[NP]) -> list[Diagnostic]

Usage:
    from esperanto_lm.verify import Verifier, NPAgreement, MissingAccusative, ...
    v = Verifier([NPAgreement(), MissingAccusative(), AffixCompat()])
    diags = v.verify("la ruĝa kato vidis la hundon.")
    for d in diags:
        print(d)

The tokenizer uses Lark for sentence and NP/PP chunking, backed by regex
over Esperanto's regular morphological endings. Token POS + features are
derived from endings; root is stripped back to the bare root using
the official root dictionary in esperanto_lm.morphology when available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from lark import Lark, Transformer, v_args

from esperanto_lm.morphology import (
    get_roots, get_prefixes, get_suffixes,
    get_other_words, get_correlatives,
    decompose as _morph_decompose, classify_morpheme,
)


# ---- Part-of-speech endings -------------------------------------------

# Verb tense/mood endings — these are the FINAL endings, everything before is stem.
VERB_ENDINGS = {
    "i":  ("INF",  {"mood": "inf"}),
    "as": ("VERB", {"tense": "pres"}),
    "is": ("VERB", {"tense": "past"}),
    "os": ("VERB", {"tense": "fut"}),
    "us": ("VERB", {"tense": "cond"}),
    "u":  ("VERB", {"mood": "imp"}),
}

# Participle suffixes — must attach to verb roots; followed by -a/-o/-e etc.
PARTICIPLE_SUFFIXES = {"ant", "int", "ont", "at", "it", "ot"}

# Directional prepositions — can take accusative when motion toward is meant.
DIRECTIONAL_PREPS = {
    "en", "sur", "sub", "super", "antaŭ", "post", "apud",
    "inter", "ekster", "ĉe", "tra", "trans", "malantaŭ",
    "ĉirkaŭ", "ĝis", "preter", "ene", "kontraŭ",
}
# Non-directional prepositions — should never take accusative after them.
NONDIR_PREPS = {
    "de", "al", "por", "pri", "pro", "kun", "sen", "per", "el",
    "da", "krom", "laŭ", "anstataŭ",
    "dum",  # "during" — temporal, never directional
    "je",  # non-directional, denotes time/rate/instance
    "malgraŭ", "po",
    # "kontraŭ" with contact/impact verbs takes acc ("frapu kontraŭ pordon") —
    # treat as directional. Same for positional preps listed in DIRECTIONAL.
}
ALL_PREPS = DIRECTIONAL_PREPS | NONDIR_PREPS

ARTICLE = "la"
CONJUNCTIONS = {"kaj", "sed", "aŭ", "nek", "ĉar", "se", "ke", "ol", "dum", "kvankam"}

# Possessive determiners (all forms — sg/pl, nom/acc)
POSSESSIVES = {
    "mia", "miaj", "mian", "miajn",
    "via", "viaj", "vian", "viajn",
    "lia", "liaj", "lian", "liajn",
    "ŝia", "ŝiaj", "ŝian", "ŝiajn",
    "ĝia", "ĝiaj", "ĝian", "ĝiajn",
    "nia", "niaj", "nian", "niajn",
    "ilia", "iliaj", "ilian", "iliajn",
    "sia", "siaj", "sian", "siajn",
    "onia", "oniaj", "onian", "oniajn",
}

# Esperanto's correlatives and common closed-class words that aren't in root dict
CLOSED_CLASS: set[str] = {
    # Pronouns + their accusative forms
    "mi", "vi", "li", "ŝi", "ĝi", "ni", "ili", "oni", "si",
    "min", "vin", "lin", "ŝin", "ĝin", "nin", "ilin", "onin", "sin",
}
# Augment from vortaro at import time — covers correlatives (kio/tiu/iu/ĉiu/neniu...)
# and "other" closed-class words (prepositions, particles, numerals, etc.).
# Words that are also in ALL_PREPS or CONJUNCTIONS get the more specific POS
# in analyze_word; everything else stays generic 'Closed'.
CLOSED_CLASS.update(get_correlatives())
CLOSED_CLASS.update(get_other_words())
# All correlative inflected forms (kio/kion, tia/tian/tiaj/tiajn, etc.)
_CORR_BASES = get_correlatives()
for base in _CORR_BASES:
    if base.endswith("o"):
        CLOSED_CLASS.update({base + "n"})
    elif base.endswith("a"):
        CLOSED_CLASS.update({base + "n", base + "j", base + "jn"})
    elif base.endswith("u"):
        CLOSED_CLASS.update({base + "n", base + "j", base + "jn"})
    elif base.endswith("e"):
        CLOSED_CLASS.update({base + "n"})  # kien (where to)

# Known intransitive and transitive verb roots. Incomplete — serves as hints.
# -igi is always transitive, -iĝi always intransitive (handled in analysis).
KNOWN_TRANSITIVE = {
    "vidi", "aŭdi", "havi", "manĝi", "trinki", "legi", "skribi", "fari",
    "diri", "peti", "demandi", "respondi", "voki", "ami", "malami", "ŝati",
    "voli", "povi", "devi", "scii", "kompreni", "pensi", "kredi", "memori",
    "trovi", "perdi", "aĉeti", "vendi", "doni", "preni", "porti", "ĵeti",
    "lavi", "fermi", "malfermi", "rompi", "konstrui", "vidi", "rigardi",
    "serĉi", "kaŝi", "kapti", "pagi", "kaŭzi", "veki", "kuŝigi", "okupi",
    "lerni", "instrui", "ricevi",
    "komenci", "fini", "ĉesi",  # transitive: 'X komencas Y', 'X finas Y'
    "domini", "uzi", "krei", "doni", "preni", "konstrui", "viziti",
    "renkonti", "amindumi", "vesti", "konduki", "sekvi", "atingi",
    "eniri", "eliri", "trairi", "transiri",  # motion-through verbs take direct obj
    "travivi", "klarigi", "ekspliki", "listigi", "klasifiki", "publikigi",
    "aliri",  # "to access/approach" — transitive in modern EO
    "supervivi", "postvivi", "transvivi",  # survive X
    "naĝi",  # "naĝi flanknaĝon" — swim a specific stroke
    "konsistigi",  # "konsistigas grupon" — "make up"
    "rigardi",  # "rigardas ŝablonojn" — already in set? check
    "suriri",  # "suriri boaton" — to board/mount
    "flugi",  # "flugis la komandan modulon" — literary transitive
}

# Time / measure roots whose accusative form is adverbial (not a real object).
# Stored as bare roots (no -o ending) since that's what tokenizer produces.
ADVERBIAL_TIME_NOUNS = {
    "hor", "minut", "sekund", "tag", "nokt", "monat", "jar",
    "semajn", "foj", "lund", "mard", "merkred", "ĵaŭd", "vendred",
    "sabat", "dimanĉ", "sezon", "somer", "vintr", "printemp", "aŭtun",
    "jardek", "jarcent", "jarmil", "maten", "vesper", "posttagmez",
    "generaci", "ludperiod", "epoĥ", "erao", "era",
    "temp",  # "vivis longan tempon" — generic time
    "paŝ",  # "iras 5 paŝojn antaŭen" — adverbial distance/count
    "mejl",  # "marŝis mejlojn" — adverbial distance
    "metr", "kilometr", "fut", "centimetr", "kilogram", "litr",
    "etaĝ",  # "tri etaĝojn alta" — measure of height
    "distanc",  # "moviĝas ajnan distancon" — measure of distance
    "spir",  # "daŭris nur unu spiron" — metaphorical duration
    "fojon",  # "alian fojon" — occurrence count (stem already in list)
}
KNOWN_INTRANSITIVE = {
    "iri", "veni", "kuri", "salti", "marŝi", "flugi", "naĝi", "stari",
    "sidi", "kuŝi", "dormi", "vivi", "morti", "esti", "fali", "leviĝi",
    "ridi", "plori", "krii", "bruli", "pluvi", "neĝi", "tondri",
    "ekzisti", "okazi", "daŭri", "finiĝi",
    # NB: 'komenci', 'fini', 'ĉesi' are TRANSITIVE — moved to KNOWN_TRANSITIVE.
}

# Modal/auxiliary verbs that take infinitives — they don't license direct
# accusative objects; their "object" is the infinitive's object.
MODAL_VERBS = {"povi", "devi", "voli", "scii", "kapabli", "rajti", "intenci",
               "decidi", "promesi", "provi", "ŝati"}
COPULA = {"esti"}


# ---- Affix POS signatures ---------------------------------------------

# Each suffix: (allowed_input_POSes, output_POS)
# POS labels: N=noun, A=adj, V=verb, Adv=adverb, Any=any
SUFFIX_SIG: dict[str, tuple[set[str], str]] = {
    "ist":  ({"N", "V", "A"}, "N"),
    "ul":   ({"A", "N"},       "N"),
    "ej":   ({"N", "V", "A"}, "N"),
    "in":   ({"N"},            "N"),
    "eg":   ({"N", "A", "V", "Adv"}, None),   # preserves POS
    "et":   ({"N", "A", "V", "Adv"}, None),
    "ar":   ({"N"},            "N"),
    "er":   ({"N"},            "N"),
    "id":   ({"N"},            "N"),
    "aĵ":   ({"N", "A", "V"}, "N"),
    "ec":   ({"A", "N"},       "N"),
    "ig":   ({"A", "V", "N"}, "V"),
    "iĝ":   ({"A", "V", "N"}, "V"),
    "ind":  ({"V"},            "A"),   # must be verb
    "ebl":  ({"V"},            "A"),
    "em":   ({"V", "N"},       "A"),
    "end":  ({"V"},            "A"),
    "ad":   ({"V", "N", "A", "Adv"}, None),  # -ad attaches broadly; strict
    # check misfires on loanwords whose decomposition over-splits (e.g.
    # "dominado" parsed as "dom+in+ad" even though `domin` is a root).
    "ant":  ({"V"},            "A"),   # participles
    "int":  ({"V"},            "A"),
    "ont":  ({"V"},            "A"),
    "at":   ({"V"},            "A"),
    "it":   ({"V"},            "A"),
    "ot":   ({"V"},            "A"),
    "estr": ({"N", "V"},       "N"),
    "uj":   ({"N"},            "N"),
}

# Prefix signatures: (allowed_input_POSes,)
PREFIX_SIG: dict[str, set[str]] = {
    "mal":  {"A", "Adv", "V", "N"},
    "re":   {"V"},
    "pra":  {"N"},
    "mis":  {"V"},
    "ek":   {"V"},
    "dis":  {"V"},
    "ĉef":  {"N"},
    "bo":   {"N"},   # kinship
    "ge":   {"N"},   # bisexual collective
    "vic":  {"N"},
    "eks":  {"N"},
    "fi":   {"N", "A"},
}


# ---- Data classes -----------------------------------------------------

@dataclass
class Token:
    text: str
    idx: int                     # word index in the sentence
    char_span: tuple[int, int]   # start/end char in original text
    pos: str                     # N, A, V, Adv, Prep, Conj, Pron, Num, Det, Punct, Closed, Unknown
    number: str | None = None    # "sg" / "pl"
    case: str | None = None      # "nom" / "acc"
    tense: str | None = None     # for verbs
    mood: str | None = None
    root: str | None = None      # bare root (stem) if decomposable
    prefixes: list[str] = field(default_factory=list)
    suffixes: list[str] = field(default_factory=list)
    ending: str | None = None    # -o, -a, -e, -i, -as, -is, -os, -us, -u, or None
    raw_lower: str = ""

    @property
    def is_noun(self): return self.pos == "N"
    @property
    def is_adj(self):  return self.pos == "A"
    @property
    def is_verb(self): return self.pos == "V"
    @property
    def is_inf(self):  return self.pos == "INF"
    @property
    def is_adv(self):  return self.pos == "Adv"
    @property
    def is_prep(self): return self.pos == "Prep"


@dataclass
class NP:
    """A noun phrase: contiguous span of DET/ADJ/NUM + head noun (optionally followed by post-modifier)."""
    tokens: list[Token]
    head: Token
    start_idx: int
    end_idx: int

    @property
    def number(self): return self.head.number
    @property
    def case(self):   return self.head.case


@dataclass
class Clause:
    """A clause = chunk of tokens between clause boundaries (conjunctions, subordinators, punctuation)."""
    start_idx: int   # token index (inclusive)
    end_idx: int     # token index (exclusive)
    tokens: list[Token]
    nps: list[NP]                  # NPs falling inside this clause
    verb: Token | None = None      # finite verb of this clause
    subject: NP | None = None      # nominative NP not inside a PP, near the verb
    is_subordinate: bool = False   # introduced by ke/kiu/kiam/...
    intro_word: str | None = None  # the subordinator/conjunction that introduced it


@dataclass
class Diagnostic:
    check: str
    level: str           # "error" / "warning" / "note"
    message: str
    token_idx: int       # primary offending token
    span: tuple[int, int] | None = None  # optional NP or phrase span

    def __str__(self) -> str:
        return f"[{self.level}:{self.check}] token#{self.token_idx}: {self.message}"


# ---- Morphological analysis -------------------------------------------

# Word: letter sequence, optionally hyphenated. Includes Esperanto chars
# (ĉĝĥĵŝŭ) plus common Latin-script diacritics from foreign names so they
# don't fragment Stryjeńska / Gaïa / Széchenyi mid-word.
_WORD_RE = re.compile(
    r"[a-zĉĝĥĵŝŭàáâãäåæèéêëìíîïñòóôõöøùúûüýÿœßšžčćďěľňřťůąęłńśźż]+"
    r"(?:-[a-zĉĝĥĵŝŭàáâãäåæèéêëìíîïñòóôõöøùúûüýÿœßšžčćďěľňřťůąęłńśźż]+)*",
    re.IGNORECASE,
)


def _strip_ending(word: str) -> tuple[str, str | None, str | None, str | None]:
    """Return (stem, ending, number, case) using Esperanto's final-vowel morphology.

    Requires stem length ≥ 2 after stripping — prevents degenerate analyses
    like "von" → ("v", "on", "sg", "acc").
    """
    w = word.lower()
    MIN_STEM = 2

    def _ok(stem):
        return len(stem) >= MIN_STEM
    for end in ("as", "is", "os", "us"):
        if w.endswith(end) and _ok(w[:-len(end)]):
            return w[:-len(end)], end, None, None
    for end in ("u",):
        if w.endswith(end) and _ok(w[:-len(end)]) and not w.endswith(("oj", "aj")):
            return w[:-len(end)], end, None, None
    if w.endswith("ojn") and _ok(w[:-3]):
        return w[:-3], "ojn", "pl", "acc"
    if w.endswith("ajn") and _ok(w[:-3]):
        return w[:-3], "ajn", "pl", "acc"
    if w.endswith("oj") and _ok(w[:-2]):
        return w[:-2], "oj", "pl", "nom"
    if w.endswith("aj") and _ok(w[:-2]):
        return w[:-2], "aj", "pl", "nom"
    if w.endswith("on") and _ok(w[:-2]):
        return w[:-2], "on", "sg", "acc"
    if w.endswith("an") and _ok(w[:-2]):
        return w[:-2], "an", "sg", "acc"
    if w.endswith("en") and _ok(w[:-2]):
        return w[:-2], "en", None, "acc"
    if w.endswith("o") and _ok(w[:-1]):
        return w[:-1], "o", "sg", "nom"
    if w.endswith("a") and _ok(w[:-1]):
        return w[:-1], "a", "sg", "nom"
    if w.endswith("e") and _ok(w[:-1]):
        return w[:-1], "e", None, None
    if w.endswith("i") and _ok(w[:-1]):
        return w[:-1], "i", None, None
    return w, None, None, None


def _decompose_stem(stem: str) -> tuple[list[str], str, list[str]]:
    """Split a stem into (prefixes, root, suffixes) using the known root/affix dictionary.

    Greedy: strip known prefixes from start, known suffixes from end, keep remainder as root.
    Works on lowercase stems (no grammatical ending).
    """
    prefixes: list[str] = []
    suffixes: list[str] = []
    roots = get_roots()
    pres = get_prefixes()
    sufs = get_suffixes()

    # Only strip if we can reduce stem to an EXACT known root.
    # Try all (prefix, suffix-chain) splits and pick the one whose remaining
    # middle is in the roots dict.
    best = None  # (prefixes_used, root, suffixes_used)
    # Enumerate prefix counts 0..2 and suffix counts 0..3
    for np in range(0, 3):
        for ns in range(0, 4):
            s = stem
            pre_taken: list[str] = []
            suf_taken: list[str] = []
            # Try to strip np prefixes from front
            ok = True
            for _ in range(np):
                matched_p = None
                for p in sorted(pres, key=len, reverse=True):
                    if s.startswith(p) and len(s) > len(p) + 1:
                        matched_p = p
                        break
                if not matched_p:
                    ok = False
                    break
                pre_taken.append(matched_p)
                s = s[len(matched_p):]
            if not ok:
                continue
            # Try to strip ns suffixes from back
            for _ in range(ns):
                matched_s = None
                for sx in sorted(sufs, key=len, reverse=True):
                    if s.endswith(sx) and len(s) > len(sx) + 1:
                        matched_s = sx
                        break
                if not matched_s:
                    ok = False
                    break
                suf_taken.insert(0, matched_s)
                s = s[:-len(matched_s)]
            if not ok:
                continue
            # Middle must be an EXACT root of length ≥ 2
            if len(s) >= 2 and s in roots:
                # Score: (fewer affixes, then prefer prefixes, then longer root)
                score = (np + ns, -np, -len(s))
                if best is None or score < best[3]:
                    best = (pre_taken, s, suf_taken, score)

    if best is not None:
        pre, rt, suf = best[0], best[1], best[2]
        # Guard against over-decomposition: Latin loans like "prioritat",
        # "hidratad", "ordinit" get shredded (pri+or+it+at, hidr+at+ad,
        # ord+in+it) because short unrelated roots sit inside them.
        # Accept a participle decomposition when it's plausibly a real verb:
        #   (a) single participle suffix with a reasonably-long root, OR
        #   (b) the implied verb form is in our KNOWN lists.
        # Otherwise treat the stem as a single opaque loan-root.
        if any(s in PARTICIPLE_SUFFIXES for s in suf):
            idx_part = next(i for i, s in enumerate(suf) if s in PARTICIPLE_SUFFIXES)
            pre_part = suf[:idx_part]
            post_part = suf[idx_part + 1:]
            verb_candidates = {rt + "i", rt + "".join(pre_part) + "i",
                               "".join(pre) + rt + "i",
                               "".join(pre) + rt + "".join(pre_part) + "i"}
            known_verb = any(c in KNOWN_TRANSITIVE or c in KNOWN_INTRANSITIVE
                             for c in verb_candidates)
            # Shape heuristic: a real participle has nothing chained after it
            # and nothing but its own root before it; root must be reasonably long.
            clean_shape = (not pre_part and not post_part and len(rt) >= 3)
            if not (known_verb or clean_shape):
                return [], stem, []
        return pre, rt, suf
    # No decomposition found — return stem as root, no affixes
    return [], stem, []


def _pos_from_ending(ending: str | None, lower: str, suffixes: list[str]) -> str:
    """Determine POS from grammatical ending.

    Participles (-ant/-int/-.../-at/.../-ot) + adjective ending → still adjective POS
    (but check will verify verb stem).
    """
    if ending is None:
        if lower in CLOSED_CLASS:
            return "Closed"
        if lower in ALL_PREPS:
            return "Prep"
        if lower in CONJUNCTIONS:
            return "Conj"
        return "Unknown"
    if ending in ("as", "is", "os", "us", "u"):
        return "V"
    if ending == "i":
        return "INF"
    if ending in ("o", "on", "oj", "ojn"):
        return "N"
    if ending in ("a", "an", "aj", "ajn"):
        return "A"
    if ending in ("e", "en"):
        return "Adv"
    return "Unknown"


def analyze_word(word: str, idx: int, char_span: tuple[int, int],
                 sentence_start: bool = False) -> Token:
    """Full morphological analysis of a single word.

    sentence_start: if True, a capitalized word here might be a normal sentence
        starter (not necessarily a proper noun). Otherwise, a capitalized word
        that doesn't decompose cleanly is probably a proper noun — tag POS as
        'Proper' so NP/case checks treat it as a noun that agrees with anything.
    """
    lower = word.lower()
    # Possessive determiners (mia/via/lia/ŝia/...) at sentence start are
    # capitalized but aren't proper nouns — make sure they still hit the
    # closed-class branch.
    if lower in CLOSED_CLASS or lower in ALL_PREPS or lower in CONJUNCTIONS \
            or lower == ARTICLE or lower in POSSESSIVES:
        pos = "Prep" if lower in ALL_PREPS else (
              "Conj" if lower in CONJUNCTIONS else (
              "Det" if lower == ARTICLE else "Closed"))
        # Annotate subject pronouns with number so agreement checks work.
        # 'vi' is ambiguous (sg/pl) — leave as None to skip number check.
        pronoun_number = {
            "mi": "sg", "li": "sg", "ŝi": "sg", "ĝi": "sg", "oni": "sg", "si": "sg",
            "ni": "pl", "ili": "pl",
        }
        return Token(text=word, idx=idx, char_span=char_span, pos=pos,
                     number=pronoun_number.get(lower),
                     case="nom" if lower in pronoun_number else None,
                     raw_lower=lower)

    # Heuristic: capitalized word → proper noun (Maria, Klara, Petro, Niagara).
    # Names dominate the use-case even when the root is a valid dictionary entry.
    # Case is normalized to nom (apparent acc on names like "Štefan" is just a
    # coincidence of the suffix), but NUMBER is preserved from the ending —
    # capitalized common words at sentence start ("Kverkoj", "Rezultoj")
    # really are plural and downstream agreement needs that.
    # Skip when the ending is verbal (-as/-is/-os/-us/-u/-i) or adverbial (-e/-en) —
    # those are normal common-word forms AT SENTENCE START ("Pripensu...",
    # "Estas...", "Hejmen iru.").
    # Mid-sentence capitals (not at sentence start) remain proper nouns even
    # with these endings — "Rascali", "Bonzi", "Neri" etc. are proper names.
    if word[0].isupper():
        stem_test, end, num, _case = _strip_ending(lower)
        verbal_endings = ("as", "is", "os", "us", "u", "i", "e", "en")
        # At sentence start a capitalized word with a verbal/adverbial ending
        # is usually a normal common word ("Pripensu...", "Estas...", "Hejmen
        # iru..."). But proper names sometimes end the same way ("Frisbee",
        # "Tesla" — note `e`/`a`). Fall through to regular analysis only when
        # the stem decomposes to a known Esperanto root; otherwise keep it
        # classified as a proper noun.
        fall_through = False
        if sentence_start and end in verbal_endings:
            _, root_candidate, _ = _decompose_stem(stem_test)
            if root_candidate and root_candidate in get_roots():
                fall_through = True
        if fall_through:
            pass  # fall through to regular analysis
        else:
            tok = Token(text=word, idx=idx, char_span=char_span, pos="N",
                        number=num or "sg", case="nom",
                        root=stem_test, ending=end, raw_lower=lower)
            tok.is_proper = True  # type: ignore
            return tok

    stem, ending, number, case = _strip_ending(lower)
    prefixes, root, suffixes = _decompose_stem(stem)
    pos = _pos_from_ending(ending, lower, suffixes)

    tok = Token(
        text=word, idx=idx, char_span=char_span, pos=pos,
        number=number, case=case,
        root=root if root else stem,
        prefixes=prefixes, suffixes=suffixes, ending=ending,
        raw_lower=lower,
    )
    if ending in ("as", "is", "os", "us", "u"):
        tok.tense = {"as": "pres", "is": "past", "os": "fut", "us": "cond"}.get(ending)
        if ending == "u":
            tok.mood = "imp"
    if ending == "i":
        tok.mood = "inf"
    return tok


# ---- Lark grammar (chunking only — all morphology done above) ---------

_GRAMMAR = r"""
start: (sentence | PUNCT)+

sentence: element+ END_PUNCT?

element: np | pp | verb | conj | unknown

np: det? (adj | part_adj)* noun (adj | part_adj)*
pp: prep np

det: "la"i
noun: NOUN_TOK
adj: ADJ_TOK
part_adj: PART_TOK
verb: VERB_TOK
prep: PREP_TOK
conj: CONJ_TOK
unknown: WORD

NOUN_TOK: /[a-zĉĝĥĵŝŭ]+o(?:j)?n?/
ADJ_TOK: /[a-zĉĝĥĵŝŭ]+a(?:j)?n?/
PART_TOK: /[a-zĉĝĥĵŝŭ]*(?:ant|int|ont|at|it|ot)a(?:j)?n?/
VERB_TOK: /[a-zĉĝĥĵŝŭ]+(?:as|is|os|us|u|i)/
PREP_TOK.3: "de"i | "al"i | "por"i | "pri"i | "pro"i | "kun"i | "sen"i | "per"i | "el"i | "da"i | "krom"i | "laŭ"i | "anstataŭ"i | "en"i | "sur"i | "sub"i | "super"i | "antaŭ"i | "post"i | "apud"i | "inter"i | "ekster"i | "ĉe"i | "tra"i | "trans"i
CONJ_TOK.3: "kaj"i | "sed"i | "aŭ"i | "nek"i | "ĉar"i | "se"i | "ke"i | "ol"i | "dum"i | "kvankam"i
WORD: /[a-zĉĝĥĵŝŭ]+/
END_PUNCT: "."|"!"|"?"
PUNCT: /[,;:—\-\(\)"']/

%ignore /\s+/
"""


# ---- Tokenize + extract NPs -------------------------------------------

_SENT_BOUNDARY = re.compile(r"[.!?]\s*")


def tokenize(text: str) -> list[Token]:
    """Morphologically analyze every word. Uses simple regex split, not Lark."""
    # Stash original text on each token so checks can examine inter-token
    # punctuation (colons, newlines, parens, quotes) without re-deriving spans.
    sent_starts = {0}
    for m in _SENT_BOUNDARY.finditer(text):
        sent_starts.add(m.end())

    # Track which token indices have a comma/dash IMMEDIATELY BEFORE them —
    # used for list-item detection AND clause-boundary detection.
    # Includes em-dash (—) and en-dash (–) which are clause separators in EO.
    comma_positions: list[int] = [i for i, ch in enumerate(text) if ch in ",;—–"]

    tokens: list[Token] = []
    for i, m in enumerate(_WORD_RE.finditer(text)):
        in_sent_window = m.start() in sent_starts or any(
            s <= m.start() < s + 3 for s in sent_starts
        )
        # A period only starts a new sentence when followed by a capitalized
        # word — otherwise it's an abbreviation ("Nr. 5 kaj", "Mr. Smith").
        is_start = (i == 0) or (in_sent_window and m.group(0)[:1].isupper())
        tok = analyze_word(m.group(0), i, (m.start(), m.end()), sentence_start=is_start)
        tok.is_sentence_start = is_start  # type: ignore
        tokens.append(tok)

    # Precompute cumulative paren depth at every character. Also track paired
    # quote state (single/double/curly quotes) so quoted phrases are treated
    # like parentheticals for subject finding.
    paren_depth = [0] * (len(text) + 1)
    d = 0
    in_dq = False  # inside "..." or curly double-quote pair
    in_sq = False  # inside '...' or curly single-quote pair
    for ci, ch in enumerate(text):
        if ch in "([{":
            d += 1
        elif ch in ")]}":
            d = max(0, d - 1)
        # Paired double-quote-like characters
        if ch in '"“”':
            in_dq = not in_dq
        # Paired single-quote-like characters — guard against apostrophes
        # by requiring whitespace/bracket on one side.
        if ch in "'‘’":
            prev_ch = text[ci - 1] if ci > 0 else " "
            next_ch = text[ci + 1] if ci + 1 < len(text) else " "
            if in_sq:
                # Closing: allow if surrounded by letter-then-space/punct
                if not next_ch.isalpha() or next_ch in ".,;:!?":
                    in_sq = False
            else:
                # Opening: require space/bracket before
                if not prev_ch.isalpha():
                    in_sq = True
        effective = d + (1 if in_dq else 0) + (1 if in_sq else 0)
        paren_depth[ci + 1] = effective
    # Mark tokens with positional attributes
    for i, tok in enumerate(tokens):
        start_c, end_c = tok.char_span
        tok.has_comma_before = any(start_c > c and (i == 0 or tokens[i-1].char_span[1] <= c) for c in comma_positions)
        tok.has_comma_after = any(end_c <= c and (i == len(tokens)-1 or tokens[i+1].char_span[0] > c) for c in comma_positions)
        # Stronger break: colon, newline, parens, quotes
        prev_text = text[tokens[i-1].char_span[1]:start_c] if i > 0 else text[:start_c]
        # Hard break: colon, newline, slash, dash-as-list-separator, an
        # enumeration marker like "4.", or sentence-terminal punctuation
        # ?/! (treating them as breaks even before a lowercase word like
        # "...?" demandis la infano).
        tok.has_hard_break_before = (
            any(ch in prev_text for ch in ":\n\r/?!")
            or " - " in prev_text or " – " in prev_text or " — " in prev_text
            or bool(re.search(r"\b\d+\.\s", prev_text))
        )
        tok.has_paren_before = any(ch in prev_text for ch in "()[]\"'“”‘’«»„‚")
        tok.is_in_parens = paren_depth[start_c] > 0
    return tokens


def extract_nps(tokens: list[Token]) -> list[NP]:
    """Contiguous runs of (la | adj | det | correlative-determiner) + noun."""
    # Correlative determiners: kiu/tiu/ĉiu/iu/neniu and number/case forms.
    # Plus possessives (mia, via, lia, ŝia, ĝia, nia, ilia, sia, onia + forms).
    # Plus quantifiers/numbers like "iuj"/"tiuj"/"ĉiuj"/"neniuj".
    correlative_dets = {b for b in (get_correlatives() or set()) if b.endswith(("u", "a"))}
    correlative_dets.update({b + "n" for b in correlative_dets if b.endswith(("u", "a"))})
    correlative_dets.update({b + "j" for b in correlative_dets if b.endswith(("u", "a"))})
    correlative_dets.update({b + "jn" for b in correlative_dets if b.endswith(("u", "a"))})
    nps: list[NP] = []
    for i, tok in enumerate(tokens):
        if tok.pos != "N":
            continue
        start = i
        j = i - 1
        saw_det = False
        while j >= 0:
            prev = tokens[j]
            # Stop at clause boundaries — comma, dash, sentence start, or any
            # non-modifier token. Without this, adjectives from prior clauses
            # latch onto unrelated heads ("granda — centoj" → granda + centoj).
            if getattr(tokens[j + 1], "has_comma_before", False):
                break
            if j + 1 < len(tokens) and getattr(tokens[j + 1], "is_sentence_start", False):
                break
            if getattr(tokens[j + 1], "has_hard_break_before", False):
                break
            # Don't cross a paren/quote boundary. "(sieĝita) 26 fojojn" —
            # sieĝita is a parenthetical gloss, not a modifier of fojojn.
            if getattr(tokens[j + 1], "has_paren_before", False):
                break
            # Also stop if the proposed modifier is *inside* parens but the
            # head we're building isn't (or vice versa).
            if getattr(prev, "is_in_parens", False) != getattr(tok, "is_in_parens", False):
                break
            is_correlative_det = prev.pos == "Closed" and prev.raw_lower in correlative_dets
            is_det = prev.pos == "Det"
            # Once we've absorbed a Det (`la`), don't walk back further —
            # another adjective behind `la` belongs to a different NP.
            # "freŝaj | la tutan tempon" → two NPs, not one.
            if saw_det and not is_det:
                break
            # Stop at intensifiers (pli/plej/tre/tro/...) — they mark the start
            # of a predicate-adj phrase, not a modifier chain.
            INTENSIFIERS = {"pli", "plej", "tre", "tro", "iom",
                             "tiom", "kiom", "sufiĉe", "tiel"}
            if prev.raw_lower in INTENSIFIERS:
                break
            if (prev.pos in ("A", "Det")
                or is_correlative_det
                or (prev.pos == "Closed" and prev.raw_lower.endswith(("a", "aj", "an", "ajn")))):
                # An adj with mismatched case is not a modifier of this head.
                # Several shapes:
                # - "pli rimarkebla mallongan tempon" — predicate adj +
                #   adverbial acc (preceding intensifier signals predicate).
                # - "substancon nomatan laktato" — `nomatan` is a participle
                #   modifying an earlier noun; `laktato` is an appositive name
                #   in nom. Participle signals the naming/apposition pattern.
                if prev.pos == "A" and prev.case and tok.case \
                        and prev.case != tok.case:
                    if j - 1 >= 0 and tokens[j - 1].raw_lower in INTENSIFIERS:
                        break
                    # Participle adj with case mismatch → appositive pattern.
                    prev_is_participle = (
                        (prev.suffixes and prev.suffixes[-1] in PARTICIPLE_SUFFIXES)
                        or (prev.root and any(prev.root.endswith(p)
                                                for p in PARTICIPLE_SUFFIXES))
                    )
                    if prev_is_participle:
                        break
                # Correlative-det must agree in case with head. "al neniu
                # klarigon" — `neniu` (nom) isn't a modifier of `klarigon`
                # (acc); they belong to different phrases.
                if is_correlative_det and tok.case == "acc" \
                        and not prev.raw_lower.endswith(("n", "jn")):
                    break
                start = j
                if is_det:
                    saw_det = True
                j -= 1
            else:
                break
        nps.append(NP(tokens=tokens[start:i+1], head=tok, start_idx=start, end_idx=i))
    return nps


# Subordinators that introduce clauses
SUBORDINATORS = {
    "ke", "kiel", "kiu", "kiuj", "kiun", "kiujn", "kies", "kio", "kion",
    "kiam", "kie", "kien", "kial", "kiom", "kvankam", "kvazaŭ",
    "se", "ĉar", "dum", "ĝis", "antaŭ", "post", "ĉu",
}
# Coordinating conjunctions
COORDINATORS = {"kaj", "sed", "aŭ", "nek"}

# Subject pronouns (nominative). 'min/vin/...' (acc) excluded — never a subject.
SUBJECT_PRONOUNS = {"mi", "vi", "li", "ŝi", "ĝi", "ni", "ili", "oni", "si"}

# Correlative substantives that can act as singular subjects:
# "ĉiu [el ili] estas ...", "iu venis", "neniu scias", "tiu estas granda".
# Excludes -io forms (kio/tio/io/ĉio/nenio) which already imply sg by shape,
# and plural -uj forms (kiuj/tiuj/iuj/ĉiuj/neniuj) which are pl.
CORRELATIVE_SUBJECTS_SG = {"ĉiu", "iu", "tiu", "kiu", "neniu",
                            "ĉio", "io", "tio", "kio", "nenio"}
CORRELATIVE_SUBJECTS_PL = {"ĉiuj", "iuj", "tiuj", "kiuj", "neniuj"}


def _compute_pp_inside(tokens: list[Token], nps: list[NP]) -> set[int]:
    """Return set of NP-head token indices that fall inside any prepositional phrase.

    A PP starts at a Prep and extends through all coordinated NPs (joined by
    kaj/aŭ/nek) until the next clause boundary (comma, dash, subordinator,
    finite verb, or sentence end).
    """
    nps_by_start = {np.start_idx: np for np in nps}
    inside: set[int] = set()
    n = len(tokens)
    i = 0
    while i < n:
        if tokens[i].pos != "Prep":
            i += 1
            continue
        # Scan forward absorbing NPs
        j = i + 1
        while j < n:
            t = tokens[j]
            # Stop at clause/sentence boundary
            if t.has_comma_before and j > i + 1:
                break
            if getattr(t, "is_sentence_start", False):
                break
            if getattr(t, "has_hard_break_before", False):
                break
            if t.pos in ("V", "INF"):
                break
            if t.raw_lower in SUBORDINATORS and j > i + 1:
                break
            # Try NP-at-j first — if an NP starts at this position (possibly
            # with a correlative determiner like "tiu bela loko"), absorb it.
            np = nps_by_start.get(j)
            if np is not None:
                inside.add(np.head.idx)
                j = np.end_idx + 1
                # Absorb consecutive proper-noun NPs — multi-word proper
                # names ("New Jersey", "Pierre Shale") are extracted as
                # separate NPs but all belong to the same PP.
                while j < n and not tokens[j].has_comma_before \
                        and tokens[j].pos not in ("V", "INF", "Prep") \
                        and tokens[j].raw_lower not in SUBORDINATORS:
                    next_np = nps_by_start.get(j)
                    if next_np is None:
                        break
                    if not getattr(next_np.head, "is_proper", False):
                        break
                    inside.add(next_np.head.idx)
                    j = next_np.end_idx + 1
                # Absorb comma-separated coordinated PP objects
                # ("de aero, fuelo kaj sparko").
                while j < n and tokens[j].has_comma_before:
                    if tokens[j].pos in ("V", "INF", "Prep"):
                        break
                    if tokens[j].raw_lower in SUBORDINATORS:
                        break
                    comma_np = nps_by_start.get(j)
                    if comma_np is None:
                        break
                    # Don't absorb a parallel complex NP (N followed by its
                    # own Prep).
                    after = comma_np.end_idx + 1
                    if after < n and tokens[after].pos == "Prep":
                        break
                    inside.add(comma_np.head.idx)
                    j = comma_np.end_idx + 1
                if j < n and tokens[j].raw_lower in {"kaj", "aŭ", "nek"}:
                    # Only absorb across kaj if the coordinated item is a
                    # simple NP (no following Prep — otherwise it's a
                    # parallel complex NP, not a coordinated PP object).
                    next_after_kaj = j + 1
                    next_np2 = nps_by_start.get(next_after_kaj)
                    if next_np2 is not None:
                        after_np2 = next_np2.end_idx + 1
                        if after_np2 < n and tokens[after_np2].pos == "Prep":
                            break
                    j += 1
                    continue
                break
            # Bare correlative pronoun (no NP built because nothing nominal
            # follows): "sur kiu serio estas" — `kiu` is the PP object, but
            # `serio` starts a new relative clause.
            if t.raw_lower in (SUBJECT_PRONOUNS | CORRELATIVE_SUBJECTS_SG
                                | CORRELATIVE_SUBJECTS_PL):
                break
            j += 1
        i = j if j > i else i + 1
    return inside


def _find_subject(tokens: list[Token], nps: list[NP], clause_start: int,
                  clause_end: int, verb_idx: int | None,
                  pp_inside: set[int]) -> Token | None:
    """Find subject within [clause_start, clause_end). Considers nominative NPs
    and subject pronouns. Prefers the last candidate before the verb (or first
    in the clause if no verb). Skips NPs that sit inside a `kiel`-comparison
    phrase ("urboj kiel Milan" — Milan is comparison, not subject).
    """
    # Mark token indices that belong to a subordinate/comparison span so
    # they aren't picked as subjects of the outer verb:
    #   - `kiel X` comparison: "urboj kiel Milan estas" — Milan not subject
    #   - `kiu/kiuj/kies/kiun/...` relative clause: "ludantoj kiuj vivis"
    #     — kiuj heads a relative clause modifying ludantoj
    kiel_span: set[int] = set()
    RELATIVE_WORDS = {"kiu", "kiuj", "kiun", "kiujn", "kies",
                       "kiam", "kie", "kien"}
    k_idx = clause_start
    while k_idx < clause_end:
        tok_raw = tokens[k_idx].raw_lower
        if tok_raw == "kiel":
            m = k_idx + 1
            saw_n = False
            while m < clause_end:
                if tokens[m].has_comma_before or tokens[m].pos in ("V", "INF"):
                    break
                # Allow "kaj" + more comparanda: "kiel X kaj Y"
                if saw_n and tokens[m].raw_lower == "kaj":
                    kiel_span.add(m)
                    saw_n = False
                    m += 1
                    continue
                kiel_span.add(m)
                if tokens[m].pos == "N":
                    saw_n = True
                m += 1
            k_idx = m
            continue
        if tok_raw in RELATIVE_WORDS and k_idx > clause_start:
            # Relative clause extends from this pronoun to the next comma or
            # sentence-level break; its contents are not subjects of the outer.
            m = k_idx
            kiel_span.add(m)
            m += 1
            while m < clause_end:
                if tokens[m].has_comma_before:
                    break
                kiel_span.add(m)
                m += 1
            k_idx = m
            continue
        k_idx += 1
    candidates: list[Token] = []
    for k in range(clause_start, clause_end):
        if k in kiel_span:
            continue
        t = tokens[k]
        # Parentheticals are apposition/aside, not subjects.
        if getattr(t, "is_in_parens", False):
            continue
        # Skip pronouns/correlatives that sit inside a PP. Walk back through
        # the closed-class demonstrative/modifier `ĉi` too. The Prep may lie
        # in a previous clause (relative: "sur kiu ..."), so this walks across
        # the clause boundary.
        m = k - 1
        while m >= 0 and tokens[m].raw_lower == "ĉi":
            m -= 1
        if m >= 0 and tokens[m].pos == "Prep":
            if t.raw_lower in SUBJECT_PRONOUNS \
                    or t.raw_lower in CORRELATIVE_SUBJECTS_SG \
                    or t.raw_lower in CORRELATIVE_SUBJECTS_PL:
                continue
        if t.raw_lower in SUBJECT_PRONOUNS \
                or t.raw_lower in CORRELATIVE_SUBJECTS_SG \
                or t.raw_lower in CORRELATIVE_SUBJECTS_PL:
            # Annotate the correlative with its implied number so downstream
            # agreement checks see it.
            if t.raw_lower in CORRELATIVE_SUBJECTS_PL and t.number is None:
                t.number = "pl"
            elif t.raw_lower in CORRELATIVE_SUBJECTS_SG and t.number is None:
                t.number = "sg"
            candidates.append(t)
            continue
        np = next((p for p in nps if p.start_idx == k), None)
        if np is None:
            continue
        if np.case != "nom":
            continue
        if np.head.idx in pp_inside:
            continue
        candidates.append(np.head)
    if not candidates:
        return None
    if verb_idx is not None:
        before = [c for c in candidates if c.idx < verb_idx]
        if before:
            return before[-1]
    return candidates[0]


def extract_clauses(tokens: list[Token], nps: list[NP]) -> list[Clause]:
    """Split tokens into clauses based on conjunctions, subordinators, and punctuation.

    A clause boundary is:
    - a coordinator (kaj/sed/aŭ/nek)
    - a subordinator (ke/kiel/kiu/...)
    - a comma, semicolon, colon, em-dash, or quote (preceding the next clause)

    Each clause records:
    - its tokens
    - the NPs falling inside its span
    - its main verb (first finite V token in the clause)
    - its subject (first nom NP not inside a PP, in the clause)
    """
    if not tokens:
        return []

    boundaries = [0]  # starting positions
    intro_words: dict[int, str] = {}  # boundary_idx → intro word
    is_subord: dict[int, bool] = {}

    sentence_starts: set[int] = set()
    last_boundary_start = 0
    # If the first token is itself a subordinator, mark it — clauses starting
    # at idx 0 still need their intro_word ("Kiel klarigas…", "Kvankam…").
    if tokens and tokens[0].raw_lower in SUBORDINATORS:
        is_subord[0] = True
        intro_words[0] = tokens[0].raw_lower
    for i, tok in enumerate(tokens):
        if i == 0:
            continue
        is_coord = tok.raw_lower in COORDINATORS
        is_sub = tok.raw_lower in SUBORDINATORS
        # Sentence boundary: hard reset
        if getattr(tok, "is_sentence_start", False):
            boundaries.append(i)
            sentence_starts.add(i)
            last_boundary_start = i
        # Hard break (colon, newline, slash, numbered list) — clause boundary
        elif getattr(tok, "has_hard_break_before", False):
            boundaries.append(i)
            last_boundary_start = i
        # Comma boundary: new clause starts at THIS token
        elif tok.has_comma_before:
            boundaries.append(i)
            last_boundary_start = i
        elif is_coord or is_sub:
            # Only insert a coordinator boundary if the left segment has a
            # finite verb — otherwise `kaj`/`sed`/`aŭ` is coordinating NPs,
            # not clauses ("Flava kaj Ruĝa estis ligitaj" must NOT split).
            if is_coord:
                left_has_verb = any(
                    tokens[k].pos == "V" for k in range(last_boundary_start, i)
                )
                if not left_has_verb:
                    continue
            boundaries.append(i + 1 if is_coord else i)
            last_boundary_start = i + 1 if is_coord else i
            if is_sub:
                is_subord[i] = True
                intro_words[i] = tok.raw_lower

    boundaries = sorted(set(boundaries))
    boundaries.append(len(tokens))

    pp_inside_global = _compute_pp_inside(tokens, nps)

    clauses: list[Clause] = []
    prev_subject_token: Token | None = None
    prev_subject_was_compound: bool = False
    for bi in range(len(boundaries) - 1):
        s, e = boundaries[bi], boundaries[bi + 1]
        if s >= e:
            continue
        clause_tokens = tokens[s:e]
        clause_nps = [np for np in nps if s <= np.start_idx < e]

        # Find verb (first finite V or imperative)
        verb = None
        for t in clause_tokens:
            if t.pos == "V":
                verb = t
                break

        # Find subject using shared helper (considers nominative NPs and
        # subject pronouns; respects PP-inside via global pp_inside set).
        subj_tok = _find_subject(tokens, nps, s, e,
                                 verb.idx if verb else None, pp_inside_global)

        # Reset elision at sentence boundaries — never carry a subject across
        # a period. Also reset across hard breaks (colon/newline) which usually
        # introduce a new top-level item ("Nefarendaj:\nEstu silenta").
        if s in sentence_starts:
            prev_subject_token = None
            prev_subject_was_compound = False
        elif s < len(tokens) and getattr(tokens[s], "has_hard_break_before", False):
            prev_subject_token = None
            prev_subject_was_compound = False

        # Subject elision: if no subject and clause introduced by a coordinator
        # (kaj/sed/aŭ/nek), inherit the previous clause's subject. But not if
        # the clause itself has an INF that serves as its (implicit) subject —
        # "X estas belaj kaj protekti ilin estas nobla" — second clause's
        # subject is the infinitive clause `protekti ilin`, not elided from X.
        intro = intro_words.get(s)
        elided = False
        has_inf_subject = any(
            t.pos == "INF" and (i_ := t.idx) < (verb.idx if verb else e)
            for t in clause_tokens
        )
        if subj_tok is None and bi > 0 and prev_subject_token is not None \
                and not has_inf_subject:
            if s - 1 >= 0 and tokens[s - 1].raw_lower in COORDINATORS:
                subj_tok = prev_subject_token
                elided = True

        # Detect compound subject within this clause: a `kaj` outside any PP
        # joining two N-bearing halves.
        this_is_compound = False
        if subj_tok is not None and subj_tok.idx >= s:
            for kk in range(s, min(e, verb.idx if verb else e)):
                if tokens[kk].raw_lower != "kaj":
                    continue
                # Check PP / INF membership. A reliable signal that kaj is
                # PP-internal: the next NP head after kaj falls inside a PP.
                in_pp = False
                in_inf = False
                for m in range(kk - 1, s - 1, -1):
                    if tokens[m].has_comma_before or tokens[m].pos == "V":
                        break
                    if tokens[m].pos == "INF":
                        in_inf = True
                        break
                    if tokens[m].pos == "Prep":
                        in_pp = True
                        break
                if in_pp:
                    for m in range(kk + 1, min(e, verb.idx if verb else e)):
                        if tokens[m].pos == "N":
                            if tokens[m].idx not in pp_inside_global:
                                in_pp = False
                            break
                if in_pp or in_inf:
                    continue
                # Widen left search: comma-separated list items ("A, B, C, kaj D")
                # end up in separate clauses. Walk back past clause boundary
                # through comma-preceded N tokens to find the full list.
                left_start = s
                while left_start > 0 and tokens[left_start - 1].pos == "N" \
                        and tokens[left_start].has_comma_before:
                    left_start -= 1
                    # Skip back past any preceding Det/A of that NP
                    while left_start > 0 and tokens[left_start - 1].pos in ("A", "Det"):
                        left_start -= 1
                # Subject pronouns (li/ŝi/ni/ili/tiu/ĉiu/etc) count as N for
                # compound detection: "li kaj la mapo estis malaperintaj".
                def _subject_like(m: int) -> bool:
                    t = tokens[m]
                    if t.pos == "N" and t.idx not in pp_inside_global:
                        return True
                    if t.raw_lower in SUBJECT_PRONOUNS \
                            or t.raw_lower in CORRELATIVE_SUBJECTS_SG \
                            or t.raw_lower in CORRELATIVE_SUBJECTS_PL:
                        return True
                    return False
                left_has_n = any(_subject_like(m) for m in range(left_start, kk))
                right_has_n = any(_subject_like(m)
                                   for m in range(kk + 1, min(e, verb.idx if verb else e)))
                if left_has_n and right_has_n:
                    this_is_compound = True
                    break

        # Wrap subject token in a virtual NP so callers always see an NP-shaped
        # subject. If subject is a noun NP head, find the matching NP; else
        # build a single-token NP. When the subject is compound (or was elided
        # from a compound clause), force the head token's number to pl.
        subject_np: NP | None = None
        effective_compound = this_is_compound or (elided and prev_subject_was_compound)
        if subj_tok is not None:
            if effective_compound and subj_tok.number != "pl":
                import copy
                subj_tok = copy.copy(subj_tok)
                subj_tok.number = "pl"
            subject_np = next((np for np in clause_nps if np.head is subj_tok), None)
            if subject_np is None:
                subject_np = NP(tokens=[subj_tok], head=subj_tok,
                                start_idx=subj_tok.idx, end_idx=subj_tok.idx)
            prev_subject_token = subj_tok
            prev_subject_was_compound = effective_compound

        clauses.append(Clause(
            start_idx=s, end_idx=e, tokens=clause_tokens, nps=clause_nps,
            verb=verb, subject=subject_np,
            is_subordinate=is_subord.get(s, False),
            intro_word=intro,
        ))

    return clauses


# ---- Check base -------------------------------------------------------

class Check(Protocol):
    name: str
    def check(self, tokens: list[Token], nps: list[NP]) -> list[Diagnostic]: ...


# ---- Concrete checks --------------------------------------------------

class NPAgreement:
    name = "np-agreement"
    def check(self, tokens, nps):
        out = []
        for np in nps:
            head = np.head
            # Skip if head looks like a proper noun — agreement is loose there
            if getattr(head, "is_proper", False):
                continue
            for mod in np.tokens:
                if mod is head:
                    continue
                if mod.pos != "A":
                    continue
                # Skip proper-noun modifiers too
                if getattr(mod, "is_proper", False):
                    continue
                # FIX 2: participles after a form of 'esti' (or before an acc noun)
                # are verbal, not modifiers — don't force case agreement.
                # Detect participle either from decomposition OR from stem ending
                # (handles dictionary-entry participles like "subtenanta" whose
                # whole stem matches a root).
                is_participle = bool(mod.suffixes) and mod.suffixes[-1] in PARTICIPLE_SUFFIXES
                if not is_participle and mod.root:
                    is_participle = any(mod.root.endswith(p) for p in PARTICIPLE_SUFFIXES)
                if is_participle:
                    # FIX A: participle is verbal if preceded by ANY form of esti
                    # (finite OR infinitive: "estas X-inta", "esti X-inta")
                    if mod.idx > 0 and tokens[mod.idx - 1].root == "est":
                        continue
                    # FIX C: participle followed by accusative noun governs it
                    # as an object (not as a modifier of head). Skip case check.
                    if head.case == "acc" and mod.idx == np.start_idx:
                        # Treat the noun as the participle's object → no agreement needed
                        continue
                # FIX 8: possessive determiners can be plural when the NP is
                # coordinated with "kaj" + another NP.
                if mod.raw_lower in POSSESSIVES and mod.number == "pl" and head.number == "sg":
                    if head.idx + 1 < len(tokens) and tokens[head.idx + 1].raw_lower == "kaj":
                        if head.idx + 2 < len(tokens) and tokens[head.idx + 2].pos in ("N", "A"):
                            continue
                if mod.number and head.number and mod.number != head.number:
                    # Distributive coordination: "(la) usona kaj eŭropa
                    # merkatoj" — each sg adj describes one of the pl
                    # referents (cf. PMEG: "la Blanka kaj la Nigra Maroj").
                    # NP extraction stopped at `kaj`, so check the tokens
                    # immediately before this NP: if they end in `kaj`/`aŭ`
                    # preceded by another sg adj, treat as distributive.
                    if head.number == "pl" and mod.number == "sg":
                        k = np.start_idx - 1
                        is_distributive = False
                        while k >= 0 and tokens[k].pos in ("A", "Det"):
                            if tokens[k].pos == "A":
                                k -= 1
                                continue
                            k -= 1
                        if k >= 0 and tokens[k].raw_lower in {"kaj", "aŭ"} \
                                and k - 1 >= 0 and tokens[k - 1].pos == "A":
                            is_distributive = True
                        if is_distributive:
                            continue
                    out.append(Diagnostic(
                        self.name, "error",
                        f"'{mod.text}' ({mod.number}) does not agree in number with head '{head.text}' ({head.number})",
                        mod.idx, (np.start_idx, np.end_idx)))
                if mod.case and head.case and mod.case != head.case:
                    out.append(Diagnostic(
                        self.name, "error",
                        f"'{mod.text}' ({mod.case}) does not agree in case with head '{head.text}' ({head.case})",
                        mod.idx, (np.start_idx, np.end_idx)))
        return out


class PredicateAdjAgreement:
    """After esti + adjective, adjective must agree in NUMBER only (never acc)."""
    name = "predicate-adj"
    def check(self, tokens, nps, clauses=None):
        out = []
        # If we have clause info, restrict subject lookup to within-clause.
        clause_of_tok: dict[int, Clause] = {}
        if clauses:
            for cl in clauses:
                for t in cl.tokens:
                    clause_of_tok[t.idx] = cl

        for i, tok in enumerate(tokens):
            if tok.pos != "V" or tok.root not in {"est"}:
                continue
            # Only pure `esti` (not derived verbs like `estigi` = to cause to be).
            # estigi/estiĝi have their own semantics (cause/become) and often
            # govern an acc object rather than a predicate adjective.
            if tok.suffixes:
                continue
            # FIX 2: stop at infinitive — "estas eble dokumenti X-acc" means
            # the infinitive governs the acc object; the adjective is "eble" (nom).
            # We want to take the adj that appears BEFORE any infinitive.
            j = i + 1
            while j < len(tokens) and tokens[j].pos not in ("A", "N", "Closed", "Punct", "INF"):
                j += 1
            if j >= len(tokens) or tokens[j].pos != "A":
                continue
            pred = tokens[j]
            # If the adj is followed (through any number of further adjs,
            # possibly coordinated with commas or `kaj`) by a noun, it's a
            # modifier of a nominal predicate rather than a predicate adj.
            # Examples: "estas la perfekta celloko", "estas bongusta, krema
            # frukto", "estas la ĉefa ekonomia centro".
            k2 = j + 1
            while k2 < len(tokens):
                if tokens[k2].pos == "A":
                    k2 += 1
                    continue
                if tokens[k2].raw_lower == "kaj":
                    k2 += 1
                    continue
                break
            if k2 < len(tokens) and tokens[k2].pos == "N":
                continue
            # Substantivized adj: "la X-a" acts as an NP, not a predicate adj.
            # Covers "la oficialaj lingvoj estas la angla kaj la franca".
            # Walk back from the pred through intensifiers/adjs to find `la`.
            kk = j - 1
            while kk >= 0 and (tokens[kk].pos == "A"
                                or tokens[kk].raw_lower in {"plej", "pli", "tre", "tro"}):
                kk -= 1
            if kk >= 0 and tokens[kk].pos == "Det":
                continue
            # FIX 5: find subject by walking the NP list — pick the FIRST nominative NP
            # whose head is NOT inside a preposition phrase. This handles
            # "multaj homoj kun tiu nomo estas konataj" — subject is 'homoj', not 'nomo'.
            pp_heads: set[int] = set()
            for k, t in enumerate(tokens[:i]):
                if t.pos == "Prep":
                    # Mark the next noun as inside a PP
                    for np in nps:
                        if np.start_idx == k + 1:
                            pp_heads.add(np.head.idx)
                            break
            # Use the clause's subject. If we have clause information and the
            # clause has no identifiable subject (relative pronoun, ellipsis),
            # SKIP the check entirely — guessing across clauses produces
            # massive false positives (relative-clause antecedent picked as
            # subject of the wrong predicate).
            cl = clause_of_tok.get(i)
            if cl is not None:
                if cl.subject is None:
                    continue
                subj = cl.subject.head
            else:
                subj = None
                for np in nps:
                    if np.start_idx >= i:
                        break
                    if np.case == "nom" and np.head.idx not in pp_heads:
                        subj = np.head
                if subj is None:
                    continue
            # FIX 7: compound subject "X kaj Y" — plural. Only `kaj` forces
            # plural: "aŭ"/"nek" typically single out one alternative and
            # agree in the singular. Also require the `kaj` to be *outside*
            # a PP — "kontakto inter X kaj Y estas" has its `kaj` inside a
            # PP, so X/Y aren't the subject.
            compound_plural = False
            cl_lo = cl.start_idx if cl is not None else 0
            cl_hi = cl.end_idx if cl is not None else len(tokens)
            pp_set = _compute_pp_inside(tokens, nps)
            for k in range(cl_lo, min(cl_hi, i)):
                if tokens[k].raw_lower != "kaj":
                    continue
                # Skip if kaj is PP-internal or sits inside an infinitival
                # complement. A reliable signal that kaj is PP-internal: the
                # next NP head after kaj falls inside some PP (the same one
                # whose objects are being coordinated).
                in_pp = False
                in_inf = False
                for m in range(k - 1, cl_lo - 1, -1):
                    if tokens[m].has_comma_before or tokens[m].pos == "V":
                        break
                    if tokens[m].pos == "INF":
                        in_inf = True
                        break
                    if tokens[m].pos == "Prep":
                        in_pp = True
                        break
                # Cross-check via pp_inside: if the next N after kaj is
                # inside a PP, the kaj is coordinating PP objects.
                if in_pp:
                    for m in range(k + 1, min(cl_hi, i)):
                        if tokens[m].pos == "N":
                            if tokens[m].idx not in pp_set:
                                in_pp = False
                            break
                if in_pp or in_inf:
                    continue
                left_has_n = any(tokens[m].pos == "N" and tokens[m].idx not in pp_set
                                  for m in range(cl_lo, k))
                right_has_n = any(tokens[m].pos == "N" and tokens[m].idx not in pp_set
                                   for m in range(k + 1, min(cl_hi, i)))
                if left_has_n and right_has_n:
                    compound_plural = True
                    break
            if compound_plural:
                # Override subject number for check
                import copy
                subj = copy.copy(subj)
                subj.number = "pl"
            if pred.case == "acc":
                out.append(Diagnostic(
                    self.name, "error",
                    f"predicate adjective '{pred.text}' after '{tok.text}' must NOT be accusative",
                    pred.idx))
            # Proper-noun subjects have idiosyncratic number only when the
            # ending gives no clear number marker. "Milimetroj" (pl ending -oj)
            # is unambiguously plural even if tagged proper; but "Beatles",
            # "Smokies", "Mountains", "Jersey" don't carry Esperanto number
            # marking, so we can't tell.
            if getattr(subj, "is_proper", False) \
                    and subj.ending not in ("oj", "ojn", "aj", "ajn"):
                continue
            if pred.number and subj.number and pred.number != subj.number:
                # Distributive predicate adj: pl subject with coordinated sg
                # adjs as predicate, each referring to one item.
                # "La koloroj estas ruĝa, blua kaj verda."
                if subj.number == "pl" and pred.number == "sg":
                    # Look for adj, [comma/kaj, adj]+ pattern starting at pred
                    k_forward = pred.idx + 1
                    has_coord_adj = False
                    while k_forward < len(tokens):
                        t2 = tokens[k_forward]
                        if t2.raw_lower in {"kaj", "aŭ"} \
                                or t2.has_comma_before:
                            k_forward += 1
                            # Skip coord words
                            if k_forward < len(tokens) and tokens[k_forward].raw_lower in {"kaj", "aŭ"}:
                                k_forward += 1
                            if k_forward < len(tokens) and tokens[k_forward].pos == "A":
                                has_coord_adj = True
                                k_forward += 1
                                continue
                        break
                    if has_coord_adj:
                        continue
                out.append(Diagnostic(
                    self.name, "error",
                    f"predicate adjective '{pred.text}' ({pred.number}) does not agree with subject '{subj.text}' ({subj.number})",
                    pred.idx))
        return out


class ParticipleAgreement:
    """Participles used as adjectives follow NP / predicate-adj rules.

    We already treat participles as A-POS (since they end in -a/-aj/etc after
    participle suffix). This check specifically verifies that the morpheme
    before -a is one of the participle suffixes and that the stem before that
    is a verb; otherwise flag the participle form as suspicious.
    """
    name = "participle"
    def check(self, tokens, nps):
        out = []
        for tok in tokens:
            if tok.pos != "A" or not tok.suffixes:
                continue
            if tok.suffixes[-1] not in PARTICIPLE_SUFFIXES:
                continue
            # Check that the remaining stem (root + any suffixes before participle)
            # resolves to a verb root
            root = tok.root or ""
            # Simple sanity: root should be in known verbs/roots, or have verb-compatible stem.
            # Too restrictive to require KNOWN_*, so only flag if root is clearly nominal
            # e.g., the participle attaches to a proper noun or short unknown stem.
            if root and root not in get_roots():
                # unknown root — can't verify, pass silently
                continue
        return out


class MissingAccusative:
    """Transitive verb followed by a nominative NP that isn't inside a PP."""
    name = "missing-accusative"
    def check(self, tokens, nps, clauses=None):
        out = []
        # Use the shared PP helper so coordinated NPs ("al pli alta X") and
        # NPs separated from their Prep by modifiers are all detected.
        head_inside_pp = _compute_pp_inside(tokens, nps)
        inside_pp: set[int] = set()
        for np in nps:
            if np.head.idx in head_inside_pp:
                for j in range(np.start_idx, np.end_idx + 1):
                    inside_pp.add(j)
        # Map verb-token-idx → clause for clause-bounded NP search
        clause_of: dict[int, Clause] = {}
        if clauses:
            for cl in clauses:
                for t in cl.tokens:
                    clause_of[t.idx] = cl
        for i, tok in enumerate(tokens):
            if tok.pos != "V":
                continue
            # Skip modal verbs — 'povas/devas/volas/scias' + INF — the object
            # belongs to the infinitive, not the modal.
            if tok.root and tok.root + "i" in MODAL_VERBS:
                # Check that the next non-trivial token is INF or another verb form
                for k in range(i + 1, min(i + 5, len(tokens))):
                    if tokens[k].pos == "INF":
                        # skip this verb entirely
                        break
                else:
                    pass  # no inf found, but still skip modal — usually safe
                continue
            is_transitive = _is_transitive_verb(tok)
            if is_transitive is None or not is_transitive:
                continue
            # Restrict NPs to within the clause if available
            cl = clause_of.get(i)
            # Postposed subject (VS/VSO word order). Esperanto freely allows
            # the subject to follow the verb — "Diris la instruisto", "Venis
            # la gastoj", "Sekvas la limdato", "— demandis la knabeto".
            # If no subject candidate appears before the verb in this clause,
            # the nominative NP that follows IS the subject, not a mis-cased
            # object. (A pre-verb subject means the post-verb nom would be a
            # real object-case error.)
            if cl is not None:
                has_subj_before = any(
                    (t.pos == "N" and t.case == "nom"
                        and t.idx not in _compute_pp_inside(tokens, nps))
                    or t.raw_lower in SUBJECT_PRONOUNS
                    or t.raw_lower in CORRELATIVE_SUBJECTS_SG
                    or t.raw_lower in CORRELATIVE_SUBJECTS_PL
                    for t in cl.tokens if t.idx < i
                )
                if not has_subj_before:
                    continue
            candidate_nps = cl.nps if cl else nps
            for np in candidate_nps:
                if np.start_idx <= i:
                    continue
                if np.start_idx in inside_pp:
                    break
                if getattr(np.head, "is_proper", False):
                    break
                # FIX E: skip if a subordinate-clause introducer intervenes.
                # Includes 'ke', kiel, kiu, kies, kio, kien, etc., or quote/colon.
                SUBORD_INTROS = {"ke", "kiel", "kiu", "kio", "kies", "kiam",
                                  "kie", "kien", "kial", "kiom", "kvankam"}
                clause_break = False
                for k in range(i + 1, np.start_idx):
                    t = tokens[k]
                    if t.raw_lower in SUBORD_INTROS:
                        clause_break = True
                        break
                # Also break on punctuation between (colon, quote)
                if not clause_break and i + 1 < np.start_idx:
                    # check raw text region
                    text_between_start = tokens[i].char_span[1]
                    text_between_end = tokens[np.start_idx].char_span[0] if np.start_idx < len(tokens) else 0
                if clause_break:
                    break
                # FIX D: OVS — if there's already an accusative noun BEFORE this
                # verb in the same clause, the verb has its object; the nominative
                # NP after is likely the subject (postposed).
                # Accusative pronouns + correlative accusatives all count as
                # objects satisfying the transitive verb.
                ACC_PRONOUNS = {
                    "min", "vin", "lin", "ŝin", "ĝin", "nin", "ilin", "onin", "sin",
                    "kion", "kiun", "kiujn", "tion", "tiun", "tiujn",
                    "ion", "iun", "iujn", "ĉion", "ĉiun", "ĉiujn",
                    "nenion", "neniun", "neniujn",
                    # numeric/quantifier accusatives:
                    "kelkajn", "multajn", "plurajn", "iujn",
                    "unun", "du",  # "du" is invariable but acts as acc too
                }
                cl_lo = cl.start_idx if cl is not None else max(0, i - 5)
                seen_acc_before = any(
                    (tokens[k].pos == "N" and tokens[k].case == "acc")
                    or tokens[k].raw_lower in ACC_PRONOUNS
                    for k in range(max(cl_lo, i - 5), i)
                )
                if seen_acc_before:
                    break
                # Object satisfied: an acc N or acc pronoun appears between
                # the verb and the candidate nominative NP.
                acc_between = any(
                    (tokens[k].pos == "N" and tokens[k].case == "acc")
                    or tokens[k].raw_lower in ACC_PRONOUNS
                    for k in range(i + 1, np.start_idx)
                )
                if acc_between:
                    break
                # Verb + INF — the infinitive takes the object
                # ("serĉas dungi en tiu kariero").
                inf_between = any(tokens[k].pos == "INF"
                                   for k in range(i + 1, np.start_idx))
                if inf_between:
                    break
                # "pli/malpli ... ol X" — X is the comparison standard in
                # nominative, not an object. "atingas pli ol pura forto".
                ol_between = any(tokens[k].raw_lower == "ol"
                                  for k in range(i + 1, np.start_idx))
                if ol_between:
                    break
                if np.case == "nom":
                    out.append(Diagnostic(
                        self.name, "warning",
                        f"transitive verb '{tok.text}' has nominative object '{np.head.text}' — expected accusative '-n'",
                        np.head.idx, (np.start_idx, np.end_idx)))
                break
        return out


class IntransitiveAccusative:
    """Intransitive verb followed by an accusative NP (unless directional or
    intervening infinitive)."""
    name = "intransitive-acc"
    def check(self, tokens, nps, clauses=None):
        out = []
        clause_of: dict[int, Clause] = {}
        if clauses:
            for cl in clauses:
                for t in cl.tokens:
                    clause_of[t.idx] = cl
        for i, tok in enumerate(tokens):
            if tok.pos != "V":
                continue
            # Skip 'esti' forms entirely — copula commonly takes adverbial
            # accusative ("estas du metrojn alta", "estas tri jarojn pli aĝa").
            if tok.root == "est":
                continue
            is_transitive = _is_transitive_verb(tok)
            if is_transitive is None or is_transitive:
                continue
            # FIX 4: if any infinitive appears between the verb and the next NP,
            # the infinitive "captures" the object — skip this verb.
            cl = clause_of.get(i)
            candidate_nps = cl.nps if cl else nps
            next_np = None
            for np in candidate_nps:
                if np.start_idx <= i:
                    continue
                next_np = np
                break
            if next_np is None:
                continue
            # Scan between the verb and the NP for an infinitive or preposition
            # or an adverbial participle ("sidis pripensante ... planojn" —
            # `pripensante` takes the acc object, not `sidis`).
            captured = False
            for k in range(i + 1, next_np.start_idx):
                if tokens[k].pos == "INF" or tokens[k].pos == "Prep":
                    captured = True
                    break
                # Adverbial participle (-ante/-inte/-ante) takes acc object
                if tokens[k].pos == "Adv" and tokens[k].root \
                        and any(tokens[k].root.endswith(p) for p in PARTICIPLE_SUFFIXES):
                    captured = True
                    break
                # Or stem ends in ant/int/ont/at/it/ot + e
                if tokens[k].pos == "Adv" and tokens[k].raw_lower[:-1].endswith(
                        tuple(PARTICIPLE_SUFFIXES)):
                    captured = True
                    break
                # Other coordinated verb (X kaj Y) — second verb may be transitive
                if tokens[k].pos == "V" and tokens[k] is not tok:
                    captured = True
                    break
            for t in next_np.tokens:
                if t.pos == "A" and t.suffixes and t.suffixes[-1] in PARTICIPLE_SUFFIXES:
                    captured = True
                    break
            if captured:
                continue
            # Cognate-object idiom: "vivi vivon", "kanti kanton", "dormi dormon"
            if tok.root and next_np.head.root == tok.root:
                continue
            # Adverbial accusative of time/measure: "vekiĝas ĉiujn horojn",
            # "kuras dek minutojn", "altas tri metrojn"
            if next_np.head.root in ADVERBIAL_TIME_NOUNS:
                continue
            if next_np.case == "acc":
                out.append(Diagnostic(
                    self.name, "warning",
                    f"intransitive verb '{tok.text}' has accusative object '{next_np.head.text}' — unexpected",
                    next_np.head.idx, (next_np.start_idx, next_np.end_idx)))
        return out


class PrepositionCase:
    """Non-directional prepositions should never take accusative after them."""
    name = "preposition-case"
    # These non-directional preps commonly take accusative either via ellipsis
    # (the accusative belongs to an implicit verb, not the prep) or because
    # they're being used adverbially as approximators/range markers alongside
    # measure accusatives:
    # "anstataŭ la kalkanojn" = "instead of [hitting] the heels."
    # "krom la restoraciojn" = "except [for] the restaurants."
    # "ĉirkaŭ 30 sekundojn" = "approximately 30 seconds" (adverbial).
    # "ĝis 2 minutojn" = "up to 2 minutes" (range marker, measure acc).
    ELLIPSIS_ALLOWED = {"anstataŭ", "krom", "ĉirkaŭ", "ĝis", "preter", "po"}
    def check(self, tokens, nps):
        out = []
        for i, tok in enumerate(tokens):
            if tok.pos != "Prep":
                continue
            if tok.raw_lower not in NONDIR_PREPS:
                continue
            if tok.raw_lower in self.ELLIPSIS_ALLOWED:
                continue
            for np in nps:
                if np.start_idx == i + 1:
                    # Skip if a sentence/hard break separates the Prep from
                    # the NP ("de 1969. Tiun nokton" — nokton is a new
                    # sentence, not governed by the prior `de`).
                    first_tok = tokens[np.start_idx]
                    if getattr(first_tok, "is_sentence_start", False) \
                            or getattr(first_tok, "has_hard_break_before", False):
                        break
                    # Skip proper nouns — foreign names with -en/-on endings
                    # often look like Esperanto accusatives but aren't inflected.
                    if getattr(np.head, "is_proper", False):
                        break
                    if np.case == "acc":
                        out.append(Diagnostic(
                            self.name, "error",
                            f"preposition '{tok.text}' must be followed by nominative, got accusative '{np.head.text}'",
                            np.head.idx, (np.start_idx, np.end_idx)))
                    break
        return out


class WrongEndings:
    """Heuristic checks for wrong POS-endings in common positions:
      - noun immediately before another noun (first should be -a)
      - adjective before a verb with no head noun in between (should be -e adverb)
    """
    name = "wrong-ending"
    def check(self, tokens, nps):
        out = []
        n = len(tokens)
        for i, tok in enumerate(tokens):
            if tok.pos == "N" and i + 1 < n and tokens[i+1].pos == "N":
                # Skip proper-noun apposition
                if getattr(tok, "is_proper", False) or getattr(tokens[i+1], "is_proper", False):
                    continue
                next_tok = tokens[i+1]
                # FIX 3a: same-root duplication ("paŝon post paŝo"-style)
                if tok.root and tok.root == next_tok.root:
                    continue
                # FIX 3b: second noun starts an idiomatic "X-acc post X-nom" phrase.
                # Pattern: tokens[i+1] (N-acc) + tokens[i+2] (Prep like post/pro/per)
                #          + tokens[i+3] (N-nom with same root as tokens[i+1])
                if next_tok.case == "acc" and i + 3 < n:
                    bridge = tokens[i+2]
                    third = tokens[i+3]
                    if bridge.pos == "Prep" and bridge.raw_lower in {"post", "pro", "per", "kun"} and \
                       third.pos == "N" and third.root and third.root == next_tok.root:
                        continue
                # FIX 9: list items separated by comma — "agrikulturo, historio"
                # Skip if comma between the two nouns.
                if next_tok.has_comma_before or tok.has_comma_after:
                    continue
                # Skip if a hard break (colon, newline) sits between them —
                # bullet lists ("- Akvo: likvaĵo") and key:value notation.
                if getattr(next_tok, "has_hard_break_before", False):
                    continue
                # Skip across parens/quotes — "jonoj (fragmentoj)" apposition.
                if getattr(next_tok, "has_paren_before", False) or \
                   getattr(tok, "has_paren_before", False):
                    continue
                # Skip if the second noun is an adverbial measure/time noun —
                # "ŝanĝas korpojn 4 fojojn" (4 isn't tokenized, so korpojn and
                # fojojn end up adjacent) is valid usage, not wrong-ending.
                if next_tok.root in ADVERBIAL_TIME_NOUNS:
                    continue
                if tok.case == next_tok.case and tok.number == next_tok.number:
                    out.append(Diagnostic(
                        self.name, "warning",
                        f"consecutive nouns '{tok.text} {next_tok.text}' — first should likely be adjective (-a)",
                        tok.idx))
            if tok.pos == "A" and i + 1 < n and tokens[i+1].pos in ("V", "INF"):
                # Skip if this "adjective" is a capitalized word (proper noun)
                if tok.text and tok.text[0].isupper():
                    continue
                # Adj + INF is a legitimate Esperanto construction ("preta
                # montri", "ekscitita iri", "kapabla detekti") — predicative
                # adj with a purpose infinitive. The adv-form would change
                # the meaning, so don't flag.
                if tokens[i+1].pos == "INF":
                    continue
                # Skip if the adj is inside parens/quotes — it's a quoted
                # mention of a term, not a predicate adj in running text.
                # "terminoj kiel 'esence kompleta' estis uzataj" — 'kompleta'
                # is inside quotes.
                if getattr(tok, "is_in_parens", False):
                    continue
                # FIX 1: comma-separated list ends with "kaj ADJ"
                if i > 0 and (tokens[i-1].raw_lower == "kaj" or tok.has_comma_before):
                    continue
                # FIX 7b: predicate-adj after esti
                if i > 0 and tokens[i-1].pos == "V" and tokens[i-1].root == "est":
                    continue
                # Respect clause boundary: comma, hard break, or sentence start
                # before the next-token verb means the adj belongs to a prior
                # clause/sentence ("...energia.\nEvitu plurtaskadon").
                next_tok = tokens[i+1]
                if (next_tok.has_comma_before
                    or getattr(next_tok, "has_hard_break_before", False)
                    or getattr(next_tok, "is_sentence_start", False)):
                    continue
                # Substantivized adj: walk back through adjs and adverb
                # intensifiers (plej/pli/tre/tro) — if we reach a
                # determiner-like token, the adj is acting as a noun.
                # Accepts `la` (Det), plus correlative dets/possessives in the
                # closed class ("kies", "ties", "mia", "tia", "ĉia"...).
                INTENSIFIERS = {"plej", "pli", "tre", "tro", "iom", "sufiĉe", "tiel"}
                k = i - 1
                while k >= 0 and (tokens[k].pos == "A"
                                   or tokens[k].raw_lower in INTENSIFIERS):
                    k -= 1
                if k >= 0:
                    prev = tokens[k]
                    if prev.pos == "Det":
                        continue
                    if prev.pos == "Closed" and prev.raw_lower.endswith(
                        ("a", "aj", "an", "ajn", "es")
                    ):
                        continue
                    # Correlative subjects acting as determiners/quantifiers:
                    # "neniu alia povis" — neniu + substantivized adj alia.
                    if prev.raw_lower in CORRELATIVE_SUBJECTS_SG \
                            or prev.raw_lower in CORRELATIVE_SUBJECTS_PL:
                        continue
                # Post-nominal adj complement — the adj modifies the entity
                # just named and the following verb/INF is its complement, not
                # a main verb needing an adverb. Covers:
                #   "mekanismo kapabla detekti" (N + adj + INF)
                #   "Resti sana permesos" (INF + adj + V)
                #   "sentu vin libera miksi" (acc pronoun + adj + INF)
                #   plural/acc adjs agreeing with a preceding acc noun
                if tok.case == "acc" or tok.number == "pl":
                    continue
                ACC_PRON = {"min", "vin", "lin", "ŝin", "ĝin", "nin", "ilin", "onin", "sin"}
                if i > 0:
                    # Walk back through intensifiers (pli/plej/tre/tro/...) to
                    # find the "real" preceding token.
                    INTENSIFIERS_WE = {"pli", "plej", "tre", "tro", "iom",
                                        "sufiĉe", "tiom", "kiom", "ju", "des"}
                    bi = i - 1
                    while bi >= 0 and tokens[bi].raw_lower in INTENSIFIERS_WE:
                        bi -= 1
                    if bi >= 0:
                        prev = tokens[bi]
                        if prev.pos in ("N", "INF") or prev.raw_lower in ACC_PRON:
                            continue
                # VS predicate-adj word order: "adj + V + N-nom" — the adj is
                # predicate complement agreeing with the postposed subject.
                # "Tre populara estas la Taĝ-Mahalo", "Ju pli alta estas la
                # nombro", "Bela estas la vespero" — all valid patterns.
                j_after = i + 2
                while j_after < n and tokens[j_after].pos in ("A", "Det", "Closed"):
                    j_after += 1
                if j_after < n and tokens[j_after].pos == "N" \
                        and tokens[j_after].case == "nom":
                    continue
                # "la plej / pli / tre + adj + esti/INF" — substantivized pattern
                # (la plej amuza spekti = the funniest [thing] to watch).
                if i >= 2 and tokens[i-1].raw_lower in {"plej", "pli", "tre", "tro"} \
                   and tokens[i-2].pos == "Det":
                    continue
                # "kiom/tiom adj estas X" — indirect-question / correlative
                # construction ("determini kiom aĝa estas elfosejo",
                # "priskribi kiom granda estas regiono").
                if i >= 1 and tokens[i-1].raw_lower in {"kiom", "tiom", "iom", "ĉiom"}:
                    continue
                out.append(Diagnostic(
                    self.name, "warning",
                    f"adjective '{tok.text}' before verb '{tokens[i+1].text}' — should probably be adverb (-e)",
                    tok.idx))
        return out


class AffixCompat:
    """Walk the suffix chain checking POS compatibility."""
    name = "affix-compat"
    def check(self, tokens, nps):
        out = []
        roots = get_roots()
        for tok in tokens:
            if not (tok.suffixes or tok.prefixes):
                continue
            root = tok.root or ""
            if root not in roots:
                continue
            # Start with root's "natural" POS — we approximate:
            #   unknown → any; -o root → N; -a root → A; -i root → V
            # Since root dictionary lacks POS info, treat as Any for permissive check.
            current_pos = "Any"
            # Prefixes check against the bare root
            for pre in tok.prefixes:
                if pre not in PREFIX_SIG:
                    continue
                allowed = PREFIX_SIG[pre]
                if current_pos != "Any" and current_pos not in allowed:
                    out.append(Diagnostic(
                        self.name, "warning",
                        f"prefix '{pre}-' in '{tok.text}' expects {allowed}, got {current_pos}",
                        tok.idx))
            # Suffix chain — each transforms POS
            prev_was_participle = False
            for suf in tok.suffixes:
                if suf not in SUFFIX_SIG:
                    continue
                allowed, output = SUFFIX_SIG[suf]
                if suf in PARTICIPLE_SUFFIXES:
                    # Check root + i as known verb, OR root+previous-suffixes+i as known verb
                    # (handles dom+in+ant where 'domini' is known but 'dom' isn't)
                    stem_before = root
                    seen = tok.suffixes[:tok.suffixes.index(suf)]
                    for prev_suf in seen:
                        stem_before += prev_suf
                    candidates = {root + "i", stem_before + "i"}
                    root_verb = any(c in KNOWN_TRANSITIVE or c in KNOWN_INTRANSITIVE
                                     for c in candidates)
                    if current_pos != "V" and current_pos != "Any" and not root_verb:
                        out.append(Diagnostic(
                            self.name, "error",
                            f"participle suffix '-{suf}-' in '{tok.text}' must attach to a verb",
                            tok.idx))
                    prev_was_participle = True
                    current_pos = output or current_pos
                    continue
                # FIX 5: if previous suffix was a participle (-ant/-int/-at/-it/-ot)
                # and this suffix expects N, allow it. The participle + nominal
                # ending forms an agent noun (spektant-o = spectator), which can
                # validly take -ar / -in / -ej etc.
                if prev_was_participle and "N" in allowed:
                    prev_was_participle = False
                    current_pos = output or "N"
                    continue
                if current_pos != "Any" and current_pos not in allowed:
                    out.append(Diagnostic(
                        self.name, "warning",
                        f"suffix '-{suf}-' in '{tok.text}' expects {allowed}, got {current_pos}",
                        tok.idx))
                prev_was_participle = False
                current_pos = output or current_pos
        return out


# ---- Helpers ----------------------------------------------------------

def _is_transitive_verb(tok: Token) -> bool | None:
    """Return True/False/None (unknown) for verb transitivity."""
    if tok.pos not in ("V", "INF"):
        return None
    # If the word ends in -igi / -igi-form, it's transitive; -iĝi is intransitive.
    if "ig" in tok.suffixes:
        return True
    if "iĝ" in tok.suffixes:
        return False
    # -ebl- / -ind- / -end- forms derive adjectives from verbs; when further
    # conjugated (troveblas = "are findable"), the resulting form is a
    # predicative/passive construction, not a transitive verb in its own right.
    if any(s in {"ebl", "ind", "end"} for s in tok.suffixes):
        return False
    # Build infinitive candidates from root + prefix-attached forms.
    # "eniros" decomposes to prefix=en, root=ir → check both "iri" and "eniri",
    # since prefix can flip transitivity (iri intransitive, eniri transitive).
    root = tok.root or ""
    candidates = [root + "i"]
    if tok.prefixes:
        full = "".join(tok.prefixes) + root + "i"
        candidates.insert(0, full)  # prefer prefixed form
    for inf in candidates:
        if inf in KNOWN_TRANSITIVE:
            return True
        if inf in KNOWN_INTRANSITIVE:
            return False
    return None


# ---- Claim extraction -------------------------------------------------

# Derivational suffixes in Esperanto carry semantic relations. When a word
# ends in one of these, the word itself is implicitly asserting a relation
# to its base root — "komponisto" = (person, role-of, kompon).
SUFFIX_RELATIONS: dict[str, str] = {
    "ist":  "role-of",        # doer/practitioner
    "ul":   "person-with",    # person with quality
    "in":   "female-of",      # female counterpart
    "id":   "offspring-of",   # child/descendant
    "ej":   "place-for",      # place for activity
    "il":   "tool-for",       # instrument
    "ar":   "group-of",       # collection
    "estr": "leader-of",      # head/chief
    "ec":   "quality-of",     # abstract quality
    "aĵ":   "thing-of",       # concrete instance
    "an":   "member-of",      # member/inhabitant
    "uj":   "container-of",   # container, country, or holder
}


@dataclass
class Claim:
    """A structural assertion extracted from the text.

    `source` identifies which rule produced it:
      - "copula": S estas P (esti + predicate)
      - "transitive": S V-as O (subject + transitive verb + acc object)
      - "pp-relation": NP1 prep NP2 (noun phrase + preposition)

    `clause_idx` identifies which clause the claim came from. Claims from the
    same clause with the same (subj, rel) are coordinated objects, not
    contradictions ("ĉiuj havu rajton aliron" — both rajt and alir are
    objects of the same `havu`).
    """
    subj: str
    rel: str
    obj: str
    source: str
    span: tuple[int, int]   # char span in the original text
    clause_idx: int = -1    # -1 for suffix claims, which aren't clause-bound
    confidence: float = 1.0


def _tok_stem(tok: Token) -> str:
    """Full stem (prefix + root + suffixes) of a token, lowercased.

    Normalizes across case/number (strips grammatical ending) while
    preserving lexical content — `ĉefurbo`/`ĉefurbon` → `ĉefurb`,
    `lernejestroj` → `lernejestr`.
    """
    if tok.root:
        stem = "".join(tok.prefixes) + tok.root + "".join(tok.suffixes)
        if stem:
            return stem
    # Fallback: strip grammatical ending from raw_lower
    lower = tok.raw_lower
    stem, _, _, _ = _strip_ending(lower)
    return stem if len(stem) >= 2 else lower


def _np_text(np: NP) -> str:
    return _tok_stem(np.head)


def _tok_text(tok: Token) -> str:
    return _tok_stem(tok)


def extract_claims(text: str) -> list[Claim]:
    """Extract structural claims from text using the verifier's parser output.

    Returns a list of `Claim` objects — one per (copula, transitive verb,
    PP-relation, derivational suffix) pattern the parser can identify.
    All subject/object strings are lowercase surface forms.
    """
    tokens = tokenize(text)
    nps = extract_nps(tokens)
    clauses = extract_clauses(tokens, nps)
    out: list[Claim] = []

    # --- 1. Copula claims (S estas P) --------------------------------
    for ci, cl in enumerate(clauses):
        v = cl.verb
        if v is None or v.root != "est" or v.suffixes:
            continue
        subj = cl.subject
        if subj is None:
            continue
        # Find predicate, preferring a head NOUN over a pre-nominal adjective.
        # "Beethoven estis fama komponisto" → predicate is komponisto, not fama.
        pred_tok = None
        first_adj = None
        for k in range(v.idx + 1, cl.end_idx):
            t = tokens[k]
            if t.has_comma_before and k > v.idx + 1:
                break
            if t.pos == "A" and first_adj is None:
                first_adj = t
                continue
            if t.pos == "N":
                pred_tok = t
                break
            if t.pos in ("V", "INF", "Prep"):
                break
        if pred_tok is None:
            pred_tok = first_adj
        if pred_tok is None:
            continue
        out.append(Claim(
            subj=_np_text(subj),
            rel="IS",
            obj=_tok_text(pred_tok),
            source="copula",
            span=(subj.head.char_span[0], pred_tok.char_span[1]),
            clause_idx=ci,
        ))

    # --- 2. Transitive claims (S V-as O-acc) --------------------------
    # Emit a claim for every accusative NP that follows the verb, up to
    # the next PP or clause boundary. Handles apposition like
    # "Francio havas ĉefurbon Parizon" → both (francio, hav, ĉefurb)
    # and (francio, hav, pariz).
    for ci, cl in enumerate(clauses):
        v = cl.verb
        if v is None or v.root == "est":
            continue
        subj = cl.subject
        if subj is None:
            continue
        for np in cl.nps:
            if np.start_idx <= v.idx:
                continue
            if np.case != "acc":
                continue
            if any(tokens[k].pos == "Prep" for k in range(v.idx + 1, np.start_idx)):
                break
            out.append(Claim(
                subj=_np_text(subj),
                rel=v.root or v.raw_lower,
                obj=_np_text(np),
                source="transitive",
                span=(subj.head.char_span[0], np.head.char_span[1]),
                clause_idx=ci,
            ))

    # --- 3. PP-relation claims (NP1 prep NP2) --------------------------
    # For every Prep, emit a relation. The left anchor is either:
    #   (a) the NP immediately preceding the Prep ("ĉefurbo de Francio")
    #   (b) the subject of the verb that precedes the Prep ("Beethoven
    #       mortis en Vieno") when no NP intervenes
    pp_inside = _compute_pp_inside(tokens, nps)
    # Map token idx → containing clause index
    tok_clause = {}
    for ci, cl in enumerate(clauses):
        for t in cl.tokens:
            tok_clause[t.idx] = ci
    for i, tok in enumerate(tokens):
        if tok.pos != "Prep":
            continue
        right_np = next((np for np in nps if np.start_idx == i + 1), None)
        if right_np is None:
            continue
        ci = tok_clause.get(i, -1)
        # (a) NP-anchor
        left_np = next((np for np in nps if np.end_idx == i - 1), None)
        if left_np is not None and left_np.head.idx not in pp_inside:
            out.append(Claim(
                subj=_np_text(left_np),
                rel=tok.raw_lower,
                obj=_np_text(right_np),
                source="pp-relation",
                span=(left_np.head.char_span[0], right_np.head.char_span[1]),
                clause_idx=ci,
            ))
            continue
        # (b) Verb-subject anchor: "Beethoven mortis en Vieno"
        if i - 1 >= 0 and tokens[i - 1].pos == "V":
            v = tokens[i - 1]
            cl = next((c for c in clauses if c.verb is v), None)
            if cl is not None and cl.subject is not None:
                out.append(Claim(
                    subj=_np_text(cl.subject),
                    rel=f"{v.root}+{tok.raw_lower}" if v.root else tok.raw_lower,
                    obj=_np_text(right_np),
                    source="pp-relation",
                    span=(cl.subject.head.char_span[0], right_np.head.char_span[1]),
                    confidence=0.8,
                    clause_idx=ci,
                ))

    # NOTE: suffix-relation claims (productive morphology, e.g. -ist =
    # role-of) were previously emitted here but removed — they're
    # tautological with the spelling and don't carry any signal beyond
    # what stem comparison already gives. The SUFFIX_RELATIONS table is
    # kept above for use by stem-aliased matching if needed.

    return out


def _normalize_x_system(s: str) -> str:
    """Normalize Zamenhof x-system substitutions (cx/gx/hx/jx/sx/ux) to
    ĉ/ĝ/ĥ/ĵ/ŝ/ŭ so that spelling variants canonicalize."""
    return (s.replace("cx", "ĉ").replace("gx", "ĝ").replace("hx", "ĥ")
             .replace("jx", "ĵ").replace("sx", "ŝ").replace("ux", "ŭ"))


def claim_tuples(text: str) -> set[tuple[str, str, str]]:
    """Return the set of (subj, rel, obj) tuples, with x-system normalized."""
    out = set()
    for c in extract_claims(text):
        out.add((_normalize_x_system(c.subj),
                 _normalize_x_system(c.rel),
                 _normalize_x_system(c.obj)))
    return out


def claim_overlap(generated: str, reference: str) -> float:
    """Fraction of reference claims that appear in the generation.

    Treats the copula `IS` relation as bidirectional — "Parizo IS ĉefurbo"
    and "ĉefurbo IS Parizo" are semantically equivalent.

    Usable as a QA reward — ranges from 0.0 (no claim match) to 1.0
    (every gold claim recovered).
    """
    ref = claim_tuples(reference)
    gen = claim_tuples(generated)
    if not ref:
        return 0.0
    matched = 0
    for s, r, o in ref:
        if (s, r, o) in gen:
            matched += 1
        elif r == "IS" and (o, r, s) in gen:
            matched += 1
    return matched / len(ref)


def claim_entity_pairs(text: str) -> set[tuple[str, str]]:
    """Return unordered (subj, obj) pairs from the text's claims.

    Weaker but more permissive than `claim_tuples` — paraphrases with
    different verb structure (e.g. "Parizo estas la ĉefurbo de Francio"
    vs "Francio havas ĉefurbon Parizon") still share the same entity
    pairs. Suitable as a paraphrase-tolerant QA reward.
    """
    pairs = set()
    for c in extract_claims(text):
        a = _normalize_x_system(c.subj)
        b = _normalize_x_system(c.obj)
        if a != b:
            pairs.add(tuple(sorted([a, b])))
    return pairs


def claim_entity_overlap(generated: str, reference: str) -> float:
    """Paraphrase-tolerant QA reward.

    Fraction of gold entity pairs (from claim subjects/objects) that
    appear in the generation. Softer than `claim_overlap`, which also
    requires the relation to match.
    """
    ref = claim_entity_pairs(reference)
    gen = claim_entity_pairs(generated)
    if not ref:
        return 0.0
    return len(ref & gen) / len(ref)


def claim_contradictions(text: str) -> list[tuple[str, str, str, str]]:
    """Return contradictions across different clauses.

    A contradiction is two claims sharing (subj, rel) but with different
    obj values, **and originating from different clauses**. Multiple
    objects within one clause are coordinated/appositive ("ĉiuj havu
    rajton aliron" = "all should have right and access") and not
    contradictory.

    Returns tuples (subj, rel, obj_a, obj_b).
    """
    # Map (subj, rel) → list of (clause_idx, obj). A pair of entries with
    # different objs AND different clause_idx is a contradiction.
    by_key: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for c in extract_claims(text):
        subj = _normalize_x_system(c.subj)
        rel = _normalize_x_system(c.rel)
        obj = _normalize_x_system(c.obj)
        by_key.setdefault((subj, rel), []).append((c.clause_idx, obj))

    bad: list[tuple[str, str, str, str]] = []
    for (subj, rel), entries in by_key.items():
        # Group objs by clause; cross-clause objs are candidates for contradiction
        clause_objs: dict[int, set[str]] = {}
        for ci, obj in entries:
            clause_objs.setdefault(ci, set()).add(obj)
        # Compare distinct clauses pairwise
        clauses_with_objs = [(ci, objs) for ci, objs in clause_objs.items() if ci >= 0]
        for i in range(len(clauses_with_objs)):
            for j in range(i + 1, len(clauses_with_objs)):
                ci_a, objs_a = clauses_with_objs[i]
                ci_b, objs_b = clauses_with_objs[j]
                # If neither clause's objs are a subset of the other's,
                # there's at least one mutually-exclusive obj.
                only_a = objs_a - objs_b
                only_b = objs_b - objs_a
                if only_a and only_b:
                    bad.append((subj, rel, sorted(only_a)[0], sorted(only_b)[0]))
                    break
            else:
                continue
            break
    return bad


# ---- Verifier pipeline ------------------------------------------------

class Verifier:
    """Composable pipeline of checks."""
    def __init__(self, checks: Iterable[Check] | None = None):
        self.checks: list[Check] = list(checks) if checks is not None else DEFAULT_CHECKS

    def verify(self, text: str) -> list[Diagnostic]:
        tokens = tokenize(text)
        nps = extract_nps(tokens)
        clauses = extract_clauses(tokens, nps)
        diags: list[Diagnostic] = []
        for chk in self.checks:
            # Pass clauses if the check accepts a third arg
            try:
                diags.extend(chk.check(tokens, nps, clauses))
            except TypeError:
                diags.extend(chk.check(tokens, nps))
        return diags


DEFAULT_CHECKS: list[Check] = [
    NPAgreement(),
    PredicateAdjAgreement(),
    ParticipleAgreement(),
    MissingAccusative(),
    IntransitiveAccusative(),
    PrepositionCase(),
    WrongEndings(),
    AffixCompat(),
]


# ---- Optional lexicon check (for reward models, confabulation detection) ----

_STEM_FREQ_PATH = Path(__file__).resolve().parent.parent.parent / "resources" / "stem_freq.json"
_STEM_FREQ: dict[str, int] | None = None


def _load_stem_freq() -> dict[str, int]:
    global _STEM_FREQ
    if _STEM_FREQ is not None:
        return _STEM_FREQ
    if _STEM_FREQ_PATH.exists():
        import json
        with open(_STEM_FREQ_PATH) as f:
            _STEM_FREQ = json.load(f)
    else:
        _STEM_FREQ = {}
    return _STEM_FREQ


from functools import lru_cache as _lru_cache


def _pieces_all_known(pieces: list[str]) -> bool:
    roots = get_roots()
    pres = get_prefixes()
    sufs = get_suffixes()
    for p in pieces:
        if p in roots or p in pres or p in sufs:
            continue
        if classify_morpheme(p) in ("ending", "particle"):
            continue
        return False
    return True


# Graduated frequency thresholds by stem length. Short stems accumulate
# false positives from morphological artifacts (e.g. "ti" is the stripped
# pronoun stem with freq ~336k; "jd" is typo noise with freq ~170) so we
# demand a high bar for them, but rare real compounds like `kvantumfizik`
# can pass with single-digit freq.
_FREQ_THRESHOLDS = {2: 10**9, 3: 1000}  # 2-char: never accept on freq alone


@_lru_cache(maxsize=50_000)
def _is_known_no_freq(stem: str, freq_threshold: int) -> bool:
    """Stricter validity: vortaro or compound decomposition only (no freq).

    Used for hyphenated-part validation where we don't want a high corpus
    frequency (e.g. from English-in-Wikipedia) to validate non-Esperanto
    pieces.
    """
    if len(stem) < 2:
        return True
    roots = get_roots()
    if stem in roots:
        return True
    pieces = _morph_decompose(stem)
    if _pieces_all_known(pieces):
        return True
    for i in range(2, len(stem) - 1):
        left, right = stem[:i], stem[i:]
        if len(left) < 2 or len(right) < 2:
            continue
        if _is_known_no_freq(left, freq_threshold) and _is_known_no_freq(right, freq_threshold):
            return True
        if left.endswith("o") and len(left) > 2:
            l2 = left[:-1]
            if len(l2) >= 2 and _is_known_no_freq(l2, freq_threshold) \
                    and _is_known_no_freq(right, freq_threshold):
                return True
    return False


@_lru_cache(maxsize=200_000)
def is_known_stem(stem: str, freq_threshold: int = 3) -> bool:
    """Return True if `stem` is a recognizable Esperanto word part.

    Validity tiers:
    1. Vortaro root (curated).
    2. Corpus frequency ≥ length-graduated threshold.
    3. Morphological decomposition into all-known pieces.
    4. Recursive compound split (with optional `-o-` linker).
    5. Hyphenated compound: each piece is individually known. (Only the
       last piece is a stem; earlier pieces retain their word endings,
       so we try each piece both as-is and with one trailing vowel removed.)
    """
    if len(stem) < 2:
        return True
    # Hyphenated compounds: "tako-ŝelo" → stem "tako-ŝel" → ["tako", "ŝel"].
    # Each piece must be individually Esperanto — we require vortaro or
    # compound-decomposition evidence (not freq-only), because English
    # words picked up from our Wikipedia corpus could otherwise validate
    # hyphenated English phrases like "arcade-style".
    if "-" in stem:
        parts = [p for p in stem.split("-") if p]
        if len(parts) >= 2:
            def _part_known(p: str) -> bool:
                for candidate in (p, p[:-1] if len(p) >= 3 and p[-1] in "oaeuin" else None,
                                  p[:-2] if len(p) >= 4 and p[-2:] in ("on", "an", "en", "un",
                                                                         "oj", "aj") else None):
                    if candidate is None:
                        continue
                    if _is_known_no_freq(candidate, freq_threshold):
                        return True
                return False
            return all(_part_known(p) for p in parts)
    roots = get_roots()
    if stem in roots:
        return True
    # Length-graduated frequency threshold.
    thr = _FREQ_THRESHOLDS.get(len(stem), freq_threshold)
    freq = _load_stem_freq().get(stem, 0)
    if freq >= thr:
        return True
    pieces = _morph_decompose(stem)
    if _pieces_all_known(pieces):
        return True
    # Recursive compound splits (with optional -o- linker). Require each
    # piece to be at least 2 chars — single-letter fragments are never a
    # valid compound member.
    for i in range(2, len(stem) - 1):
        left, right = stem[:i], stem[i:]
        if len(left) < 2 or len(right) < 2:
            continue
        if is_known_stem(left, freq_threshold) and is_known_stem(right, freq_threshold):
            return True
        if left.endswith("o") and len(left) > 2:
            l2 = left[:-1]
            if len(l2) >= 2 and is_known_stem(l2, freq_threshold) \
                    and is_known_stem(right, freq_threshold):
                return True
    return False


class LexiconCheck:
    """Flag words whose stem isn't recognizable as Esperanto.

    Non-default check. Intended for reward-signal use (GRPO) or auditing
    model generations for confabulated vocabulary.

    Skips proper nouns, closed-class words, and very short stems. Uses a
    three-tier validity signal: vortaro membership, corpus frequency, and
    recursive compound splitting.
    """
    name = "lexicon"

    def __init__(self, freq_threshold: int = 3):
        self.freq_threshold = freq_threshold

    def check(self, tokens, nps):
        out = []
        for tok in tokens:
            if tok.pos not in ("N", "A", "V", "INF", "Adv"):
                continue
            if getattr(tok, "is_proper", False):
                continue
            stem = tok.root
            if not stem or len(stem) < 3:
                continue
            if not is_known_stem(stem, self.freq_threshold):
                out.append(Diagnostic(
                    self.name, "warning",
                    f"'{tok.text}' → stem '{stem}' not recognized as Esperanto "
                    f"(not in vortaro, freq<{self.freq_threshold}, no compound split)",
                    tok.idx))
        return out


def unknown_word_rate(text: str, freq_threshold: int = 3) -> float:
    """Return the fraction of content words whose stem is unknown.

    Convenient for reward signals — use `-unknown_word_rate(text)` as a
    penalty term. Counts only N / A / V / INF / Adv tokens, excluding
    proper nouns and stems shorter than 3 characters.
    """
    tokens = tokenize(text)
    total = 0
    unknown = 0
    for tok in tokens:
        if tok.pos not in ("N", "A", "V", "INF", "Adv"):
            continue
        if getattr(tok, "is_proper", False):
            continue
        stem = tok.root
        if not stem or len(stem) < 3:
            continue
        total += 1
        if not is_known_stem(stem, freq_threshold):
            unknown += 1
    return unknown / total if total else 0.0
