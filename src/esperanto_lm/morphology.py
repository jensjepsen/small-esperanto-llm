"""Rule-based Esperanto morpheme decomposer.

Root dictionary sourced from the parseo project:
https://github.com/rieselhilfe/parseo (vortaro.json)
Contains ~4,600 official Esperanto roots, prefixes, and suffixes
from the Akademia Vortaro / Universala Vortaro tradition.
"""

import json
from pathlib import Path

# --- Load official root dictionary ---

_VORTARO_PATH = Path(__file__).resolve().parent.parent.parent / "resources" / "vortaro.json"
_ROOTS: set[str] | None = None
_DICT_PREFIXES: set[str] | None = None
_DICT_SUFFIXES: set[str] | None = None


def _load_vortaro():
    global _ROOTS, _DICT_PREFIXES, _DICT_SUFFIXES
    if _ROOTS is not None:
        return

    if not _VORTARO_PATH.exists():
        _ROOTS = set()
        _DICT_PREFIXES = set()
        _DICT_SUFFIXES = set()
        return

    with open(_VORTARO_PATH) as f:
        d = json.load(f)

    _ROOTS = set()
    for group in d.get("radikoj", {}).values():
        for root in group:
            _ROOTS.add(root.lower())

    _DICT_PREFIXES = {p.lower() for p in d.get("prefiksoj", {})}
    _DICT_SUFFIXES = {s.lower() for s in d.get("sufiksoj", {})}


def get_roots() -> set[str]:
    _load_vortaro()
    return _ROOTS


def get_prefixes() -> set[str]:
    _load_vortaro()
    return _DICT_PREFIXES


def get_suffixes() -> set[str]:
    _load_vortaro()
    return _DICT_SUFFIXES


# --- Closed-class words that should not be decomposed ---

CORRELATIVES = {
    "kio", "kiu", "kia", "kiel", "kiam", "kie", "kien", "kial", "kiom", "kies",
    "tio", "tiu", "tia", "tiel", "tiam", "tie", "tien", "tial", "tiom", "ties",
    "io", "iu", "ia", "iel", "iam", "ie", "ien", "ial", "iom", "ies",
    "ĉio", "ĉiu", "ĉia", "ĉiel", "ĉiam", "ĉie", "ĉien", "ĉial", "ĉiom", "ĉies",
    "nenio", "neniu", "nenia", "neniel", "neniam", "nenie", "nenien", "nenial", "neniom", "nenies",
}

PRONOUNS = {
    "mi", "vi", "li", "ŝi", "ĝi", "ni", "ili", "oni", "si",
    "mia", "via", "lia", "ŝia", "ĝia", "nia", "ilia", "sia",
    "min", "vin", "lin", "ŝin", "ĝin", "nin", "ilin",
}

PREPOSITIONS = {
    "al", "anstataŭ", "antaŭ", "apud", "ĉe", "ĉirkaŭ", "da", "de",
    "dum", "ekster", "el", "en", "ĝis", "inter", "je", "kontraŭ",
    "kun", "laŭ", "malgraŭ", "per", "po", "por", "post", "preter",
    "pri", "pro", "sen", "sub", "super", "sur", "tra", "trans",
}

CONJUNCTIONS = {
    "kaj", "aŭ", "sed", "ĉar", "ke", "ĉu", "do", "ja", "jen",
    "jes", "ne", "nek", "nu", "ol", "plus", "se", "tamen",
}

PARTICLES = {
    "ajn", "almenaŭ", "ankaŭ", "ankoraŭ", "apenaŭ", "baldaŭ",
    "des", "eĉ", "jam", "ĵus", "kvazaŭ", "mem", "nur", "plej",
    "pli", "plu", "preskaŭ", "tre", "tro", "tuj",
}

NUMERALS = {
    "unu", "du", "tri", "kvar", "kvin", "ses", "sep", "ok", "naŭ",
    "dek", "cent", "mil",
}

ARTICLES = {"la"}

DO_NOT_DECOMPOSE = (
    CORRELATIVES | PRONOUNS | PREPOSITIONS | CONJUNCTIONS
    | PARTICLES | NUMERALS | ARTICLES
)

# Grammatical endings — atomic units only.
# Compound endings (oj, ajn, etc.) are split into separate tokens
# so the model learns o/a/e/i/u, j (plural), n (accusative) independently.
ENDINGS = [
    # Verb forms
    "as", "is", "os", "us",
    # Participles
    "anta", "inta", "onta",
    "ata", "ita", "ota",
    "ante", "inte", "onte",
    "ate", "ite", "ote",
    # Basic endings
    "o", "a", "e", "i", "u",
    # Plural and accusative as separate tokens
    "j", "n",
]

# Minimum root length after stripping
MIN_ROOT_LENGTH = 2


def classify_morpheme(morpheme: str) -> str:
    """Classify a morpheme as prefix, suffix, ending, or root."""
    prefixes = get_prefixes()
    suffixes = get_suffixes()
    if morpheme in prefixes:
        return "prefix"
    if morpheme in suffixes:
        return "suffix"
    if morpheme in ENDINGS:
        return "ending"
    if morpheme in DO_NOT_DECOMPOSE:
        return "particle"
    return "root"


def decompose_tagged(word: str) -> list[tuple[str, str]]:
    """Decompose a word and return (morpheme, type) pairs.

    Types: prefix, root, suffix, ending, particle
    """
    morphemes = decompose(word)
    if len(morphemes) == 1 and morphemes[0].lower() in DO_NOT_DECOMPOSE:
        return [(morphemes[0], "particle")]

    tagged = []
    for m in morphemes:
        tagged.append((m, classify_morpheme(m)))
    return tagged


def _try_split_stem(stem: str, roots: set[str], prefixes: set[str],
                    suffixes: set[str]) -> list[str] | None:
    """Try to split a stem into known roots, prefixes, and suffixes.

    Returns a list of morphemes if successful, None if the stem
    can't be fully decomposed into known parts.
    """
    if not stem:
        return []

    # Direct root match
    if stem in roots:
        return [stem]

    # Try prefix + remainder
    for prefix in sorted(prefixes, key=len, reverse=True):
        if stem.startswith(prefix) and len(stem) > len(prefix):
            rest = _try_split_stem(stem[len(prefix):], roots, prefixes, suffixes)
            if rest is not None:
                return [prefix] + rest

    # Try remainder + suffix
    for suffix in sorted(suffixes, key=len, reverse=True):
        if stem.endswith(suffix) and len(stem) > len(suffix):
            rest = _try_split_stem(stem[:-len(suffix)], roots, prefixes, suffixes)
            if rest is not None:
                return rest + [suffix]

    # Try splitting into two known roots (compound word)
    for i in range(MIN_ROOT_LENGTH, len(stem) - MIN_ROOT_LENGTH + 1):
        left = stem[:i]
        right = stem[i:]
        if left in roots:
            rest = _try_split_stem(right, roots, prefixes, suffixes)
            if rest is not None:
                return [left] + rest

    return None


def decompose(word: str) -> list[str]:
    """Decompose an Esperanto word into morphemes.

    Uses the official root dictionary to validate splits.
    Falls back to heuristic splitting for unknown words.

    Returns a list of morphemes, e.g.:
        "malbela" → ["mal", "bel", "a"]
        "lernejo" → ["lern", "ej", "o"]
        "gepatroj" → ["ge", "patr", "oj"]
        "malbonfarintoj" → ["mal", "bon", "far", "int", "oj"]
        "ĉefurbo" → ["ĉef", "urb", "o"]
    """
    lower = word.lower()

    # Don't decompose closed-class words
    if lower in DO_NOT_DECOMPOSE:
        return [lower]

    roots = get_roots()
    prefixes = get_prefixes()
    suffixes = get_suffixes()

    # Strip grammatical endings right to left: first n (accusative),
    # then j (plural), then the base ending (o/a/e/i/u or verb form)
    stem = lower
    trailing = []

    if stem.endswith("n") and len(stem) > MIN_ROOT_LENGTH + 1:
        trailing.append("n")
        stem = stem[:-1]

    if stem.endswith("j") and len(stem) > MIN_ROOT_LENGTH + 1:
        trailing.append("j")
        stem = stem[:-1]

    ending = None
    for e in ENDINGS:
        if e in ("j", "n"):
            continue  # already handled above
        if stem.endswith(e) and len(stem) - len(e) >= MIN_ROOT_LENGTH:
            ending = e
            stem = stem[:-len(e)]
            break

    # Try dictionary-based decomposition of the stem
    parts = _try_split_stem(stem, roots, prefixes, suffixes)

    if parts is not None:
        result = parts
        if ending:
            result.append(ending)
        result.extend(reversed(trailing))
        return result

    # Fallback: return as a single root (unknown word)
    result = [stem]
    if ending:
        result.append(ending)
    result.extend(reversed(trailing))
    return result


def decompose_text(text: str) -> list[str]:
    """Decompose a text into morphemes, preserving word boundaries."""
    words = text.split()
    result = []
    for word in words:
        # Strip punctuation
        clean = word.strip(".,;:!?\"'()[]{}–—-")
        if not clean:
            continue
        result.extend(decompose(clean))
    return result
