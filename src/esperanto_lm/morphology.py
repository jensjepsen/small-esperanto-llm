"""Rule-based Esperanto morpheme decomposer."""

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

# --- Affixes ---

PREFIXES = [
    "mal", "re", "ek", "ge", "dis", "mis", "bo", "pra",
    "fi", "eks", "ne", "sen",
]

SUFFIXES = [
    "ist", "ej", "il", "ig", "iĝ", "ul", "in", "ar",
    "ebl", "ind", "em", "eg", "et", "aĉ", "id", "an",
    "estr", "ism", "ec", "aĵ", "er", "uj", "ad", "ĉj", "nj",
    "end", "obl", "on", "op",
]

# Grammatical endings (order matters — check longer first)
ENDINGS = [
    # Accusative plural
    "ojn", "ajn",
    # Plural
    "oj", "aj",
    # Accusative
    "on", "an", "en", "in",
    # Verb forms
    "as", "is", "os", "us",
    "anta", "inta", "onta",
    "ata", "ita", "ota",
    "ante", "inte", "onte",
    "ate", "ite", "ote",
    # Basic endings
    "o", "a", "e", "i", "u",
    # Plural j (after noun/adj ending already stripped — shouldn't happen
    # but included for safety)
    "j", "n",
]

# Minimum root length after stripping
MIN_ROOT_LENGTH = 2


def classify_morpheme(morpheme: str) -> str:
    """Classify a morpheme as prefix, suffix, ending, or root."""
    if morpheme in [p for p in PREFIXES]:
        return "prefix"
    if morpheme in [s for s in SUFFIXES]:
        return "suffix"
    if morpheme in [e for e in ENDINGS]:
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


def decompose(word: str) -> list[str]:
    """Decompose an Esperanto word into morphemes.

    Returns a list of morphemes, e.g.:
        "malbela" → ["mal", "bel", "a"]
        "lernejo" → ["lern", "ej", "o"]
        "gepatroj" → ["ge", "patr", "o", "j"]
    """
    lower = word.lower()

    # Don't decompose closed-class words
    if lower in DO_NOT_DECOMPOSE:
        return [lower]

    morphemes = []
    remaining = lower

    # Strip prefixes
    found_prefix = True
    while found_prefix:
        found_prefix = False
        for prefix in PREFIXES:
            if remaining.startswith(prefix) and len(remaining) > len(prefix) + MIN_ROOT_LENGTH:
                morphemes.append(prefix)
                remaining = remaining[len(prefix):]
                found_prefix = True
                break

    # Strip ending
    ending = None
    for e in ENDINGS:
        if remaining.endswith(e) and len(remaining) - len(e) >= MIN_ROOT_LENGTH:
            ending = e
            remaining = remaining[:-len(e)]
            break

    # Strip suffixes from the remaining stem (right to left)
    suffixes_found = []
    found_suffix = True
    while found_suffix:
        found_suffix = False
        for suffix in SUFFIXES:
            if remaining.endswith(suffix) and len(remaining) - len(suffix) >= MIN_ROOT_LENGTH:
                suffixes_found.append(suffix)
                remaining = remaining[:-len(suffix)]
                found_suffix = True
                break

    # Build result: prefixes + root + suffixes (reversed) + ending
    morphemes.append(remaining)
    morphemes.extend(reversed(suffixes_found))
    if ending:
        morphemes.append(ending)

    return morphemes


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
