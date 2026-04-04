"""Wikidata factoid template system for generating Esperanto training text."""

import random
from collections import defaultdict
from dataclasses import dataclass


# --- Property whitelist grouped by entity type ---

PERSON_PROPERTIES = {
    "okupo", "ŝtataneco", "naskiĝloko", "mortloko", "naskiĝdato",
    "mortdato", "edz(in)o", "infano", "patro", "patrino", "familia nomo",
    "antaŭnomo", "lingvo uzata", "honorigo", "membro de", "lernejo",
    "partio", "religio",
}

PLACE_PROPERTIES = {
    "lando", "komuna limo kun", "troviĝas en administra unuo",
    "ĝemelurbo", "loĝantaro", "horzono", "lingvo uzata", "parto de",
    "havas parton", "plej alta punkto", "ĉefurbo", "oficialaj lingvoj",
}

GENERAL_PROPERTIES = {
    "estas", "parto de", "havas parton", "membro de", "lando",
    "lingvo uzata", "honorigo", "komuna limo kun",
    "troviĝas en administra unuo",
}

ALL_USEFUL_PROPERTIES = PERSON_PROPERTIES | PLACE_PROPERTIES | GENERAL_PROPERTIES

# --- Entity type detection ---

PERSON_INDICATORS = {"homo", "persono", "sportisto", "politikisto", "aktoro",
                     "verkisto", "muzikisto", "sciencisto", "huamn", "human",
                     "ino", "viro", "ludanto", "reĝo", "prezidento"}
PLACE_INDICATORS = {"urbo", "komunumo", "lando", "insulo", "provinco",
                    "distrikto", "regiono", "vilaĝo", "lago", "rivero", "monto",
                    "ŝtato", "kontinento", "kantono"}
FEMALE_INDICATORS = {"ino", "aktorino", "kantistino", "reĝino", "princino",
                     "sportistino", "verkistino", "dancistino"}

TRIVIAL_INSTANCE_OF = {"homo", "human", "huamn", "persono"}

# Properties where the continuation should use pronouns, not full names
PRONOUN_PROPERTIES = {
    "okupo", "ŝtataneco", "naskiĝloko", "mortloko", "membro de",
    "honorigo", "lernejo", "partio", "religio", "lando", "estas",
    "troviĝas en administra unuo", "komuna limo kun", "ĝemelurbo",
    "parto de",
}

# Properties where the template references {subj} by name (possessive constructions)
NAME_PROPERTIES = {
    "patro", "patrino", "edz(in)o", "infano", "plej alta punkto",
    "ĉefurbo", "antaŭnomo", "familia nomo",
}


# --- Sentence templates ---

@dataclass
class Template:
    pattern: str
    needs_pronoun: bool = True


OPENING_TEMPLATES = {
    "estas": [
        Template("{subj} estas {obj}.", needs_pronoun=False),
    ],
    "okupo": [
        Template("{subj} estas {obj}.", needs_pronoun=False),
        Template("{subj} estas {obj} laŭ profesio.", needs_pronoun=False),
    ],
    "lando": [
        Template("{subj} troviĝas en {obj}.", needs_pronoun=False),
        Template("{subj} estas en {obj}.", needs_pronoun=False),
    ],
    "troviĝas en administra unuo": [
        Template("{subj} troviĝas en {obj}.", needs_pronoun=False),
        Template("{subj} situas en {obj}.", needs_pronoun=False),
    ],
    "ŝtataneco": [
        Template("{subj} estas ŝtatano de {obj}.", needs_pronoun=False),
        Template("{subj} devenas el {obj}.", needs_pronoun=False),
    ],
    "naskiĝloko": [
        Template("{subj} naskiĝis en {obj}.", needs_pronoun=False),
    ],
}

CONTINUATION_TEMPLATES = {
    "estas": [
        Template("{pron} estas {obj}."),
        Template("{pron} estas ankaŭ {obj}."),
    ],
    "okupo": [
        Template("{pron} estas {obj} laŭ profesio."),
        Template("{pron} estas {obj}."),
    ],
    "lando": [
        Template("{pron} troviĝas en {obj}."),
        Template("{pron} situas en la lando {obj}."),
    ],
    "troviĝas en administra unuo": [
        Template("{pron} troviĝas en {obj}."),
        Template("{pron} apartenas al {obj}."),
    ],
    "ŝtataneco": [
        Template("{pron} estas ŝtatano de {obj}."),
    ],
    "naskiĝloko": [
        Template("{pron} naskiĝis en {obj}."),
    ],
    "mortloko": [
        Template("{pron} mortis en {obj}."),
        Template("{pron} forpasis en {obj}."),
    ],
    "komuna limo kun": [
        Template("{pron} limas kun {obj}."),
        Template("{pron} havas komunan limon kun {obj}."),
    ],
    "ĝemelurbo": [
        Template("{pron} estas ĝemelurbo de {obj}."),
        Template("{pron} estas ĝemeligita kun {obj}."),
    ],
    "lingvo uzata": [
        Template("oni parolas la {obj}n en {subj}."),
        Template("la {obj} estas parolata en {subj}."),
        Template("{pron} uzas la {obj}n."),
    ],
    "membro de": [
        Template("{pron} estas membro de {obj}."),
        Template("{pron} membras en {obj}."),
    ],
    "parto de": [
        Template("{pron} estas parto de {obj}."),
        Template("{pron} apartenas al {obj}."),
    ],
    "havas parton": [
        Template("{obj} estas parto de {subj}."),
    ],
    "honorigo": [
        Template("{pron} ricevis la honoron {obj}."),
        Template("{pron} estis premiita per {obj}."),
    ],
    "edz(in)o": [
        Template("{pron} edziniĝis kun {obj}."),
    ],
    "infano": [
        Template("{pron} havas infanon nomatan {obj}."),
        Template("{obj} estas infano de {subj}."),
    ],
    "patro": [
        Template("la patro de {subj} estas {obj}."),
    ],
    "patrino": [
        Template("la patrino de {subj} estas {obj}."),
    ],
    "plej alta punkto": [
        Template("la plej alta punkto de {subj} estas {obj}."),
    ],
    "ĉefurbo": [
        Template("la ĉefurbo estas {obj}."),
        Template("{pron} havas la ĉefurbon {obj}."),
    ],
    "partio": [
        Template("{pron} estas ano de la partio {obj}."),
        Template("{pron} apartenas al la partio {obj}."),
    ],
    "lernejo": [
        Template("{pron} studis ĉe {obj}."),
        Template("{pron} lernis ĉe {obj}."),
    ],
    "religio": [
        Template("{pron} praktikas la religion {obj}."),
    ],
    "antaŭnomo": [
        Template("la antaŭnomo de {subj} estas {obj}."),
    ],
    "familia nomo": [
        Template("la familia nomo de {subj} estas {obj}."),
    ],
}

DEFAULT_CONTINUATION = [
    Template("la {prop} de {subj} estas {obj}."),
]


# --- Connectors ---

ADDITIVE_CONNECTORS = [
    "Krome, ", "Ankaŭ ", "Plue, ", "Cetere, ",
    "Plie, ", "Menciindas ke ", "Notinde, ",
    "Indas mencii ke ", "",
]


# --- Semantic ordering ---

PROPERTY_ORDER = {
    "estas": 0,
    "okupo": 1,
    "lando": 10,
    "troviĝas en administra unuo": 11,
    "parto de": 12,
    "ĉefurbo": 13,
    "ŝtataneco": 20,
    "naskiĝloko": 21,
    "lernejo": 22,
    "partio": 24,
    "membro de": 25,
    "honorigo": 26,
    "patro": 30,
    "patrino": 31,
    "edz(in)o": 32,
    "infano": 33,
    "mortloko": 40,
    "lingvo uzata": 50,
    "loĝantaro": 51,
    "horzono": 52,
    "plej alta punkto": 53,
    "komuna limo kun": 60,
    "ĝemelurbo": 61,
    "havas parton": 62,
    "antaŭnomo": 70,
    "familia nomo": 71,
}


def detect_entity_type(facts: list[dict]) -> str:
    for fact in facts:
        if fact["property"] == "estas":
            val = fact["value"].lower()
            if any(ind in val for ind in PERSON_INDICATORS):
                return "person"
            if any(ind in val for ind in PLACE_INDICATORS):
                return "place"
    fact_props = {f["property"] for f in facts}
    if fact_props & {"naskiĝloko", "mortloko", "okupo", "ŝtataneco"}:
        return "person"
    if fact_props & {"komuna limo kun", "ĝemelurbo", "loĝantaro"}:
        return "place"
    return "other"


def get_pronoun(entity_type: str, facts: list[dict]) -> str:
    if entity_type == "person":
        for fact in facts:
            val = fact["value"].lower()
            if fact["property"] == "estas" and any(f in val for f in FEMALE_INDICATORS):
                return "ŝi"
            if fact["property"] == "okupo" and any(f in val for f in FEMALE_INDICATORS):
                return "ŝi"
        return "li"
    return "ĝi"


def _is_likely_english(text: str) -> bool:
    english_words = {"the ", " of ", " and ", " in ", " for ", " with ", " from "}
    lower = text.lower()
    return any(w in lower for w in english_words)


MAX_VALUE_LENGTH = 80  # Skip values longer than this (e.g. long award names)


def _looks_broken(text: str) -> bool:
    """Detect obviously broken values from Wikidata."""
    # Double letters at end (e.g. "Napoleonn")
    if len(text) > 2 and text[-1] == text[-2] and text[-1].isalpha():
        return True
    # Too short to be meaningful
    if len(text) < 2:
        return True
    # Too long — torpedoes readability
    if len(text) > MAX_VALUE_LENGTH:
        return True
    return False


def _is_trivial(fact: dict, entity_type: str) -> bool:
    if fact["property"] == "estas":
        val = fact["value"].lower()
        if entity_type == "person" and any(t in val for t in TRIVIAL_INSTANCE_OF):
            return True
    return False


def _strip_lingvo_suffix(value: str) -> str:
    """Remove trailing ' lingvo' from language names to avoid redundancy."""
    if value.endswith(" lingvo"):
        return value[:-7]
    if value.endswith(" Lingvo"):
        return value[:-7]
    return value


def filter_facts(facts: list[dict], entity_label: str = "",
                 entity_type: str = "other") -> list[dict]:
    seen = {}
    entity_lower = entity_label.lower()
    for fact in facts:
        prop = fact["property"]
        if prop not in ALL_USEFUL_PROPERTIES:
            continue
        if prop in seen:
            continue
        if _is_likely_english(fact["value"]):
            continue
        if _looks_broken(fact["value"]):
            continue
        if fact["value"].lower() == entity_lower:
            continue
        if _is_trivial(fact, entity_type):
            continue
        # Clean up language names
        if prop == "lingvo uzata":
            fact = {**fact, "value": _strip_lingvo_suffix(fact["value"])}
        seen[prop] = fact
    return list(seen.values())


def _capitalize_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def render_sentence(template: Template, subj: str, obj: str,
                    pron: str, prop: str) -> str:
    return template.pattern.format(
        subj=subj, obj=obj, pron=pron, prop=prop
    )


def _sort_facts(facts: list[dict]) -> list[dict]:
    return sorted(facts, key=lambda f: PROPERTY_ORDER.get(f["property"], 99))


def _pick_connector(used_connectors: set) -> str:
    """Pick a connector not yet used in this paragraph."""
    available = [c for c in ADDITIVE_CONNECTORS if c not in used_connectors]
    if not available:
        # All used — fall back to empty connector
        return ""
    return random.choice(available)


def generate_paragraph(entity_label: str, facts: list[dict],
                       min_sentences: int = 3, max_sentences: int = 5) -> str | None:
    entity_type = detect_entity_type(facts)
    useful_facts = filter_facts(facts, entity_label, entity_type)
    if len(useful_facts) < 2:
        return None

    pron = get_pronoun(entity_type, facts)
    pron_cap = pron[0].upper() + pron[1:]
    label = _capitalize_first(entity_label)

    n_sentences = min(random.randint(min_sentences, max_sentences), len(useful_facts))

    sorted_facts = _sort_facts(useful_facts)

    # Prefer facts with opening templates for the first position
    opening_facts = [f for f in sorted_facts if f["property"] in OPENING_TEMPLATES]

    if opening_facts:
        first_fact = opening_facts[0]
        remaining_pool = [f for f in sorted_facts if f is not first_fact]
    else:
        first_fact = sorted_facts[0]
        remaining_pool = sorted_facts[1:]

    remaining_count = min(n_sentences - 1, len(remaining_pool))
    selected_remaining = random.sample(remaining_pool, remaining_count)
    selected_remaining = _sort_facts(selected_remaining)

    sentences = []

    # --- Opening sentence: always uses entity name ---
    prop = first_fact["property"]
    templates = OPENING_TEMPLATES.get(prop, [])
    if not templates:
        templates = CONTINUATION_TEMPLATES.get(prop, DEFAULT_CONTINUATION)
    template = random.choice(templates)
    sentence = render_sentence(template, label, first_fact["value"],
                               pron_cap, prop)
    sentences.append(sentence)
    name_used_in_opening = True

    # --- Continuation sentences ---
    used_connectors: set[str] = set()

    for i, fact in enumerate(selected_remaining):
        prop = fact["property"]
        templates = CONTINUATION_TEMPLATES.get(prop, DEFAULT_CONTINUATION)
        template = random.choice(templates)

        connector = _pick_connector(used_connectors)
        used_connectors.add(connector)

        if connector:
            use_pron = pron
        else:
            use_pron = pron_cap

        # Use pronoun for most properties; templates with {subj} in
        # NAME_PROPERTIES will use the entity name via the template itself
        sentence = render_sentence(template, label, fact["value"],
                                   use_pron, prop)

        if not connector:
            sentence = _capitalize_first(sentence)

        sentences.append(connector + sentence)

    return " ".join(sentences)


def generate_variants(entity_label: str, facts: list[dict],
                      n_variants: int = 3) -> list[str]:
    results = []
    for _ in range(n_variants):
        paragraph = generate_paragraph(entity_label, facts)
        if paragraph:
            results.append(paragraph)
    return results


# --- Cross-entity comparison templates ---

# Properties suitable for comparison between entities of the same type
COMPARABLE_PROPERTIES = {
    # Places
    "lando", "troviĝas en administra unuo", "ĉefurbo", "lingvo uzata",
    "komuna limo kun", "parto de", "horzono",
    # People
    "okupo", "ŝtataneco", "naskiĝloko", "mortloko", "lernejo",
    "membro de", "honorigo",
    # General
    "estas",
}

# Templates for comparing two entities on the same property
COMPARISON_SAME = {
    "okupo": [
        "Kaj {a} kaj {b} estas {val}.",
        "Ambaŭ {a} kaj {b} laboras kiel {val}.",
        "{a} estas {val}, same kiel {b}.",
    ],
    "ŝtataneco": [
        "Kaj {a} kaj {b} devenas el {val}.",
        "Ambaŭ {a} kaj {b} estas ŝtatanoj de {val}.",
        "{a} kaj {b} ambaŭ devenas el {val}.",
    ],
    "lando": [
        "Kaj {a} kaj {b} troviĝas en {val}.",
        "Ambaŭ {a} kaj {b} situas en {val}.",
    ],
    "estas": [
        "Kaj {a} kaj {b} estas {val}.",
        "Ambaŭ {a} kaj {b} estas {val}.",
    ],
    "membro de": [
        "Kaj {a} kaj {b} estas membroj de {val}.",
        "Ambaŭ {a} kaj {b} membras en {val}.",
    ],
    "lingvo uzata": [
        "Kaj en {a} kaj en {b} oni parolas la {val}n.",
        "La {val} estas parolata kaj en {a} kaj en {b}.",
    ],
}

# Templates for contrasting two entities on the same property
COMPARISON_DIFF = {
    "okupo": [
        "{a} estas {val_a}, dum {b} estas {val_b}.",
        "{a} estas {val_a} laŭ profesio, dum {b} estas {val_b}.",
    ],
    "ŝtataneco": [
        "{a} devenas el {val_a}, dum {b} devenas el {val_b}.",
        "{a} estas ŝtatano de {val_a}, sed {b} estas ŝtatano de {val_b}.",
    ],
    "naskiĝloko": [
        "{a} naskiĝis en {val_a}, dum {b} naskiĝis en {val_b}.",
        "{a} naskiĝis en {val_a}, sed {b} en {val_b}.",
    ],
    "mortloko": [
        "{a} mortis en {val_a}, dum {b} mortis en {val_b}.",
    ],
    "lando": [
        "{a} troviĝas en {val_a}, dum {b} troviĝas en {val_b}.",
        "{a} situas en {val_a}, sed {b} en {val_b}.",
    ],
    "ĉefurbo": [
        "La ĉefurbo de {a} estas {val_a}, dum la ĉefurbo de {b} estas {val_b}.",
        "{a} havas la ĉefurbon {val_a}, sed {b} havas {val_b}n.",
    ],
    "lingvo uzata": [
        "En {a} oni parolas la {val_a}n, dum en {b} oni parolas la {val_b}n.",
        "{a} uzas la {val_a}n, sed {b} uzas la {val_b}n.",
    ],
    "estas": [
        "{a} estas {val_a}, dum {b} estas {val_b}.",
    ],
    "lernejo": [
        "{a} studis ĉe {val_a}, dum {b} studis ĉe {val_b}.",
    ],
    "horzono": [
        "{a} uzas la horzonon {val_a}, dum {b} uzas {val_b}.",
    ],
    "membro de": [
        "{a} membras en {val_a}, dum {b} membras en {val_b}.",
    ],
    "honorigo": [
        "{a} ricevis la honoron {val_a}, dum {b} ricevis {val_b}.",
    ],
}

# Linking phrases between comparison sentences
COMPARISON_LINKS = [
    "Tamen, ", "Aliflanke, ", "Kompare, ",
    "Kontraste, ",
]

SIMILARITY_LINKS = [
    "Simile, ", "Same, ", "Ankaŭ ", "Kiel {a}, ",
]


def _get_fact_value(facts: list[dict], prop: str) -> str | None:
    """Get the value for a property from a fact list."""
    for fact in facts:
        if fact["property"] == prop:
            val = fact["value"]
            if prop == "lingvo uzata":
                val = _strip_lingvo_suffix(val)
            return val
    return None


def _get_instance_classes(facts: list[dict]) -> set[str]:
    """Get all 'estas' values for an entity (its classes)."""
    return {
        f["value"].lower() for f in facts
        if f["property"] == "estas" and not _is_likely_english(f["value"])
    }


def find_comparable_pairs(entities: list[dict]) -> list[tuple[dict, dict, list[str]]]:
    """Find pairs of entities that share comparable properties and class."""
    # Index entities by type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for entity in entities:
        etype = detect_entity_type(entity["facts"])
        by_type[etype].append(entity)

    pairs = []
    for etype, ents in by_type.items():
        if len(ents) < 2:
            continue
        # Sample pairs rather than doing all combinations
        n_pairs = min(len(ents) * 3, 10000)
        for _ in range(n_pairs):
            a, b = random.sample(ents, 2)

            # Require at least one shared instance_of class
            classes_a = _get_instance_classes(a["facts"])
            classes_b = _get_instance_classes(b["facts"])
            if not (classes_a & classes_b):
                continue

            # Find shared comparable properties (excluding "estas" itself
            # since we already know they share a class)
            a_props = {f["property"] for f in a["facts"]} & COMPARABLE_PROPERTIES
            b_props = {f["property"] for f in b["facts"]} & COMPARABLE_PROPERTIES
            shared = (a_props & b_props) - {"estas"}
            if len(shared) >= 2:
                pairs.append((a, b, list(shared)))

    return pairs


def generate_comparison(entity_a: dict, entity_b: dict,
                        shared_props: list[str],
                        min_comparisons: int = 2,
                        max_comparisons: int = 4) -> str | None:
    """Generate a comparison paragraph between two entities."""
    label_a = _capitalize_first(entity_a["label"])
    label_b = _capitalize_first(entity_b["label"])

    type_a = detect_entity_type(entity_a["facts"])
    facts_a = filter_facts(entity_a["facts"], entity_a["label"], type_a)
    facts_b = filter_facts(entity_b["facts"], entity_b["label"],
                           detect_entity_type(entity_b["facts"]))

    # Build comparison sentences
    n = min(random.randint(min_comparisons, max_comparisons), len(shared_props))
    selected_props = random.sample(shared_props, n)
    selected_props = sorted(selected_props,
                            key=lambda p: PROPERTY_ORDER.get(p, 99))

    sentences = []
    used_links: set[str] = set()

    for prop in selected_props:
        val_a = _get_fact_value(facts_a, prop)
        val_b = _get_fact_value(facts_b, prop)

        if val_a is None or val_b is None:
            continue
        if any(_is_likely_english(v) or _looks_broken(v)
               for v in (val_a, val_b)):
            continue

        is_first = len(sentences) == 0

        if val_a.lower() == val_b.lower():
            # Same value — use similarity template
            templates = COMPARISON_SAME.get(prop)
            if not templates:
                continue
            sentence = random.choice(templates).format(
                a=label_a, b=label_b, val=val_a,
            )
            if not is_first:
                available = [l for l in SIMILARITY_LINKS if l not in used_links]
                if available:
                    link = random.choice(available)
                    used_links.add(link)
                    link = link.format(a=label_a, b=label_b)
                    sentence = link + sentence
        else:
            # Different values — use contrast template
            templates = COMPARISON_DIFF.get(prop)
            if not templates:
                continue
            sentence = random.choice(templates).format(
                a=label_a, b=label_b, val_a=val_a, val_b=val_b,
            )
            if not is_first:
                # Don't add a link if the sentence already contains
                # a discourse word (from templates like "Kvankam...")
                discourse_words = ("Kvankam ", "Dum ", "Sed ", "Malgraŭ ")
                has_discourse = any(
                    sentence.startswith(w) or
                    f" {w.strip().lower()} " in f" {sentence.lower()} "
                    for w in discourse_words
                )
                if not has_discourse:
                    available = [l for l in COMPARISON_LINKS
                                 if l not in used_links]
                    if available:
                        link = random.choice(available)
                        used_links.add(link)
                        sentence = link + sentence

        sentences.append(sentence)

    if len(sentences) < 2:
        return None

    return " ".join(sentences)
