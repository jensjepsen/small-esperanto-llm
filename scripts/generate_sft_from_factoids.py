"""Generate SFT instruction/response pairs from Wikidata factoid entities.

Generates both single-turn Q&A and multi-turn conversations grounded
in real facts from the extracted Wikidata entities.
"""

import argparse
import json
import random
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from esperanto_lm.factoids import detect_entity_type

console = Console()

DEFAULT_INPUT = Path("/mnt/data2/wikidata5m/eo_factoids_v2/eo_factoids.jsonl")
DEFAULT_OUTPUT = Path("data/sft/sft_factoid.jsonl")

# Question templates per property: (question, answer, followup_question)
# followup_question is used in multi-turn when the entity is already established
QA_TEMPLATES = {
    "ĉefurbo": {
        "initial": [
            ("Kio estas la ĉefurbo de {entity}?", "La ĉefurbo de {entity} estas {value}."),
            ("Nomu la ĉefurbon de {entity}.", "La ĉefurbo de {entity} estas {value}."),
            ("Kiu urbo estas la ĉefurbo de {entity}?", "{value} estas la ĉefurbo de {entity}."),
        ],
        "followup": [
            ("Kio estas la ĉefurbo?", "La ĉefurbo estas {value}."),
            ("Kaj kiu estas la ĉefurbo?", "La ĉefurbo estas {value}."),
            ("Kaj la ĉefurbo?", "{value}."),
        ],
    },
    "lando": {
        "initial": [
            ("En kiu lando troviĝas {entity}?", "{entity} troviĝas en {value}."),
            ("Kie situas {entity}?", "{entity} situas en {value}."),
            ("Al kiu lando apartenas {entity}?", "{entity} apartenas al {value}."),
        ],
        "followup": [
            ("En kiu lando?", "En {value}."),
            ("Kie ĝi troviĝas?", "Ĝi troviĝas en {value}."),
            ("En kiu lando ĝi situas?", "En {value}."),
        ],
    },
    "okupo": {
        "initial": [
            ("Kio estas la profesio de {entity}?", "{entity} estas {value}."),
            ("Kiu estas {entity}?", "{entity} estas {value}."),
            ("Per kio {entity} okupiĝas?", "{entity} estas {value} laŭ profesio."),
        ],
        "followup": [
            ("Kio estas {poss} profesio?", "{pron_cap} estas {value}."),
            ("Kion {pron} faras?", "{pron_cap} estas {value}."),
            ("Kaj la profesio?", "{pron_cap} estas {value} laŭ profesio."),
        ],
    },
    "ŝtataneco": {
        "initial": [
            ("El kiu lando devenas {entity}?", "{entity} devenas el {value}."),
            ("Kio estas la ŝtataneco de {entity}?", "{entity} estas ŝtatano de {value}."),
        ],
        "followup": [
            ("El kiu lando {pron} devenas?", "{pron_cap} devenas el {value}."),
            ("Kaj la ŝtataneco?", "{pron_cap} estas ŝtatano de {value}."),
            ("El kiu lando?", "El {value}."),
        ],
    },
    "naskiĝloko": {
        "initial": [
            ("Kie naskiĝis {entity}?", "{entity} naskiĝis en {value}."),
            ("En kiu urbo naskiĝis {entity}?", "{entity} naskiĝis en {value}."),
        ],
        "followup": [
            ("Kie {pron} naskiĝis?", "{pron_cap} naskiĝis en {value}."),
            ("Kaj kie {pron} naskiĝis?", "En {value}."),
            ("La naskiĝloko?", "En {value}."),
        ],
    },
    "mortloko": {
        "initial": [
            ("Kie mortis {entity}?", "{entity} mortis en {value}."),
            ("En kiu urbo {entity} forpasis?", "{entity} forpasis en {value}."),
        ],
        "followup": [
            ("Kaj kie {pron} mortis?", "{pron_cap} mortis en {value}."),
            ("Kie {pron} forpasis?", "En {value}."),
        ],
    },
    "estas": {
        "initial": [
            ("Kio estas {entity}?", "{entity} estas {value}."),
            ("Priskribu {entity}n.", "{entity} estas {value}."),
        ],
        "followup": [],
    },
    "lingvo uzata": {
        "initial": [
            ("Kiun lingvon oni parolas en {entity}?", "En {entity} oni parolas la {value}n."),
            ("Kiu lingvo estas uzata en {entity}?", "La {value} estas uzata en {entity}."),
        ],
        "followup": [
            ("Kiun lingvon oni parolas tie?", "Oni parolas la {value}n."),
            ("Kaj la lingvo?", "La {value}."),
        ],
    },
    "komuna limo kun": {
        "initial": [
            ("Kun kiu lando limas {entity}?", "{entity} limas kun {value}."),
            ("Kiuj estas la najbaroj de {entity}?", "{entity} havas komunan limon kun {value}."),
        ],
        "followup": [
            ("Kun kiu lando ĝi limas?", "Ĝi limas kun {value}."),
            ("Kaj la najbaroj?", "Ĝi limas kun {value}."),
        ],
    },
    "parto de": {
        "initial": [
            ("De kio {entity} estas parto?", "{entity} estas parto de {value}."),
            ("Al kio apartenas {entity}?", "{entity} apartenas al {value}."),
        ],
        "followup": [
            ("De kio ĝi estas parto?", "Ĝi estas parto de {value}."),
            ("Al kio ĝi apartenas?", "Al {value}."),
        ],
    },
    "membro de": {
        "initial": [
            ("De kio {entity} estas membro?", "{entity} estas membro de {value}."),
            ("En kio {entity} membras?", "{entity} membras en {value}."),
        ],
        "followup": [
            ("De kio {pron} estas membro?", "{pron_cap} estas membro de {value}."),
            ("Ĉu {pron} membras ie?", "Jes, {pron} membras en {value}."),
        ],
    },
    "edz(in)o": {
        "initial": [
            ("Kun kiu edziniĝis {entity}?", "{entity} edziniĝis kun {value}."),
            ("Kiu estas la edzo aŭ edzino de {entity}?", "La geedzo de {entity} estas {value}."),
        ],
        "followup": [
            ("Kun kiu {pron} edziniĝis?", "{pron_cap} edziniĝis kun {value}."),
            ("Kaj la geedzo?", "{value}."),
        ],
    },
    "patro": {
        "initial": [
            ("Kiu estas la patro de {entity}?", "La patro de {entity} estas {value}."),
        ],
        "followup": [
            ("Kiu estas {poss} patro?", "{poss_cap} patro estas {value}."),
            ("Kaj la patro?", "{value}."),
        ],
    },
    "patrino": {
        "initial": [
            ("Kiu estas la patrino de {entity}?", "La patrino de {entity} estas {value}."),
        ],
        "followup": [
            ("Kaj la patrino?", "{poss_cap} patrino estas {value}."),
            ("Kiu estas {poss} patrino?", "{value}."),
        ],
    },
    "plej alta punkto": {
        "initial": [
            ("Kio estas la plej alta punkto de {entity}?", "La plej alta punkto de {entity} estas {value}."),
        ],
        "followup": [
            ("Kio estas la plej alta punkto?", "La plej alta punkto estas {value}."),
            ("Kaj la plej alta loko?", "{value}."),
        ],
    },
    "honorigo": {
        "initial": [
            ("Kiun honoron ricevis {entity}?", "{entity} ricevis la honoron {value}."),
            ("Ĉu {entity} ricevis premion?", "Jes, {entity} ricevis {value}."),
        ],
        "followup": [
            ("Ĉu {pron} ricevis iun honoron?", "Jes, {pron} ricevis {value}."),
            ("Kaj honoroj?", "{pron_cap} ricevis {value}."),
        ],
    },
    "lernejo": {
        "initial": [
            ("Kie studis {entity}?", "{entity} studis ĉe {value}."),
            ("En kiu lernejo lernis {entity}?", "{entity} lernis ĉe {value}."),
        ],
        "followup": [
            ("Kie {pron} studis?", "{pron_cap} studis ĉe {value}."),
            ("Kaj la studado?", "Ĉe {value}."),
        ],
    },
    "ĝemelurbo": {
        "initial": [
            ("Kiu estas ĝemelurbo de {entity}?", "{entity} estas ĝemeligita kun {value}."),
        ],
        "followup": [
            ("Ĉu ĝi havas ĝemelurbon?", "Jes, ĝi estas ĝemeligita kun {value}."),
            ("Kaj ĝemelurboj?", "Ĝi estas ĝemeligita kun {value}."),
        ],
    },
    # Quantity/date properties (from v2 extraction)
    "loĝantaro": {
        "initial": [
            ("Kiom da loĝantoj havas {entity}?", "{entity} havas {value} loĝantojn."),
            ("Kiom granda estas la loĝantaro de {entity}?", "La loĝantaro de {entity} estas {value}."),
            ("Kiom da homoj loĝas en {entity}?", "En {entity} loĝas {value} homoj."),
        ],
        "followup": [
            ("Kiom da loĝantoj?", "Ĝi havas {value} loĝantojn."),
            ("Kaj kiom da homoj loĝas tie?", "Tie loĝas {value} homoj."),
            ("Kaj la loĝantaro?", "{value} loĝantoj."),
        ],
    },
    "naskiĝdato": {
        "initial": [
            ("Kiam naskiĝis {entity}?", "{entity} naskiĝis en {value}."),
            ("Kiu estas la naskiĝdato de {entity}?", "{entity} naskiĝis la {value}."),
        ],
        "followup": [
            ("Kiam {pron} naskiĝis?", "{pron_cap} naskiĝis en {value}."),
            ("La naskiĝdato?", "En {value}."),
        ],
    },
    "mortdato": {
        "initial": [
            ("Kiam mortis {entity}?", "{entity} mortis en {value}."),
            ("Kiam forpasis {entity}?", "{entity} forpasis en {value}."),
        ],
        "followup": [
            ("Kiam {pron} mortis?", "{pron_cap} mortis en {value}."),
            ("Kaj kiam {pron} forpasis?", "En {value}."),
        ],
    },
    "areo": {
        "initial": [
            ("Kiom granda estas {entity}?", "{entity} havas areon de {value} kvadrataj kilometroj."),
            ("Kiom estas la areo de {entity}?", "La areo de {entity} estas {value} km²."),
        ],
        "followup": [
            ("Kiom granda ĝi estas?", "Ĝi havas areon de {value} kvadrataj kilometroj."),
            ("Kaj la areo?", "{value} km²."),
        ],
    },
    "supermara alteco": {
        "initial": [
            ("Kiom alta estas {entity}?", "{entity} estas {value} metrojn alta."),
            ("Kio estas la alteco de {entity}?", "La alteco de {entity} estas {value} metroj super la marnivelo."),
        ],
        "followup": [
            ("Kiom alta?", "{value} metrojn."),
            ("Kaj la alteco?", "{value} metroj super la marnivelo."),
        ],
    },
    "dato de fondo aŭ kreo": {
        "initial": [
            ("Kiam estis fondita {entity}?", "{entity} estis fondita en {value}."),
            ("Kiam kreiĝis {entity}?", "{entity} kreiĝis en {value}."),
        ],
        "followup": [
            ("Kiam ĝi estis fondita?", "En {value}."),
            ("Kaj la fondo-dato?", "En {value}."),
        ],
    },
}

SKIP_PROPERTIES = {
    "antaŭnomo", "familia nomo", "horzono", "poŝtkodo",
    "loka telefonkodo", "oficiala retejo",
}

SKIP_CLASSES = {
    "taksono", "vikimedia apartigilo", "vikimedia kategorio",
    "vikimedia ŝablono", "familia nomo", "jaro", "taxonomy template",
    "wikimedia human name disambiguation page", "wikimedia topic category",
    "vira persona nomo", "virina persona nomo", "affixed family name",
    "asteroido", "galaksio", "stelo", "supernovao",
}

# Trivial "estas" values that shouldn't be used as answers
TRIVIAL_ESTAS = {"homo", "human", "huamn", "persono"}

def _pronoun(entity_type: str) -> str:
    """Get the right pronoun for an entity type."""
    if entity_type == "person":
        return "li"
    return "ĝi"


def _pron_possessive(entity_type: str) -> str:
    if entity_type == "person":
        return "lia"
    return "ĝia"


# Transition phrases for switching entities in multi-turn
ENTITY_TRANSITIONS = [
    "Kaj pri {entity}?",
    "Diru al mi pri {entity}.",
    "Kio pri {entity}?",
    "Kaj {entity}?",
]


def _is_likely_english(text: str) -> bool:
    english_words = {"the ", " of ", " and ", " in ", " for ", " with ", " from "}
    return any(w in text.lower() for w in english_words)


def _capitalize(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def _strip_lingvo(value: str) -> str:
    if value.endswith(" lingvo"):
        return value[:-7]
    return value


def _should_skip(entity: dict) -> bool:
    for fact in entity["facts"]:
        if fact["property"] == "estas" and fact["value"].lower() in SKIP_CLASSES:
            return True
    return False


def _usable_facts(entity: dict) -> list[dict]:
    """Get facts that have QA templates and aren't junk."""
    facts = []
    seen_props = set()
    for fact in entity["facts"]:
        prop = fact["property"]
        value = str(fact["value"])
        if prop in SKIP_PROPERTIES or prop not in QA_TEMPLATES:
            continue
        if prop in seen_props:
            continue
        if _is_likely_english(value) or len(value) > 80 or len(value) < 2:
            continue
        # Skip trivial "estas homo" type answers
        if prop == "estas" and value.lower() in TRIVIAL_ESTAS:
            continue
        if prop == "lingvo uzata":
            fact = {**fact, "value": _strip_lingvo(value)}
        # Clean up quantity values — strip Wikidata unit URLs
        if " http" in value:
            value = value.split(" http")[0]
            fact = {**fact, "value": value}
        seen_props.add(prop)
        facts.append(fact)
    return facts


def generate_single_turn(entity: dict) -> list[dict]:
    """Generate single-turn Q&A pairs."""
    label = _capitalize(entity["label"])
    facts = _usable_facts(entity)
    entity_type = detect_entity_type(entity["facts"])
    pairs = []

    for fact in facts:
        templates = QA_TEMPLATES[fact["property"]]["initial"]
        if not templates:
            continue
        q_tmpl, a_tmpl = random.choice(templates)
        value = str(fact["value"])
        pair = {
            "messages": [
                {"role": "user", "content": _format_template(q_tmpl, label, value, entity_type)},
                {"role": "assistant", "content": _format_template(a_tmpl, label, value, entity_type)},
            ]
        }
        pairs.append(pair)

    return pairs


def _format_template(template: str, entity: str, value: str, entity_type: str) -> str:
    """Format a template with entity, value, and correct pronouns."""
    pron = _pronoun(entity_type)
    poss = _pron_possessive(entity_type)
    return template.format(
        entity=entity, value=value,
        pron=pron, pron_cap=pron[0].upper() + pron[1:],
        poss=poss, poss_cap=poss[0].upper() + poss[1:],
    )


def generate_multi_turn(entity: dict, all_entities: list[dict] | None = None) -> dict | None:
    """Generate a multi-turn conversation about an entity.

    If all_entities is provided, may include a comparison follow-up
    with a related entity.
    """
    label = _capitalize(entity["label"])
    facts = _usable_facts(entity)
    if not facts:
        return None

    entity_type = detect_entity_type(entity["facts"])
    selected = random.sample(facts, min(random.randint(1, 5), len(facts)))
    messages = []

    for i, fact in enumerate(selected):
        prop = fact["property"]
        value = str(fact["value"])
        templates = QA_TEMPLATES[prop]

        if i == 0:
            if not templates["initial"]:
                continue
            q_tmpl, a_tmpl = random.choice(templates["initial"])
        else:
            if templates["followup"]:
                q_tmpl, a_tmpl = random.choice(templates["followup"])
            elif templates["initial"]:
                q_tmpl, a_tmpl = random.choice(templates["initial"])
            else:
                continue

        messages.append({"role": "user", "content": _format_template(q_tmpl, label, value, entity_type)})
        messages.append({"role": "assistant", "content": _format_template(a_tmpl, label, value, entity_type)})

    # Optionally add a comparison follow-up (~30% of the time)
    if all_entities and random.random() < 0.3:
        comp_msg = _add_comparison_followup(entity, label, entity_type, all_entities)
        if comp_msg:
            messages.extend(comp_msg)

    if len(messages) < 2:
        return None
    return {"messages": messages}


def _add_comparison_followup(entity: dict, label: str, entity_type: str,
                             all_entities: list[dict]) -> list[dict] | None:
    """Add a comparison question at the end of a multi-turn conversation."""
    # Find a comparable property
    for prop, config in COMPARABLE_QUANTITIES.items():
        val = _get_numeric_value(entity["facts"], prop)
        if val is None or val == 0:
            continue

        # Find another entity of the same type with the same property
        candidates = []
        for other in random.sample(all_entities, min(200, len(all_entities))):
            if other["id"] == entity["id"]:
                continue
            if detect_entity_type(other["facts"]) != entity_type:
                continue
            other_val = _get_numeric_value(other["facts"], prop)
            if other_val is not None and other_val > 0:
                candidates.append(other)
                if len(candidates) >= 5:
                    break

        if not candidates:
            continue

        other = random.choice(candidates)
        other_label = _capitalize(other["label"])
        other_val = _get_numeric_value(other["facts"], prop)
        val_a = f"{val:,.0f}".replace(",", " ")
        val_b = f"{other_val:,.0f}".replace(",", " ")

        # Pick a comparison question
        comparison_questions = {
            "loĝantaro": [
                f"Ĉu {label} estas pli granda ol {other_label}?",
                f"Kiu estas pli granda, {label} aŭ {other_label}?",
            ],
            "supermara alteco": [
                f"Ĉu {label} estas pli alta ol {other_label}?",
                f"Kiu estas pli alta, {label} aŭ {other_label}?",
            ],
            "areo": [
                f"Ĉu {label} estas pli granda laŭ areo ol {other_label}?",
            ],
        }

        questions = comparison_questions.get(prop)
        if not questions:
            continue

        question = random.choice(questions)

        if val > other_val:
            if "Ĉu" in question:
                answer = f"Jes, {label} estas pli granda. {label} havas {val_a}, dum {other_label} havas {val_b}."
            else:
                answer = f"{label} estas pli granda. {label} havas {val_a}, dum {other_label} havas {val_b}."
        else:
            if "Ĉu" in question:
                answer = f"Ne, {other_label} estas pli granda. {other_label} havas {val_b}, dum {label} havas {val_a}."
            else:
                answer = f"{other_label} estas pli granda. {other_label} havas {val_b}, dum {label} havas {val_a}."

        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

    return None


def generate_cross_entity(entity_a: dict, entity_b: dict) -> dict | None:
    """Generate a conversation that switches between two entities."""
    label_a = _capitalize(entity_a["label"])
    label_b = _capitalize(entity_b["label"])
    type_a = detect_entity_type(entity_a["facts"])
    type_b = detect_entity_type(entity_b["facts"])
    facts_a = _usable_facts(entity_a)
    facts_b = _usable_facts(entity_b)

    if len(facts_a) < 2 or len(facts_b) < 2:
        return None

    props_a = {f["property"] for f in facts_a}
    props_b = {f["property"] for f in facts_b}
    shared = props_a & props_b

    if len(shared) < 2:
        return None

    messages = []
    shared_list = list(shared)
    random.shuffle(shared_list)

    # Start with entity A
    first_prop = shared_list[0]
    fact_a = next(f for f in facts_a if f["property"] == first_prop)
    templates = QA_TEMPLATES[first_prop]
    if not templates["initial"]:
        return None
    q_tmpl, a_tmpl = random.choice(templates["initial"])
    val_a = str(fact_a["value"])
    messages.append({"role": "user", "content": _format_template(q_tmpl, label_a, val_a, type_a)})
    messages.append({"role": "assistant", "content": _format_template(a_tmpl, label_a, val_a, type_a)})

    # Transition to entity B
    transition = random.choice(ENTITY_TRANSITIONS)
    messages.append({"role": "user", "content": transition.format(entity=label_b)})

    # Answer about B
    fact_b = next(f for f in facts_b if f["property"] == first_prop)
    a_tmpl_b = random.choice(templates["initial"])[1]
    val_b = str(fact_b["value"])
    messages.append({"role": "assistant", "content": _format_template(a_tmpl_b, label_b, val_b, type_b)})

    # Follow-up about B
    if len(shared_list) > 1:
        second_prop = shared_list[1]
        fact_b2 = next(f for f in facts_b if f["property"] == second_prop)
        templates2 = QA_TEMPLATES[second_prop]
        if templates2["followup"]:
            q2, a2 = random.choice(templates2["followup"])
        elif templates2["initial"]:
            q2, a2 = random.choice(templates2["initial"])
        else:
            return {"messages": messages} if len(messages) >= 4 else None
        val_b2 = str(fact_b2["value"])
        messages.append({"role": "user", "content": _format_template(q2, label_b, val_b2, type_b)})
        messages.append({"role": "assistant", "content": _format_template(a2, label_b, val_b2, type_b)})

    return {"messages": messages} if len(messages) >= 4 else None


# Properties that can be numerically compared
COMPARABLE_QUANTITIES = {
    "loĝantaro": {
        "questions": [
            ("Kiu estas pli granda, {a} aŭ {b}?", "larger"),
            ("Kiu estas pli malgranda, {a} aŭ {b}?", "smaller"),
            ("Kiu havas pli da loĝantoj, {a} aŭ {b}?", "larger"),
            ("Kiu havas malpli da loĝantoj, {a} aŭ {b}?", "smaller"),
            ("Ĉu {a} estas pli granda ol {b}?", "yesno"),
        ],
        "answer_larger": "{larger} estas pli granda ol {smaller}. {larger} havas {val_larger} loĝantojn, dum {smaller} havas {val_smaller} loĝantojn.",
        "answer_smaller": "{smaller} estas pli malgranda ol {larger}. {smaller} havas {val_smaller} loĝantojn, dum {larger} havas {val_larger} loĝantojn.",
        "answer_yes": "Jes, {a} estas pli granda ol {b}. {a} havas {val_a} loĝantojn, dum {b} havas {val_b} loĝantojn.",
        "answer_no": "Ne, {b} estas pli granda ol {a}. {b} havas {val_b} loĝantojn, dum {a} havas {val_a} loĝantojn.",
        "answer_equal": "{a} kaj {b} havas proksimume la saman nombron da loĝantoj.",
    },
    "areo": {
        "questions": [
            ("Kiu estas pli granda laŭ areo, {a} aŭ {b}?", "larger"),
            ("Kiu estas pli malgranda laŭ areo, {a} aŭ {b}?", "smaller"),
            ("Kiu havas pli grandan areon, {a} aŭ {b}?", "larger"),
        ],
        "answer_larger": "{larger} estas pli granda ol {smaller} laŭ areo. {larger} havas areon de {val_larger} km², dum {smaller} havas areon de {val_smaller} km².",
        "answer_smaller": "{smaller} estas pli malgranda ol {larger} laŭ areo. {smaller} havas areon de {val_smaller} km², dum {larger} havas areon de {val_larger} km².",
        "answer_equal": "{a} kaj {b} havas proksimume la saman areon.",
    },
    "supermara alteco": {
        "questions": [
            ("Kiu estas pli alta, {a} aŭ {b}?", "larger"),
            ("Kiu estas pli malalta, {a} aŭ {b}?", "smaller"),
            ("Kiu situas pli alte, {a} aŭ {b}?", "larger"),
        ],
        "answer_larger": "{larger} estas pli alta ol {smaller}. {larger} estas {val_larger} metrojn alta, dum {smaller} estas {val_smaller} metrojn alta.",
        "answer_smaller": "{smaller} estas pli malalta ol {larger}. {smaller} estas {val_smaller} metrojn alta, dum {larger} estas {val_larger} metrojn alta.",
        "answer_equal": "{a} kaj {b} havas proksimume la saman altecon.",
    },
}


def _get_numeric_value(facts: list[dict], prop: str) -> float | None:
    """Extract a numeric value for a property, stripping units."""
    for fact in facts:
        if fact["property"] == prop:
            val = str(fact["value"]).split(" http")[0].strip()
            try:
                return float(val)
            except ValueError:
                return None
    return None


def generate_comparison(entity_a: dict, entity_b: dict) -> dict | None:
    """Generate a comparison Q&A between two entities on a numeric property."""
    label_a = _capitalize(entity_a["label"])
    label_b = _capitalize(entity_b["label"])

    for prop, config in COMPARABLE_QUANTITIES.items():
        val_a = _get_numeric_value(entity_a["facts"], prop)
        val_b = _get_numeric_value(entity_b["facts"], prop)

        if val_a is None or val_b is None:
            continue
        if val_a == 0 or val_b == 0:
            continue

        va = f"{val_a:,.0f}".replace(",", " ")
        vb = f"{val_b:,.0f}".replace(",", " ")

        # Pick a random question variant
        q_tmpl, q_type = random.choice(config["questions"])
        question = q_tmpl.format(a=label_a, b=label_b)

        if abs(val_a - val_b) / max(val_a, val_b) < 0.05:
            answer = config["answer_equal"].format(a=label_a, b=label_b)
        elif q_type == "yesno":
            if val_a > val_b:
                answer = config["answer_yes"].format(
                    a=label_a, b=label_b, val_a=va, val_b=vb)
            else:
                answer = config["answer_no"].format(
                    a=label_a, b=label_b, val_a=va, val_b=vb)
        elif q_type == "smaller":
            if val_a < val_b:
                answer = config["answer_smaller"].format(
                    larger=label_b, smaller=label_a,
                    val_larger=vb, val_smaller=va)
            else:
                answer = config["answer_smaller"].format(
                    larger=label_a, smaller=label_b,
                    val_larger=va, val_smaller=vb)
        else:  # "larger"
            if val_a > val_b:
                answer = config["answer_larger"].format(
                    larger=label_a, smaller=label_b,
                    val_larger=va, val_smaller=vb)
            else:
                answer = config["answer_larger"].format(
                    larger=label_b, smaller=label_a,
                    val_larger=vb, val_smaller=va)

        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }

    return None


SUPERLATIVE_TEMPLATES = {
    "loĝantaro": {
        "largest": [
            ("Kiu estas la plej granda urbo en {group}?",
             "La plej granda urbo en {group} estas {entity}, kun {value} loĝantoj."),
            ("Kiu urbo en {group} havas la plej multajn loĝantojn?",
             "{entity} havas la plej multajn loĝantojn en {group}, kun {value} loĝantoj."),
        ],
        "smallest": [
            ("Kiu estas la plej malgranda urbo en {group}?",
             "La plej malgranda urbo en {group} estas {entity}, kun nur {value} loĝantoj."),
        ],
    },
    "supermara alteco": {
        "largest": [
            ("Kiu estas la plej alta loko en {group}?",
             "La plej alta loko en {group} estas {entity}, je {value} metroj super la maro."),
        ],
        "smallest": [
            ("Kiu estas la plej malalta loko en {group}?",
             "La plej malalta loko en {group} estas {entity}, je {value} metroj super la maro."),
        ],
    },
    "areo": {
        "largest": [
            ("Kiu estas la plej granda laŭ areo en {group}?",
             "La plej granda laŭ areo en {group} estas {entity}, kun areo de {value} km²."),
        ],
        "smallest": [
            ("Kiu estas la plej malgranda laŭ areo en {group}?",
             "La plej malgranda laŭ areo en {group} estas {entity}, kun areo de nur {value} km²."),
        ],
    },
}


def generate_superlatives(entities: list[dict], max_count: int = 5000) -> list[dict]:
    """Generate superlative Q&A by grouping entities by country/region."""
    # Group entities by country
    by_country: dict[str, list[tuple[str, dict]]] = {}
    for entity in entities:
        label = _capitalize(entity["label"])
        for fact in entity["facts"]:
            if fact["property"] == "lando":
                country = fact["value"]
                if not _is_likely_english(country):
                    if country not in by_country:
                        by_country[country] = []
                    by_country[country].append((label, entity))
                break

    pairs = []
    for prop, templates in SUPERLATIVE_TEMPLATES.items():
        for country, ents in by_country.items():
            if len(ents) < 5:
                continue

            # Find entities with this numeric property
            with_values = []
            for label, entity in ents:
                val = _get_numeric_value(entity["facts"], prop)
                if val is not None and val > 0:
                    with_values.append((label, val))

            if len(with_values) < 3:
                continue

            # Largest
            with_values.sort(key=lambda x: -x[1])
            largest_label, largest_val = with_values[0]
            val_str = f"{largest_val:,.0f}".replace(",", " ")

            q_tmpl, a_tmpl = random.choice(templates["largest"])
            pairs.append({
                "messages": [
                    {"role": "user", "content": q_tmpl.format(group=country, entity=largest_label, value=val_str)},
                    {"role": "assistant", "content": a_tmpl.format(group=country, entity=largest_label, value=val_str)},
                ]
            })

            # Smallest
            smallest_label, smallest_val = with_values[-1]
            val_str = f"{smallest_val:,.0f}".replace(",", " ")

            if templates.get("smallest"):
                q_tmpl, a_tmpl = random.choice(templates["smallest"])
                pairs.append({
                    "messages": [
                        {"role": "user", "content": q_tmpl.format(group=country, entity=smallest_label, value=val_str)},
                        {"role": "assistant", "content": a_tmpl.format(group=country, entity=smallest_label, value=val_str)},
                    ]
                })

            if len(pairs) >= max_count:
                break
        if len(pairs) >= max_count:
            break

    return pairs[:max_count]


def main():
    parser = argparse.ArgumentParser(description="Generate SFT data from factoids")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-conversations", type=int, default=100000,
                        help="Max conversations (1-5 turns each)")
    parser.add_argument("--max-cross", type=int, default=20000,
                        help="Max cross-entity conversations")
    parser.add_argument("--max-compare", type=int, default=20000,
                        help="Max comparison Q&A pairs")
    parser.add_argument("--max-superlative", type=int, default=10000,
                        help="Max superlative Q&A pairs")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    console.print(f"[bold green]Reading entities from {args.input}")

    # Read all usable entities, then shuffle for unbiased sampling
    entities = []
    with open(args.input) as f:
        for line in f:
            entity = json.loads(line)
            if not _should_skip(entity) and len(_usable_facts(entity)) >= 1:
                entities.append(entity)

    random.shuffle(entities)
    console.print(f"[bold]Usable entities:[/] {len(entities):,}")

    conv_count = 0
    cross_count = 0
    compare_count = 0
    superlative_count = 0

    with open(args.output, "w") as out:
        # Conversations (1-5 turns, optionally with comparison follow-up)
        console.print("[bold green]Generating conversations...")
        random.shuffle(entities)
        for entity in entities:
            if conv_count >= args.max_conversations:
                break
            conv = generate_multi_turn(entity, all_entities=entities)
            if conv:
                out.write(json.dumps(conv, ensure_ascii=False) + "\n")
                conv_count += 1

        # Cross-entity conversations
        console.print("[bold green]Generating cross-entity conversations...")
        random.shuffle(entities)
        for i in range(0, len(entities) - 1, 2):
            if cross_count >= args.max_cross:
                break
            conv = generate_cross_entity(entities[i], entities[i + 1])
            if conv:
                out.write(json.dumps(conv, ensure_ascii=False) + "\n")
                cross_count += 1

        # Comparative Q&A
        console.print("[bold green]Generating comparison pairs...")
        random.shuffle(entities)
        # Group by entity type for meaningful comparisons
        by_type: dict[str, list[dict]] = {}
        for entity in entities:
            etype = detect_entity_type(entity["facts"])
            if etype not in by_type:
                by_type[etype] = []
            by_type[etype].append(entity)

        for etype, ents in by_type.items():
            if len(ents) < 2:
                continue
            random.shuffle(ents)
            for i in range(0, len(ents) - 1, 2):
                if compare_count >= args.max_compare:
                    break
                conv = generate_comparison(ents[i], ents[i + 1])
                if conv:
                    out.write(json.dumps(conv, ensure_ascii=False) + "\n")
                    compare_count += 1
            if compare_count >= args.max_compare:
                break

        # Superlative Q&A
        console.print("[bold green]Generating superlative pairs...")
        superlatives = generate_superlatives(entities, max_count=args.max_superlative)
        for pair in superlatives:
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            superlative_count += 1

    console.print()
    console.print(f"[bold]Conversations:[/] {conv_count:,}")
    console.print(f"[bold]Cross-entity:[/] {cross_count:,}")
    console.print(f"[bold]Comparisons:[/] {compare_count:,}")
    console.print(f"[bold]Superlatives:[/] {superlative_count:,}")
    total = conv_count + cross_count + compare_count + superlative_count
    console.print(f"[bold]Total:[/] {total:,}")
    console.print(f"[bold green]Saved to {args.output}")

    # Show samples
    console.print("\n[bold]Sample short conversation:")
    with open(args.output) as f:
        for line in f:
            pair = json.loads(line)
            if len(pair["messages"]) == 2:
                for msg in pair["messages"]:
                    prefix = "  Q:" if msg["role"] == "user" else "  A:"
                    console.print(f"{prefix} {msg['content']}")
                console.print()
                break

    console.print("[bold]Sample long conversation:")
    with open(args.output) as f:
        for line in f:
            pair = json.loads(line)
            if len(pair["messages"]) >= 6:
                for msg in pair["messages"]:
                    prefix = "  Q:" if msg["role"] == "user" else "  A:"
                    console.print(f"{prefix} {msg['content']}")
                console.print()
                break

    console.print("[bold]Sample cross-entity:")
    lines = open(args.output).readlines()
    for line in reversed(lines):
        pair = json.loads(line)
        if len(pair["messages"]) >= 6:
            for msg in pair["messages"]:
                prefix = "  Q:" if msg["role"] == "user" else "  A:"
                console.print(f"{prefix} {msg['content']}")
            console.print()
            break


if __name__ == "__main__":
    main()
