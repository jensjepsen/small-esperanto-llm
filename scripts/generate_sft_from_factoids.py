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
        "initial_q": [
            "Kio estas la ĉefurbo de {entity}?",
            "Nomu la ĉefurbon de {entity}.",
            "Kiu urbo estas la ĉefurbo de {entity}?",
            "Kie estas la registaro de {entity}?",
            "Kiu estas la ĉefa urbo de {entity}?",
            "Diru al mi la ĉefurbon de {entity}.",
        ],
        "initial_a": [
            "La ĉefurbo de {entity} estas {value}.",
            "{value}.",
            "La ĉefurbo estas {value}.",
            "{value} estas la ĉefurbo de {entity}.",
            "Tiu urbo estas {value}.",
            "La registaro sidas en {value}.",
            "La ĉefa urbo de {entity} nomiĝas {value}.",
            "{value} funkcias kiel la ĉefurbo de {entity}.",
        ],
        "followup_q": [
            "Kio estas la ĉefurbo?",
            "Kaj kiu estas la ĉefurbo?",
            "Kaj la ĉefurbo?",
            "Kiu estas la ĉefurbo?",
            "Kie sidas la registaro?",
        ],
        "followup_a": [
            "La ĉefurbo estas {value}.",
            "{value}.",
            "La ĉefurbo nomiĝas {value}.",
            "En {value}.",
        ],
    },
    "lando": {
        "initial_q": [
            "En kiu lando troviĝas {entity}?",
            "Kie situas {entity}?",
            "Al kiu lando apartenas {entity}?",
            "En kiu ŝtato estas {entity}?",
            "Kie en la mondo troviĝas {entity}?",
            "Al kiu lando {entity} apartenas?",
        ],
        "initial_a": [
            "{entity} troviĝas en {value}.",
            "En {value}.",
            "{entity} situas en {value}.",
            "{entity} apartenas al {value}.",
            "Al {value}.",
            "{entity} estas en {value}.",
            "Ĝi estas en {value}.",
            "Vi trovos {entity}n en {value}.",
        ],
        "followup_q": [
            "En kiu lando?",
            "Kie ĝi troviĝas?",
            "En kiu lando ĝi situas?",
            "En kiu lando ĝi estas?",
            "Kie?",
        ],
        "followup_a": [
            "En {value}.",
            "Ĝi troviĝas en {value}.",
            "Ĝi situas en {value}.",
            "Ĝi estas en {value}.",
        ],
    },
    "okupo": {
        "initial_q": [
            "Kio estas la profesio de {entity}?",
            "Kiu estas {entity}?",
            "Per kio {entity} okupiĝas?",
            "Kiel {entity} vivtenas sin?",
            "Kion {entity} faras profesie?",
            "Kia profesio estas tiu de {entity}?",
        ],
        "initial_a": [
            "{entity} estas {value}.",
            "{value}.",
            "Temas pri {value}.",
            "{entity} estas {value} laŭ profesio.",
            "{pron_cap} laboras kiel {value}.",
            "{pron_cap} estas konata kiel {value}.",
            "La profesio de {entity} estas {value}.",
            "{entity} profesie estas {value}.",
        ],
        "followup_q": [
            "Kio estas {poss} profesio?",
            "Kion {pron} faras?",
            "Kaj la profesio?",
            "Kiel {pron} laboras?",
            "Kia profesio?",
        ],
        "followup_a": [
            "{pron_cap} estas {value}.",
            "{value}.",
            "{pron_cap} laboras kiel {value}.",
            "{pron_cap} profesie estas {value}.",
        ],
    },
    "ŝtataneco": {
        "initial_q": [
            "El kiu lando devenas {entity}?",
            "Kio estas la ŝtataneco de {entity}?",
            "De kie {entity} venas?",
            "Kiu estas la devenlando de {entity}?",
        ],
        "initial_a": [
            "{entity} devenas el {value}.",
            "{entity} estas ŝtatano de {value}.",
            "El {value}.",
            "{entity} venas el {value}.",
            "La devenlando de {entity} estas {value}.",
            "{pron_cap} estas el {value}.",
        ],
        "followup_q": [
            "El kiu lando {pron} devenas?",
            "Kaj la ŝtataneco?",
            "El kiu lando?",
            "De kie {pron} venas?",
        ],
        "followup_a": [
            "{pron_cap} devenas el {value}.",
            "{pron_cap} estas ŝtatano de {value}.",
            "El {value}.",
            "{pron_cap} venas el {value}.",
        ],
    },
    "naskiĝloko": {
        "initial_q": [
            "Kie naskiĝis {entity}?",
            "En kiu urbo naskiĝis {entity}?",
            "Kio estas la naskiĝloko de {entity}?",
            "Kie {entity} venis al la mondo?",
        ],
        "initial_a": [
            "{entity} naskiĝis en {value}.",
            "En {value}.",
            "La naskiĝloko de {entity} estas {value}.",
            "{entity} venis al la mondo en {value}.",
            "{value} estas la naskiĝloko de {entity}.",
        ],
        "followup_q": [
            "Kie {pron} naskiĝis?",
            "Kaj kie {pron} naskiĝis?",
            "La naskiĝloko?",
            "Kaj la naskiĝloko?",
        ],
        "followup_a": [
            "{pron_cap} naskiĝis en {value}.",
            "En {value}.",
            "{value}.",
        ],
    },
    "mortloko": {
        "initial_q": [
            "Kie mortis {entity}?",
            "En kiu urbo {entity} forpasis?",
            "Kie {entity} pasigis siajn lastajn tagojn?",
        ],
        "initial_a": [
            "{entity} mortis en {value}.",
            "{entity} forpasis en {value}.",
            "En {value}.",
            "{pron_cap} forpasis en {value}.",
        ],
        "followup_q": [
            "Kaj kie {pron} mortis?",
            "Kie {pron} forpasis?",
            "Kaj la mortloko?",
        ],
        "followup_a": [
            "{pron_cap} mortis en {value}.",
            "En {value}.",
            "{value}.",
        ],
    },
    "estas": {
        "initial_q": [
            "Kio estas {entity}?",
            "Priskribu {entity}n.",
            "Rakontu al mi pri {entity}.",
            "Ĉu vi konas {entity}n?",
            "Kion vi scias pri {entity}?",
            "Klarigu kio estas {entity}.",
            "Mi volas scii pri {entity}.",
            "Diru al mi kio estas {entity}.",
        ],
        "initial_a": [
            "{entity} estas {value}.",
            "Oni povas priskribi {entity}n kiel {value}.",
            "Temas pri {value}.",
            "Mi povas diri, ke {entity} estas {value}.",
            "Jes, {entity} estas {value}.",
            "{entity} estas konata kiel {value}.",
            "Simple dirite, {entity} estas {value}.",
            "Laŭ miaj informoj, {entity} estas {value}.",
            "{value} — jen kio estas {entity}.",
        ],
        "followup_q": [],
        "followup_a": [],
    },
    "lingvo uzata": {
        "initial_q": ["Kiun lingvon oni parolas en {entity}?", "Kiu lingvo estas uzata en {entity}?"],
        "initial_a": ["En {entity} oni parolas la {value}n.", "La {value} estas uzata en {entity}.", "La {value}n."],
        "followup_q": ["Kiun lingvon oni parolas tie?", "Kaj la lingvo?"],
        "followup_a": ["Oni parolas la {value}n.", "La {value}.", "La {value}n."],
    },
    "komuna limo kun": {
        "initial_q": ["Kun kiu lando limas {entity}?", "Kiuj estas la najbaroj de {entity}?"],
        "initial_a": ["{entity} limas kun {value}.", "{entity} havas komunan limon kun {value}.", "Kun {value}."],
        "followup_q": ["Kun kiu lando ĝi limas?", "Kaj la najbaroj?"],
        "followup_a": ["Ĝi limas kun {value}.", "Kun {value}."],
    },
    "parto de": {
        "initial_q": ["De kio {entity} estas parto?", "Al kio apartenas {entity}?"],
        "initial_a": ["{entity} estas parto de {value}.", "{entity} apartenas al {value}.", "Al {value}."],
        "followup_q": ["De kio ĝi estas parto?", "Al kio ĝi apartenas?"],
        "followup_a": ["Ĝi estas parto de {value}.", "Al {value}."],
    },
    "membro de": {
        "initial_q": ["De kio {entity} estas membro?", "En kio {entity} membras?"],
        "initial_a": ["{entity} estas membro de {value}.", "{entity} membras en {value}."],
        "followup_q": ["De kio {pron} estas membro?", "Ĉu {pron} membras ie?"],
        "followup_a": ["{pron_cap} estas membro de {value}.", "Jes, {pron} membras en {value}."],
    },
    "edz(in)o": {
        "initial_q": ["Kun kiu edziniĝis {entity}?", "Kiu estas la edzo aŭ edzino de {entity}?"],
        "initial_a": ["{entity} edziniĝis kun {value}.", "La geedzo de {entity} estas {value}.", "{value}."],
        "followup_q": ["Kun kiu {pron} edziniĝis?", "Kaj la geedzo?"],
        "followup_a": ["{pron_cap} edziniĝis kun {value}.", "{value}."],
    },
    "patro": {
        "initial_q": ["Kiu estas la patro de {entity}?"],
        "initial_a": ["La patro de {entity} estas {value}.", "{value}."],
        "followup_q": ["Kiu estas {poss} patro?", "Kaj la patro?"],
        "followup_a": ["{poss_cap} patro estas {value}.", "{value}."],
    },
    "patrino": {
        "initial_q": ["Kiu estas la patrino de {entity}?"],
        "initial_a": ["La patrino de {entity} estas {value}.", "{value}."],
        "followup_q": ["Kaj la patrino?", "Kiu estas {poss} patrino?"],
        "followup_a": ["{poss_cap} patrino estas {value}.", "{value}."],
    },
    "plej alta punkto": {
        "initial_q": ["Kio estas la plej alta punkto de {entity}?"],
        "initial_a": ["La plej alta punkto de {entity} estas {value}.", "{value}."],
        "followup_q": ["Kio estas la plej alta punkto?", "Kaj la plej alta loko?"],
        "followup_a": ["La plej alta punkto estas {value}.", "{value}."],
    },
    "honorigo": {
        "initial_q": ["Kiun honoron ricevis {entity}?", "Ĉu {entity} ricevis premion?"],
        "initial_a": ["{entity} ricevis la honoron {value}.", "Jes, {entity} ricevis {value}."],
        "followup_q": ["Ĉu {pron} ricevis iun honoron?", "Kaj honoroj?"],
        "followup_a": ["Jes, {pron} ricevis {value}.", "{pron_cap} ricevis {value}."],
    },
    "lernejo": {
        "initial_q": ["Kie studis {entity}?", "En kiu lernejo lernis {entity}?"],
        "initial_a": ["{entity} studis ĉe {value}.", "{entity} lernis ĉe {value}.", "Ĉe {value}."],
        "followup_q": ["Kie {pron} studis?", "Kaj la studado?"],
        "followup_a": ["{pron_cap} studis ĉe {value}.", "Ĉe {value}."],
    },
    "ĝemelurbo": {
        "initial_q": ["Kiu estas ĝemelurbo de {entity}?"],
        "initial_a": ["{entity} estas ĝemeligita kun {value}.", "Ĝi estas ĝemeligita kun {value}."],
        "followup_q": ["Ĉu ĝi havas ĝemelurbon?", "Kaj ĝemelurboj?"],
        "followup_a": ["Jes, ĝi estas ĝemeligita kun {value}.", "Kun {value}."],
    },
    # Quantity/date properties (from v2 extraction)
    "loĝantaro": {
        "initial_q": [
            "Kiom da loĝantoj havas {entity}?",
            "Kiom granda estas la loĝantaro de {entity}?",
            "Kiom da homoj loĝas en {entity}?",
            "Kio estas la loĝantaro de {entity}?",
            "Kiom granda estas {entity} laŭ loĝantaro?",
        ],
        "initial_a": [
            "{entity} havas {value} loĝantojn.",
            "La loĝantaro de {entity} estas {value}.",
            "En {entity} loĝas {value} homoj.",
            "{value} loĝantoj.",
            "Proksimume {value} homoj loĝas en {entity}.",
            "La loĝantaro nombras {value}.",
            "Tie loĝas ĉirkaŭ {value} homoj.",
        ],
        "followup_q": ["Kiom da loĝantoj?", "Kaj kiom da homoj loĝas tie?", "Kaj la loĝantaro?", "Kiom da homoj?"],
        "followup_a": ["Ĝi havas {value} loĝantojn.", "Tie loĝas {value} homoj.", "{value} loĝantoj.", "{value}.", "Proksimume {value}."],
    },
    "naskiĝdato": {
        "initial_q": [
            "Kiam naskiĝis {entity}?",
            "Kiu estas la naskiĝdato de {entity}?",
            "En kiu jaro naskiĝis {entity}?",
            "Kiam {entity} venis al la mondo?",
        ],
        "initial_a": [
            "{entity} naskiĝis en {value}.",
            "{entity} naskiĝis la {value}.",
            "En {value}.",
            "La naskiĝdato estas {value}.",
            "{entity} venis al la mondo en {value}.",
        ],
        "followup_q": ["Kiam {pron} naskiĝis?", "La naskiĝdato?", "En kiu jaro?"],
        "followup_a": ["{pron_cap} naskiĝis en {value}.", "En {value}.", "{value}.", "La {value}."],
    },
    "mortdato": {
        "initial_q": ["Kiam mortis {entity}?", "Kiam forpasis {entity}?"],
        "initial_a": ["{entity} mortis en {value}.", "{entity} forpasis en {value}.", "En {value}."],
        "followup_q": ["Kiam {pron} mortis?", "Kaj kiam {pron} forpasis?"],
        "followup_a": ["{pron_cap} mortis en {value}.", "En {value}.", "{value}."],
    },
    "areo": {
        "initial_q": ["Kiom granda estas {entity}?", "Kiom estas la areo de {entity}?"],
        "initial_a": ["{entity} havas areon de {value} kvadrataj kilometroj.", "La areo de {entity} estas {value} km².", "{value} km²."],
        "followup_q": ["Kiom granda ĝi estas?", "Kaj la areo?"],
        "followup_a": ["Ĝi havas areon de {value} kvadrataj kilometroj.", "{value} km².", "{value}."],
    },
    "supermara alteco": {
        "initial_q": ["Kiom alta estas {entity}?", "Kio estas la alteco de {entity}?"],
        "initial_a": ["{entity} estas {value} metrojn alta.", "La alteco de {entity} estas {value} metroj super la marnivelo.", "{value} metrojn."],
        "followup_q": ["Kiom alta?", "Kaj la alteco?"],
        "followup_a": ["{value} metrojn.", "{value} metroj super la marnivelo.", "{value}."],
    },
    "dato de fondo aŭ kreo": {
        "initial_q": ["Kiam estis fondita {entity}?", "Kiam kreiĝis {entity}?"],
        "initial_a": ["{entity} estis fondita en {value}.", "{entity} kreiĝis en {value}.", "En {value}."],
        "followup_q": ["Kiam ĝi estis fondita?", "Kaj la fondo-dato?"],
        "followup_a": ["Ĝi estis fondita en {value}.", "En {value}.", "{value}."],
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


def _make_followup(template: str) -> str:
    """Convert an initial template to a followup by replacing entity with pronoun."""
    return (template
            .replace("{entity}", "{pron_cap}")
            .replace(" de {pron_cap}", "")  # "La ĉefurbo de X" → "La ĉefurbo"
            .replace("{pron_cap}n ", "")     # "Priskribu Xn" → "Priskribu"
            )


# Auto-generate followup_q and followup_a from initial templates if not provided
for _prop, _tmpls in QA_TEMPLATES.items():
    if not _tmpls.get("followup_q"):
        _tmpls["followup_q"] = [_make_followup(q) for q in _tmpls["initial_q"]]
    if not _tmpls.get("followup_a"):
        _tmpls["followup_a"] = [_make_followup(a) for a in _tmpls["initial_a"]]


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
    selected = random.sample(facts, min(random.randint(2, 5), len(facts)))
    messages = []

    for i, fact in enumerate(selected):
        prop = fact["property"]
        value = str(fact["value"])
        templates = QA_TEMPLATES[prop]

        if i == 0:
            if not templates["initial_q"]:
                continue
            q_tmpl = random.choice(templates["initial_q"])
            a_tmpl = random.choice(templates["initial_a"])
        else:
            if templates.get("followup_q"):
                q_tmpl = random.choice(templates["followup_q"])
                a_tmpl = random.choice(templates["followup_a"])
            elif templates["initial_q"]:
                q_tmpl = random.choice(templates["initial_q"])
                a_tmpl = random.choice(templates["initial_a"])
            else:
                continue

        messages.append({"role": "user", "content": _format_template(q_tmpl, label, value, entity_type)})
        messages.append({"role": "assistant", "content": _format_template(a_tmpl, label, value, entity_type)})

    # Optionally add a comparison follow-up (~50% of the time)
    if all_entities and random.random() < 0.5:
        comp_msg = _add_comparison_followup(entity, label, entity_type, all_entities)
        if comp_msg:
            messages.extend(comp_msg)

    if len(messages) < 4:
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

        # Build questions that naturally produce both yes and no answers
        comparison_qa = {
            "loĝantaro": {
                "q_pos": [  # questions where entity is bigger → yes
                    f"Ĉu {label} estas pli granda ol {other_label}?",
                    f"Ĉu {label} havas pli da loĝantoj ol {other_label}?",
                ],
                "q_neg": [  # questions where entity is smaller → no
                    f"Ĉu {label} estas pli granda ol {other_label}?",
                    f"Ĉu {label} havas pli da loĝantoj ol {other_label}?",
                ],
                "q_open": [  # open questions
                    f"Kiu estas pli granda, {label} aŭ {other_label}?",
                    f"Kiu havas pli da loĝantoj, {label} aŭ {other_label}?",
                    f"Kiu estas pli malgranda, {label} aŭ {other_label}?",
                ],
                "a_yes": [
                    f"Jes, {label} estas pli granda. {label} havas {val_a} loĝantojn, dum {other_label} havas {val_b}.",
                    f"Jes. {label} havas {val_a} loĝantojn, do ĝi estas pli granda ol {other_label} kun {val_b}.",
                ],
                "a_no": [
                    f"Ne, {other_label} estas pli granda. {other_label} havas {val_b} loĝantojn, dum {label} havas nur {val_a}.",
                    f"Ne, fakte {other_label} estas pli granda kun {val_b} loĝantoj kontraŭ {val_a} por {label}.",
                    f"Ne. {label} havas {val_a} loĝantojn, sed {other_label} havas {val_b}, do {other_label} estas pli granda.",
                ],
                "a_open_bigger": [
                    f"{label} estas pli granda. {label} havas {val_a} loĝantojn, dum {other_label} havas {val_b}.",
                    f"{label}, kun {val_a} loĝantoj. {other_label} havas nur {val_b}.",
                ],
                "a_open_smaller": [
                    f"{other_label} estas pli granda. {other_label} havas {val_b} loĝantojn, dum {label} havas {val_a}.",
                    f"{label} estas pli malgranda. {label} havas {val_a} loĝantojn, dum {other_label} havas {val_b}.",
                ],
            },
            "supermara alteco": {
                "q_pos": [f"Ĉu {label} estas pli alta ol {other_label}?"],
                "q_neg": [f"Ĉu {label} estas pli alta ol {other_label}?"],
                "q_open": [
                    f"Kiu estas pli alta, {label} aŭ {other_label}?",
                    f"Kiu estas pli malalta, {label} aŭ {other_label}?",
                ],
                "a_yes": [f"Jes, {label} estas pli alta je {val_a} metroj, dum {other_label} estas je {val_b} metroj."],
                "a_no": [
                    f"Ne, {other_label} estas pli alta. {other_label} estas je {val_b} metroj, dum {label} estas je {val_a} metroj.",
                    f"Ne. {label} estas nur {val_a} metrojn alta, sed {other_label} estas {val_b} metrojn.",
                ],
                "a_open_bigger": [f"{label} estas pli alta je {val_a} metroj, dum {other_label} estas je {val_b}."],
                "a_open_smaller": [f"{other_label} estas pli alta je {val_b} metroj, dum {label} estas je {val_a}."],
            },
            "areo": {
                "q_pos": [f"Ĉu {label} estas pli granda laŭ areo ol {other_label}?"],
                "q_neg": [f"Ĉu {label} estas pli granda laŭ areo ol {other_label}?"],
                "q_open": [f"Kiu estas pli granda laŭ areo, {label} aŭ {other_label}?"],
                "a_yes": [f"Jes, {label} havas areon de {val_a} km², dum {other_label} havas nur {val_b} km²."],
                "a_no": [
                    f"Ne, {other_label} estas pli granda laŭ areo. {other_label} havas {val_b} km², dum {label} havas {val_a} km².",
                    f"Ne. {label} havas areon de nur {val_a} km², sed {other_label} havas {val_b} km².",
                ],
                "a_open_bigger": [f"{label} estas pli granda kun areo de {val_a} km², kontraŭ {val_b} km² por {other_label}."],
                "a_open_smaller": [f"{other_label} estas pli granda laŭ areo: {val_b} km² kontraŭ {val_a} km² por {label}."],
            },
        }

        qa = comparison_qa.get(prop)
        if not qa:
            continue

        # Decide question type: yes/no or open
        if random.random() < 0.5:
            # Yes/no question
            if val > other_val:
                question = random.choice(qa["q_pos"])
                answer = random.choice(qa["a_yes"])
            else:
                question = random.choice(qa["q_neg"])
                answer = random.choice(qa["a_no"])
        else:
            # Open question
            question = random.choice(qa["q_open"])
            if val > other_val:
                answer = random.choice(qa["a_open_bigger"])
            else:
                answer = random.choice(qa["a_open_smaller"])

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
    if not templates["initial_q"]:
        return None
    q_tmpl = random.choice(templates["initial_q"])
    a_tmpl = random.choice(templates["initial_a"])
    val_a = str(fact_a["value"])
    messages.append({"role": "user", "content": _format_template(q_tmpl, label_a, val_a, type_a)})
    messages.append({"role": "assistant", "content": _format_template(a_tmpl, label_a, val_a, type_a)})

    # Transition to entity B
    transition = random.choice(ENTITY_TRANSITIONS)
    messages.append({"role": "user", "content": transition.format(entity=label_b)})

    # Answer about B
    fact_b = next(f for f in facts_b if f["property"] == first_prop)
    a_tmpl_b = random.choice(templates["initial_a"])
    val_b = str(fact_b["value"])
    messages.append({"role": "assistant", "content": _format_template(a_tmpl_b, label_b, val_b, type_b)})

    # Follow-up about B
    if len(shared_list) > 1:
        second_prop = shared_list[1]
        fact_b2 = next(f for f in facts_b if f["property"] == second_prop)
        templates2 = QA_TEMPLATES[second_prop]
        if templates2.get("followup_q"):
            q2 = random.choice(templates2["followup_q"])
            a2 = random.choice(templates2["followup_a"])
        elif templates2["initial_q"]:
            q2 = random.choice(templates2["initial_q"])
            a2 = random.choice(templates2["initial_a"])
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

        # Superlative Q&A
        console.print("[bold green]Generating superlative pairs...")
        superlatives = generate_superlatives(entities, max_count=args.max_superlative)
        for pair in superlatives:
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            superlative_count += 1

    console.print()
    console.print(f"[bold]Conversations:[/] {conv_count:,}")
    console.print(f"[bold]Cross-entity:[/] {cross_count:,}")
    console.print(f"[bold]Superlatives:[/] {superlative_count:,}")
    total = conv_count + cross_count + superlative_count
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
