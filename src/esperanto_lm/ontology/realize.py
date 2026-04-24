"""Template-based surface realization with controlled variation.

One Esperanto sentence per event, plus scene-setting sentences for the
initial relations. Templates key off **role structure** (which roles the
verb has), not the verb's lemma — Esperanto's regular morphology means
the same template covers many verbs.

Decisions:
  - Past tense throughout (-is on every verb).
  - Article tracking: strict first-mention indefinite, subsequent definite,
    applied uniformly including locations and entities introduced in
    setup. Persons are capitalized names with no article.
  - Pronouns (li/ŝi) substitute for subsequent person mentions when:
      (a) the person was mentioned earlier, AND
      (b) only one person in the trace shares the pronoun's gender, AND
      (c) the per-trace RNG flips a coin in favor.
  - Causally-linked events get a connective: Tial / Sekve / Pro tio /
    juxtaposition. Variant chosen by the per-trace RNG.
  - Multiple templates per relation type (en / sur / havi). Variant
    chosen by the per-trace RNG.
  - **Use-instrument fusion**: the trace contains a logically-distinct
    `uzi(...)` event followed by the signature-resolved verb event
    (e.g. `tranĉi(...)`) caused by it. In prose those read as one
    action, so we skip the `uzi` sentence.

All variation is driven by an optional `rng` parameter on
`realize_trace`. When `rng=None` the realizer is fully deterministic
(picks the first variant of every choice), so existing tests don't
need to be updated.
"""
from __future__ import annotations

import random
from typing import Optional

from .causal import Event, Trace
from .loader import Lexicon, resolve_signature


# ---- pronoun map ---------------------------------------------------------

# Per-name gender for pronoun substitution. Matches the PERSON_NAMES list
# in sampler.py. If a name isn't here, no pronoun substitution happens.
PRONOUN_OF_NAME: dict[str, str] = {
    "petro": "li", "johano": "li", "pavel": "li", "mikael": "li",
    "maria": "ŝi", "anna": "ŝi", "klara": "ŝi", "sara": "ŝi",
    "elena": "ŝi", "lidia": "ŝi",
}

# Probability of substituting a pronoun for a subsequent person mention
# (when the substitution is unambiguous). Conservative — repetition is
# better than confusion when this fires too aggressively.
PRONOUN_RATE = 0.55


# ---- variation pools ----------------------------------------------------

# Templates take (a_form, b_form, tense_suffix) where tense_suffix is "is"
# (past) or "as" (present). The verb stem in each template is followed by
# {t} so it inflects naturally.
RELATION_TEMPLATES = {
    "en": [
        lambda a, b, t: f"{a} est{t} en {b}.",
        lambda a, b, t: f"En {b} est{t} {a}.",
        lambda a, b, t: f"Est{t} {a} en {b}.",
    ],
    "sur": [
        lambda a, b, t: f"{a} kuŝ{t} sur {b}.",
        lambda a, b, t: f"Sur {b} kuŝ{t} {a}.",
        lambda a, b, t: f"Sur {b} est{t} {a}.",
    ],
    "havi": [
        lambda a, b_acc, t: f"{a} hav{t} {b_acc}.",
        lambda a, b_acc, t: f"{a} ten{t} {b_acc}.",
    ],
}

CONNECTIVES = ["Tial", "Sekve", "Pro tio", ""]   # "" = juxtaposition

# Per-trace verb tense. 50/50 past/present. Future and conditional don't
# fit completed-event narratives well; skipped intentionally.
TENSES = ["is", "as"]


def _pick_tense(rng: Optional[random.Random]) -> str:
    """Pick a tense suffix once per trace. rng=None defaults to past
    (matches the old, pre-variation behavior — keeps existing tests
    passing without modification)."""
    if rng is None:
        return "is"
    return rng.choice(TENSES)


def _pick(rng: Optional[random.Random], options: list):
    """Pick from options. With rng=None, returns the first (deterministic)."""
    if rng is None:
        return options[0]
    return rng.choice(options)


# ---- morphology helpers --------------------------------------------------

def inflect(verb_lemma: str, tense: str) -> str:
    """tranĉi + 'is' → tranĉis;  tranĉi + 'as' → tranĉas. Strip final -i,
    append the tense suffix. Works for any of -is / -as / -os / -us /
    bare -u / -i, though the realizer only uses past and present."""
    if verb_lemma.endswith("i"):
        return verb_lemma[:-1] + tense
    return verb_lemma + tense


def past_tense(verb_lemma: str) -> str:
    """Backwards-compat alias. Prefer `inflect(lemma, 'is')` in new code."""
    return inflect(verb_lemma, "is")


# Pronouns that take -n in accusative the same way nouns do.
_PRONOUNS_BASE = {"li", "ŝi", "ĝi", "ili", "mi", "vi", "ni"}


def to_accusative(noun_form: str) -> str:
    """Append -n. Handles `la X` -> `la Xn` and pronouns (li → lin)."""
    if " " in noun_form:
        head, _, tail = noun_form.rpartition(" ")
        return f"{head} {to_accusative(tail)}"
    if noun_form in _PRONOUNS_BASE:
        return noun_form + "n"
    if noun_form.endswith(("o", "oj", "a", "aj")):
        return noun_form + "n"
    return noun_form


# ---- entity naming -------------------------------------------------------

def _pronoun_unambiguous(name: str, trace: Trace) -> bool:
    """True iff `name` has a gender pronoun and no other person in the
    trace shares it."""
    pronoun = PRONOUN_OF_NAME.get(name)
    if pronoun is None:
        return False
    for eid, ent in trace.entities.items():
        if eid == name or ent.entity_type != "person":
            continue
        if PRONOUN_OF_NAME.get(eid) == pronoun:
            return False
    return True


def name_for(
    entity, mentioned: set[str], *,
    scene_location_id: Optional[str] = None,
    rng: Optional[random.Random] = None,
    trace: Optional[Trace] = None,
) -> str:
    """How to refer to this entity given who's been mentioned.

    Persons: subsequent mention may be replaced with li/ŝi when the
    pronoun is unambiguous (only one person of that gender in trace) and
    the rng's coin flip lands.

    Other entities: bare lemma on first mention, 'la X' on subsequent.
    The scene-location entity is definite from first mention.
    """
    if entity.entity_type == "person":
        name = entity.id
        if (rng is not None and trace is not None
                and entity.id in mentioned
                and _pronoun_unambiguous(name, trace)
                and rng.random() < PRONOUN_RATE):
            return PRONOUN_OF_NAME[name]
        return name.capitalize()
    lemma = entity.concept_lemma
    if entity.id == scene_location_id:
        return f"la {lemma}"
    if entity.id in mentioned:
        return f"la {lemma}"
    return lemma


# ---- relation → scene-setting sentence -----------------------------------

def render_relation(
    rel, trace: Trace, lexicon: Lexicon,
    mentioned: set[str], *,
    scene_location_id: Optional[str] = None,
    rng: Optional[random.Random] = None,
    tense: str = "is",
) -> Optional[str]:
    a = trace.entity(rel.args[0])
    b = trace.entity(rel.args[1]) if len(rel.args) > 1 else None
    if a is None or b is None:
        return None

    name_kw = dict(scene_location_id=scene_location_id, rng=rng, trace=trace)

    a_form = name_for(a, mentioned, **name_kw)
    mentioned.add(a.id)

    if rel.relation == "havi":
        b_form = to_accusative(name_for(b, mentioned, **name_kw))
        template = _pick(rng, RELATION_TEMPLATES["havi"])
        sent = template(a_form, b_form, tense)
    elif rel.relation in ("en", "sur"):
        b_form = name_for(b, mentioned, **name_kw)
        template = _pick(rng, RELATION_TEMPLATES[rel.relation])
        sent = template(a_form, b_form, tense)
    else:
        return None
    mentioned.add(b.id)
    return sent


# ---- event → action sentence ---------------------------------------------

def render_event(
    event: Event, trace: Trace, lexicon: Lexicon,
    mentioned: set[str], *,
    scene_location_id: Optional[str] = None,
    rng: Optional[random.Random] = None,
    tense: str = "is",
) -> Optional[str]:
    action = lexicon.actions.get(event.action)
    if action is None:
        return None

    role_names = {r.name for r in action.roles}
    name_kw = dict(scene_location_id=scene_location_id, rng=rng, trace=trace)

    if "agent" in role_names and event.roles.get("agent"):
        subject_id = event.roles["agent"]
        subject_role_name = "agent"
    elif "theme" in role_names and event.roles.get("theme"):
        subject_id = event.roles["theme"]
        subject_role_name = "theme"
    else:
        return None

    subject = trace.entity(subject_id)
    if subject is None:
        return None
    subject_form = name_for(subject, mentioned, **name_kw)
    mentioned.add(subject.id)

    parts = [subject_form, inflect(event.action, tense)]

    if subject_role_name == "agent" and event.roles.get("theme"):
        theme = trace.entity(event.roles["theme"])
        if theme is not None:
            parts.append(to_accusative(name_for(theme, mentioned, **name_kw)))
            mentioned.add(theme.id)

    if event.roles.get("instrument"):
        instr = trace.entity(event.roles["instrument"])
        if instr is not None:
            parts.append(f"per {name_for(instr, mentioned, **name_kw)}")
            mentioned.add(instr.id)

    if event.roles.get("recipient"):
        recip = trace.entity(event.roles["recipient"])
        if recip is not None:
            parts.append(f"al {name_for(recip, mentioned, **name_kw)}")
            mentioned.add(recip.id)

    if event.roles.get("location"):
        loc = trace.entity(event.roles["location"])
        if loc is not None:
            prep = "en" if loc.entity_type == "location" else "sur"
            parts.append(f"{prep} {name_for(loc, mentioned, **name_kw)}")
            mentioned.add(loc.id)

    return " ".join(parts) + "."


# ---- use-instrument fusion -----------------------------------------------

def _use_instrument_skip_set(trace: Trace, lexicon: Lexicon) -> set[str]:
    """Identify `uzi` events whose synthesized verb-event we'll render
    instead. Those `uzi` events get skipped in prose."""
    by_id = {e.id: e for e in trace.events}
    skip: set[str] = set()
    for e2 in trace.events:
        if not e2.caused_by or len(e2.caused_by) != 1:
            continue
        e1 = by_id.get(e2.caused_by[0])
        if e1 is None or e1.action != "uzi":
            continue
        instr_id = e1.roles.get("instrument")
        if not instr_id:
            continue
        instr = trace.entity(instr_id)
        if instr is None:
            continue
        instr_concept = lexicon.concepts.get(instr.concept_lemma)
        if instr_concept is None:
            continue
        source = resolve_signature(lexicon, instr_concept)
        if source is None or source.lemma != e2.action:
            continue
        skip.add(e1.id)
    return skip


# ---- formatting (connectives, capitalization) ----------------------------

def _starts_with_proper_noun(sentence: str) -> bool:
    if not sentence:
        return False
    first = sentence.split(" ", 1)[0]
    if not first or not first[0].isupper():
        return False
    return first.lower() not in {"la", "ĉi", "tiu", "tio", "tial", "kaj",
                                 "sekve", "pro", "en", "sur", "estis"}


def format_with_connective(
    sentence: str, has_cause_in_prose: bool, is_first: bool,
    rng: Optional[random.Random] = None,
) -> str:
    if not is_first and has_cause_in_prose:
        connective = _pick(rng, CONNECTIVES)
        if not connective:
            # Juxtaposition — same handling as if it had no cause: just a
            # normal capitalized sentence.
            return sentence[0].upper() + sentence[1:]
        if _starts_with_proper_noun(sentence):
            return f"{connective} {sentence}"
        return f"{connective} {sentence[0].lower()}{sentence[1:]}"
    return sentence[0].upper() + sentence[1:]


# ---- synthetic grounding -------------------------------------------------

def _synthetic_scene_grounding(
    trace: Trace, scene_location_id: Optional[str],
    mentioned: set[str], rng: Optional[random.Random] = None,
    tense: str = "is",
) -> list[str]:
    """Synthetic 'X estis en la SCENE.' setup lines for non-person event
    participants that lack any relation grounding.

    Skips entities with `created_at_event is not None` — those came into
    existence mid-trace and shouldn't be pre-introduced in setup. Their
    first mention is handled by the event rendering that creates them
    (see `render_event` and the appearance-line emission in
    `realize_trace`)."""
    if scene_location_id is None or scene_location_id not in trace.entities:
        return []
    scene_ent = trace.entities[scene_location_id]

    in_relations: set[str] = set()
    for r in trace.relations:
        in_relations.update(r.args)

    in_events: set[str] = set()
    for ev in trace.events:
        for v in ev.roles.values():
            if isinstance(v, str):
                in_events.add(v)

    sentences: list[str] = []
    name_kw = dict(scene_location_id=scene_location_id, rng=rng, trace=trace)
    for eid in sorted(in_events):
        if eid == scene_location_id:
            continue
        if eid in in_relations:
            continue
        ent = trace.entities.get(eid)
        if ent is None or ent.entity_type == "person":
            continue
        # Skip mid-trace-created entities — their introduction belongs
        # to the event that creates them.
        if ent.created_at_event is not None:
            continue
        ent_form = name_for(ent, mentioned, **name_kw)
        mentioned.add(ent.id)
        scene_form = name_for(scene_ent, mentioned, **name_kw)
        mentioned.add(scene_ent.id)
        template = _pick(rng, RELATION_TEMPLATES["en"])
        sentences.append(template(ent_form, scene_form, tense))
    return sentences


# ---- created-entity appearance lines (Step 5) ---------------------------

def _render_appearance_line(
    created_entity, mentioned: set[str], scene_location_id: Optional[str],
    rng: Optional[random.Random], trace: Trace, tense: str,
) -> str:
    """Render a one-line existential introduction for a newly-created
    entity. Form: 'Aperis X.' (verb-subject order, intransitive — subject
    in nominative). Marks the entity as mentioned so subsequent
    references use definite form.
    """
    name_kw = dict(scene_location_id=scene_location_id, rng=rng, trace=trace)
    # name_for returns bare lemma on first mention (entity isn't in
    # mentioned yet), then we add it.
    ent_form = name_for(created_entity, mentioned, **name_kw)
    mentioned.add(created_entity.id)
    verb = inflect("aperi", tense)   # aperi → aperis / aperas
    return f"{verb.capitalize()} {ent_form}."


# ---- top-level -----------------------------------------------------------

def realize_trace(
    trace: Trace, lexicon: Lexicon, *,
    scene_location_id: Optional[str] = None,
    rng: Optional[random.Random] = None,
    tense: Optional[str] = None,
    setup_relations: Optional[list] = None,
) -> str:
    """Render the full trace as a paragraph of Esperanto.

    `setup_relations`: if provided, renders scene-setup sentences from
    this list instead of `trace.relations`. Use case: rules that
    modify relations (e.g. `preni` transferring ownership via
    remove/add) leave `trace.relations` in the *post*-trace state.
    Snapshotting relations before running the engine and passing the
    snapshot here makes the rendered setup reflect the starting world
    rather than the final one.

    `rng`: optional per-trace RNG for surface variation (templates,
    connectives, pronouns, tense). When None the realizer picks the
    first variant of every choice (deterministic; backwards-compatible
    with old tests, which all use past tense).

    `tense`: explicit override ('is' for past, 'as' for present). When
    None, picked once per trace from the rng (50/50 past/present), or
    defaults to 'is' if rng is also None.
    """
    if tense is None:
        tense = _pick_tense(rng)

    mentioned: set[str] = set()
    raw: list[tuple[str, bool]] = []   # (text, has_cause_in_prose)

    for sent in _synthetic_scene_grounding(
            trace, scene_location_id, mentioned, rng, tense):
        raw.append((sent, False))

    relations_for_setup = (setup_relations if setup_relations is not None
                           else trace.relations)
    for rel in relations_for_setup:
        sent = render_relation(rel, trace, lexicon, mentioned,
                               scene_location_id=scene_location_id,
                               rng=rng, tense=tense)
        if sent:
            raw.append((sent, False))

    skip_ids = _use_instrument_skip_set(trace, lexicon)
    rendered_event_ids: set[str] = set()
    for ev in trace.events:
        if ev.id in skip_ids:
            continue
        sent = render_event(ev, trace, lexicon, mentioned,
                            scene_location_id=scene_location_id,
                            rng=rng, tense=tense)
        if sent is None:
            continue
        has_cause_in_prose = any(
            c in rendered_event_ids for c in ev.caused_by)
        raw.append((sent, has_cause_in_prose))
        rendered_event_ids.add(ev.id)
        # Step 5: each created entity gets an appearance line right after
        # the event that creates it. First mention; mark mentioned so
        # later events use definite reference.
        for created in ev.creates:
            if created.id in mentioned:
                continue
            app = _render_appearance_line(
                created, mentioned, scene_location_id, rng, trace, tense)
            # Appearance lines aren't causally connected — they're a
            # narrative consequence of the previous sentence, not a new
            # event in the cascade. Render without a connective.
            raw.append((app, False))

    out = [
        format_with_connective(s, caused, is_first=(i == 0), rng=rng)
        for i, (s, caused) in enumerate(raw)
    ]
    return " ".join(out)
