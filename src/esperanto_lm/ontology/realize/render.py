"""Message → prose. Dispatches on Message type; reuses the old
realizer's templates, articles, pronouns, and morphology. A new
`CoordinatedMessage` renderer emits children as 'X [v1] kaj [v2]',
dropping the shared subject on the second verb phrase.

Kept deliberately self-contained — `plan_messages` + `render_messages`
are the only seams. New message kinds become new render methods.
"""
from __future__ import annotations

import random
from typing import Optional

from ..causal import Event, RelationAssertion, Trace
from ..loader import Lexicon
from .messages import (
    AppearanceMessage,
    CoordinatedMessage,
    DestructionMessage,
    EventMessage,
    Message,
    RelationAddedMessage,
    RelationMessage,
    RelationRemovedMessage,
    SceneGroundingMessage,
    SubordinatedMessage,
)


# =================== morphology + naming helpers ==================

PRONOUN_OF_NAME: dict[str, str] = {
    "petro": "li", "johano": "li", "pavel": "li", "mikael": "li",
    "maria": "ŝi", "anna": "ŝi", "klara": "ŝi", "sara": "ŝi",
    "elena": "ŝi", "lidia": "ŝi",
}

PRONOUN_RATE = 0.55

_PRONOUNS_BASE = {"li", "ŝi", "ĝi", "ili", "mi", "vi", "ni"}


def inflect(verb_lemma: str, tense: str) -> str:
    """tranĉi + 'is' → tranĉis. Regular Esperanto: strip final -i,
    append tense suffix."""
    if verb_lemma.endswith("i"):
        return verb_lemma[:-1] + tense
    return verb_lemma + tense


def past_tense(verb_lemma: str) -> str:
    """Backwards-compat alias. Prefer `inflect(lemma, 'is')`."""
    return inflect(verb_lemma, "is")


def to_accusative(noun_form: str) -> str:
    """Append -n. Handles 'la X' → 'la Xn' and pronouns (li → lin)."""
    if " " in noun_form:
        head, _, tail = noun_form.rpartition(" ")
        return f"{head} {to_accusative(tail)}"
    if noun_form in _PRONOUNS_BASE:
        return noun_form + "n"
    if noun_form.endswith(("o", "oj", "a", "aj")):
        return noun_form + "n"
    return noun_form


# =================== templates ==================

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
    "apud": [
        lambda a, b, t: f"{a} est{t} apud {b}.",
        lambda a, b, t: f"Apud {b} est{t} {a}.",
    ],
}

CONNECTIVES = ["Tial", "Sekve", "Pro tio", ""]
TENSES = ["is", "as"]


def _pick(rng: Optional[random.Random], options: list):
    return options[0] if rng is None else rng.choice(options)


def _pick_tense(rng: Optional[random.Random]) -> str:
    return "is" if rng is None else rng.choice(TENSES)


# =================== naming / article tracker ==================

def _pronoun_unambiguous(name: str, trace: Trace) -> bool:
    pronoun = PRONOUN_OF_NAME.get(name)
    if pronoun is None:
        return False
    for eid, ent in trace.entities.items():
        if eid == name or ent.entity_type != "person":
            continue
        if PRONOUN_OF_NAME.get(eid) == pronoun:
            return False
    return True


def _name_for(
    entity, mentioned: set[str], *,
    scene_location_id: Optional[str] = None,
    rng: Optional[random.Random] = None,
    trace: Optional[Trace] = None,
) -> str:
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


# =================== context ==================

class _Ctx:
    """Per-trace rendering state. Not a dataclass because `mentioned`
    mutates in place and we don't want accidental copies."""
    __slots__ = ("trace", "lexicon", "mentioned", "rng", "tense",
                 "scene_location_id", "rendered_event_ids")

    def __init__(self, trace, lexicon, *, scene_location_id, rng, tense):
        self.trace = trace
        self.lexicon = lexicon
        self.mentioned: set[str] = set()
        self.rng = rng
        self.tense = tense
        self.scene_location_id = scene_location_id
        self.rendered_event_ids: set[str] = set()

    def name_for(self, entity) -> str:
        return _name_for(
            entity, self.mentioned,
            scene_location_id=self.scene_location_id,
            rng=self.rng, trace=self.trace)


# =================== per-message renderers ==================

def _render_scene_grounding(m: SceneGroundingMessage, ctx: _Ctx) -> Optional[str]:
    if ctx.scene_location_id is None:
        return None
    scene_ent = ctx.trace.entities.get(ctx.scene_location_id)
    ent = ctx.trace.entities.get(m.entity_id)
    if scene_ent is None or ent is None:
        return None
    ent_form = ctx.name_for(ent)
    ctx.mentioned.add(ent.id)
    scene_form = ctx.name_for(scene_ent)
    ctx.mentioned.add(scene_ent.id)
    template = _pick(ctx.rng, RELATION_TEMPLATES["en"])
    return template(ent_form, scene_form, ctx.tense)


def _render_relation(m: RelationMessage, ctx: _Ctx) -> Optional[str]:
    rel = m.relation
    a = ctx.trace.entity(rel.args[0])
    b = ctx.trace.entity(rel.args[1]) if len(rel.args) > 1 else None
    if a is None or b is None:
        return None
    a_form = ctx.name_for(a)
    ctx.mentioned.add(a.id)
    if rel.relation == "havi":
        b_form = to_accusative(ctx.name_for(b))
        template = _pick(ctx.rng, RELATION_TEMPLATES["havi"])
        sent = template(a_form, b_form, ctx.tense)
    elif rel.relation in ("en", "sur", "apud"):
        b_form = ctx.name_for(b)
        template = _pick(ctx.rng, RELATION_TEMPLATES[rel.relation])
        sent = template(a_form, b_form, ctx.tense)
    else:
        return None
    ctx.mentioned.add(b.id)
    return sent


def _render_event(m: EventMessage, ctx: _Ctx) -> Optional[str]:
    return _render_event_phrase(m.event, ctx, drop_subject=False)


def _render_event_phrase(
    ev: Event, ctx: _Ctx, *, drop_subject: bool,
) -> Optional[str]:
    """Render one event. When `drop_subject=True`, emit only the verb
    phrase (no subject) — used by CoordinatedMessage for children
    past the first."""
    action = ctx.lexicon.actions.get(ev.action)
    if action is None:
        return None
    role_names = {r.name for r in action.roles}

    # Subject / impersonal dispatch.
    if "agent" in role_names and ev.roles.get("agent"):
        subject_id = ev.roles["agent"]
        subject_role_name = "agent"
    elif "theme" in role_names and ev.roles.get("theme"):
        subject_id = ev.roles["theme"]
        subject_role_name = "theme"
    elif "location" in role_names and ev.roles.get("location"):
        # Impersonal — no subject to drop even in coordination.
        loc = ctx.trace.entity(ev.roles["location"])
        if loc is None:
            return None
        loc_form = ctx.name_for(loc)
        ctx.mentioned.add(loc.id)
        return f"En {loc_form} {inflect(ev.action, ctx.tense)}."
    else:
        return None

    subject = ctx.trace.entity(subject_id)
    if subject is None:
        return None

    parts: list[str] = []
    if not drop_subject:
        subject_form = ctx.name_for(subject)
        ctx.mentioned.add(subject.id)
        parts.append(subject_form)
    else:
        # Skip subject; but still mark as mentioned for downstream
        # anaphora consistency.
        ctx.mentioned.add(subject.id)
    parts.append(inflect(ev.action, ctx.tense))

    if subject_role_name == "agent" and ev.roles.get("theme"):
        theme = ctx.trace.entity(ev.roles["theme"])
        if theme is not None:
            parts.append(to_accusative(ctx.name_for(theme)))
            ctx.mentioned.add(theme.id)

    if ev.roles.get("instrument"):
        instr = ctx.trace.entity(ev.roles["instrument"])
        if instr is not None:
            parts.append(f"per {ctx.name_for(instr)}")
            ctx.mentioned.add(instr.id)

    if ev.roles.get("recipient"):
        recip = ctx.trace.entity(ev.roles["recipient"])
        if recip is not None:
            parts.append(f"al {ctx.name_for(recip)}")
            ctx.mentioned.add(recip.id)

    if ev.roles.get("location"):
        loc = ctx.trace.entity(ev.roles["location"])
        if loc is not None:
            prep = "en" if loc.entity_type == "location" else "sur"
            parts.append(f"{prep} {ctx.name_for(loc)}")
            ctx.mentioned.add(loc.id)

    if ev.roles.get("destination"):
        dest = ctx.trace.entity(ev.roles["destination"])
        if dest is not None:
            parts.append(f"al {ctx.name_for(dest)}")
            ctx.mentioned.add(dest.id)

    # Strip trailing period for coordination (caller adds one).
    sent = " ".join(parts)
    if not drop_subject:
        sent += "."
    return sent


def _render_appearance(m: AppearanceMessage, ctx: _Ctx) -> Optional[str]:
    ent = ctx.trace.entities.get(m.entity_id)
    if ent is None or ent.id in ctx.mentioned:
        return None
    form = ctx.name_for(ent)
    ctx.mentioned.add(ent.id)
    verb = inflect("aperi", ctx.tense)
    return f"{verb.capitalize()} {form}."


def _render_relation_removed(
    m: RelationRemovedMessage, ctx: _Ctx,
) -> Optional[str]:
    """Narrate a relation that no longer holds. Shape depends on the
    relation — havi removal says 'ne plu havis', en removal is usually
    implicit in the move verb and omitted."""
    if m.relation != "havi":
        # en/sur removals are implied by the corresponding move event
        # ('Petro iris al salono' → reader infers he's no longer in
        # the kitchen). Skip narration.
        return None
    owner = ctx.trace.entities.get(m.args[0])
    theme = ctx.trace.entities.get(m.args[1])
    if owner is None or theme is None:
        return None
    owner_form = ctx.name_for(owner)
    ctx.mentioned.add(owner.id)
    theme_form = to_accusative(ctx.name_for(theme))
    ctx.mentioned.add(theme.id)
    return f"{owner_form} ne plu hav{ctx.tense} {theme_form}."


def _render_relation_added(
    m: RelationAddedMessage, ctx: _Ctx,
) -> Optional[str]:
    """Narrate a new relation. Skip when already implied by the
    triggering event — for havi, we DO narrate because the new owner
    may be a different phrase than the event's subject."""
    if m.relation != "havi":
        return None
    owner = ctx.trace.entities.get(m.args[0])
    theme = ctx.trace.entities.get(m.args[1])
    if owner is None or theme is None:
        return None
    owner_form = ctx.name_for(owner)
    ctx.mentioned.add(owner.id)
    theme_form = to_accusative(ctx.name_for(theme))
    ctx.mentioned.add(theme.id)
    return f"Nun {owner_form} hav{ctx.tense} {theme_form}."


def _render_destruction(
    m: DestructionMessage, ctx: _Ctx,
) -> Optional[str]:
    ent = ctx.trace.entities.get(m.entity_id)
    if ent is None:
        return None
    form = ctx.name_for(ent)
    ctx.mentioned.add(ent.id)
    verb = inflect("malaperi", ctx.tense)
    return f"{form} {verb}."


def _render_coordinated(
    m: CoordinatedMessage, ctx: _Ctx,
) -> Optional[str]:
    """Children share a subject. Render first child fully, then each
    subsequent as a bare verb phrase joined with 'kaj'."""
    if not m.children:
        return None
    first = m.children[0]
    if not isinstance(first, EventMessage):
        # Only coordinating events in this slice.
        return None
    first_sent = _render_event_phrase(first.event, ctx, drop_subject=False)
    if first_sent is None:
        return None
    first_body = first_sent.rstrip(".")
    rest_bodies: list[str] = []
    for child in m.children[1:]:
        if not isinstance(child, EventMessage):
            continue
        phrase = _render_event_phrase(
            child.event, ctx, drop_subject=True)
        if phrase is None:
            continue
        rest_bodies.append(phrase)
    if not rest_bodies:
        return first_body + "."
    return first_body + " kaj " + " kaj ".join(rest_bodies) + "."


# =================== connectives + capitalization ==================

def _starts_with_proper_noun(sentence: str) -> bool:
    if not sentence:
        return False
    first = sentence.split(" ", 1)[0]
    if not first or not first[0].isupper():
        return False
    return first.lower() not in {"la", "ĉi", "tiu", "tio", "tial", "kaj",
                                 "sekve", "pro", "en", "sur", "estis",
                                 "estas", "nun", "aperis", "aperas"}


def _format_with_connective(
    sentence: str, has_cause_in_prose: bool, is_first: bool,
    rng: Optional[random.Random] = None,
) -> str:
    if not is_first and has_cause_in_prose:
        connective = _pick(rng, CONNECTIVES)
        if not connective:
            return sentence[0].upper() + sentence[1:]
        if _starts_with_proper_noun(sentence):
            return f"{connective} {sentence}"
        return f"{connective} {sentence[0].lower()}{sentence[1:]}"
    return sentence[0].upper() + sentence[1:]


# =================== dispatch ==================

def _render_subordinated(
    m: SubordinatedMessage, ctx: _Ctx,
) -> Optional[str]:
    """`<main>, <conjunction> <subordinate>.` One sentence, comma-
    joined. Subordinate clause drops its final period and starts
    lowercase since it's no longer at sentence boundary."""
    main_sent = _render_message(m.main, ctx)
    if main_sent is None:
        return None
    sub_sent = _render_message(m.subordinate, ctx)
    if sub_sent is None:
        return main_sent
    main_body = main_sent.rstrip(".")
    sub_body = sub_sent.rstrip(".")
    sub_body = sub_body[0].lower() + sub_body[1:] if sub_body else sub_body
    return f"{main_body}, {m.conjunction} {sub_body}."


def _render_message(m: Message, ctx: _Ctx) -> Optional[str]:
    """Central dispatch — used by SubordinatedMessage's recursive
    render as well as the top-level pipeline."""
    render_fn = _DISPATCH.get(type(m))
    if render_fn is None:
        return None
    return render_fn(m, ctx)


def _collect_rendered_event_ids(m: Message, out: set[str]) -> None:
    """Walk a message tree and add every contained EventMessage's
    event id to `out`. Used by the top-level loop so a follow-on
    message whose cause is nested inside a CoordinatedMessage or
    SubordinatedMessage still triggers a connective."""
    if isinstance(m, EventMessage):
        out.add(m.event.id)
    elif isinstance(m, CoordinatedMessage):
        for c in m.children:
            _collect_rendered_event_ids(c, out)
    elif isinstance(m, SubordinatedMessage):
        _collect_rendered_event_ids(m.main, out)
        _collect_rendered_event_ids(m.subordinate, out)


_DISPATCH = {
    SceneGroundingMessage: _render_scene_grounding,
    RelationMessage: _render_relation,
    EventMessage: _render_event,
    AppearanceMessage: _render_appearance,
    RelationRemovedMessage: _render_relation_removed,
    RelationAddedMessage: _render_relation_added,
    DestructionMessage: _render_destruction,
    CoordinatedMessage: _render_coordinated,
    SubordinatedMessage: _render_subordinated,
}


def render_messages(
    messages: list[Message], trace: Trace, lexicon: Lexicon, *,
    scene_location_id: Optional[str] = None,
    rng: Optional[random.Random] = None,
    tense: Optional[str] = None,
) -> str:
    if tense is None:
        tense = _pick_tense(rng)
    ctx = _Ctx(trace, lexicon,
               scene_location_id=scene_location_id, rng=rng, tense=tense)

    raw: list[tuple[str, bool]] = []   # (text, has_cause_in_prose)
    for msg in messages:
        render_fn = _DISPATCH.get(type(msg))
        if render_fn is None:
            continue
        sent = render_fn(msg, ctx)
        if sent is None:
            continue
        has_cause = (
            msg.cause_event_id is not None
            and msg.cause_event_id in ctx.rendered_event_ids)
        raw.append((sent, has_cause))
        # Track rendered events for connective causation lookups.
        _collect_rendered_event_ids(msg, ctx.rendered_event_ids)

    out = [
        _format_with_connective(s, caused, is_first=(i == 0), rng=rng)
        for i, (s, caused) in enumerate(raw)
    ]
    return " ".join(out)
