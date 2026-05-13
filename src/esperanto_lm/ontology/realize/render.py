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


_VERB_META_CACHE: dict | None = None


def _verb_metadata() -> dict:
    """Lazily-built classification of verbs from the rule structure
    (consumption / transfer / acquisition / instrument-quantified).
    Replaces the hand-curated action-name lists this module used to
    keep — any new verb whose rule uses `consume_one` or `transfer_n`
    is automatically picked up. Cached on first call."""
    global _VERB_META_CACHE
    if _VERB_META_CACHE is None:
        from ..dsl.introspect import (
            acquisition_verbs, consumption_verbs,
            instrument_quantified_verbs, transfer_verbs,
        )
        from ..dsl.rules import DEFAULT_DSL_RULES
        rules = list(DEFAULT_DSL_RULES)
        _VERB_META_CACHE = {
            "consumption": consumption_verbs(rules),
            "transfer": transfer_verbs(rules),
            "acquisition": acquisition_verbs(rules),
            "instrument_quantified": instrument_quantified_verbs(rules),
        }
    return _VERB_META_CACHE
from .messages import (
    AppearanceMessage,
    CoordinatedMessage,
    DestructionMessage,
    EventMessage,
    GroupedRelationMessage,
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


def _surface_verb_lemma(ev, trace, lex) -> str:
    """Pick the verb lemma to render. For most events this is just
    ev.action. For `fari` events, swap to the instrument concept's
    functional_signature when bound — so bulko-with-forno reads
    "bakas" rather than "faras", and martelo-driven construction
    reads "marteli". Sandwich-style assemblies (no instrument) stay
    on "fari"."""
    if ev.action != "fari":
        return ev.action
    instr_eid = ev.roles.get("instrument")
    if not instr_eid:
        return "fari"
    instr_ent = trace.entities.get(instr_eid)
    if instr_ent is None:
        return "fari"
    concept = lex.concepts.get(instr_ent.concept_lemma)
    if concept is None:
        return "fari"
    fs = concept.properties.get("functional_signature", ())
    return fs[0] if fs else "fari"


def past_tense(verb_lemma: str) -> str:
    """Backwards-compat alias. Prefer `inflect(lemma, 'is')`."""
    return inflect(verb_lemma, "is")


def to_accusative(noun_form: str) -> str:
    """Inflect a noun phrase for accusative. Multi-word phrases inflect
    every nominal/adjectival element: 'la fragila pomo' → 'la fragilan
    pomon', 'la du fragilaj pomoj' → 'la du fragilajn pomojn'. Articles
    ('la') and numerals ('tri', 'dek') stay invariant; pronouns and
    -o/-oj/-a/-aj endings get -n."""
    if " " in noun_form:
        return " ".join(to_accusative(w) for w in noun_form.split(" "))
    if noun_form == "la":
        return noun_form
    if noun_form in _PRONOUNS_BASE:
        return noun_form + "n"
    if noun_form.endswith(("o", "oj", "a", "aj")):
        return noun_form + "n"
    return noun_form


def to_plural(noun_form: str) -> str:
    """Append plural -j to the head noun. 'pomo' → 'pomoj';
    'la pomo' → 'la pomoj'. Plural article 'la' stays singular form
    in Esperanto (la is invariant)."""
    if " " in noun_form:
        head, _, tail = noun_form.rpartition(" ")
        return f"{head} {to_plural(tail)}"
    if noun_form.endswith("o"):
        return noun_form + "j"
    if noun_form.endswith("a"):
        return noun_form + "j"
    return noun_form


# Esperanto cardinals 1-99. For corpus purposes, scenes won't have
# more than ~10 of anything, but the table covers wider just in case.
_ESPERANTO_DIGIT = {
    0: "nul", 1: "unu", 2: "du", 3: "tri", 4: "kvar", 5: "kvin",
    6: "ses", 7: "sep", 8: "ok", 9: "naŭ",
}
_ESPERANTO_TEN = {
    1: "dek", 2: "dudek", 3: "tridek", 4: "kvardek", 5: "kvindek",
    6: "sesdek", 7: "sepdek", 8: "okdek", 9: "naŭdek",
}


def int_to_esperanto(n: int) -> str:
    """Render an integer 0-99 as an Esperanto numeral. Falls back to
    the digit string for out-of-range values (shouldn't happen for
    our corpus)."""
    if 0 <= n <= 9:
        return _ESPERANTO_DIGIT[n]
    if 10 <= n <= 99:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return _ESPERANTO_TEN[tens]
        return _ESPERANTO_TEN[tens] + _ESPERANTO_DIGIT[ones]
    return str(n)


def _entity_count(entity) -> int:
    """Read the count slot, defaulting to 1. Stored as a string in
    properties (slot vocabulary is string-list)."""
    raw = entity.properties.get("count", ["1"])
    if not raw:
        return 1
    try:
        return int(raw[0])
    except (TypeError, ValueError):
        return 1


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
TENSES = ["is", "as", "os"]


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


ADJECTIVE_RATE = 0.30  # per-mention probability of attaching an adjective
ALIAS_RATE = 0.20      # per back-reference probability of using a category alias


def _concept_aliases(
    concept_lemma: str, lex: Lexicon, *,
    entity_id: Optional[str] = None,
    derived=None,
) -> list[str]:
    """Transitively walk `concept.category` to collect superordinate
    lemmas that can stand in for `concept_lemma` on a back-reference.
    pomo → frukto → manĝaĵo: a `pomo` entity yields ["frukto", "manĝaĵo"].

    When `entity_id` and `derived` are given, also folds in any
    contextual category labels the engine has tagged on the entity —
    e.g. a viro entity in `gepatro(this, ?)` carries the derived
    `patro` label. Derived labels each get their own concept-category
    chain walked so "patro → viro" still surfaces as a deeper alias.

    Cycle-safe via a visit set. Returns lemmas in the order encountered
    (most-specific parent first), so the caller's random choice
    naturally surfaces both levels with similar frequency."""
    out: list[str] = []
    visited: set[str] = {concept_lemma}
    frontier = [concept_lemma]
    if derived is not None and entity_id is not None:
        for label in derived.categories_for(entity_id):
            if label not in visited:
                visited.add(label)
                out.append(label)
                frontier.append(label)
    while frontier:
        cur = frontier.pop(0)
        concept = lex.concepts.get(cur)
        if concept is None:
            continue
        for parent in getattr(concept, "category", ()):
            if parent in visited:
                continue
            visited.add(parent)
            out.append(parent)
            frontier.append(parent)
    return out


def _pick_adjective(
    entity, *, lexicon, rng: Optional[random.Random],
    history: Optional[dict[str, set[str]]] = None,
) -> Optional[str]:
    """Return a slot value to render attributively for `entity`, or None.

    Picks among slots flagged `adjectival: true` in the lexicon whose
    value is set on this entity. Persons and entities without rng are
    skipped — adjectives on bare names ("la malsata Maria") read
    stilted. The `history` map lets the caller cycle through different
    slots across multiple mentions of the same entity, so we don't get
    "fragila pomo … fragila pomo" two clauses apart.
    """
    if rng is None or lexicon is None:
        return None
    if entity.entity_type == "person":
        return None
    if rng.random() >= ADJECTIVE_RATE:
        return None
    candidates: list[tuple[str, str]] = []   # (slot, value)
    for slot_name, values in entity.properties.items():
        slot_def = lexicon.slots.get(slot_name)
        if slot_def is None or not getattr(slot_def, "adjectival", False):
            continue
        if not values:
            continue
        value = values[0]
        # Skip unmarked / default values — saying "fortika lampo" or
        # "luma valo" or "sata Maria" is noise. Only marked deviations
        # (fragila, malluma, malsata) carry information worth surfacing.
        if getattr(slot_def, "unmarked", None) == value:
            continue
        candidates.append((slot_name, value))
    if not candidates:
        return None
    used = (history.get(entity.id, set())
            if history is not None else set())
    fresh = [(s, v) for (s, v) in candidates if s not in used]
    pool = fresh if fresh else candidates
    slot, value = rng.choice(pool)
    if history is not None:
        history.setdefault(entity.id, set()).add(slot)
    return value


def _inflect_adjective(adj: str, *, plural: bool) -> str:
    """Apply -j for plural agreement. Accusative -n is added later by
    `to_accusative` (called on the rendered phrase); we don't want to
    double-decline if we add it here."""
    return adj + "j" if plural else adj


def _name_for(
    entity, mentioned: set[str], *,
    scene_location_id: Optional[str] = None,
    rng: Optional[random.Random] = None,
    trace: Optional[Trace] = None,
    count_override: Optional[int] = None,
    lexicon: Optional[Lexicon] = None,
    adjective_history: Optional[dict[str, set[str]]] = None,
    alias_history: Optional[dict[str, set[str]]] = None,
    derived=None,
) -> str:
    if entity.entity_type == "person":
        name = entity.id
        # Derived-category alias on back-reference: a viro entity in
        # `gepatro(this, ?)` carries the derived `patro` label and can
        # be referred to as "la patro" instead of by name. Same
        # ALIAS_RATE / cycle-through-history shape as the non-person
        # case below. Falls through to pronoun / name when the
        # rng / mentioned / derived gates don't all line up.
        if (rng is not None
                and entity.id in mentioned
                and alias_history is not None
                and derived is not None
                and rng.random() < ALIAS_RATE):
            cats = list(derived.categories_for(entity.id))
            if cats:
                used = alias_history.get(entity.id, set())
                fresh = [c for c in cats if c not in used]
                pool = fresh if fresh else cats
                chosen = rng.choice(pool)
                alias_history.setdefault(entity.id, set()).add(chosen)
                return f"la {chosen}"
        if (rng is not None and trace is not None
                and entity.id in mentioned
                and _pronoun_unambiguous(name, trace)
                and rng.random() < PRONOUN_RATE):
            return PRONOUN_OF_NAME[name]
        return name.capitalize()
    lemma = entity.concept_lemma
    # On back-references, sometimes substitute a superordinate
    # category alias for the bare lemma — "la pomo" → "la frukto".
    # First mentions keep the specific lemma so the model gets a
    # binding anchor; back-references vary so it learns the
    # genus-species coreference. Cycle through the available
    # parents per entity via `alias_history` so we don't repeat.
    # Folds in derived categories (e.g. role-from-relation labels)
    # alongside concept.category so contextual roles surface here too.
    if (rng is not None and lexicon is not None
            and entity.id in mentioned
            and alias_history is not None
            and rng.random() < ALIAS_RATE):
        aliases = _concept_aliases(
            lemma, lexicon,
            entity_id=entity.id, derived=derived)
        if aliases:
            used = alias_history.get(entity.id, set())
            fresh = [a for a in aliases if a not in used]
            pool = fresh if fresh else aliases
            chosen = rng.choice(pool)
            alias_history.setdefault(entity.id, set()).add(chosen)
            lemma = chosen
    # `count_override` is set by event-rendering for consumption /
    # transfer verbs that operate on N units of a stack: "Maria manĝis
    # du pomojn" when Maria's stack of 5 is being decremented by 2.
    # The override forces the rendering to use the action's quantity
    # rather than the entity's full count. Override of 1 → singular
    # ("la pomon"); override > 1 → plural with that numeral.
    natural_count = _entity_count(entity)
    count = (count_override
             if count_override is not None
             else natural_count)
    # Partial-quantity transfers / consumptions render without "la":
    # "Maria prenis du pomojn" reads better than "la du pomojn" when
    # only 2 of a 4-stack move. Full-stack actions keep "la" via the
    # mentioned/scene-location path.
    is_partial = (
        count_override is not None
        and count_override < natural_count)
    # Optional attributive adjective. Picked from the entity's
    # adjectival-flagged slot values; rate-limited per call. Cycles
    # slots across mentions via `adjective_history`. Skipped for
    # persons inside `_pick_adjective`.
    adj = _pick_adjective(
        entity, lexicon=lexicon, rng=rng, history=adjective_history)
    if count > 1:
        plural_lemma = to_plural(lemma)
        numeral = int_to_esperanto(count)
        adj_part = (
            f"{_inflect_adjective(adj, plural=True)} "
            if adj else "")
        if (not is_partial
                and (entity.id == scene_location_id
                     or entity.id in mentioned)):
            return f"la {numeral} {adj_part}{plural_lemma}"
        return f"{numeral} {adj_part}{plural_lemma}"
    adj_part = f"{adj} " if adj else ""
    if entity.id == scene_location_id or entity.id in mentioned:
        return f"la {adj_part}{lemma}"
    return f"{adj_part}{lemma}"


# =================== context ==================

NONPERSON_PRONOUN_RATE = 0.7


class _Ctx:
    """Per-trace rendering state. Not a dataclass because `mentioned`
    mutates in place and we don't want accidental copies."""
    __slots__ = ("trace", "lexicon", "mentioned", "rng", "tense",
                 "scene_location_id", "rendered_event_ids",
                 "last_nonperson", "adjective_history",
                 "alias_history", "derived")

    def __init__(self, trace, lexicon, *, scene_location_id, rng, tense,
                 derived=None):
        self.trace = trace
        self.lexicon = lexicon
        self.mentioned: set[str] = set()
        self.rng = rng
        self.tense = tense
        self.scene_location_id = scene_location_id
        self.derived = derived
        self.rendered_event_ids: set[str] = set()
        # Tracks the most recently mentioned non-person entity id.
        # Cleared to None whenever a *different* non-person is mentioned,
        # so `ĝi`/`ĝin` substitution only fires when the referent is
        # unambiguous.
        self.last_nonperson: Optional[str] = None
        # entity_id -> set of slot names already used as adjectives.
        # `_pick_adjective` consults this to cycle slots across
        # mentions instead of always picking the same one.
        self.adjective_history: dict[str, set[str]] = {}
        # entity_id -> set of category aliases already used in lemma
        # substitution. Cycles parent aliases the same way so an
        # entity tagged with category=["frukto"] and (transitively)
        # "manĝaĵo" alternates between both.
        self.alias_history: dict[str, set[str]] = {}

    def name_for(self, entity, *,
                 count_override: Optional[int] = None) -> str:
        return _name_for(
            entity, self.mentioned,
            scene_location_id=self.scene_location_id,
            rng=self.rng, trace=self.trace,
            count_override=count_override,
            lexicon=self.lexicon,
            adjective_history=self.adjective_history,
            alias_history=self.alias_history,
            derived=self.derived)

    def theme_form(self, entity, *,
                   count_override: Optional[int] = None) -> str:
        """Render a theme-position entity, pronominalizing to `ĝin`
        when the entity is the currently-salient non-person referent
        AND it's been mentioned before. Only fires for non-persons;
        persons use the existing `li`/`ŝi` pathway via `name_for`.

        `count_override` forces a specific count for the rendering
        — used by event rendering when the verb operates on N units
        of a stack (e.g. manĝi.quantity=2 → "du pomojn")."""
        if (entity.entity_type != "person"
                and entity.id in self.mentioned
                and self.last_nonperson == entity.id
                and self.rng is not None
                and self.rng.random() < NONPERSON_PRONOUN_RATE):
            return "ĝin"
        return to_accusative(self.name_for(
            entity, count_override=count_override))

    def mark_nonperson_mention(self, entity) -> None:
        """Call whenever a non-person entity is mentioned in rendered
        prose. Pins the salient-referent slot so the next theme-
        position mention of the same entity can pronominalize — or
        invalidates the slot if it's a different entity."""
        if entity.entity_type == "person":
            return
        self.last_nonperson = entity.id

    def note_mention(self, entity) -> None:
        """Record an entity as mentioned in rendered prose. Adds to
        the `mentioned` set for article/pronoun resolution AND
        updates the non-person salient slot for `ĝi`/`ĝin`
        substitution."""
        self.mentioned.add(entity.id)
        self.mark_nonperson_mention(entity)


# =================== per-message renderers ==================

def _render_entity_quality(m, ctx: _Ctx) -> Optional[str]:
    """'La sofo estas malseka.' — predicative attribution. The
    quality lemma is already in adjective form (Esperanto -a ending),
    no inflection needed for predicative use (predicative adjective
    stays in nominative singular form to match the singular subject).

    When the (slot, quality_lemma) has a producing verb in the
    lexicon's `state_verbs` index AND rng allows, contracts the
    predicate to verbal form ("la pordo ŝlositas"). Stilted but
    valid Esperanto and gives the model lexical variation."""
    ent = ctx.trace.entities.get(m.entity_id)
    if ent is None:
        return None
    form = ctx.name_for(ent)
    ctx.note_mention(ent)
    predicate = _state_predicate(m.slot, m.quality_lemma, ctx)
    return f"{form} {predicate}."


# Probability of contracting "estas X-ita" → "X-itas" when a verbal
# form is available. Stilted but grammatical; the variation is for
# the language model's benefit. Independent of the agent-state vs
# theme-state distinction — both can contract.
_VERBAL_PREDICATE_RATE = 0.30


def _state_predicate(
    slot: Optional[str], quality_lemma: str, ctx: _Ctx,
) -> str:
    """Render the copula+quality predicate for a state.

    Default: "estas X" (or estis/estos by tense). When the (slot,
    quality_lemma) has a producing verb AND rng rolls under
    _VERBAL_PREDICATE_RATE, returns the verbal-contracted form
    (drops the trailing -a of the participle and appends the tense
    suffix: "ŝlosita" → "ŝlositas"). The slot is None when the
    caller doesn't have it; in that case we fall back to the
    adjectival form."""
    copula = f"est{ctx.tense}"
    if slot is None or ctx.rng is None or ctx.lexicon is None:
        return f"{copula} {quality_lemma}"
    verbs = ctx.lexicon.state_verbs.get((slot, quality_lemma))
    if not verbs:
        return f"{copula} {quality_lemma}"
    if not quality_lemma.endswith("a"):
        return f"{copula} {quality_lemma}"
    if ctx.rng.random() >= _VERBAL_PREDICATE_RATE:
        return f"{copula} {quality_lemma}"
    return f"{quality_lemma[:-1]}{ctx.tense}"


def _render_scene_grounding(m: SceneGroundingMessage, ctx: _Ctx) -> Optional[str]:
    if ctx.scene_location_id is None:
        return None
    scene_ent = ctx.trace.entities.get(ctx.scene_location_id)
    ent = ctx.trace.entities.get(m.entity_id)
    if scene_ent is None or ent is None:
        return None
    ent_form = ctx.name_for(ent)
    ctx.note_mention(ent)
    scene_form = ctx.name_for(scene_ent)
    ctx.note_mention(scene_ent)
    template = _pick(ctx.rng, RELATION_TEMPLATES["en"])
    return template(ent_form, scene_form, ctx.tense)


def _render_relation(m: RelationMessage, ctx: _Ctx) -> Optional[str]:
    rel = m.relation
    a = ctx.trace.entity(rel.args[0])
    b = ctx.trace.entity(rel.args[1]) if len(rel.args) > 1 else None
    if a is None or b is None:
        return None
    # Renderable-relation gate. Without this, non-renderable
    # relations (havas_parton, subjekto, objekto, ...) still triggered
    # name_for + note_mention as a side effect, marking entities as
    # "already mentioned" before any prose actually named them. That
    # caused first-reference pronominalization: "Ŝi estos en la
    # koridoro" before "Sara" had ever surfaced. Bail early so
    # mentions only happen when the message produces text.
    if rel.relation not in ("havi", "sur", "en", "apud"):
        return None
    a_form = ctx.name_for(a)
    ctx.note_mention(a)
    if rel.relation == "havi":
        b_form = to_accusative(ctx.name_for(b))
        template = _pick(ctx.rng, RELATION_TEMPLATES["havi"])
        sent = template(a_form, b_form, ctx.tense)
    elif rel.relation == "sur":
        # Read the contained's posture: intrinsic (glaso=staranta,
        # libro=kuŝanta) or derived (sittable/lieable/imposes_pose
        # all write to posture via derivations). One uniform path —
        # no per-relation lookup logic, the data layer carries it.
        b_form = ctx.name_for(b)
        verb_root = _contextual_posture_verb_root(a, ctx)
        if verb_root is not None:
            flip = ctx.rng.random() < 0.5 if ctx.rng is not None else False
            sent = (f"{a_form} {verb_root}{ctx.tense} sur {b_form}."
                    if flip
                    else f"Sur {b_form} {verb_root}{ctx.tense} {a_form}.")
        else:
            template = _pick(ctx.rng, RELATION_TEMPLATES["sur"])
            sent = template(a_form, b_form, ctx.tense)
    elif rel.relation in ("en", "apud"):
        b_form = ctx.name_for(b)
        # If the contained has a derived posture other than the slot's
        # unmarked default, render with that posture's verb — gives
        # "Lidia naĝas en la lago" via `animate_swimming_when_in_water_body`.
        # Apud-cases naturally fall through: the only derivations that
        # set posture fire on `sur`/`en` containment, not `apud`, so
        # an animate apud the lake stays at the unmarked posture and
        # we render the bland template. No relation-literal gating here.
        verb_root = _contextual_posture_verb_root(a, ctx)
        if verb_root is not None:
            flip = ctx.rng.random() < 0.5 if ctx.rng is not None else False
            sent = (f"{a_form} {verb_root}{ctx.tense} {rel.relation} {b_form}."
                    if flip
                    else f"{rel.relation.capitalize()} {b_form} {verb_root}{ctx.tense} {a_form}.")
        else:
            template = _pick(ctx.rng, RELATION_TEMPLATES[rel.relation])
            sent = template(a_form, b_form, ctx.tense)
    else:
        return None
    ctx.note_mention(b)
    return sent


def _append_precondition_clause(
    sentence: str,
    precondition: Optional[tuple[str, str, str]],
    ctx: _Ctx,
) -> str:
    """Fold an active precondition into a sentence as a `ĉar` clause:
    "Maria manĝis la panon ĉar ŝi estis malsata." The precondition
    entity has typically just been mentioned (it's the event's
    subject or theme), so name_for will produce a pronoun where
    appropriate. No-op if precondition is None or the entity is
    missing."""
    if precondition is None:
        return sentence
    eid, slot, quality_lemma = precondition
    ent = ctx.trace.entities.get(eid)
    if ent is None:
        return sentence
    ent_form = ctx.name_for(ent)
    predicate = _state_predicate(slot, quality_lemma, ctx)
    base = sentence[:-1] if sentence.endswith(".") else sentence
    return f"{base} ĉar {ent_form} {predicate}."


def _render_event(m: EventMessage, ctx: _Ctx) -> Optional[str]:
    sentence = _render_event_phrase(
        m.event, ctx, drop_subject=False,
        source_entity_id=m.source_entity_id)
    if sentence is None:
        return None
    return _append_precondition_clause(sentence, m.precondition, ctx)


def _render_fakto_as_ke_clause(fakto_ent, ctx: _Ctx,
                               *, mode: str = "assertion") -> Optional[str]:
    """Unfold a fakto entity into a subordinate clause.
    Reads the fakto's pri_relacio (still a property — string-valued)
    and the subjekto/objekto relations.

    `mode="assertion"` (default, for rakonti/etc.):
      en  → "ke X estas en Y"
      sur → "ke X estas sur Y"
      havi → "ke X havas Y" (Y in accusative)
    `mode="question"` (for demandi):
      en/sur → "kie estas X"
      havi → "kiu havas X"
    Returns None if any field is missing or the relation isn't one
    we know how to surface yet."""
    def _unwrap_property(slot):
        v = fakto_ent.properties.get(slot, [None])
        return v[0] if isinstance(v, list) and v else v
    rel = _unwrap_property("pri_relacio")
    subj_id = None
    obj_id = None
    for r in ctx.trace.relations:
        if r.args[0] != fakto_ent.id:
            continue
        if r.relation == "subjekto":
            subj_id = r.args[1]
        elif r.relation == "objekto":
            obj_id = r.args[1]
    if rel is None or subj_id is None or obj_id is None:
        return None
    subj_ent = ctx.trace.entity(subj_id)
    obj_ent = ctx.trace.entity(obj_id)
    if subj_ent is None or obj_ent is None:
        return None
    subj_form = ctx.name_for(subj_ent)
    obj_form = ctx.name_for(obj_ent)
    ctx.note_mention(subj_ent)
    ctx.note_mention(obj_ent)
    if mode == "question":
        copula = f"est{ctx.tense}"
        if rel in ("en", "sur"):
            return f"kie {copula} {subj_form}"
        if rel == "havi":
            verb = "havis" if ctx.tense == "is" else "havas"
            return f"kiu {verb} {to_accusative(subj_form)}"
        return None
    if rel == "en":
        copula = f"est{ctx.tense}"
        return f"ke {subj_form} {copula} en {obj_form}"
    if rel == "sur":
        copula = f"est{ctx.tense}"
        return f"ke {subj_form} {copula} sur {obj_form}"
    if rel == "havi":
        verb = "havis" if ctx.tense == "is" else "havas"
        return f"ke {subj_form} {verb} {to_accusative(obj_form)}"
    return None


# Surface forms for direct quotation. Picked uniformly per direct-
# quote event so trace prose mixes all three styles. All three are
# attested in Esperanto literature; em-dash is the most common in
# narrative dialogue, guillemets in printed prose, single quotes
# come from translated/colloquial texts. Keeping them all gives
# the trainer richer formatting cues.
_QUOTE_STYLES = ("single", "em_dash", "guillemets")
# Probability per speech-act event of choosing direct quote over the
# default ke-/kie-clause indirect form. Tuned to keep roughly one in
# three speech events as direct quote — enough to be a clear signal
# without crowding out the indirect form.
_DIRECT_QUOTE_PROB = 0.30


def _render_fakto_as_quote_body(fakto_ent, ctx: _Ctx,
                                *, mode: str = "assertion") -> Optional[str]:
    """Standalone utterance form of a fakto, suitable for embedding in
    a direct-quote frame ("Maria diris al Petro: '...'").

    Same semantic content as `_render_fakto_as_ke_clause` but:
      - no leading `ke`/`kie`/`kiu` subordinator
      - first letter capitalized
      - terminal `.` (assertion) or `?` (question)
      - tense is always present (`estas`/`havas`) — direct speech is
        in-the-moment from the speaker's perspective, regardless of
        the surrounding narrative's past-tense framing.

    Returns None for fakto shapes we don't know how to surface yet
    (caller falls back to the ke-clause path)."""
    def _unwrap_property(slot):
        v = fakto_ent.properties.get(slot, [None])
        return v[0] if isinstance(v, list) and v else v
    rel = _unwrap_property("pri_relacio")
    subj_id = None
    obj_id = None
    for r in ctx.trace.relations:
        if r.args[0] != fakto_ent.id:
            continue
        if r.relation == "subjekto":
            subj_id = r.args[1]
        elif r.relation == "objekto":
            obj_id = r.args[1]
    if rel is None or subj_id is None or obj_id is None:
        return None
    subj_ent = ctx.trace.entity(subj_id)
    obj_ent = ctx.trace.entity(obj_id)
    if subj_ent is None or obj_ent is None:
        return None
    subj_form = ctx.name_for(subj_ent)
    obj_form = ctx.name_for(obj_ent)
    ctx.note_mention(subj_ent)
    ctx.note_mention(obj_ent)

    def _cap(s: str) -> str:
        return s[0].upper() + s[1:] if s else s

    if mode == "question":
        if rel in ("en", "sur"):
            return f"Kie estas {subj_form}?"
        if rel == "havi":
            return f"Kiu havas {to_accusative(subj_form)}?"
        return None
    # assertion
    if rel == "en":
        return f"{_cap(subj_form)} estas en {obj_form}."
    if rel == "sur":
        return f"{_cap(subj_form)} estas sur {obj_form}."
    if rel == "havi":
        return f"{_cap(subj_form)} havas {to_accusative(obj_form)}."
    return None


def _render_peti_request_body(theme_ent, ctx: _Ctx) -> Optional[str]:
    """Standalone imperative request for `peti`. Returns "Donu al mi
    la libron." (give-me + accusative theme). Tense is invariant —
    Esperanto imperative uses the -u suffix regardless of the
    surrounding narrative tense."""
    if theme_ent is None:
        return None
    theme_form = ctx.name_for(theme_ent)
    return f"Donu al mi {to_accusative(theme_form)}."


def _render_voki_call_body(theme_ent, ctx: _Ctx) -> Optional[str]:
    """Standalone vocative call for `voki`. Returns "Petro!" — just
    the called entity's name with terminal exclamation. The optional
    "venu!" follow-up isn't included; the bare vocative reads more
    natural for a single-event call."""
    if theme_ent is None:
        return None
    return f"{ctx.name_for(theme_ent)}!"


def _wrap_direct_quote(body: str, style: str) -> str:
    """Apply one of three Esperanto dialog conventions to `body`. The
    leading `:` glues onto the preceding recipient phrase so the join
    lands as `al Maria: 'X'` / `al Maria: — X` / `al Maria: «X»`."""
    if style == "single":
        return f": '{body}'"
    if style == "em_dash":
        return f": — {body}"
    if style == "guillemets":
        return f": «{body}»"
    return f": {body}"


_VERB_LOCATION_PREP_CACHE: dict[int, dict[str, str]] = {}


_POSE_VERB_ROOT_CACHE: dict[str, str] = {}


# Agent-state slots considered for verbal rendering, in order of
# preference when multiple are marked. sleep_state wins over posture
# because "Maria dormas" is more informative than "Maria kuŝas" for
# a sleeping Maria. Extension is data-driven — the loader's
# `state_verbs` index identifies slots whose action effects target
# the agent role; we list them here in priority order. (Hardcoding
# the priority is deliberate: nothing in the slot config expresses
# salience ordering, and it's a small list.)
_AGENT_STATE_SLOTS_BY_PREFERENCE: tuple[str, ...] = (
    "sleep_state", "posture",
)


def _contextual_posture_verb_root(entity, ctx) -> Optional[str]:
    """Verb root for an entity's most-informative agent-state, picking
    a verb over the bland `estas` template.

    Resolution per slot in preference order (sleep_state, posture):
      1. Intrinsic value (declared in concept.properties — glaso=
         posture:staranta, libro=posture:kuŝanta). Explicit opt-ins;
         the verb is always informative even if it's the slot's
         unmarked default, so we render with it unconditionally.
      2. Derived value (computed by derivations — posture:naĝanta
         for an animate en water_body, posture:penda for a fruit
         sur arbo, posture:sidanta for an animate sur sittable).
         Render with the verb unless it's the slot's unmarked
         default, since `animate_default_standing` writes staranta
         to every animate's derived state to satisfy iri's posture
         precondition — that default isn't worth rendering as "X
         staras en Y" everywhere.

    Returns the first slot with a non-default value's verb root,
    or None if no slot has informative state. Caller falls back to
    the bland `estas` template."""
    for slot_name in _AGENT_STATE_SLOTS_BY_PREFERENCE:
        intrinsic = entity.properties.get(slot_name, [])
        if intrinsic:
            return _state_verb_root(slot_name, intrinsic[0], ctx.lexicon)
        if ctx.derived is None:
            continue
        value = ctx.derived.properties.get((entity.id, slot_name))
        if not value:
            continue
        slot = ctx.lexicon.slots.get(slot_name)
        unmarked = slot.unmarked if slot is not None else None
        if value == unmarked:
            continue
        return _state_verb_root(slot_name, value, ctx.lexicon)
    return None


def _state_verb_root(slot: str, value: str, lexicon=None) -> str:
    """Verb stem for an entity-as-AGENT state participle, ready for
    tense suffix composition.

    Resolution:
      1. Lexicon's `agent_state_verbs[(slot, value)]` — the verb
         whose effect declares it produces this state on the agent
         role. Covers (posture, sidanta) → sidi, (sleep_state,
         dormanta) → dormi, (sleep_state, vekita) → vekiĝi
         (intransitive becoming, NOT veki — which targets theme).
         The stem is the lemma with the trailing -i stripped, NOT
         the morphological root: vekiĝi → vekiĝ (so vekiĝ+as =
         vekiĝas), since the morph parser would strip -iĝ as an
         affix and yield just `vek` (which would render as the
         transitive `vekas`).
      2. Morph parse — for derivation-produced values that no
         action effect declares: (posture, naĝanta) → naĝ,
         (posture, penda) → pend. Affix-stripping is what we want
         here, since the participle's morphology IS the verb stem.
    """
    cached = _POSE_VERB_ROOT_CACHE.get(value)
    if cached is not None:
        return cached
    if lexicon is not None:
        verbs = lexicon.agent_state_verbs.get((slot, value))
        if verbs:
            stem = verbs[0][:-1] if verbs[0].endswith("i") else verbs[0]
            _POSE_VERB_ROOT_CACHE[value] = stem
            return stem
    from ..morph import DefaultMorphParser
    root = DefaultMorphParser().parse(value).root
    _POSE_VERB_ROOT_CACHE[value] = root
    return root


def _verb_location_preposition(action_lemma: str, lex) -> Optional[str]:
    """Return 'en' or 'sur' if some DSL rule fires on this action and
    adds an en/sur relation between vars bound to the action's theme
    and location roles. None when no rule adds either — caller falls
    back to the entity-type heuristic.

    Cached per-lexicon so we don't walk DEFAULT_DSL_RULES on every
    event render. Cache invalidates with the lex object identity."""
    cached = _VERB_LOCATION_PREP_CACHE.get(id(lex))
    if cached is None:
        from ..dsl.rules import DEFAULT_DSL_RULES
        from ..dsl.effects import AddRelation
        from ..dsl.patterns import EventPattern, BindPattern, Var
        cached = {}
        for rule in DEFAULT_DSL_RULES:
            if not isinstance(rule.when, EventPattern):
                continue
            verb = rule.when.action
            theme_var = rule.when.role_patterns.get("theme")
            loc_var = rule.when.role_patterns.get("location")
            theme_id = id(_extract_var(theme_var)) if theme_var else None
            loc_id_var = id(_extract_var(loc_var)) if loc_var else None
            if theme_id is None or loc_id_var is None:
                continue
            effects = (rule.then if isinstance(rule.then, (list, tuple))
                       else [rule.then])
            for eff in effects:
                if not isinstance(eff, AddRelation):
                    continue
                if eff.relation not in ("en", "sur"):
                    continue
                if (len(eff.args) >= 2
                        and isinstance(eff.args[0], Var)
                        and isinstance(eff.args[1], Var)
                        and id(eff.args[0]) == theme_id
                        and id(eff.args[1]) == loc_id_var):
                    cached[verb] = eff.relation
                    break
        _VERB_LOCATION_PREP_CACHE[id(lex)] = cached
    return cached.get(action_lemma)


def _extract_var(pattern):
    """Pull the underlying Var out of a possibly-bound pattern.
    Returns the Var or None if pattern doesn't carry one."""
    from ..dsl.patterns import BindPattern, Var
    if isinstance(pattern, Var):
        return pattern
    if isinstance(pattern, BindPattern):
        return pattern.target
    return None


def _render_learned_fakto(ev, ctx) -> Optional[str]:
    """If the event's theme is a text linked via priskribas to a fakto,
    render `kaj eksciis ke <fakto>` as a tail clause. The reader sees
    what information was extracted from reading. Verb-agnostic: any
    event whose theme is a priskribas-source qualifies, but in practice
    fires for legi (the only knowledge-extraction-from-text verb).

    No-op when:
      - The theme isn't in the trace, or has no priskribas link.
      - The agent already konas the fakto pre-event (no new learning
        to narrate)."""
    theme_id = ev.roles.get("theme")
    agent_id = ev.roles.get("agent")
    if theme_id is None or agent_id is None:
        return None
    fakto_id = None
    for r in ctx.trace.relations:
        if r.relation == "priskribas" and r.args[0] == theme_id:
            fakto_id = r.args[1]
            break
    if fakto_id is None:
        return None
    fakto = ctx.trace.entities.get(fakto_id)
    if fakto is None:
        return None
    ke_clause = _render_fakto_as_ke_clause(fakto, ctx, mode="assertion")
    if ke_clause is None:
        return None
    return f"kaj eksci{ctx.tense} {ke_clause}"


def _render_event_phrase(
    ev: Event, ctx: _Ctx, *, drop_subject: bool,
    drop_theme: bool = False,
    source_entity_id: Optional[str] = None,
) -> Optional[str]:
    """Render one event. `drop_subject=True` omits the subject (used
    by CoordinatedMessage for non-leading children). `drop_theme=True`
    omits the direct object (used when a later coordinated child
    carries the same theme and will render it). `source_entity_id`
    appends "de <source>" after the theme — acquisition verbs use
    this to name the previous owner without a separate clause."""
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
        ctx.note_mention(loc)
        return f"En {loc_form} {inflect(_surface_verb_lemma(ev, ctx.trace, ctx.lexicon), ctx.tense)}."
    else:
        return None

    subject = ctx.trace.entity(subject_id)
    if subject is None:
        return None

    parts: list[str] = []
    if not drop_subject:
        subject_form = ctx.name_for(subject)
        ctx.note_mention(subject)
        parts.append(subject_form)
    else:
        # Skip subject; but still mark as mentioned for downstream
        # anaphora consistency.
        ctx.note_mention(subject)
    parts.append(inflect(_surface_verb_lemma(ev, ctx.trace, ctx.lexicon),
                          ctx.tense))

    recipient_handled = False

    # peti renders as "petis Petron pri pano" — recipient as accusative
    # direct object, theme as `pri` prepositional phrase. The standard
    # ditransitive shape ("petis la panon al Petro", parallel to doni)
    # is grammatical but not idiomatic for asking-verbs in Esperanto.
    # When the theme is elided in coordination (peti+manĝi sharing
    # pano), drop the "pri theme" but keep the accusative recipient.
    peti_handled = False
    if (subject_role_name == "agent"
            and ev.action == "peti"
            and ev.roles.get("recipient")):
        recip = ctx.trace.entity(ev.roles["recipient"])
        if recip is not None:
            # Direct-quote variant: "petis Petron: «Donu al mi la
            # libron.»". Same per-event probability + style choice
            # as the rakonti/demandi quote path. Falls back to the
            # default "petis Petron pri X" form on coin-low or when
            # the theme isn't unfoldable.
            theme_ent_for_quote = (ctx.trace.entity(ev.roles["theme"])
                                   if ev.roles.get("theme")
                                   and not drop_theme else None)
            quote_phrase = None
            if (theme_ent_for_quote is not None
                    and ctx.rng is not None
                    and ctx.rng.random() < _DIRECT_QUOTE_PROB):
                body = _render_peti_request_body(theme_ent_for_quote, ctx)
                if body is not None:
                    style = ctx.rng.choice(_QUOTE_STYLES)
                    quote_phrase = _wrap_direct_quote(body, style)
            if quote_phrase is not None:
                parts.append(
                    f"{to_accusative(ctx.name_for(recip))}{quote_phrase}")
                ctx.note_mention(recip)
                ctx.note_mention(theme_ent_for_quote)
                ctx.mark_nonperson_mention(theme_ent_for_quote)
            else:
                parts.append(to_accusative(ctx.name_for(recip)))
                ctx.note_mention(recip)
                if ev.roles.get("theme") and not drop_theme:
                    theme = ctx.trace.entity(ev.roles["theme"])
                    if theme is not None:
                        parts.append(f"pri {ctx.name_for(theme)}")
                        ctx.note_mention(theme)
                        ctx.mark_nonperson_mention(theme)
            recipient_handled = True
            peti_handled = True

    if not peti_handled and subject_role_name == "agent" and ev.roles.get("theme") and not drop_theme:
        # Reflexive shortcut: agent == theme on a `reflexive_ok` verb
        # renders as "sin" (third-person reflexive accusative). Avoids
        # "Maria sekigis Marian" — Esperanto reads that as Maria
        # drying a different person named Marian, not herself. The
        # planner only proposes agent==theme bindings for verbs that
        # opt in via the schema flag, so this gate is sufficient.
        _action_def = ctx.lexicon.actions.get(ev.action)
        is_reflexive = (
            ev.roles.get("agent") == ev.roles.get("theme")
            and _action_def is not None
            and getattr(_action_def, "reflexive_ok", False))
        if is_reflexive:
            parts.append("sin")
            ctx.note_mention(ctx.trace.entity(ev.roles["theme"]))
            theme = None  # already rendered
        else:
            theme = ctx.trace.entity(ev.roles["theme"])
        if theme is not None:
            # voki: theme is the called person; render as vocative
            # ("vokis: «Petro!»") instead of the bare accusative
            # ("vokis Petron"). Same per-event probability + style
            # choice as the rakonti/peti quote paths.
            if ev.action == "voki":
                quote_phrase = None
                if (ctx.rng is not None
                        and ctx.rng.random() < _DIRECT_QUOTE_PROB):
                    body = _render_voki_call_body(theme, ctx)
                    if body is not None:
                        style = ctx.rng.choice(_QUOTE_STYLES)
                        quote_phrase = _wrap_direct_quote(body, style)
                if quote_phrase is not None:
                    # Strip the leading ": " — voki has no recipient
                    # to anchor it onto, so the call follows the verb
                    # directly: "vokis — Petro!" / "vokis 'Petro!'".
                    parts.append(quote_phrase[2:])
                    ctx.note_mention(theme)
                    theme = None
        if theme is not None:
            # Abstract themes (faktos) with a recipient unfold as a
            # `al RECIP ke ...` clause: "rakontis al Petro ke la
            # libro estas en la breto" instead of the literal "la
            # fakton" accusative. The fakto's pri_* properties give
            # us the underlying relation.
            ke = None
            quote_phrase = None
            if (theme.entity_type == "abstract"
                    and ev.roles.get("recipient")):
                # demandi (ask) renders the fakto as a question
                # (kie/kiu) rather than an assertion (ke ...).
                mode = "question" if ev.action == "demandi" else "assertion"
                # Per-event coin flip: with `_DIRECT_QUOTE_PROB`,
                # render as direct quote ("Maria diris al Petro:
                # 'La libro estas en la breto.'") instead of the
                # default ke-clause indirect form. Surface form
                # picked uniformly from `_QUOTE_STYLES` so traces
                # mix all three Esperanto dialog conventions.
                if (ctx.rng is not None
                        and ctx.rng.random() < _DIRECT_QUOTE_PROB):
                    body = _render_fakto_as_quote_body(
                        theme, ctx, mode=mode)
                    if body is not None:
                        style = ctx.rng.choice(_QUOTE_STYLES)
                        quote_phrase = _wrap_direct_quote(body, style)
                if quote_phrase is None:
                    ke = _render_fakto_as_ke_clause(theme, ctx, mode=mode)
            if quote_phrase is not None:
                recip = ctx.trace.entity(ev.roles["recipient"])
                if recip is not None:
                    parts.append(f"al {ctx.name_for(recip)}{quote_phrase}")
                    ctx.note_mention(recip)
                    recipient_handled = True
                else:
                    parts.append(quote_phrase)
            elif ke is not None:
                recip = ctx.trace.entity(ev.roles["recipient"])
                if recip is not None:
                    parts.append(f"al {ctx.name_for(recip)}")
                    ctx.note_mention(recip)
                    recipient_handled = True
                parts.append(ke)
            else:
                # Consumption verbs (rules using `consume_one`) and
                # transfer verbs (rules using `transfer_n`) both render
                # at the event's quantity — qty=1 forces singular even
                # when the source stack has count > 1 ("Maria manĝis la
                # pomon", "Anna prenis la piron de la pirarbo"), qty>1
                # produces the plural numeral ("Maria prenis du pomojn").
                # Without this, default qty=1 fell back to the entity's
                # natural_count and rendered preni-from-stacked-source
                # as a collective ("prenis la tri pirojn") even when
                # the engine's TransferN had only split off one unit —
                # the prose contradicted the actual world-state. Both
                # verb sets are derived from rule structure — see
                # dsl/introspect.
                _ev_qty = getattr(ev, "quantity", 1)
                _verb_meta = _verb_metadata()
                if ev.action in _verb_meta["consumption"]:
                    _count_override = _ev_qty
                elif ev.action in _verb_meta["transfer"]:
                    _count_override = _ev_qty
                else:
                    _count_override = None
                parts.append(ctx.theme_form(
                    theme, count_override=_count_override))
                ctx.note_mention(theme)
                ctx.mark_nonperson_mention(theme)
                if source_entity_id is not None:
                    source_ent = ctx.trace.entities.get(source_entity_id)
                    if source_ent is not None:
                        parts.append(f"de {ctx.name_for(source_ent)}")
                        ctx.note_mention(source_ent)
                        ctx.mark_nonperson_mention(source_ent)

    if ev.roles.get("instrument"):
        instr = ctx.trace.entity(ev.roles["instrument"])
        if instr is not None:
            # When the verb's rule transfers the instrument as one of
            # the moved stacks (aĉeti's coins go to the seller, vendi's
            # come from the buyer), match the instrument's count to
            # ev.quantity so "per tri moneroj" mirrors "tri pomojn".
            _ev_qty = getattr(ev, "quantity", 1)
            _verb_meta = _verb_metadata()
            instr_count = (
                _ev_qty
                if (ev.action in _verb_meta["instrument_quantified"]
                    and _ev_qty > 1)
                else None)
            parts.append(
                f"per {ctx.name_for(instr, count_override=instr_count)}")
            ctx.note_mention(instr)

    # `fari.parts` (kind="list") renders as "el X, Y kaj Z" — the
    # source materials the constructed entity is made from. Only fari
    # uses a list role today, so the role-name check is sufficient;
    # if more variadic verbs land, generalize via role_spec.kind.
    parts_role = ev.roles.get("parts")
    if isinstance(parts_role, (list, tuple)) and parts_role:
        rendered_parts: list[str] = []
        for peid in parts_role:
            pent = ctx.trace.entity(peid)
            if pent is None:
                continue
            rendered_parts.append(ctx.name_for(pent))
            ctx.note_mention(pent)
            ctx.mark_nonperson_mention(pent)
        if rendered_parts:
            if len(rendered_parts) == 1:
                joined = rendered_parts[0]
            else:
                joined = ", ".join(rendered_parts[:-1]) + (
                    f" kaj {rendered_parts[-1]}")
            parts.append(f"el {joined}")

    if ev.roles.get("recipient") and not recipient_handled:
        recip_id = ev.roles["recipient"]
        # When the recipient IS the prior owner narrated via
        # source_entity_id ("de Petro" already on the event), suppress
        # the "al recipient" tail so we don't duplicate the referent.
        # This is what makes aĉeti read as "aĉetis du pomojn de Petro"
        # without trailing "al Petro" — recipient and source coincide
        # for acquisition verbs whose theme came from the recipient.
        if recip_id != source_entity_id:
            recip = ctx.trace.entity(recip_id)
            if recip is not None:
                parts.append(f"al {ctx.name_for(recip)}")
                ctx.note_mention(recip)

    if ev.roles.get("location"):
        loc_id = ev.roles["location"]
        loc = ctx.trace.entity(loc_id)
        if loc is not None:
            # Pick the preposition by looking at what the rule that
            # fires on this verb actually adds — declarative source
            # of truth that works regardless of whether DSL has been
            # run on the trace yet. Falls back to the entity-type
            # heuristic ("location → en, else sur") when no rule
            # explicitly adds en/sur for this verb's theme/location.
            #
            # Without this, planti(theme=semo, location=tero) would
            # render "sur tero" because tero is a substance — but the
            # planti_plants_theme rule actually adds en(theme, tero).
            prep = _verb_location_preposition(ev.action, ctx.lexicon)
            if prep is None:
                prep = "en" if loc.entity_type == "location" else "sur"
            parts.append(f"{prep} {ctx.name_for(loc)}")
            ctx.note_mention(loc)

    if ev.roles.get("destination"):
        dest = ctx.trace.entity(ev.roles["destination"])
        if dest is not None:
            parts.append(f"al {ctx.name_for(dest)}")
            ctx.note_mention(dest)

    # `legi` and other text-extraction verbs: append the discovered
    # fakto as a "kaj eksciis ke ..." tail clause so the prose tells
    # the reader WHAT was learned, not just that something was read.
    # Driven by priskribas(theme, fakto) in the trace — verb-agnostic;
    # any verb with a text theme that has a priskribas link gets this.
    learned = _render_learned_fakto(ev, ctx)
    if learned is not None:
        parts.append(learned)

    # Strip trailing period for coordination (caller adds one).
    sent = " ".join(parts)
    if not drop_subject and not sent.endswith((".", "?", "!", "'", "»")):
        # Direct-quote events end with the embedded utterance's own
        # terminal punctuation (`?`, `!`, `.`) plus an optional quote
        # closer (`'`, `»`). Skipping `.` here avoids `?.` / `'.` /
        # `».` artifacts; the quote closer serves as the sentence
        # terminator visually.
        sent += "."
    return sent


def _render_appearance(m: AppearanceMessage, ctx: _Ctx) -> Optional[str]:
    ent = ctx.trace.entities.get(m.entity_id)
    if ent is None or ent.id in ctx.mentioned:
        return None
    form = ctx.name_for(ent)
    ctx.note_mention(ent)
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
    ctx.note_mention(owner)
    theme_form = to_accusative(ctx.name_for(theme))
    ctx.note_mention(theme)
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
    ctx.note_mention(owner)
    theme_form = to_accusative(ctx.name_for(theme))
    ctx.note_mention(theme)
    return f"Nun {owner_form} hav{ctx.tense} {theme_form}."


def _render_destruction(
    m: DestructionMessage, ctx: _Ctx,
) -> Optional[str]:
    ent = ctx.trace.entities.get(m.entity_id)
    if ent is None:
        return None
    form = ctx.name_for(ent)
    ctx.note_mention(ent)
    verb = inflect("malaperi", ctx.tense)
    return f"{form} {verb}."


def _render_coordinated(
    m: CoordinatedMessage, ctx: _Ctx,
) -> Optional[str]:
    """Children share a subject. Render first child fully, then each
    subsequent as a bare verb phrase joined with 'kaj'.

    Object elision: when a run of consecutive children share the same
    theme, the theme renders only on the last of that run. "Ŝi kuiras
    la panon kaj manĝas la panon" → "Ŝi kuiras kaj manĝas la panon".
    Applies per-run so mixed-theme sequences like kuiras(pano) +
    manĝas(pano) + satiĝas(None) still read naturally — the first
    two share elision, the third stands on its own.
    """
    if not m.children:
        return None
    # Compute the elide-run: which children's themes are suppressed?
    elide_flags = _compute_theme_elision(m.children)

    first = m.children[0]
    if not isinstance(first, EventMessage):
        return None
    first_sent = _render_event_phrase(
        first.event, ctx, drop_subject=False,
        drop_theme=elide_flags[0],
        source_entity_id=first.source_entity_id)
    if first_sent is None:
        return None
    first_body = first_sent.rstrip(".")
    # Inline the first child's `ĉar` clause directly after its verb
    # phrase rather than waiting until the end of the coordinated
    # chain. Without this, the precondition reads as if it explained
    # the LAST verb in the chain — e.g.
    #   "malfermis la pordon kaj eniris kaj metis Annan ĉar la pordo
    #    estis fermita"
    # which attaches the reason to `metis`. Inlining gives:
    #   "malfermis la pordon ĉar la pordo estis fermita kaj eniris
    #    kaj metis Annan"
    # — the reason now sits beside the verb it actually motivates.
    if first.precondition is not None:
        first_with_clause = _append_precondition_clause(
            first_body + ".", first.precondition, ctx)
        first_body = first_with_clause.rstrip(".")
    rest_bodies: list[str] = []
    for idx, child in enumerate(m.children[1:], start=1):
        if not isinstance(child, EventMessage):
            continue
        phrase = _render_event_phrase(
            child.event, ctx, drop_subject=True,
            drop_theme=elide_flags[idx],
            source_entity_id=child.source_entity_id)
        if phrase is None:
            continue
        if isinstance(child, EventMessage) and child.precondition is not None:
            phrase = _append_precondition_clause(
                phrase + ".", child.precondition, ctx).rstrip(".")
        rest_bodies.append(phrase)
    if not rest_bodies:
        return first_body + "."
    return first_body + " kaj " + " kaj ".join(rest_bodies) + "."


def _compute_theme_elision(children) -> list[bool]:
    """For each child, True if its theme should be suppressed on
    rendering (subsumed into a later child's theme). Only elides
    within a run of consecutive children that share the *same*
    theme entity id.
    """
    themes: list[Optional[str]] = []
    for c in children:
        if isinstance(c, EventMessage):
            themes.append(c.event.roles.get("theme"))
        else:
            themes.append(None)
    flags = [False] * len(children)
    for i in range(len(children)):
        theme = themes[i]
        if theme is None:
            continue
        # Is there a later consecutive sibling with the same theme?
        # If yes, elide this one.
        if i + 1 < len(children) and themes[i + 1] == theme:
            flags[i] = True
    return flags


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

def _render_grouped_relation(
    m: GroupedRelationMessage, ctx: _Ctx,
) -> Optional[str]:
    """'En la kuirejo estas tablo, glaso, kaj korbo.' / 'Sur la breto
    estas libro kaj papero.' Container fronts the sentence; contents
    are listed with comma + 'kaj' for the final entry."""
    container = ctx.trace.entities.get(m.container_id)
    if container is None:
        return None
    container_form = ctx.name_for(container)
    ctx.note_mention(container)

    contained_forms: list[str] = []
    for cid in m.contained_ids:
        ent = ctx.trace.entities.get(cid)
        if ent is None:
            continue
        contained_forms.append(ctx.name_for(ent))
        ctx.note_mention(ent)
    if not contained_forms:
        return None

    if len(contained_forms) == 1:
        list_str = contained_forms[0]
    elif len(contained_forms) == 2:
        list_str = f"{contained_forms[0]} kaj {contained_forms[1]}"
    else:
        head = ", ".join(contained_forms[:-1])
        list_str = f"{head}, kaj {contained_forms[-1]}"

    prep = "En" if m.relation == "en" else "Sur"
    # Single-contained animates with a derived posture → use that
    # posture's verb ("En la lago naĝas Lidia") instead of the bland
    # "estas" copula. Mirrors the contextual-posture path in
    # `_render_relation`. Multi-contained groups stay on "estas" —
    # you can't conjugate a single posture verb across mixed contents.
    verb = "est" + ctx.tense
    if len(m.contained_ids) == 1:
        only = ctx.trace.entities.get(m.contained_ids[0])
        if only is not None and ctx.lexicon.types.is_subtype(
                only.entity_type, "animate"):
            posture_root = _contextual_posture_verb_root(only, ctx)
            if posture_root is not None:
                verb = f"{posture_root}{ctx.tense}"
    return f"{prep} {container_form} {verb} {list_str}."


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


from .messages import EntityQualityMessage  # late import to avoid cycle

_DISPATCH = {
    SceneGroundingMessage: _render_scene_grounding,
    EntityQualityMessage: _render_entity_quality,
    RelationMessage: _render_relation,
    GroupedRelationMessage: _render_grouped_relation,
    EventMessage: _render_event,
    AppearanceMessage: _render_appearance,
    RelationRemovedMessage: _render_relation_removed,
    RelationAddedMessage: _render_relation_added,
    DestructionMessage: _render_destruction,
    CoordinatedMessage: _render_coordinated,
    SubordinatedMessage: _render_subordinated,
}


def _render_world_preamble(ctx: _Ctx) -> Optional[str]:
    """Surface the trace-wide `mondo` singleton's marked state as one
    or more sentences at the trace opening.

    Emits time-of-day when marked (mateno/vespero/nokto) and weather
    when marked (pluva/neĝa). Unmarked values (tempo_de_tago=tago,
    weather=serena) stay silent, since speakers don't say "estis tago"
    any more than they say "the room had four walls".

    Tempo_de_tago renders as a copular noun ("Estis nokto.").
    Weather renders as an impersonal verb derived from the adjective
    root ("Pluvis." from `pluva`, "Neĝis." from `neĝa`) — natural
    Esperanto for atmospheric conditions. Tense matches the trace's
    narrative tense throughout."""
    mondo = ctx.trace.entities.get("mondo")
    if mondo is None:
        return None
    parts: list[str] = []

    val = mondo.properties.get("tempo_de_tago")
    if isinstance(val, list):
        val = val[0] if val else None
    if val is not None:
        slot_def = ctx.lexicon.slots.get("tempo_de_tago")
        unmarked = slot_def.unmarked if slot_def is not None else "tago"
        if val != unmarked:
            copula = f"est{ctx.tense}"
            parts.append(f"{copula.capitalize()} {val}.")

    weather = mondo.properties.get("weather")
    if isinstance(weather, list):
        weather = weather[0] if weather else None
    if weather is not None:
        slot_def = ctx.lexicon.slots.get("weather")
        unmarked = slot_def.unmarked if slot_def is not None else "serena"
        if weather != unmarked:
            # Adjective root → impersonal verb. `pluva` → `pluv` →
            # `pluvis`/`pluvas`/`pluvos`. Trim a trailing -a only;
            # other endings are left to fail loudly so a typo in the
            # vocabulary doesn't silently produce ungrammatical output.
            root = weather[:-1] if weather.endswith("a") else weather
            parts.append(f"{root.capitalize()}{ctx.tense}.")

    if not parts:
        return None
    return " ".join(parts)


def render_messages(
    messages: list[Message], trace: Trace, lexicon: Lexicon, *,
    scene_location_id: Optional[str] = None,
    rng: Optional[random.Random] = None,
    tense: Optional[str] = None,
    derived=None,
) -> str:
    if tense is None:
        tense = _pick_tense(rng)
    ctx = _Ctx(trace, lexicon,
               scene_location_id=scene_location_id, rng=rng, tense=tense,
               derived=derived)

    raw: list[tuple[str, bool]] = []   # (text, has_cause_in_prose)
    # Trace-wide preamble from the `mondo` singleton (currently just
    # tempo_de_tago). Appears once at the trace opening when the
    # value is marked (mateno/vespero/nokto) — `tago` is the unmarked
    # default and stays silent. cause=False since it's setting, not
    # consequence.
    preamble = _render_world_preamble(ctx)
    if preamble is not None:
        raw.append((preamble, False))

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
