"""Pydantic schemas for the Esperanto lexical-semantic ontology.

Primitives:
  PropertySlot  — owns vocabulary + which entity types it applies to.
  Concept       — noun sense: entity type + property bundle.
  Relation      — preposition / relational verb schema.
  Action        — verb sense: roles + effects.
  Affix         — typed operator (e.g. -il-) used at lexicon-load time.

Property values are stored uniformly as `dict[str, list[str]]` so that scalar
and multi-valued slots share the same shape. Validation against slots is done
in the loader, not on the model itself, because schemas don't know the slot
registry.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    """Immutable base — the lexicon is read-only after load."""
    model_config = ConfigDict(frozen=True, extra="forbid")


class PersonName(_Frozen):
    """A given-name string and its associated gender lemma. The gender
    is a concept-lemma label (`viro` / `virino`) that scene seeders
    use to filter person-concept choices when materializing a named
    entity, so "Maria" picks from virino-categorized concepts and
    "Petro" from viro-categorized ones."""
    name: str
    gender: str


class PropertySlot(_Frozen):
    """A property slot. The single source of truth for what values are
    legal on what entity types."""
    name: str
    # None = open vocabulary (e.g. functional_signature whose values are
    # verb lemmas resolved at load time).
    vocabulary: Optional[list[str]] = None
    applies_to: list[str]                       # entity-type names
    scalar: bool = True
    # Marks the slot as transient state that varies at instance time.
    # If True AND the concept (after bake) declares this slot, the
    # sampler picks a uniformly-random value from `vocabulary` for
    # each new entity instance — the concept's authored/derived value
    # is just an opt-in marker that the slot is meaningful here, not
    # a "default" that random respects.
    #
    # Use for transient state (hunger, openness, lock_state, ...).
    # Leave False for identity slots (made_of, parts, fragility,
    # state_of_matter, ...) — those describe what the concept *is*
    # and shouldn't vary across instances.
    #
    # Requires vocabulary to be non-null and scalar=True; open-vocab
    # and multi-valued slots can't be uniformly randomized.
    varies: bool = False
    # When True, the slot applies to EVERY concept of its `applies_to`
    # type without per-concept declaration — a default-derivation
    # like `animate_has_hunger` materializes it for all instances at
    # runtime. The regression sampler's concept-matching filters skip
    # the "concept must declare this slot" check for pervasive slots.
    #
    # Examples (pervasive=True): hunger, thirst, sleep_state, posture,
    # wetness, cleanliness, temperature, mood (when added).
    # Examples (pervasive=False — opt-in via concept declaration or
    # parts-derivation): openness, lock_state, power_state, fullness,
    # attachment, water_state — these only apply to specific concepts
    # (pordo gets openness, motoro gets power_state, glaso gets
    # fullness, ...).
    pervasive: bool = False
    # When True, the slot's vocabulary words can be rendered as
    # attributive adjectives on entities that hold them — "fragila pomo",
    # "malsata Maria". Slot values must already be -a/participle adjective
    # forms (fragila, malsata, ŝlosita); boolean tag slots like
    # `is_clothing=yes` are not adjectival even though they're scalar.
    # Realizer reads this flag to decide which slot values to surface
    # as adjectives in noun phrases.
    adjectival: bool = False
    # The slot's unmarked / default value — what speakers leave
    # implicit. The renderer skips adjective rendering when an entity's
    # value matches this, so "fortika lampo" / "luma valo" / "sata
    # Maria" don't surface as noise. Marked values (fragila, malluma,
    # malsata) still render. None means every value is potentially
    # marked (e.g. hazard: akra and glita are both noteworthy).
    unmarked: Optional[str] = None
    # Optional sampling weights aligned 1:1 with `vocabulary`. When
    # set, `_randomize_state` uses `rng.choices(vocabulary, weights)`
    # instead of uniform `rng.choice`. Lets the world bias toward
    # narratively-common values (e.g. nokto rare relative to tago)
    # without inventing parallel mechanisms. None = uniform.
    weights: Optional[list[float]] = None


class ConceptPart(_Frozen):
    """A sub-entity that an instance of the host concept comes with —
    auto-materialized by the sampler at instance time. Used for
    meronymy where the part has its own state (a door's lock, a
    person's hand) and may be acted upon independently."""
    concept: str                                # part's concept lemma
    relation: str = "havas_parton"              # whole→part relation
    # Where the part instance lives in the scene. "intrinsic" → no
    # `en` placement (samloke with the host is derived). Other shapes
    # ("same_container" for detachable parts, etc.) can be added when
    # needed.
    placement: Literal["intrinsic"] = "intrinsic"
    # For constructable-recipe parts: per-part property requirements
    # the planner must satisfy before fari fires. Maps slot name → list
    # of acceptable values. Used for "teo's akvo must be bolanta": the
    # planner must boli the akvo before fari(teo). Empty = no extra
    # requirement beyond havi(agent, part).
    requires: dict[str, list[str]] = Field(default_factory=dict)


class Concept(_Frozen):
    """Noun sense. Property values are validated against the slot registry
    by the loader."""
    lemma: str
    entity_type: str
    properties: dict[str, list[str]] = Field(default_factory=dict)
    # Sub-entity parts: each instance gets its own materialized part
    # entities. Used for meronymy where the part has independent
    # state (a person's piedo, a door's seruro) and can be the target
    # of verbs in its own right. Distinct from old-style capability
    # markers — the part is a real entity, not a slot value.
    parts: list[ConceptPart] = Field(default_factory=list)
    # For constructable concepts: tool concepts the agent uses to
    # make this entity. Each entry names a tool lemma (e.g. "forno"
    # for bulko); the planner picks one and binds it as fari's
    # instrument role. The verb to render in surface text comes from
    # the picked tool's `functional_signature`. Empty/missing means
    # no tool needed — the construction is performed with bare hands
    # (sandviĉo, salato).
    crafted_with: list[str] = Field(default_factory=list)
    # Superordinate concept lemmas this one is-a. Walks transitively at
    # rendering time so a `papago` can be referred to as "birdo" or
    # "animalo" if `papago.category=["birdo"]` and
    # `birdo.category=["animalo"]`. Plain string lemmas — no validation
    # that the parent concept exists, so tests can use ad-hoc taxa
    # without padding the lexicon. List form allows multi-inheritance
    # ("seĝo" might be both "meblo" and "lignaĵo") though most concepts
    # will have 0 or 1 parents.
    category: list[str] = Field(default_factory=list)
    # When True, this concept is a category stub — semantically a real
    # type (an artifact, an animal, …) but not directly instantiated
    # in scenes. Carries shared properties for its children to inherit
    # (e.g. lignaĵo holds made_of=wood for all wooden things). Scene
    # selection skips these so we don't get incoherent prose like
    # "Estis meblo en la salono"; the children render normally.
    is_category_stub: bool = False
    # Marks concepts produced by composition rather than authored on disk.
    derived: bool = False
    # Provenance for derived concepts: which lemma(s) + affix(es) produced
    # this. Empty for authored concepts.
    derived_from: Optional[dict[str, str]] = None


class Quality(_Frozen):
    """Adjectival lexical item — describes a state or attribute of an
    entity. Lives in the lexicon alongside Concepts but renders as an
    adjective (Esperanto -a ending) rather than a noun.

    Slot vocabularies reference Quality lemmas: e.g. `lock_state`
    accepts `["ŝlosita", "malŝlosita"]` where each value is a
    registered Quality. Effects writing a slot set its value to a
    Quality lemma; role bindings constrain on Quality lemmas. The
    realizer renders Quality lemmas directly via the standard
    adjective inflection (-a → -an, -aj, -ajn).

    Many qualities are auto-derived from verbs via participle affixes
    (-it- for transitive passive past: ŝlosi → ŝlosita). Others are
    authored as base qualities (varma, granda, ruĝa) and may be
    extended via mal-/-eg-/-et- composition affixes."""
    lemma: str   # adjective form, ending in -a (e.g. "ŝlosita")
    derived: bool = False
    derived_from: Optional[dict[str, str]] = None


class Relation(_Frozen):
    """Preposition / relational-verb schema. Instances live in traces.

    `arg_names` labels each positional argument ("contained"/"container"
    for `en`/`sur`, "owner"/"theme" for `havi`, etc.) so rule-DSL sites
    can write `rel("en", container=X, contained=Y)` instead of relying
    on positional order. Length must equal `arity`.
    """
    name: str
    arity: int
    arg_types: list[str]
    arg_names: list[str]
    inverse: Optional[str] = None
    symmetric: bool = False
    # Per-arg-position list of forbidden subtypes. `arg_types` declares
    # the broadest allowed type (e.g. `havi.theme = physical`); this
    # carves out narrower exceptions that would otherwise pass the
    # subtype check (you can't `havi` a `location` or `person` even
    # though both are physical). Empty/omitted = no exclusions.
    # Validated at `Trace.assert_relation`.
    arg_excludes: list[list[str]] = Field(default_factory=list)
    # Per-arg-position flag: when True, the arg cannot be a `parto`
    # of any other entity (no `havas_parton` edge ending at it).
    # Used by `havi.theme` to forbid owning body parts separately
    # from their host (Petro can't `havi` Mikael's mano). The check
    # at `assert_relation` is O(1) via Trace._parts_index.
    arg_not_part: list[bool] = Field(default_factory=list)
    # Per-arg-position pattern (Pattern | None). Evaluated against the
    # entity at that arg position via `entity_matches_static`. NotPattern
    # expresses "must not match" — havi.theme uses
    # `~entity(nemovebla="yes")` to forbid owning fixtures (forno, fajro,
    # kameno) without per-verb gating.
    #
    # Stored as Python Pattern objects, parsed from JSON dicts by the
    # loader (`{"not": {"entity": {"slot": "val"}}}`). Empty/omitted =
    # no patterns. Same constraint is also lifted to a static index in
    # `introspect.relation_arg_excludes` for grounding-time pruning in
    # the forward planner — one source of truth.
    arg_patterns: tuple = ()
    # Numeric comparison between two args' property values. List of
    # dicts: {"left_arg": int, "left_property": str, "op": str,
    # "right_arg": int, "right_property": str}. Asserted at relation-
    # assertion time and at planner grounding time. Same vacuous-on-
    # missing-data semantics as ComparePropertyPrecondition.
    #
    # havi uses [{"left_arg": 1, "left_property": "maso", "op": "<=",
    # "right_arg": 0, "right_property": "lift_capacity"}] — the owned
    # theme's mass must fit the owner's lift capacity. Catches preni,
    # kapti, doni, and any future verb that produces havi, in one
    # place. Mirrors arg_patterns (set-membership) for the numeric case.
    arg_compare: list[dict] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, extra="forbid",
                              arbitrary_types_allowed=True)


class RoleSpec(_Frozen):
    name: str                           # "agent", "theme", "instrument", ...
    type: str                           # entity-type constraint
    properties: dict[str, list[str]] = Field(default_factory=dict)
    # Role kinds beyond the default "single entity bound at plan time":
    #   "single"  — one entity (default)
    #   "created" — to-be-created entity of concept named by from_field
    #               in the theme concept's properties; engine mints
    #               the entity at event-firing time
    #   "list"    — N entities, one per concept named in the theme
    #               concept's `from_field`. Binds to a list var.
    kind: Literal["single", "created", "list"] = "single"
    # Which field of the *theme* concept (or another previously-bound
    # role) the grounder reads to populate this role. For kind="list"
    # this is the parts-list field (e.g. "parts"); for an instrument
    # tied to a recipe it's "crafted_with". Ignored for kind="single"
    # without a from_field source.
    from_field: Optional[str] = None
    # Soft role: planner allows leaving it unbound. Used for the
    # instrument role when a recipe has no crafted_with.
    optional: bool = False


class Effect(_Frozen):
    """A state change applied to a role-bound participant after the action
    fires. For this slice: property assignment only. Relation create/delete
    can be added later."""
    target_role: str
    property: str
    value: str


class RelationPrecondition(_Frozen):
    """A cross-role relation that must hold before the action is plannable.

    `roles` lists role names from the action's `roles` field, in the same
    positional order as the relation's `arg_names`. Example: for `manĝi`
    requiring havi(agent, theme), kind="relation", rel="havi",
    roles=["agent", "theme"] — the planner reads it as
    rel("havi", owner=<agent_eid>, theme=<theme_eid>).

    Co-location is `rel="samloke"` — `samloke` is a derived relation
    (X and Y share an `en` container), so any verb requiring two roles
    to be in the same place uses this single mechanism rather than a
    separate co_locate kind."""
    kind: Literal["relation"] = "relation"
    rel: str
    roles: list[str]


class IfPropertyPrecondition(_Frozen):
    """A conditional property requirement on a single role: if the role
    entity has `if_property=if_value`, then it must also have
    `then_property=then_value`. The gate vacuously passes when
    `if_property` is absent or holds a different value.

    Use case: malfermi's theme must be unlocked, but only if it has a
    lock at all. Things without lock_capable have no lock_state to
    check; the gate is silent for them, leaving malfermi(skatolo)
    plannable while still forcing malŝlosi → malfermi for pordo."""
    kind: Literal["if_property"] = "if_property"
    role: str
    if_property: str
    if_value: str
    then_property: str
    then_value: str


class MatchPrecondition(_Frozen):
    """Two roles must overlap on a slot value. Action is plannable
    iff `entity(role_a).slot_a ∩ entity(role_b).slot_b ≠ ∅`. Pure
    rejection — the planner skips the candidate when the match
    fails; no subgoaling, since the use case is intrinsic typed
    properties (terrain, material) that no verb changes.

    Use case: veturi requires the instrument's terrain to be one
    the destination affords. aŭto (terrain=land) → kuirejo (no
    terrain) fails; aŭto → urbo (terrain=land via vojo part)
    passes."""
    kind: Literal["match"] = "match"
    role_a: str
    slot_a: str
    role_b: str
    slot_b: str


# Discriminated union point — when more precondition kinds appear
# (e.g. quantitative comparisons, negation), add them here. The kind
# field is the tag the planner dispatches on.
Precondition = (
    RelationPrecondition | IfPropertyPrecondition | MatchPrecondition
)


class Action(_Frozen):
    """Verb sense."""
    lemma: str
    transitivity: Literal["intransitive", "transitive", "ditransitive"]
    aspect: Literal["activity", "achievement", "accomplishment", "state"]
    roles: list[RoleSpec]
    effects: list[Effect] = Field(default_factory=list)
    # Cross-role preconditions a planner must satisfy before the verb
    # is plannable. Distinct from `roles[*].properties` (which gates a
    # SINGLE role's entity properties) — these gate combinations of
    # roles via relations. Empty list = no extra constraints beyond
    # role-level ones.
    preconditions: list[Precondition] = Field(default_factory=list)
    # Affix-derivation flags. Each enables the loader's compositional
    # pass for one affix kind:
    #   derives_instrument   → -il-  (tranĉi → tranĉilo)
    #   derives_professional → -ist- (kuiri → kuiristo)
    #   derives_thing        → -aĵ-  (manĝi → manĝaĵo)
    #   derives_place        → -ej-  (kuiri → kuirejo)
    #   derives_quality      → -it-  (ŝlosi → ŝlosita; passive past
    #                                participle = the adjective for
    #                                the result-state)
    # Loader matches each flag against the affix's `trigger_flag`.
    derives_instrument: bool = False
    derives_professional: bool = False
    derives_thing: bool = False
    derives_place: bool = False
    derives_quality: bool = False
    # When True, the planner permits agent==theme bindings for this
    # action (sekigi/lavi/vesti — actions one can plausibly do to
    # oneself). The realizer renders the theme as the reflexive
    # pronoun "sin" instead of repeating the agent's name. Default
    # False rejects same-entity bindings (preni/doni/sekvi/etc., where
    # reflexive doesn't make semantic sense).
    reflexive_ok: bool = False

    @model_validator(mode="after")
    def _check_precondition_roles(self):
        role_names = {r.name for r in self.roles}
        for pc in self.preconditions:
            if isinstance(pc, RelationPrecondition):
                referenced = pc.roles
            elif isinstance(pc, MatchPrecondition):
                referenced = [pc.role_a, pc.role_b]
            else:
                referenced = [pc.role]
            for rn in referenced:
                if rn not in role_names:
                    raise ValueError(
                        f"action {self.lemma!r}: precondition references "
                        f"role {rn!r} not present in roles "
                        f"{sorted(role_names)}")
        return self


class ContainmentPattern(_Frozen):
    """Describes what concepts qualify as the `container` side of a
    ContainmentFact. All set fields are conjuncted; at least one must be
    set. Validated on load.

    Field semantics (all match a single concept):
      - sense_id: exact lemma match. The shorthand form `container: "X"`
        on a ContainmentFact is normalized to a pattern with this field.
      - entity_type: concept's entity_type is-a this type (subtype check
        via the type spine).
      - suffix: this morpheme appears as a suffix in the concept's
        morphological decomposition (e.g. "ej" matches kuirejo, laborejo,
        manĝejo).
      - property: every (slot, value) pair in this dict is satisfied by
        the concept's properties.
      - contains: SECOND-ORDER. Concept's first-order containment
        reachability includes this lemma. Evaluated in a separate
        resolver pass against pass-1 results only — no nested
        contains-of-contains. If we need another relational pattern type
        later (`affords:`, `instance_of:`), it gets its own field with
        the same two-pass structure rather than a generalized mechanism.
    """
    sense_id: Optional[str] = None
    entity_type: Optional[str] = None
    suffix: Optional[str] = None
    property: Optional[dict[str, str]] = None
    contains: Optional[str] = None
    # Match if the concept's transitive `category` chain includes this
    # lemma. Walks up via `concept.category` to follow supertypes
    # (papago → birdo → besto). Lets containment rules classify by
    # the existing taxonomy without inventing parallel marker slots —
    # adding a concept to a category automatically extends its
    # placement affordances.
    category: Optional[str] = None


class ContainmentFact(_Frozen):
    """A modal-possibility containment assertion.

    "X can be in/on/etc. Y." Absence asserts nothing. Frequency, default
    presence, and sampling density are all *sampler* concerns — the
    containment graph just describes what placements are possible.

    Exactly one of `container` (shorthand for sense_id pattern) and
    `container_pattern` must be set. Loader normalizes the shorthand.

    Same on the contained side: exactly one of `contained` (string —
    either a concept lemma or a type name from the spine) and
    `contained_pattern` must be set. Loader normalizes
    `contained: "X"` to either a sense_id pattern (if X is a known
    concept) or an entity_type pattern (if X is in the spine).

    Pattern semantics on either side: see `ContainmentPattern`.
    """
    container: Optional[str] = None
    container_pattern: Optional[ContainmentPattern] = None
    contained: Optional[str] = None
    contained_pattern: Optional[ContainmentPattern] = None
    relation: str
    # Two-tier semantics:
    #   `required: false` (default) — affordance. "This combination is
    #     permitted." Multiple affordances combine via OR; placement is
    #     legal if at least one entry permits the pair.
    #   `required: true` — constraint. "If contained matches the
    #     contained_pattern, container MUST match the container_pattern."
    #     Compose by AND (every applicable requirement must hold). A
    #     required entry ALSO doubles as an affordance (it permits its
    #     own pattern), so you don't need a sibling affordance for the
    #     same pair.
    required: bool = False
    # Slot-overlap requirement. For each slot S in this list, the
    # contained's values for S and the container's values for S must
    # intersect (set overlap ≠ ∅). Vacuously satisfied if either side
    # lacks the slot — so terrain-bearing entities (fish, vehicles)
    # are constrained to terrain-matching containers, but
    # terrain-free entities (books, hammers) aren't restricted.
    # Combines naturally with `required: true` (the intersection check
    # is the requirement); the contained_pattern / container_pattern
    # fields, if set, gate which pairs the rule applies to.
    slot_overlap: list[str] = Field(default_factory=list)


class Affix(_Frozen):
    """A typed morphological operator. Generalized for affixes that
    attach to either verbs or nouns; the loader dispatches on which
    trigger is set.

    For verb-attaching affixes (-il-, -ist-, -aĵ-, -ej-): set
    `trigger_flag` to the Action boolean that gates the derivation
    (`derives_instrument`, `derives_professional`, `derives_thing`,
    `derives_place`).

    For noun-attaching affixes (-id-, -ar-): set
    `trigger_concept_type` to the entity-type whose concepts get the
    derivation. The loader applies the affix to every concept whose
    `entity_type` is-a (subtype of) this value.

    Exactly one trigger must be set, matching `attaches_to`."""
    form: str                           # e.g. "il"
    kind: Literal["prefix", "suffix"]
    attaches_to: Literal["verb", "noun", "adjective"]
    produces: Literal["verb", "noun", "adjective"]
    output_type: str                    # entity type of the produced concept
    # Verb-attaching trigger: the Action flag that gates this affix.
    trigger_flag: Optional[str] = None
    # Noun-attaching trigger: synthesize for every concept whose
    # entity_type is-a this. Examples: `animal` for -id- (offspring)
    # and -ar- (collection).
    trigger_concept_type: Optional[str] = None
    # How to synthesize the produced concept's functional signature.
    #   "effect" = read it off the source verb's effect structure
    #              (instruments: tranĉilo's signature is the effect of
    #              tranĉi).
    #   "none"   = no functional signature (professionals, things,
    #              places, offspring, collections).
    signature_source: Literal["effect", "none"]
    # The ending appended after the affix to form the surface lemma.
    # For -il-: "o" (-> tranĉilo).
    noun_ending: Optional[str] = None
    # Default property bundle for the synthesized concept. -il- artifacts
    # are always solid, so we set state_of_matter via the affix definition
    # rather than hardcoding it in the loader. Validated against the slot
    # registry the same way authored properties are.
    output_properties: dict[str, list[str]] = Field(default_factory=dict)
