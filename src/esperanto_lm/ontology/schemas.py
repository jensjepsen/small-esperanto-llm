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


class RoleSpec(_Frozen):
    name: str                           # "agent", "theme", "instrument", ...
    type: str                           # entity-type constraint
    properties: dict[str, list[str]] = Field(default_factory=dict)


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


# Discriminated union point — when more precondition kinds appear
# (e.g. quantitative comparisons, negation), add them here. The kind
# field is the tag the planner dispatches on.
Precondition = RelationPrecondition | IfPropertyPrecondition


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

    @model_validator(mode="after")
    def _check_precondition_roles(self):
        role_names = {r.name for r in self.roles}
        for pc in self.preconditions:
            referenced = (
                pc.roles if isinstance(pc, RelationPrecondition) else [pc.role])
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
