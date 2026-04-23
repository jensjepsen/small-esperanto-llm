"""Pydantic schemas for the Esperanto lexical-semantic ontology.

Five primitives:
  PropertySlot  — owns vocabulary + which entity types it applies to.
  Concept       — noun sense: entity type + property bundle.
  Quality       — adjective sense: names a slot + value (applies_to derived).
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

from pydantic import BaseModel, ConfigDict, Field


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


class Concept(_Frozen):
    """Noun sense. Property values are validated against the slot registry
    by the loader."""
    lemma: str
    entity_type: str
    properties: dict[str, list[str]] = Field(default_factory=dict)
    # Marks concepts produced by composition rather than authored on disk.
    derived: bool = False
    # Provenance for derived concepts: which lemma(s) + affix(es) produced
    # this. Empty for authored concepts.
    derived_from: Optional[dict[str, str]] = None


class Quality(_Frozen):
    """Adjective sense. `applies_to` is *derived* from the slot's domain;
    not stored on the quality itself."""
    lemma: str
    slot: str
    value: str


class Relation(_Frozen):
    """Preposition / relational-verb schema. Instances live in traces."""
    name: str
    arity: int
    arg_types: list[str]
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


class Action(_Frozen):
    """Verb sense."""
    lemma: str
    transitivity: Literal["intransitive", "transitive", "ditransitive"]
    aspect: Literal["activity", "achievement", "accomplishment", "state"]
    roles: list[RoleSpec]
    effects: list[Effect] = Field(default_factory=list)
    # If true, this verb supports compositional instrument derivation
    # via -il-. Used by the loader; not consulted at runtime.
    derives_instrument: bool = False


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
    """A typed morphological operator. For this slice we only need -il-,
    but the schema is general enough to accept other instrument-deriving or
    role-deriving suffixes (-ej-, -ist-, -ant-, ...) without code changes."""
    form: str                           # e.g. "il"
    kind: Literal["prefix", "suffix"]
    attaches_to: Literal["verb", "noun", "adjective"]
    produces: Literal["verb", "noun", "adjective"]
    output_type: str                    # entity type of the produced concept
    # How to synthesize the produced concept's functional signature.
    # "effect" = read it off the source verb's effect structure.
    signature_source: Literal["effect"]
    # The ending appended after the affix to form the surface lemma.
    # For -il-: "o" (-> tranĉilo).
    noun_ending: Optional[str] = None
    # Default property bundle for the synthesized concept. -il- artifacts
    # are always solid, so we set state_of_matter via the affix definition
    # rather than hardcoding it in the loader. Validated against the slot
    # registry the same way authored properties are.
    output_properties: dict[str, list[str]] = Field(default_factory=dict)
