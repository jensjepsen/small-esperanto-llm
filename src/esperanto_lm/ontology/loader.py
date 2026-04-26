"""Lexicon loader.

Reads JSONL on disk into typed registries, validates references against the
slot registry and type spine, and applies compositional derivation rules
(currently just -il-) to materialize derived concepts.

Validation philosophy: fail loudly at load. Any slot-name typo, any value
outside a slot's vocabulary, any type that's not in the spine — these all
raise here, with the offending lemma in the message. Runtime code can then
trust the lexicon.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .morph import DefaultMorphParser, MorphParser
from .schemas import (
    Action,
    Affix,
    Concept,
    ContainmentFact,
    ContainmentPattern,
    Effect,
    PropertySlot,
    Quality,
    Relation,
)
from .types import TypeSpine


# Slot name used to encode the verb that an instrument's functional
# signature points at. Stored as a single-element list of the verb lemma;
# the engine resolves it to the verb's effect structure on the fly.
FUNCTIONAL_SIGNATURE = "functional_signature"


@dataclass
class Lexicon:
    """In-memory, indexed view of the loaded ontology."""
    types: TypeSpine
    slots: dict[str, PropertySlot]
    concepts: dict[str, Concept]
    relations: dict[str, Relation]
    actions: dict[str, Action]
    affixes: dict[str, Affix]                   # keyed by form (e.g. "il")
    containment: list[ContainmentFact] = field(default_factory=list)
    qualities: dict[str, Quality] = field(default_factory=dict)  # keyed by lemma

    # ---------- read helpers ----------
    def concept(self, lemma: str) -> Concept:
        return self.concepts[lemma]

    def action(self, lemma: str) -> Action:
        return self.actions[lemma]

    def slot(self, name: str) -> PropertySlot:
        return self.slots[name]

    def role_of(self, action_lemma: str, role_name: str):
        for r in self.actions[action_lemma].roles:
            if r.name == role_name:
                return r
        raise KeyError(f"action {action_lemma!r} has no role {role_name!r}")


# --------------------------- raw JSONL helpers ---------------------------

def _read_jsonl(path: Path) -> Iterable[dict]:
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON: {e}") from e


# --------------------------- validation passes ---------------------------

def _validate_property_bundle(
    *, owner: str, properties: dict[str, list[str]],
    entity_type: str, slots: dict[str, PropertySlot], spine: TypeSpine,
) -> None:
    """Check that every (slot, value) is legal on `entity_type`."""
    for slot_name, values in properties.items():
        if slot_name not in slots:
            raise ValueError(
                f"{owner}: references unknown slot {slot_name!r}")
        slot = slots[slot_name]
        # Type-applicability: at least one of slot.applies_to must be an
        # ancestor (or equal) of the owner's entity_type.
        if not any(spine.is_subtype(entity_type, a) for a in slot.applies_to):
            raise ValueError(
                f"{owner}: slot {slot_name!r} does not apply to entity_type "
                f"{entity_type!r} (slot.applies_to={slot.applies_to})")
        # Vocabulary check, unless slot has open vocabulary.
        if slot.vocabulary is not None:
            for v in values:
                if v not in slot.vocabulary:
                    raise ValueError(
                        f"{owner}: slot {slot_name!r} value {v!r} not in "
                        f"vocabulary {slot.vocabulary}")
        if slot.scalar and len(values) > 1:
            raise ValueError(
                f"{owner}: slot {slot_name!r} is scalar but got "
                f"{len(values)} values")


# --------------------------- compositional derivation ---------------------------

def _derive_concept_from_verb(
    verb: Action, affix: Affix, parser: MorphParser,
) -> Concept:
    """Compose verb + affix into a derived concept.

    Stem extraction: we use the verb lemma minus its final -i, *not* the
    morphological root from the parser. The parser strips derivational
    suffixes like -ig- and -iĝ- to find the bare etymon, which is wrong
    for compositional naming: purigi has root=pur but the conventional
    tool is purigilo (not purilo). Lemma-minus-i gives purig → purigilo,
    while still working for plain stems like tranĉi → tranĉ → tranĉilo.
    Parser is kept for future affix kinds whose derivation needs root
    stripping.

    Functional signature is attached only when `signature_source == 'effect'`
    (the -il- case): the produced artifact is defined by the verb it tools
    for. -ist-, -aĵ-, -ej- derivations don't carry a functional signature.
    """
    _ = parser  # parser unused for current affixes; see docstring
    stem = verb.lemma[:-1] if verb.lemma.endswith("i") else verb.lemma
    surface = stem + affix.form + (affix.noun_ending or "")
    properties = dict(affix.output_properties)
    if affix.signature_source == "effect":
        properties[FUNCTIONAL_SIGNATURE] = [verb.lemma]
    return Concept(
        lemma=surface,
        entity_type=affix.output_type,
        properties=properties,
        derived=True,
        derived_from={"verb": verb.lemma, "affix": affix.form},
    )


def _derive_concept_from_concept(
    parent: Concept, affix: Affix,
    slots: dict[str, PropertySlot], spine: TypeSpine,
) -> Concept:
    """Compose noun + affix into a derived concept (e.g. kato + id =
    katido). Stem extraction strips the input's final -o (Esperanto
    noun ending). No functional signature: -id-/-ar- products are
    independent of any verb.

    Property inheritance: each parent property is copied into the
    derived concept IF its slot's `applies_to` covers the affix's
    `output_type`. This means -id- (output_type=animal) inherits the
    parent animal's locomotion (slot.applies_to=animate ⊇ animal),
    while -ar- (output_type=collective) does not (locomotion doesn't
    apply to collective). Affix's `output_properties` then overlay,
    taking precedence over inherited values."""
    stem = parent.lemma[:-1] if parent.lemma.endswith("o") else parent.lemma
    surface = stem + affix.form + (affix.noun_ending or "")
    properties: dict[str, list[str]] = {}
    for slot_name, value in parent.properties.items():
        slot = slots.get(slot_name)
        if slot is None:
            continue
        if any(spine.is_subtype(affix.output_type, t) for t in slot.applies_to):
            properties[slot_name] = list(value)
    properties.update({k: list(v) for k, v in affix.output_properties.items()})
    return Concept(
        lemma=surface,
        entity_type=affix.output_type,
        properties=properties,
        derived=True,
        derived_from={"concept": parent.lemma, "affix": affix.form},
    )


def _derive_quality_from_verb(verb: Action, affix: Affix) -> Quality:
    """Compose verb + adjective affix into a Quality lemma. For -it-
    (passive past participle): ŝlosi + it + a = ŝlosita. The result
    is the adjective naming the state the verb produces."""
    stem = verb.lemma[:-1] if verb.lemma.endswith("i") else verb.lemma
    surface = stem + affix.form + (affix.noun_ending or "")
    return Quality(
        lemma=surface,
        derived=True,
        derived_from={"verb": verb.lemma, "affix": affix.form},
    )


def _derive_quality_from_quality(
    parent: Quality, affix: Affix,
) -> Quality:
    """Compose adjective + composition affix. Three patterns:
      mal- (prefix): mal + lemma → malvarma (no stem strip)
      -eg- (suffix): strip -a, add eg + a → varmega
      -et- (suffix): strip -a, add et + a → varmeta
    """
    if affix.kind == "prefix":
        surface = affix.form + parent.lemma
    else:
        # Suffix: strip the trailing -a from the parent lemma, then
        # append affix form + new ending.
        stem = parent.lemma[:-1] if parent.lemma.endswith("a") else parent.lemma
        surface = stem + affix.form + (affix.noun_ending or "")
    return Quality(
        lemma=surface,
        derived=True,
        derived_from={"quality": parent.lemma, "affix": affix.form},
    )


def _apply_quality_derivations(
    *, actions: dict[str, Action],
    qualities_in: dict[str, Quality],
    affixes: dict[str, Affix],
) -> dict[str, Quality]:
    """Synthesize Quality lemmas via adjective-producing affixes.

    Two dispatch paths:
      attaches_to=verb: iterate Actions, fire when `trigger_flag` is
        set on the verb. Used by -it- (passive past participle).
      attaches_to=adjective: iterate existing qualities, fire
        unconditionally. Used by mal-/-eg-/-et- composition. Skips
        qualities that are themselves derived from another adjective
        (no infinite recursion: malvarma stays a single mal-, not
        mal-mal-mal-).

    Already-existing qualities (authored or earlier-derived) are
    skipped silently — collisions are common in composition (mal- on
    "varma" produces "malvarma" which we may already author for
    explicit clarity), and the existing entry is the source of truth.
    """
    qualities: dict[str, Quality] = {}
    for affix in affixes.values():
        if affix.produces != "adjective":
            continue

        if affix.attaches_to == "verb":
            if not affix.trigger_flag:
                raise ValueError(
                    f"affix {affix.form!r}: adjective-from-verb requires "
                    f"trigger_flag")
            for verb in actions.values():
                if not getattr(verb, affix.trigger_flag, False):
                    continue
                q = _derive_quality_from_verb(verb, affix)
                if q.lemma in qualities or q.lemma in qualities_in:
                    continue
                qualities[q.lemma] = q

        elif affix.attaches_to == "adjective":
            # Source pool: authored qualities + qualities derived from
            # verbs (e.g. ŝlosita). Don't recurse into qualities that
            # are themselves adjective-derived — mal- on malvarma would
            # give "malmalvarma" which isn't meaningful.
            sources = list(qualities_in.values()) + [
                q for q in qualities.values()
                if not (q.derived_from and "quality" in q.derived_from)
            ]
            for parent in sources:
                if parent.derived_from and "quality" in parent.derived_from:
                    continue
                # Lexical-shape filters to avoid double-marking:
                #   mal-  on a lemma already starting "mal" or "ne" →
                #         don't (malvarma → malmalvarma is meaningless;
                #          nebrulebla → malnebrulebla = un-not-
                #          flammable is just confused).
                #   -eg-  on a lemma already ending "ega"/"eta" → don't
                #         (varmega → varmegega is super-super-warm;
                #          intensification doesn't compound usefully)
                #   -et-  on a lemma already ending "ega"/"eta" → don't
                if affix.form == "mal" and parent.lemma.startswith(("mal", "ne")):
                    continue
                if affix.form in ("eg", "et"):
                    if parent.lemma.endswith(("ega", "eta")):
                        continue
                q = _derive_quality_from_quality(parent, affix)
                if q.lemma in qualities or q.lemma in qualities_in:
                    continue
                qualities[q.lemma] = q

    return qualities


def _apply_derivations(
    *, actions: dict[str, Action], concepts: dict[str, Concept],
    affixes: dict[str, Affix], parser: MorphParser,
    slots: dict[str, PropertySlot], spine: TypeSpine,
) -> dict[str, Concept]:
    """For each affix, synthesize derived concepts. Two dispatch paths:

    - `attaches_to=verb`: iterate actions, fire when `trigger_flag`
      Boolean is set on the verb. Used by -il-, -ist-, -aĵ-, -ej-.

    - `attaches_to=noun`: iterate concepts, fire when concept's
      entity_type is-a `trigger_concept_type`. Used by -id-, -ar-.

    Adding a new affix kind is data-only when it fits one of these
    patterns; a genuinely new pattern needs a new dispatch arm here.
    """
    derived: dict[str, Concept] = {}

    def _admit(concept: Concept, source_label: str) -> None:
        _validate_property_bundle(
            owner=f"derived:{concept.lemma}",
            properties=concept.properties,
            entity_type=concept.entity_type,
            slots=slots, spine=spine,
        )
        if concept.lemma in derived:
            raise ValueError(
                f"derived concept {concept.lemma!r} produced twice "
                f"(by {source_label!r} and another source)")
        derived[concept.lemma] = concept

    for affix in affixes.values():
        if affix.produces != "noun":
            continue

        if affix.attaches_to == "verb":
            if not affix.trigger_flag:
                raise ValueError(
                    f"affix {affix.form!r}: attaches_to=verb requires "
                    f"trigger_flag")
            for verb in actions.values():
                if not getattr(verb, affix.trigger_flag, False):
                    continue
                _admit(_derive_concept_from_verb(verb, affix, parser),
                       verb.lemma)

        elif affix.attaches_to == "noun":
            if not affix.trigger_concept_type:
                raise ValueError(
                    f"affix {affix.form!r}: attaches_to=noun requires "
                    f"trigger_concept_type")
            target = affix.trigger_concept_type
            if not spine.known(target):
                raise ValueError(
                    f"affix {affix.form!r}: unknown "
                    f"trigger_concept_type {target!r}")
            for parent in concepts.values():
                if not spine.is_subtype(parent.entity_type, target):
                    continue
                _admit(_derive_concept_from_concept(
                       parent, affix, slots, spine),
                       parent.lemma)

    return derived


# --------------------------- top-level load ---------------------------

# Default data lives next to this module — `ontology/data/`. Keeps the
# lexicon under version control, lets `load_lexicon()` work with no
# argument from anywhere (not just a CWD with a `data/` symlink), and
# makes the package installable without an external data drop. Pass an
# explicit `data_dir` to override (e.g., for experiments or tests).
DATA_DIR = Path(__file__).parent / "data"


def load_lexicon(
    data_dir: Path = DATA_DIR,
    parser: MorphParser | None = None,
    *, bake_derivations: bool = True,
) -> Lexicon:
    """Load the lexicon from `data_dir`.

    `bake_derivations`: when True (default), runs the DSL's
    concept-compatible derivations once at load and writes their
    implied properties back into `concept.properties`. This closes
    the historical gap where the sampler — which scans
    concept.properties directly — couldn't see facts that only
    materialized at runtime via the DSL engine. Set to False for
    tests of the bake function itself or for inspecting raw
    on-disk concepts.
    """
    parser = parser or DefaultMorphParser()

    spine = TypeSpine.from_json(data_dir / "types.json")

    slots: dict[str, PropertySlot] = {}
    for d in _read_jsonl(data_dir / "slots.jsonl"):
        s = PropertySlot(**d)
        if s.name in slots:
            raise ValueError(f"duplicate slot {s.name!r}")
        for t in s.applies_to:
            if not spine.known(t):
                raise ValueError(
                    f"slot {s.name!r}: applies_to references unknown type "
                    f"{t!r}")
        slots[s.name] = s

    concepts: dict[str, Concept] = {}
    for d in _read_jsonl(data_dir / "concepts.jsonl"):
        c = Concept(**d)
        if c.lemma in concepts:
            raise ValueError(f"duplicate concept {c.lemma!r}")
        if not spine.known(c.entity_type):
            raise ValueError(
                f"concept {c.lemma!r}: unknown entity_type {c.entity_type!r}")
        _validate_property_bundle(
            owner=f"concept:{c.lemma}", properties=c.properties,
            entity_type=c.entity_type, slots=slots, spine=spine,
        )
        concepts[c.lemma] = c

    relations: dict[str, Relation] = {}
    for d in _read_jsonl(data_dir / "relations.jsonl"):
        r = Relation(**d)
        if r.name in relations:
            raise ValueError(f"duplicate relation {r.name!r}")
        if len(r.arg_types) != r.arity:
            raise ValueError(
                f"relation {r.name!r}: arity={r.arity} but "
                f"arg_types has {len(r.arg_types)} entries")
        if len(r.arg_names) != r.arity:
            raise ValueError(
                f"relation {r.name!r}: arity={r.arity} but "
                f"arg_names has {len(r.arg_names)} entries")
        if len(set(r.arg_names)) != len(r.arg_names):
            raise ValueError(
                f"relation {r.name!r}: arg_names must be unique, "
                f"got {r.arg_names}")
        for t in r.arg_types:
            if not spine.known(t):
                raise ValueError(
                    f"relation {r.name!r}: unknown arg type {t!r}")
        relations[r.name] = r

    actions: dict[str, Action] = {}
    for d in _read_jsonl(data_dir / "actions.jsonl"):
        a = Action(**d)
        if a.lemma in actions:
            raise ValueError(f"duplicate action {a.lemma!r}")
        for role in a.roles:
            if not spine.known(role.type):
                raise ValueError(
                    f"action {a.lemma!r} role {role.name!r}: unknown type "
                    f"{role.type!r}")
            # Property constraints on roles are validated against slots.
            _validate_property_bundle(
                owner=f"action:{a.lemma}.role.{role.name}",
                properties=role.properties, entity_type=role.type,
                slots=slots, spine=spine,
            )
        # Effects must target a known role and use a slot legal on that
        # role's type.
        role_by_name = {r.name: r for r in a.roles}
        for eff in a.effects:
            if eff.target_role not in role_by_name:
                raise ValueError(
                    f"action {a.lemma!r}: effect targets unknown role "
                    f"{eff.target_role!r}")
            target_type = role_by_name[eff.target_role].type
            _validate_property_bundle(
                owner=f"action:{a.lemma}.effect",
                properties={eff.property: [eff.value]},
                entity_type=target_type, slots=slots, spine=spine,
            )
        actions[a.lemma] = a

    affixes: dict[str, Affix] = {}
    for d in _read_jsonl(data_dir / "affixes.jsonl"):
        af = Affix(**d)
        if af.form in affixes:
            raise ValueError(f"duplicate affix {af.form!r}")
        # Adjective-producing affixes synthesize Quality lemmas, not
        # Concept entities — their `output_type` is the conventional
        # marker "quality" and isn't a real entity type. Skip the
        # spine check for those.
        if af.produces != "adjective":
            if not spine.known(af.output_type):
                raise ValueError(
                    f"affix {af.form!r}: unknown output_type "
                    f"{af.output_type!r}")
        affixes[af.form] = af

    # Compositional derivation. Synthesized concepts join the registry
    # alongside authored ones.
    derived = _apply_derivations(
        actions=actions, concepts=concepts, affixes=affixes,
        parser=parser, slots=slots, spine=spine,
    )
    for lemma, concept in derived.items():
        if lemma in concepts:
            raise ValueError(
                f"derived concept {lemma!r} collides with authored concept; "
                f"remove the authored entry — derivations are the truth")
        concepts[lemma] = concept

    # Qualities live in their own registry — they're adjective lemmas
    # referenced by slot vocabularies and rendered as adjectives by
    # the realizer. Two sources:
    #   (1) Authored in qualities.jsonl — base adjectives like
    #       "fragila", "varma", "pura" that aren't tied to a verb.
    #   (2) Auto-derived from verbs flagged `derives_quality` via the
    #       -it- (passive past participle) affix — e.g. "ŝlosita"
    #       from ŝlosi.
    qualities: dict[str, Quality] = {}
    qualities_path = data_dir / "qualities.jsonl"
    if qualities_path.exists():
        for d in _read_jsonl(qualities_path):
            q = Quality(**d)
            if q.lemma in qualities:
                raise ValueError(f"duplicate quality {q.lemma!r}")
            qualities[q.lemma] = q
    derived_qualities = _apply_quality_derivations(
        actions=actions, qualities_in=qualities, affixes=affixes)
    for lemma, q in derived_qualities.items():
        # Skip-if-exists handled inside _apply_quality_derivations now.
        # If we still see a collision here it's a bug.
        if lemma in qualities:
            raise ValueError(
                f"derived quality {lemma!r} collides; this should "
                f"have been skipped inside derivation")
        qualities[lemma] = q

    # Bake concept-compatible DSL derivations into concept properties
    # so the sampler — which scans concept.properties directly — sees
    # the same world the runtime engine does. Skip for bake-disabled
    # loads (tests of the bake function itself, raw inspection).
    if bake_derivations:
        from .dsl.bake import bake_concept_derivations
        from .dsl.rules import DEFAULT_DSL_DERIVATIONS
        concepts = bake_concept_derivations(
            concepts, DEFAULT_DSL_DERIVATIONS, spine,
            slots=slots, relations=relations)

    containment: list[ContainmentFact] = []
    containment_path = data_dir / "containment.jsonl"
    if containment_path.exists():
        for idx, d in enumerate(_read_jsonl(containment_path), start=1):
            # Normalize shorthand: container: "X" => container_pattern sense_id=X
            if "container" in d and "container_pattern" in d:
                raise ValueError(
                    f"containment[{idx}]: set exactly one of 'container' or "
                    f"'container_pattern', got both")
            if "container" in d and d["container"] is not None:
                sense = d["container"]
                d = {k: v for k, v in d.items() if k != "container"}
                d["container_pattern"] = {"sense_id": sense}
            # Symmetric normalization on the contained side. The shorthand
            # `contained: "X"` becomes a sense_id pattern if X is a concept,
            # or an entity_type pattern if X is in the spine. We keep
            # `contained` as the canonical scalar field for the common case
            # (single named concept) and synthesize `contained_pattern` only
            # when the resolver needs to expand it.
            if "contained" in d and "contained_pattern" in d:
                if d.get("contained") is not None and d.get(
                        "contained_pattern") is not None:
                    raise ValueError(
                        f"containment[{idx}]: set exactly one of "
                        f"'contained' or 'contained_pattern', got both")
            if "container_pattern" not in d or d["container_pattern"] is None:
                raise ValueError(
                    f"containment[{idx}]: missing container/container_pattern")
            fact = ContainmentFact(**d)
            _validate_containment_fact(
                fact, index=idx, relations=relations, concepts=concepts,
                slots=slots, spine=spine,
            )
            containment.append(fact)

    return Lexicon(
        types=spine, slots=slots, concepts=concepts,
        relations=relations, actions=actions, affixes=affixes,
        containment=containment, qualities=qualities,
    )


def _validate_containment_fact(
    fact: ContainmentFact, *, index: int,
    relations: dict[str, Relation],
    concepts: dict[str, Concept],
    slots: dict[str, PropertySlot],
    spine: TypeSpine,
) -> None:
    """Loud-fail validation for one ContainmentFact."""
    pat = fact.container_pattern
    if pat is None:
        raise ValueError(f"containment[{index}]: no container_pattern")
    # At least one pattern field must be set.
    if not any((pat.sense_id, pat.entity_type, pat.suffix,
                pat.property, pat.contains)):
        raise ValueError(
            f"containment[{index}]: container_pattern must set at least one "
            f"of sense_id/entity_type/suffix/property/contains")
    if pat.sense_id is not None and pat.sense_id not in concepts:
        raise ValueError(
            f"containment[{index}]: pattern.sense_id {pat.sense_id!r} is not "
            f"a known concept")
    if pat.entity_type is not None and not spine.known(pat.entity_type):
        raise ValueError(
            f"containment[{index}]: pattern.entity_type {pat.entity_type!r} "
            f"is not in the type spine")
    if pat.property is not None:
        for slot_name, value in pat.property.items():
            if slot_name not in slots:
                raise ValueError(
                    f"containment[{index}]: pattern.property references "
                    f"unknown slot {slot_name!r}")
            slot = slots[slot_name]
            if slot.vocabulary is not None and value not in slot.vocabulary:
                raise ValueError(
                    f"containment[{index}]: pattern.property value {value!r} "
                    f"not in slot {slot_name!r} vocabulary")
    if pat.contains is not None and pat.contains not in concepts:
        raise ValueError(
            f"containment[{index}]: pattern.contains {pat.contains!r} is "
            f"not a known concept")
    # Contained side: either a string (concept or type) or a pattern. At
    # least one must be set.
    if fact.contained is None and fact.contained_pattern is None:
        raise ValueError(
            f"containment[{index}]: must set 'contained' or "
            f"'contained_pattern'")
    if fact.contained is not None:
        if (fact.contained not in concepts
                and not spine.known(fact.contained)):
            raise ValueError(
                f"containment[{index}]: contained {fact.contained!r} is "
                f"neither a known concept nor a known entity type")
    if fact.contained_pattern is not None:
        cp = fact.contained_pattern
        if not any((cp.sense_id, cp.entity_type, cp.suffix, cp.property)):
            raise ValueError(
                f"containment[{index}]: contained_pattern must set at least "
                f"one of sense_id/entity_type/suffix/property")
        if cp.sense_id is not None and cp.sense_id not in concepts:
            raise ValueError(
                f"containment[{index}]: contained_pattern.sense_id "
                f"{cp.sense_id!r} is not a known concept")
        if cp.entity_type is not None and not spine.known(cp.entity_type):
            raise ValueError(
                f"containment[{index}]: contained_pattern.entity_type "
                f"{cp.entity_type!r} is not in the type spine")
        if cp.property is not None:
            for slot_name, value in cp.property.items():
                if slot_name not in slots:
                    raise ValueError(
                        f"containment[{index}]: contained_pattern.property "
                        f"references unknown slot {slot_name!r}")
                slot = slots[slot_name]
                if (slot.vocabulary is not None
                        and value not in slot.vocabulary):
                    raise ValueError(
                        f"containment[{index}]: contained_pattern.property "
                        f"value {value!r} not in slot {slot_name!r} "
                        f"vocabulary")
    if fact.relation not in relations:
        raise ValueError(
            f"containment[{index}]: unknown relation {fact.relation!r}")


# Re-export for callers that need to dig into the verb's effect via the
# functional signature stored on a derived concept.
def resolve_signature(lexicon: Lexicon, concept: Concept) -> Action | None:
    """If `concept` carries a functional_signature, return the source verb's
    Action. Otherwise None."""
    sig = concept.properties.get(FUNCTIONAL_SIGNATURE)
    if not sig:
        return None
    verb_lemma = sig[0]
    return lexicon.actions.get(verb_lemma)


def signature_effects(
    lexicon: Lexicon, concept: Concept,
) -> list[Effect]:
    """The effects the source verb of this instrument's signature produces
    (excluding effects on the instrument role itself)."""
    verb = resolve_signature(lexicon, concept)
    if verb is None:
        return []
    return [e for e in verb.effects if e.target_role != "instrument"]
