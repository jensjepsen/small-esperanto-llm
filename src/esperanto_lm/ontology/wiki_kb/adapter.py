"""Bridge between the YAGO KB and the ontology vocabulary.

The KB is a generic knowledge graph (YAGO/schema.org property and type
names). The ontology speaks Esperanto-style lemmas (`naskiĝloko`,
`ĉefurbo`, `homo`, `lando`). This module is the *only* place those two
vocabularies meet.

Caller speaks ontology, adapter translates:

  kb_lookup(kb, "yago:Wolfgang_Amadeus_Mozart", "naskiĝloko")
    → ("yago:Salzburg",)       # canonical KB value(s)

  concept_for_kb_types(("persono", "loko"))
    → "homo"                   # best matching ontology concept

  pick_grounded(kb, "urbo", rng)
    → ("yago:Paris", "Parizo") # qid + canonical EO label

When a mapping is missing, helpers return None / empty — never raise.
Callers fall back to fabrication.
"""
from __future__ import annotations

from random import Random
from typing import Sequence

from .schema import KB, EntityRec, QID


# Ontology property name → list of KB relation names that source it.
# Multiple KB relations can feed the same ontology property
# (e.g. `loko` is fed by both `schema:location` and `schema:containedInPlace`).
# Keys are the ontology vocabulary — what the rest of the codebase should ask for.
_ONT_TO_KB: dict[str, tuple[str, ...]] = {
    # Geography
    "ĉefurbo":         ("yago:capital",),
    "loko":            ("schema:location", "schema:containedInPlace"),
    "najbara":         ("yago:neighbors",),
    "fluas_al":        ("yago:flowsInto",),
    "loĝantaro":       ("yago:populationNumber",),
    "areo":            ("yago:area",),
    "alteco":          ("yago:elevation",),
    "demonimo":        ("yago:demonym",),
    # Person — biography
    "naskiĝdato":      ("schema:birthDate",),
    "naskiĝloko":      ("schema:birthPlace",),
    "mortdato":        ("schema:deathDate",),
    "mortejo":         ("schema:deathPlace",),
    "ŝtataneco":       ("schema:nationality",),
    "sekso":           ("schema:gender",),
    "parolas_lingvon": ("schema:knowsLanguage",),
    "ano_de":          ("schema:memberOf",),
    "studinto_de":     ("schema:alumniOf",),
    "ricevis_premion": ("schema:award",),
    "laboras_por":     ("schema:worksFor",),
    "geedzo":          ("schema:spouse",),
    "infano":          ("schema:children",),
    "grava_verko":     ("yago:notableWork",),
    # Creative works
    "verkinto":        ("schema:author",),
    "reĝisoro":        ("schema:director",),
    "komponinto":      ("schema:musicBy",),
    "aktoro":          ("schema:actor",),
    "produktanto":     ("schema:productionCompany",),
    "eldonejo":        ("schema:publisher",),
    "en_lingvo":       ("schema:inLanguage",),
    "kreita_dato":     ("schema:dateCreated",),
    "materialo":       ("schema:material",),
    # Organizations
    "fondinto":        ("schema:founder",),
    "fond_dato":       ("schema:foundingDate",),
    "fond_loko":       ("schema:foundingLocation",),
    "dungitaro":       ("schema:numberOfEmployees",),
    "posedita_de":     ("yago:ownedBy",),
    # Country / organization → language
    "oficiala_lingvo": ("yago:officialLanguage",),
}


# Reverse lookup, built once at module load.
_KB_TO_ONT: dict[str, str] = {
    kb_prop: ont_prop
    for ont_prop, kb_props in _ONT_TO_KB.items()
    for kb_prop in kb_props
}


# KB-bucket → ontology concept name. Used by `concept_for_kb_types` to
# answer "what is this KB entity's ontology concept?" — a reverse-
# direction utility (KB→ontology) for renderers that need to know the
# concept of a grounded entity. Forward-direction grounding (concept
# → KB candidates) goes through `KB.by_eo_tag` and needs no mapping
# at all: the lexicon's concept lemma IS the index key.
_KB_TYPE_TO_CONCEPT: dict[str, str] = {
    "persono":     "homo",
    "urbo":        "urbo",
    "monto":       "monto",
    "rivero":      "rivero",
    "lago":        "lago",
    "insulo":      "insulo",
    "stelo":       "stelo",
    "libro":       "libro",
}


# Abstract / sortal-neutral concepts that may safely ground to ANY
# entity of the corresponding KB bucket — used only when neither eo_tag
# nor bucket-name match. Restricted by LEMMA (not entity_type) on
# purpose: keys like `filo`/`princino`/`doktoro` are person-typed but
# imply specific relations/roles YAGO won't model; grounding them to a
# random person would teach the model wrong facts (Marie Curie as
# "doktoro=physician"). Stay fab unless explicitly listed here.
_ABSTRACT_FALLBACK: dict[str, tuple[str, ...]] = {
    "homo":    ("persono",),
    "persono": ("persono",),
}


def is_kb_qid(s: str) -> bool:
    """True if `s` looks like a YAGO/Wikidata QID (the values stored in
    `EntityRec.facts` for entity-valued relations)."""
    return s.startswith("yago:") or s.startswith("wd:")


def kb_label(kb: KB, qid: QID) -> str | None:
    """Canonical EO label for `qid`. None if the entity is absent or
    has no EO label."""
    rec = kb.get(qid)
    return rec.label if rec and rec.label else None


def resolve_value(kb: KB, value: str) -> str:
    """Convert a `kb_facts` value to a display string. QIDs become EO
    labels (yago:France → "Francio"); bare strings (dates, demonyms,
    decimals) pass through unchanged.

    YAGO uses a `_generic_instance` suffix for abstract-type singleton
    entities — `yago:English_language_generic_instance` represents
    "the English language as a fact target". The labelled entity is
    the bare-suffix sibling (`yago:English_language` → "angla lingvo");
    we fall back to that when the suffixed entity has no label of its
    own. If neither resolves, return the QID verbatim — better than
    dropping data."""
    if not is_kb_qid(value):
        return value
    lab = kb_label(kb, value)
    if lab is not None:
        return lab
    # Unclassified entities: try the extra-labels fallback (carries EO
    # labels for things like yago:German_language that didn't land in
    # any bucket but are valid fact targets).
    lab = kb.extra_labels.get(value)
    if lab is not None:
        return lab
    # YAGO `_generic_instance` singleton without its own EO label —
    # peek at the bare-suffix sibling (the class definition).
    if value.endswith("_generic_instance"):
        stripped = value[: -len("_generic_instance")]
        lab = kb_label(kb, stripped) or kb.extra_labels.get(stripped)
        if lab is not None:
            return lab
    return value


def resolve_values(kb: KB, values: tuple[str, ...]) -> tuple[str, ...]:
    """Bulk variant of `resolve_value` — for the tuple-of-values shape
    that `EntityInstance.kb_facts` and `kb_lookup` return."""
    return tuple(resolve_value(kb, v) for v in values)


def kb_comment(kb: KB, qid: QID) -> str:
    """One-line EO description of `qid`, or empty string."""
    rec = kb.get(qid)
    return rec.comment if rec else ""


def concept_for_kb_types(types: Sequence[str]) -> str | None:
    """Pick the ontology concept best matching the KB type set. Prefers
    the narrowest mapped type (lando over loko, urbo over loko). Returns
    None if no type in `types` has an ontology mapping."""
    # Iterate in mapping order — narrower types are declared first, so
    # the first hit is the most specific available.
    for t in types:
        c = _KB_TYPE_TO_CONCEPT.get(t)
        if c is not None:
            return c
    return None


def kb_types_for_concept(concept: str, lex=None) -> tuple[str, ...]:
    """Which KB buckets ground this ontology concept (fallback path)?

    Used only for ABSTRACT lexicon concepts that YAGO doesn't label
    with that exact lemma — e.g. `homo`, `persono`. Most concepts
    resolve via the eo_tag direct path in `pick_grounded`, which
    bypasses this function entirely. Empty tuple when the concept
    isn't on the safe-to-broaden list — caller falls back to
    fabrication, which is correct for kinship/role concepts that
    don't correspond to YAGO occupation types."""
    return _ABSTRACT_FALLBACK.get(concept, ())


def kb_lookup(
    kb: KB, qid: QID, ont_property: str,
) -> tuple[str, ...]:
    """Return canonical value(s) for `ont_property` of the entity at
    `qid`. Values may be QIDs (entity references) or bare strings
    (dates, decimals, demonyms). Empty tuple when the entity or
    property is absent.

    Multiple KB relations may feed one ontology property (e.g. `loko`
    is fed by both `schema:location` and `schema:containedInPlace`);
    values from all sources are concatenated in declared order, with
    duplicates removed."""
    kb_props = _ONT_TO_KB.get(ont_property)
    if not kb_props:
        return ()
    rec = kb.get(qid)
    if not rec:
        return ()
    seen: set[str] = set()
    out: list[str] = []
    for kp in kb_props:
        for v in rec.facts.get(kp, ()):
            if v not in seen:
                seen.add(v)
                out.append(v)
    return tuple(out)


def ontology_property_for_kb(kb_property: str) -> str | None:
    """Reverse of the property mapping — useful when iterating an
    entity's raw KB facts to surface them as ontology properties."""
    return _KB_TO_ONT.get(kb_property)


def all_ontology_properties_of(
    kb: KB, qid: QID,
) -> dict[str, tuple[str, ...]]:
    """All known ontology properties of `qid`, with their canonical
    values. Convenience for the sampler when grounding an entity:
    materialize every known fact in one call."""
    rec = kb.get(qid)
    if not rec:
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for kb_prop, values in rec.facts.items():
        ont = _KB_TO_ONT.get(kb_prop)
        if ont is None:
            continue
        # Property may already exist from another KB source; merge.
        existing = out.get(ont, ())
        merged: list[str] = list(existing)
        seen: set[str] = set(existing)
        for v in values:
            if v not in seen:
                seen.add(v)
                merged.append(v)
        out[ont] = tuple(merged)
    return out


def pick_grounded(
    kb: KB, concept: str, rng: Random, *,
    lex=None, must_have: Sequence[str] = (),
) -> tuple[QID, str] | None:
    """Sample a KB entity matching the ontology `concept`. Returns
    (qid, eo_label) or None if nothing in the KB lines up with the
    concept. Caller wraps the result into an EntityInstance with `qid`
    + `name` set.

    Three-stage match:
      1. Direct: `kb.by_eo_tag[concept]` — entities whose rdf:types
         include a YAGO class labelled with the concept's lemma in
         Esperanto. Marie Curie ↔ "fizikisto"/"kemiisto" both match
         their corresponding ontology concepts. Principal path — fully
         derived from KB labels, no hand-curation.
      2. Bucket-named lemmas: `kb.of_type(concept)`. The lexicon's
         "urbo" matches the KB bucket "urbo" directly even though
         YAGO doesn't tag any city's rdf:type with the literal label
         "urbo" (cities are typed as Comune, Capital_city, etc.).
      3. Abstract fallback: for sortal-neutral concepts (`homo`,
         `persono`) that YAGO doesn't tag by lemma OR by bucket-name,
         walk to the matching KB bucket via lexicon entity_type.

    `must_have` filters to entities with non-empty values for each
    listed ontology property — falls back to unfiltered if no
    candidate qualifies."""
    candidates: list[QID] = list(kb.by_eo_tag.get(concept, ()))
    if not candidates:
        candidates = list(kb.of_type(concept))
    if not candidates:
        # Abstract concept (homo, persono) — fall back to bucket.
        kb_types = kb_types_for_concept(concept, lex=lex)
        for t in kb_types:
            candidates.extend(kb.of_type(t))
    if not candidates:
        return None
    if must_have:
        filtered = [
            q for q in candidates
            if all(kb_lookup(kb, q, p) for p in must_have)
        ]
        candidates = filtered or candidates
    qid = rng.choice(candidates)
    label = kb_label(kb, qid)
    if label is None:
        return None
    return qid, label


def resolve_label(
    kb: KB, label: str, *, concept: str | None = None,
) -> EntityRec | None:
    """Look up an entity by EO label, with optional ontology-concept
    disambiguation. Returns the most-facts candidate matching the
    concept, or the most-facts candidate overall when `concept` is
    None. Wraps `KB.by_name` with concept↔kb-type translation so
    callers can speak ontology."""
    if concept:
        kb_types = kb_types_for_concept(concept)
        for t in kb_types:
            rec = kb.by_name(label, type_filter=t)
            if rec:
                return rec
        return None
    return kb.by_name(label)
