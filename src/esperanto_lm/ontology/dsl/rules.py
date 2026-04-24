"""DSL ports of the causal-rule library.

Phases 1–4 ported here. Each rule is a module-level value the engine
introspects via `collect_rules`. `make_use_instrument_rules(lex)`
emits one rule per instrument-capable verb from the lexicon — the
only dynamic-dispatch rule needed lexicon access in the old engine,
and the DSL replaces it with a cleanly-enumerated set.
"""
from __future__ import annotations

from ..loader import Lexicon
from .engine import Rule
from . import (
    bind, caused_by, closure, create_entity, derive, emit, entity, event,
    has_concept_field, past_event, property, rel, rule, var,
)


# ---------- causal: fragile_falls_breaks (Phase 1) ----------------------

fragile_falls_breaks = rule(
    when=event("fali",
               theme=entity(fragility="fragile", integrity="intact")
                     & bind(T := var("T"))),
    then=emit("rompiĝi", theme=T).changing(T, "integrity", "broken"),
    name="fragile_falls_breaks",
)


# ---------- causal: hungry_eats_sated (Phase 2) -------------------------
#
# Property-conditioned rule: the agent must currently be hungry. The
# emitted satiĝi event carries the hunger transition on itself
# (`.changing(...)`) so the trace shape matches the old engine —
# one event with a property_change, not two separate events.

hungry_eats_sated = rule(
    when=event("manĝi",
               agent=entity(hunger="hungry") & bind(A := var("A"))),
    then=emit("satiĝi", theme=A).changing(A, "hunger", "sated"),
    name="hungry_eats_sated",
)


# ---------- causal: broken_fragile_creates_shards (Phase 2) -------------
#
# When a fragile thing breaks, its concept-level transforms_on_break
# names what it transforms into (e.g. glaso → vitropecetoj). Spawn
# that entity and emit aperi.

broken_fragile_creates_shards = rule(
    when=event("rompiĝi", theme=bind(T_break := var("T"))),
    given=[has_concept_field(T_break, "transforms_on_break",
                             K_break := var("K"))],
    then=[
        create_entity(concept=K_break, as_var=(S_break := var("S")),
                      from_=T_break),
        emit("aperi", theme=S_break),
    ],
    name="broken_fragile_creates_shards",
)


# ---------- causal: wet_liquid_container_tips (Phase 2) -----------------
#
# Sibling shape: when a liquid falls, its concept-level
# transforms_on_spill names the puddle concept (akvo → flako). Same
# structural pattern as broken_fragile_creates_shards.

wet_liquid_container_tips = rule(
    when=event("fali", theme=bind(T_spill := var("T"))),
    given=[has_concept_field(T_spill, "transforms_on_spill",
                             K_spill := var("K"))],
    then=[
        create_entity(concept=K_spill, as_var=(S_spill := var("S")),
                      from_=T_spill),
        emit("aperi", theme=S_spill),
    ],
    name="wet_liquid_container_tips",
)


# ---------- causal: container_falls_contents_fall (Phase 3) -------------
#
# When a container falls, every non-location entity en/sur it also
# falls — once per entity (the past_event guard prevents a second
# fali when broken_container_releases_contents would also fire on
# the same contents). The non-location filter lives inside the rel
# arg via `~entity(type="location") & bind(I)`.

container_falls_contents_fall = rule(
    when=event("fali", theme=bind(C := var("C"))),
    given=[
        rel("en",
            contained=~entity(type="location") & bind(I := var("I")),
            container=C)
        | rel("sur",
              contained=~entity(type="location") & bind(I),
              container=C),
        ~past_event("fali", theme=I),
    ],
    then=emit("fali", theme=I),
    name="container_falls_contents_fall",
)


# ---------- causal: broken_container_releases_contents (Phase 3) --------

broken_container_releases_contents = rule(
    when=event("rompiĝi", theme=bind(C2 := var("C"))),
    given=[
        rel("en",
            contained=~entity(type="location") & bind(I2 := var("I")),
            container=C2)
        | rel("sur",
              contained=~entity(type="location") & bind(I2),
              container=C2),
        ~past_event("fali", theme=I2),
    ],
    then=emit("fali", theme=I2),
    name="broken_container_releases_contents",
)


# ---------- causal: carried_thing_falls_when_carrier_falls (Phase 3) ----
#
# When a person falls, everything they `havi` also falls. Fragility
# isn't checked here — a dropped book hits the floor too. Breakage
# is a separate consequence: `fragile_falls_breaks` fires on the
# resulting fali if the carried thing happens to be fragile.

carried_thing_falls_when_carrier_falls = rule(
    when=event("fali",
               theme=entity(type="person") & bind(P := var("P"))),
    given=[
        rel("havi", owner=P, theme=bind(F := var("F"))),
        ~past_event("fali", theme=F),
    ],
    then=emit("fali", theme=F),
    name="carried_thing_falls_when_carrier_falls",
)


# ---------- causal: fire_spreads_to_adjacent_flammables (Phase 3) -------
#
# Closure with max_steps=1 yields immediate neighbors only — the
# engine's outer loop drives further propagation, which preserves the
# old engine's layer-by-layer cascade depth. Without max_steps the
# whole connected component would burn in one firing, collapsing the
# cascade structure.

fire_spreads_to_adjacent_flammables = rule(
    when=event("bruli", theme=bind(B := var("B"))),
    given=[
        closure({"en", "sur"}, from_=B,
                to_=(~entity(type="location")
                     & entity(flammability="flammable")
                     & bind(N := var("N"))),
                max_steps=1),
        ~past_event("bruli", theme=N),
    ],
    then=emit("bruli", theme=N).changing(N, "presence", "consumed"),
    name="fire_spreads_to_adjacent_flammables",
)


# ---------- causal: person_slips_on_wet (Phase 3, updated) -------------
#
# Wet surfaces cause slips; sharp shards do not (stepping on broken
# glass would cut you, but that's a different event we don't model).
# Filter on `hazard="slippery"` specifically — the aperi event has to
# bring a slippery thing into existence (a flako is the canonical
# example). The hazard's location is inferred from the cause's theme —
# the entity that fell or broke to produce the puddle.

person_slips_on_wet = rule(
    when=event("aperi",
               theme=entity(hazard="slippery") & bind(H := var("H"))),
    given=[
        # The cause's theme is the origin entity (the thing that
        # fell to produce the puddle). `fali` is the expected shape;
        # `rompiĝi` is included for symmetry in case a future
        # rule produces a slippery thing from a break.
        caused_by("rompiĝi", theme=bind(O := var("O")))
        | caused_by("fali", theme=O),
        # Walk the containment chain to the surrounding location.
        closure({"en", "sur"}, from_=O,
                to_=entity(type="location") & bind(L := var("L"))),
        # Find people in that location.
        rel("en",
            contained=entity(type="person") & bind(PWH := var("P")),
            container=L),
        ~past_event("fali", theme=PWH),
    ],
    then=emit("fali", theme=PWH),
    name="person_slips_on_wet",
)


# ---------- derivation: flammable as a derived property -----------------
#
# Demo: lexicon currently tags flammability directly on ligno/libro/etc.;
# Phase 5 audits which of those tags should become derivations. This
# rule shows the target encoding so the substrate has at least one
# meaningful derivation to exercise.

flammability_from_material = derive(
    when=(entity(made_of="wood")
          | entity(made_of="paper")
          | entity(made_of="fabric")
          | entity(made_of="wicker"))
         & bind(T_w := var("T")),
    implies=property(T_w, "flammability", "flammable"),
    name="flammability_from_material",
)


# Convenience bundle: every derivation the library ships with.
# Callers assemble explicitly (as they do for rules) — no hidden
# auto-registration. Pass to `run_dsl(..., derivations=...)`.
DEFAULT_DSL_DERIVATIONS = [flammability_from_material]


# ---------- factory: use_instrument (Phase 4) ---------------------------
#
# The old engine had a single dynamic `make_use_instrument(lex)` rule
# that resolved each `uzi` event's instrument signature at match time.
# The DSL reifies this: one rule per instrument-capable verb (tranĉi,
# ŝlosi, purigi, najli, ...), generated from the lexicon. Each rule's
# `when` pins the specific functional_signature it cares about, so
# exactly one rule matches any given uzi event — same firing pattern,
# but every rule is a first-class, introspectable value.

def make_use_instrument_rules(lex: Lexicon) -> list[Rule]:
    """Emit one causal rule per `derives_instrument` verb in the
    lexicon. The synthesized event carries the verb's declared effects
    as property_changes, matching the old engine's behavior.

    Callers: `rules = DEFAULT_DSL_RULES + make_use_instrument_rules(lex)`.
    """
    out: list[Rule] = []
    for verb in lex.actions.values():
        if not verb.derives_instrument:
            continue
        theme_role = next(
            (r for r in verb.roles if r.name == "theme"), None)
        if theme_role is None:
            continue
        has_instrument_role = any(
            r.name == "instrument" for r in verb.roles)

        # Fresh variables per rule; each rule's scope is independent.
        A = var("A")
        INST = var("INST")
        TH = var("TH")

        # Theme-role entity constraints: type + any declared property
        # requirements (RoleSpec.properties is dict[str, list[str]];
        # the DSL constraint is scalar, so take the first value).
        theme_constraints: dict[str, object] = {"type": theme_role.type}
        for slot, values in (theme_role.properties or {}).items():
            if values:
                theme_constraints[slot] = values[0]

        emit_roles: dict[str, object] = {"agent": A, "theme": TH}
        if has_instrument_role:
            emit_roles["instrument"] = INST
        emission = emit(verb.lemma, **emit_roles)
        for eff in verb.effects:
            target_var = {"agent": A, "theme": TH, "instrument": INST}.get(
                eff.target_role)
            if target_var is None:
                continue
            emission = emission.changing(
                target_var, eff.property, eff.value)

        out.append(rule(
            when=event("uzi",
                       agent=bind(A),
                       instrument=entity(functional_signature=verb.lemma)
                                  & bind(INST),
                       theme=entity(**theme_constraints) & bind(TH)),
            then=emission,
            name=f"use_for_{verb.lemma}",
        ))
    return out


# Convenience bundle: all standalone rules, ordered to match the old
# engine's DEFAULT_RULES followed by its factory-produced rules. Same
# order means same firing sequence under the fixed-point loop, which
# is what the Phase-4 parity tests assert.
DEFAULT_DSL_RULES: list[Rule] = [
    fragile_falls_breaks,
    hungry_eats_sated,
    container_falls_contents_fall,
    broken_container_releases_contents,
    person_slips_on_wet,
    carried_thing_falls_when_carrier_falls,
    fire_spreads_to_adjacent_flammables,
    # Previously factory-produced; now plain values after Phase 2.
    broken_fragile_creates_shards,
    wet_liquid_container_tips,
]
