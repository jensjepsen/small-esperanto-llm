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
    add_relation, bind, caused_by, closure, create_entity, derive,
    destroy_entity, emit, entity, event, has_concept_field, past_event,
    property, rel, remove_relation, rule, var,
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


# ---------- causal: manĝi_destroys_theme --------------------------------
#
# Eating destroys the food. The intrinsic effect on manĝi already
# records presence=consumed in property_changes; this rule closes the
# loop at the lifecycle level by marking the theme as destroyed from
# that event onward. `entities_at(t)` stops returning the eaten thing.

manĝi_destroys_theme = rule(
    when=event("manĝi", theme=bind(TE := var("T"))),
    then=destroy_entity(TE),
    name="manĝi_destroys_theme",
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


# ---------- causal: preni_transfers_ownership ---------------------------
#
# `preni` (take) moves an item from whoever currently `havi`s it to the
# taker. Expressed as a pair of relation effects on the trace: remove
# the old ownership, assert the new one. No-op when the taker already
# holds the item (the remove + add cancel). When nothing else owns the
# item (e.g. picking it up from a shelf) the `rel("havi", ...)` match
# in given fails and the rule doesn't fire — a sibling rule could
# handle environment-to-agent transfer later if needed.

preni_transfers_ownership = rule(
    when=event("preni",
               agent=bind(TA := var("A")),
               theme=bind(TT := var("T"))),
    given=[
        rel("havi", owner=bind(TM := var("M")), theme=TT),
    ],
    then=[
        remove_relation("havi", TM, TT),
        add_relation("havi", TA, TT),
    ],
    name="preni_transfers_ownership",
)


# ---------- causal: ĵeti_releases_possession ----------------------------
#
# Throwing an object relinquishes possession — the thrower no longer
# `havi`s the thrown thing. If nobody catches it, that's where the
# state settles (the ball is unpossessed). Pairs with
# `kapti_takes_possession` below, which runs when someone grabs it
# out of the air.

ĵeti_releases_possession = rule(
    when=event("ĵeti",
               agent=bind(JA := var("A")),
               theme=bind(JT := var("T"))),
    given=[
        rel("havi", owner=JA, theme=JT),
    ],
    then=remove_relation("havi", JA, JT),
    name="ĵeti_releases_possession",
)


# ---------- causal: kapti_takes_possession ------------------------------
#
# Catching is acquisition. Same shape as preni_transfers_ownership,
# but no prior owner is required — the thing may be mid-flight (no
# `havi` for it after a throw). Structurally split into two rules so
# each case is independent: transfer from prior owner, or grant to
# catcher when no one holds it.

kapti_takes_possession_from_nobody = rule(
    when=event("kapti",
               agent=bind(KA := var("A")),
               theme=bind(KT := var("T"))),
    given=[
        ~rel("havi", owner=bind(var("_any")), theme=KT),
    ],
    then=add_relation("havi", KA, KT),
    name="kapti_takes_possession_from_nobody",
)

kapti_takes_possession_from_owner = rule(
    when=event("kapti",
               agent=bind(KA2 := var("A")),
               theme=bind(KT2 := var("T"))),
    given=[
        rel("havi", owner=bind(KM := var("M")), theme=KT2),
    ],
    then=[
        remove_relation("havi", KM, KT2),
        add_relation("havi", KA2, KT2),
    ],
    name="kapti_takes_possession_from_owner",
)


# ---------- causal: doni_transfers_ownership ----------------------------
#
# `doni` (give) is preni's mirror: the agent relinquishes `havi` and
# the recipient receives. Same relation-swap shape. Requires the
# agent to currently own the theme (you can't give what you don't
# have).

doni_transfers_ownership = rule(
    when=event("doni",
               agent=bind(DA := var("A")),
               theme=bind(DT := var("T")),
               recipient=bind(DR := var("R"))),
    given=[
        rel("havi", owner=DA, theme=DT),
    ],
    then=[
        remove_relation("havi", DA, DT),
        add_relation("havi", DR, DT),
    ],
    name="doni_transfers_ownership",
)


# ---------- causal: meti_places_theme -----------------------------------
#
# `meti` (put) places the theme into/onto a location. The preposition
# depends on the location's type: things go `en` rooms and `sur`
# artifacts (tables, shelves). Two rules with mutually-exclusive type
# guards keep the logic declarative — either the location is a
# `type="location"` match or it isn't.

meti_places_in_location = rule(
    when=event("meti",
               agent=bind(MPA := var("A")),
               theme=bind(MPT := var("T")),
               location=entity(type="location") & bind(MPL := var("L"))),
    then=add_relation("en", MPT, MPL),
    name="meti_places_in_location",
)

meti_places_on_surface = rule(
    when=event("meti",
               agent=bind(MSA := var("A")),
               theme=bind(MST := var("T")),
               location=~entity(type="location") & bind(MSL := var("L"))),
    then=add_relation("sur", MST, MSL),
    name="meti_places_on_surface",
)


# ---------- causal: iri_moves_agent -------------------------------------
#
# `iri` (go) moves the agent from wherever they currently `en`-reside
# to the named destination. Mirrors `preni`'s structure — swap one
# relation for another. Requires a current location; if the agent
# isn't `en` anywhere, the rule doesn't fire (a sibling rule could
# handle that case if needed).

iri_moves_agent = rule(
    when=event("iri",
               agent=bind(IA := var("A")),
               destination=bind(ID := var("D"))),
    given=[
        rel("en", contained=IA, container=bind(IO := var("O"))),
    ],
    then=[
        remove_relation("en", IA, IO),
        add_relation("en", IA, ID),
    ],
    name="iri_moves_agent",
)


# ---------- causal: veturi_moves_agent ----------------------------------
#
# `veturi` is "travel by vehicle" — same location-transfer shape as iri
# but with an instrument role for the vehicle. The rule just moves the
# agent; the vehicle is narrative (the realizer renders "per aŭto").
# If a scene needs the vehicle's location to track the agent, a
# separate rule can move it alongside — scope for later.

veturi_moves_agent = rule(
    when=event("veturi",
               agent=bind(VA := var("A")),
               destination=bind(VD := var("D"))),
    given=[
        rel("en", contained=VA, container=bind(VO := var("O"))),
    ],
    then=[
        remove_relation("en", VA, VO),
        add_relation("en", VA, VD),
    ],
    name="veturi_moves_agent",
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
        closure({"en", "sur", "apud"}, from_=B,
                to_=(~entity(type="location")
                     & entity(flammability="flammable")
                     & bind(N := var("N"))),
                max_steps=1),
        ~past_event("bruli", theme=N),
    ],
    then=emit("bruli", theme=N).changing(N, "presence", "consumed"),
    name="fire_spreads_to_adjacent_flammables",
)


# ---------- causal: rain rules ------------------------------------------
#
# `pluvi` is a weather event keyed by location. It wets everything in
# the rained-on location (property change) AND spawns a flako puddle
# so the existing slip machinery has something to trigger on. Two
# rules on the same `when` — one per cascade branch — keep each
# effect self-contained.

rain_wets_contents = rule(
    when=event("pluvi", location=bind(RL := var("L"))),
    given=[
        rel("en", contained=bind(RX := var("X")), container=RL),
        # Anyone carrying an umbrella stays dry. `has_suffix` matches
        # the concept's lemma so this is a single concept-level guard
        # rather than a whole new slot.
        ~rel("havi", owner=RX, theme=entity(has_suffix="ombrelo")),
    ],
    then=emit("_wet", theme=RX).changing(RX, "wetness", "wet"),
    name="rain_wets_contents",
)

rain_creates_puddle = rule(
    when=event("pluvi", location=bind(RPL := var("L"))),
    given=[
        # Outdoor places only — indoor rain would need a roof leak.
        entity(indoor_outdoor="outdoor") & bind(RPL),
    ],
    then=[
        create_entity(concept="flako", as_var=(RPF := var("F")), from_=RPL),
        emit("aperi", theme=RPF),
        add_relation("en", RPF, RPL),
    ],
    name="rain_creates_puddle",
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


# ---------- causal: person_slips_on_rain --------------------------------
#
# Sibling of person_slips_on_wet for the rain path. The puddle was
# created by pluvi (not fali/rompiĝi), so the cause's "theme" walk
# doesn't apply — pluvi carries its location directly. Persons en
# that location slip on the puddle.

person_slips_on_rain = rule(
    when=event("aperi",
               theme=entity(hazard="slippery") & bind(HR := var("H"))),
    given=[
        caused_by("pluvi", location=bind(LR := var("L"))),
        # Only wet persons slip — the umbrella-protected ones stay dry
        # per rain_wets_contents, so this guard filters them out
        # automatically.
        rel("en",
            contained=(entity(type="person", wetness="wet")
                       & bind(PR := var("P"))),
            container=LR),
        ~past_event("fali", theme=PR),
    ],
    then=emit("fali", theme=PR),
    name="person_slips_on_rain",
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


# ---------- derivation chain: animals → meat → edible -------------------
#
# "All animals are made of meat." "Things made of meat are edible."
# Two derivations that chain within a single cycle of the engine's
# derivation phase (fixed point). An entity of type=animal thus gains
# edibility=edible transitively — the manĝi role constraint
# `theme: edibility=edible` then matches animal themes naturally,
# via the effective_property read that sees the derived layer.

animal_is_made_of_meat = derive(
    when=entity(type="animal") & bind(T_a := var("T")),
    implies=property(T_a, "made_of", "meat"),
    name="animal_is_made_of_meat",
)

meat_is_edible = derive(
    when=entity(made_of="meat") & bind(T_m := var("T")),
    implies=property(T_m, "edibility", "edible"),
    name="meat_is_edible",
)


# Convenience bundle: every derivation the library ships with.
# Callers assemble explicitly (as they do for rules) — no hidden
# auto-registration. Pass to `run_dsl(..., derivations=...)`.
DEFAULT_DSL_DERIVATIONS = [
    flammability_from_material,
    animal_is_made_of_meat,
    meat_is_edible,
]


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
    manĝi_destroys_theme,
    container_falls_contents_fall,
    broken_container_releases_contents,
    person_slips_on_wet,
    person_slips_on_rain,
    rain_wets_contents,
    rain_creates_puddle,
    carried_thing_falls_when_carrier_falls,
    fire_spreads_to_adjacent_flammables,
    preni_transfers_ownership,
    doni_transfers_ownership,
    meti_places_in_location,
    meti_places_on_surface,
    iri_moves_agent,
    veturi_moves_agent,
    ĵeti_releases_possession,
    kapti_takes_possession_from_nobody,
    kapti_takes_possession_from_owner,
    # Previously factory-produced; now plain values after Phase 2.
    broken_fragile_creates_shards,
    wet_liquid_container_tips,
]
