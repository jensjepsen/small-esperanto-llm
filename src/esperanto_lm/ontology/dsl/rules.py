"""DSL ports of the causal-rule library.

Phases 1–4 ported here. Each rule is a module-level value the engine
introspects via `collect_rules`. Instrument-using verbs (tranĉi,
ŝlosi, purigi, najli, ...) are first-class actions with their own
`instrument` role and `effects` — they don't need a meta-verb
dispatch layer.
"""
from __future__ import annotations

from ..loader import Lexicon
from .engine import Rule
from . import (
    add_relation, bind, bind_list, category, caused_by, closure, Compare,
    consume_one, create_entity, derive, destroy_entity, emit, entity, event,
    concept_models_slot_check, for_each, has_concept_field, not_relation,
    part, past_event, property,
    rel, relation, remove_relation, rule, transfer_n, var, var_list, VarProp,
)


# ---------- causal: fragile_falls_breaks (Phase 1) ----------------------

fragile_falls_breaks = rule(
    when=event("fali",
               theme=entity(fragility="fragila", integrity="tuta")
                     & bind(T := var("T"))),
    then=emit("rompiĝi", theme=T).changing(T, "integrity", "rompita"),
    name="fragile_falls_breaks",
)


# ---------- causal: bati_breaks_fragile ---------------------------------
#
# Hitting a fragile, intact thing breaks it. Same shape as
# fragile_falls_breaks — different trigger event, identical effect —
# so the downstream cascade (broken_fragile_creates_shards →
# aperi(vitropecetoj) → person_slips_on_wet etc.) reuses transparently.

bati_breaks_fragile = rule(
    when=event("bati",
               theme=entity(fragility="fragila", integrity="tuta")
                     & bind(BT := var("T"))),
    then=emit("rompiĝi", theme=BT).changing(BT, "integrity", "rompita"),
    name="bati_breaks_fragile",
)


# ---------- causal: hungry_eats_sated (Phase 2) -------------------------
#
# Property-conditioned rule: the agent must currently be hungry. The
# emitted satiĝi event carries the hunger transition on itself
# (`.changing(...)`) so the trace shape matches the old engine —
# one event with a property_change, not two separate events.

hungry_eats_sated = rule(
    when=event("manĝi",
               agent=entity(hunger="malsata") & bind(A := var("A"))),
    then=emit("satiĝi", theme=A).changing(A, "hunger", "sata"),
    name="hungry_eats_sated",
)


# Mirror of hungry_eats_sated for the thirst axis. Emits
# `sensoifiĝi` (a coined transparent verb, sen- + soif- + -iĝ-i =
# "become un-thirsty") rather than satiĝi so the regression
# sampler's prop_pool — which keys off `action.effects[0].property`
# — picks it up as a thirst-direct entry. Prose surfaces as "Maria
# trinkis la akvon kaj sensoifiĝis."
thirsty_drinks_quenched = rule(
    when=event("trinki",
               agent=entity(thirst="soifa") & bind(TQA := var("A"))),
    then=emit("sensoifiĝi", theme=TQA).changing(TQA, "thirst", "satigita"),
    name="thirsty_drinks_quenched",
)


# ---------- causal: manĝi_destroys_theme --------------------------------
#
# Eating destroys the food. The intrinsic effect on manĝi already
# records presence=consumed in property_changes; this rule closes the
# loop at the lifecycle level by marking the theme as destroyed from
# that event onward. `entities_at(t)` stops returning the eaten thing.

manĝi_consumes_theme = rule(
    when=event("manĝi", theme=bind(TE := var("T"))),
    then=consume_one(TE),
    name="manĝi_consumes_theme",
)


# Drinking consumes the liquid the same way: countable liquids
# (a stack of bottles, a bowl with multiple cups) decrement; single
# liquids vanish. Closes the pre-existing gap where trinki left the
# liquid intact in the trace.
trinki_consumes_theme = rule(
    when=event("trinki", theme=bind(TT := var("T"))),
    then=consume_one(TT),
    name="trinki_consumes_theme",
)


# ---------- causal: morti_destroys_self ---------------------------------
#
# Dying removes the entity from the trace — `entities_at(t)` stops
# returning it after the morti event. Mirrors `manĝi_destroys_theme`
# (eaten things vanish) for the lifecycle layer. Bodies-after-death
# would need a transforms_on_death slot like fragile→shards; we don't
# model that yet — the entity is just gone.

morti_destroys_self = rule(
    when=event("morti", theme=bind(MtT := var("T"))),
    then=destroy_entity(MtT),
    name="morti_destroys_self",
)


# ---------- causal: mortigi_causes_morti --------------------------------
#
# Killing is causative dying — `mortigi` doesn't destroy the theme
# itself; it emits a separate `morti` event whose own rule does the
# destruction. This keeps causation explicit (the trace shows BOTH
# events, with morti.caused_by pointing at mortigi) and lets the
# realizer narrate "Petro mortigis la muson, kiu mortis."

mortigi_causes_morti = rule(
    when=event("mortigi", theme=bind(MgT := var("T"))),
    then=emit("morti", theme=MgT),
    name="mortigi_causes_morti",
)


# ---------- causal: detrui_destroys_theme -------------------------------
#
# Explicit destruction. Same shape as manĝi_destroys_theme but invoked
# via the `detrui` verb rather than as a side effect of eating.

detrui_destroys_theme = rule(
    when=event("detrui", theme=bind(DtT := var("T"))),
    then=destroy_entity(DtT),
    name="detrui_destroys_theme",
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
        # The spilled liquid is now contained by the puddle (the
        # depression that formed). Preserves narrative distinction
        # (biero-puddle vs akvo-puddle) and satisfies the
        # liquid-must-be-in-liquid_holder requirement: flako has
        # liquid_holder=yes, so the contained liquid is properly
        # held even after spilling.
        add_relation("en", T_spill, S_spill),
    ],
    name="wet_liquid_container_tips",
)


# ---------- causal: verŝi (pour) — agentive liquid transfer/spill -----
#
# Two outcomes split by destination type:
#   - destination is a location (kuirejo, ĝardeno) → spill cascade.
#     Emit `fali` on the liquid; existing
#     `wet_liquid_container_tips` creates a flako and
#     `person_slips_on_wet` propagates to nearby people.
#   - destination is a non-location (glaso, botelo, vazo) → clean
#     transfer. Move the liquid via `en` from its current container
#     to the destination, no spill.
# Agent loses `havi` on the poured liquid in both cases (parallel to
# `ĵeti_releases_possession`).

# ---------- causal: plenigi / malplenigi — container fill/empty --------
#
# `plenigi` fills a malplena container with a liquid the agent holds:
# moves the liquid `en` the container and releases the agent's `havi`
# of it. The fullness=plena state transition is on the action's
# `effects` (so plan_to_achieve can target glaso.fullness=plena
# directly). `malplenigi` empties a plena container by releasing its
# liquid contents — emit `fali` so the existing wet_liquid_container_tips
# cascade creates a flako on the floor.

plenigi_transfers_contents = rule(
    when=event("plenigi",
               agent=bind(PLA := var("A")),
               theme=bind(PLT := var("T")),
               instrument=bind(PLI := var("I"))),
    then=[
        remove_relation("havi", PLA, PLI),
        add_relation("en", PLI, PLT),
    ],
    name="plenigi_transfers_contents",
)

malplenigi_releases_contents = rule(
    when=event("malplenigi",
               theme=bind(MPT := var("T"))),
    given=[
        rel("en", contained=bind(MPL := var("L")), container=MPT),
        entity(state_of_matter="likva") & MPL,
    ],
    then=[
        remove_relation("en", MPL, MPT),
        emit("fali", theme=MPL),
    ],
    name="malplenigi_releases_contents",
)


# ---------- causal: surmeti / demeti — clothing on/off ------------------
#
# `surmeti` (put on) adds `vestita(agent, theme)`; `demeti` (take off)
# removes it. The vestita relation is distinct from `portas` (active
# carrying) so the porti_drop_when_carrier_falls cascade doesn't
# accidentally yank a worn hat off when the wearer trips. Worn
# clothing also stays in the agent's possession (we don't remove
# `havi`) — taking off a coat doesn't make it disappear from your
# wardrobe.

surmeti_dresses = rule(
    when=event("surmeti",
               agent=bind(SuA := var("A")),
               theme=bind(SuT := var("T"))),
    then=add_relation("vestita", SuA, SuT),
    name="surmeti_dresses",
)

demeti_undresses = rule(
    when=event("demeti",
               agent=bind(DeA := var("A")),
               theme=bind(DeT := var("T"))),
    given=[rel("vestita", wearer=DeA, garment=DeT)],
    then=remove_relation("vestita", DeA, DeT),
    name="demeti_undresses",
)


verŝi_releases_possession = rule(
    when=event("verŝi",
               agent=bind(VRA := var("A")),
               theme=bind(VRT := var("T"))),
    given=[rel("havi", owner=VRA, theme=VRT)],
    then=remove_relation("havi", VRA, VRT),
    name="verŝi_releases_possession",
)

verŝi_into_location_emits_fali = rule(
    when=event("verŝi",
               theme=bind(VFT := var("T")),
               destination=entity(type="location")
                           & bind(VFD := var("D"))),
    then=emit("fali", theme=VFT),
    name="verŝi_into_location_emits_fali",
)

verŝi_into_container_transfers = rule(
    when=event("verŝi",
               theme=bind(VCT := var("T")),
               # Destination must be a real liquid container — the
               # `fullness` slot is the marker (it applies_to=artifact
               # in slots.jsonl and is set on glaso/botelo/taso/vazo).
               # Without this gate the rule fired with animals,
               # body parts, and lumber as destinations, producing
               # nonsense `en(akvo, lupo)` assertions.
               destination=entity(fullness=...)
                           & bind(VCD := var("D"))),
    given=[rel("en", contained=VCT,
                container=bind(VCC := var("C")))],
    then=[
        remove_relation("en", VCT, VCC),
        add_relation("en", VCT, VCD),
    ],
    name="verŝi_into_container_transfers",
)


# ---------- causal: fari_creates_constructable -------------------------
#
# Construction. The agent has gathered the parts (one entity per
# declared part of the theme concept's `parts` field) and the planner
# has pre-staged the to-be-created theme as a real entity in the trace
# (so we can refer to it). The rule transfers havi to the agent and
# attaches each gathered part as a havas_parton of the new whole,
# detaching the parts from the agent's havi.
#
# Variadic over the theme concept's parts list — driven by the
# event's `parts` role (kind="list" in actions.jsonl), which the
# matcher binds as a VarList. The for_each effect iterates per
# element. Future variadic verbs (decompose, miksi, ŝarĝi) plug into
# the same pattern.

fari_creates_constructable = rule(
    when=event("fari",
               agent=bind(FA := var("A")),
               theme=bind(FT := var("T")),
               parts=bind_list(FPs := var_list("P"))),
    given=[
        rel("en", contained=FA, container=bind(FL := var("L"))),
    ],
    then=[
        add_relation("en", FT, FL),
        for_each(FPs, (FP := var("P_item")),
            add_relation("havas_parton", FT, FP),
            remove_relation("havi", FA, FP)),
    ],
    name="fari_creates_constructable",
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


# ---------- causal: porti_establishes_carrying --------------------------
#
# `porti` (carry) establishes a `portas` relation between agent and
# theme. Distinct from `havi` (ownership): you can `havi` a book left
# at home without `porti`-ing it. A scene that wants the agent to be
# carrying something at the start can either include the porti event
# (sets up portas via this rule) or assert portas directly in setup.

porti_establishes_carrying = rule(
    when=event("porti",
               agent=bind(PoA := var("A")),
               theme=bind(PoT := var("T"))),
    then=add_relation("portas", PoA, PoT),
    name="porti_establishes_carrying",
)


# ---------- causal: porti_drop_when_carrier_falls (Phase 3, updated) ----
#
# When a carrier falls, everything they `portas` also falls AND the
# portas relation is removed (the agent has lost their grip). Fragility
# isn't checked — a dropped book hits the floor too. Breakage is a
# separate consequence: `fragile_falls_breaks` fires on the resulting
# fali if the carried thing happens to be fragile.
#
# Replaces the older havi-based rule. Ownership doesn't drop on a fall
# (your books at home stay yours when you trip outside) — only active
# carrying does, which is what `portas` models.

porti_drop_when_carrier_falls = rule(
    when=event("fali", theme=bind(P := var("P"))),
    given=[
        rel("portas", carrier=P, theme=bind(F := var("F"))),
        ~past_event("fali", theme=F),
    ],
    then=[
        emit("fali", theme=F),
        remove_relation("portas", P, F),
    ],
    name="porti_drop_when_carrier_falls",
)


# ---------- causal: preni_transfers_ownership ---------------------------
#
# `preni` (take) handles unowned and owned themes uniformly via
# `transfer_n`: the engine looks up any current `havi` and either
# - swaps the owner (full transfer), or
# - splits the stack when the event's quantity < source.count.
# Quantity defaults to 1 on the event; multi-unit themes still transfer
# wholesale unless the planner sets an explicit quantity.

preni_transfers_ownership = rule(
    when=event("preni",
               agent=bind(TA := var("A")),
               theme=bind(TT := var("T"))),
    then=transfer_n(source=TT, target=TA),
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
# Catching is acquisition. Same shape as preni — `transfer_n` handles
# both the from-nobody (mid-flight) and from-prior-owner cases.

kapti_takes_possession = rule(
    when=event("kapti",
               agent=bind(KA := var("A")),
               theme=bind(KT := var("T"))),
    then=transfer_n(source=KT, target=KA),
    name="kapti_takes_possession",
)


# ---------- causal: doni_transfers_ownership ----------------------------
#
# `doni` (give) is preni's mirror: the agent relinquishes `havi` and
# the recipient receives. Action preconditions enforce the agent owns
# the theme; transfer_n handles single-unit swap and partial-stack
# split via the event's quantity.

doni_transfers_ownership = rule(
    when=event("doni",
               agent=bind(DA := var("A")),
               theme=bind(DT := var("T")),
               recipient=bind(DR := var("R"))),
    then=transfer_n(source=DT, target=DR),
    name="doni_transfers_ownership",
)


# ---------- causal: peti_transfers_ownership ----------------------------
#
# `peti` (request) is doni's cooperative inverse from the asker's
# perspective: agent asks recipient for theme. The transfer happens
# atomically — modeled as one event rather than the implied two
# (peti + doni response), the way demandi atomizes ask-and-answer.
# The action's preconditions enforce samloke(agent, recipient) and
# havi(recipient, theme); transfer_n does the swap (or partial split
# when the event carries quantity < the recipient's stack count).

peti_transfers_ownership = rule(
    when=event("peti",
               agent=bind(PEA := var("A")),
               theme=bind(PET := var("T")),
               recipient=bind(PER := var("R"))),
    then=transfer_n(source=PET, target=PEA),
    name="peti_transfers_ownership",
)


# ---------- causal: aĉeti / vendi — bidirectional transfer --------------
#
# `aĉeti` (buy) and `vendi` (sell) are exchanges: goods flow one way,
# money flows the other. Modeled as two `transfer_n` effects per rule;
# both read `event.quantity` so the buyer pays N money for N goods
# (1:1 economy — keeps the model simple while exercising partial-stack
# transfer twice per event).
#
#   aĉeti(agent=buyer, theme=goods, recipient=seller, instrument=money)
#     → buyer gets goods (theme), seller gets money (instrument).
#   vendi(agent=seller, theme=goods, recipient=buyer, instrument=money)
#     → buyer gets goods, seller gets money.
#
# Action preconditions enforce the right starting ownership in both
# directions plus samloke(agent, recipient).

aĉeti_transfers = rule(
    when=event("aĉeti",
               agent=bind(ABA := var("A")),
               theme=bind(ABT := var("T")),
               recipient=bind(ABR := var("R")),
               instrument=bind(ABI := var("I"))),
    then=[
        transfer_n(source=ABT, target=ABA),
        transfer_n(source=ABI, target=ABR),
    ],
    name="aĉeti_transfers",
)

vendi_transfers = rule(
    when=event("vendi",
               agent=bind(VSA := var("A")),
               theme=bind(VST := var("T")),
               recipient=bind(VSR := var("R")),
               instrument=bind(VSI := var("I"))),
    then=[
        transfer_n(source=VST, target=VSR),
        transfer_n(source=VSI, target=VSA),
    ],
    name="vendi_transfers",
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

# `planti` (plant) puts a held theme into the plantable location
# (tero in a garden, tero in a florpoto). Removes possession and
# adds en-containment — same shape as plenigi_transfers_contents.
# Action schema enforces havi(agent, theme) + samloke(agent, location)
# and constrains theme to category=planto, location to plantable=yes.

planti_plants_theme = rule(
    when=event("planti",
               agent=bind(PtA := var("A")),
               theme=bind(PtT := var("T")),
               location=bind(PtL := var("L"))),
    then=[
        remove_relation("havi", PtA, PtT),
        add_relation("en", PtT, PtL),
    ],
    name="planti_plants_theme",
)


# ---------- causal: ami / timi as relational state ----------------------
#
# `ami` (love) and `timi` (fear) describe a persistent emotional bond
# between an animate and a target. Modelled as relations rather than
# slot values: the truth lives in the link, not in either endpoint.
# The verb fires once and the relation persists, available to future
# planning (love-drives-co-location, fear-drives-avoidance) without
# the verb having to re-fire.

ami_creates_amas = rule(
    when=event("ami",
               agent=bind(AmA := var("A")),
               theme=bind(AmT := var("T"))),
    then=add_relation("amas", AmA, AmT),
    name="ami_creates_amas",
)

timi_creates_timas = rule(
    when=event("timi",
               agent=bind(TiA := var("A")),
               theme=bind(TiT := var("T"))),
    then=add_relation("timas", TiA, TiT),
    name="timi_creates_timas",
)


meti_places_on_surface = rule(
    when=event("meti",
               agent=bind(MSA := var("A")),
               theme=bind(MST := var("T")),
               # Destination must be a real surface (tablo/breto/sofo,
               # category=surfaco). Without this gate, meti(theme,
               # body_part) would emit nonsense `sur(forko, buŝo)`.
               # Containers (category=ujo) get the en-rule fallback
               # below; we don't double-fire here.
               location=entity(category="surfaco")
                        & bind(MSL := var("L"))),
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
        # Negation guard: don't re-fire after a downstream eniri puts
        # the agent en the destination. Without this, iri's given
        # would re-match with O=D, removing en and re-adding apud,
        # undoing the eniri's transition. Same pattern needed on
        # every travel verb whose `given` lookup is on a relation
        # that downstream events can mutate.
        ~rel("en", contained=IA, container=ID),
    ],
    then=[
        remove_relation("en", IA, IO),
        add_relation("apud", IA, ID),
    ],
    name="iri_moves_agent",
)


# ---------- causal: veni_moves_agent ------------------------------------
#
# `veni` (come) is the arrival twin of `iri` — same relation transfer,
# different framing. Useful for scenes told from the destination's
# perspective ("Petro venis al la lernejo") rather than the origin's.

veni_moves_agent = rule(
    when=event("veni",
               agent=bind(VnA := var("A")),
               destination=bind(VnD := var("D"))),
    given=[
        rel("en", contained=VnA, container=bind(VnO := var("O"))),
        ~rel("en", contained=VnA, container=VnD),
    ],
    then=[
        remove_relation("en", VnA, VnO),
        add_relation("apud", VnA, VnD),
    ],
    name="veni_moves_agent",
)


# ---------- causal: locomotion variants of iri -------------------------
#
# kuri/naĝi/flugi share iri's en-transfer mechanic but gate on a
# different locomotion value, so the planner picks a creature-fit
# verb when achieving an `en` goal: birds flugi (wings → locomotion=fly),
# fish/persons naĝi (fins or default → locomotion=swim), walkers kuri
# or iri (paws → locomotion=walk). The `_filter_candidates_by_slots`
# pass surfaces only the verbs the actor satisfies, so a serpento
# (locomotion=slither) finds none of these and would need its own
# rule — left for when slither-typed locations exist to receive it.

kuri_moves_agent = rule(
    when=event("kuri",
               agent=bind(KrA := var("A")),
               destination=bind(KrD := var("D"))),
    given=[
        rel("en", contained=KrA, container=bind(KrO := var("O"))),
        ~rel("en", contained=KrA, container=KrD),
    ],
    then=[
        remove_relation("en", KrA, KrO),
        add_relation("apud", KrA, KrD),
    ],
    name="kuri_moves_agent",
)


naĝi_moves_agent = rule(
    when=event("naĝi",
               agent=bind(NgA := var("A")),
               destination=bind(NgD := var("D"))),
    given=[
        rel("en", contained=NgA, container=bind(NgO := var("O"))),
        ~rel("en", contained=NgA, container=NgD),
    ],
    then=[
        remove_relation("en", NgA, NgO),
        add_relation("en", NgA, NgD),
    ],
    name="naĝi_moves_agent",
)


# `eniri` (enter) transitions an apud agent into en. Companion to
# the travel verbs once they switch from end-state=en to end-state
# =apud: iri puts you at the door, eniri takes you inside. The
# action's MatchPrecondition (agent.terrain ⊆ destination.terrain)
# gates which agents can enter which destinations; this rule just
# performs the relation swap on a fired event.

eniri_enters_location = rule(
    when=event("eniri",
               agent=bind(EnA := var("A")),
               theme=bind(EnT := var("T"))),
    given=[
        rel("apud", subject=EnA, neighbor=EnT),
    ],
    then=[
        remove_relation("apud", EnA, EnT),
        add_relation("en", EnA, EnT),
    ],
    name="eniri_enters_location",
)


# Mounting: agent surgrimpas an apud vehicle and ends up sur it.
# Mirror of eniri's apud→en swap. Used together with rajdi (the
# sur-companion to veturi for horses, bikes, scooters, rafts).
surgrimpi_mounts_vehicle = rule(
    when=event("surgrimpi",
               agent=bind(SgA := var("A")),
               theme=bind(SgT := var("T"))),
    given=[
        rel("apud", subject=SgA, neighbor=SgT),
    ],
    then=[
        remove_relation("apud", SgA, SgT),
        add_relation("sur", SgA, SgT),
    ],
    name="surgrimpi_mounts_vehicle",
)


flugi_moves_agent = rule(
    when=event("flugi",
               agent=bind(FgA := var("A")),
               destination=bind(FgD := var("D"))),
    given=[
        rel("en", contained=FgA, container=bind(FgO := var("O"))),
        ~rel("en", contained=FgA, container=FgD),
    ],
    then=[
        remove_relation("en", FgA, FgO),
        add_relation("apud", FgA, FgD),
    ],
    name="flugi_moves_agent",
)


# ---------- causal: sekvi_brings_agent_to_theme ------------------------
#
# Following relocates the follower to wherever the followed currently
# resides. Same en-transfer shape as iri/veni — the difference is that
# the destination isn't a named role; it's read off the theme's `en`.
# The agent's locomotion/posture/sleep gates live on the action schema,
# so the planner subgoals wakeup/stand the same way it does for iri.

sekvi_brings_agent_to_theme = rule(
    when=event("sekvi",
               agent=bind(SqA := var("A")),
               theme=bind(SqT := var("T"))),
    given=[
        rel("en", contained=SqT, container=bind(SqL := var("L"))),
        rel("en", contained=SqA, container=bind(SqO := var("O"))),
        ~rel("en", contained=SqA, container=SqL),
    ],
    then=[
        remove_relation("en", SqA, SqO),
        add_relation("apud", SqA, SqL),
        # Same fact in role-bound form so verb_postconditions can
        # surface sekvi as a producer of apud(agent, theme) in the
        # goal_index. Without this, SqL — bound from the given
        # `en(theme, *)` clause — has no action-role mapping and
        # the postcondition is silently dropped from the index.
        # Runtime semantics unchanged: samloke chains already make
        # SqA samloke SqT via the SqA-apud-SqL fact above.
        add_relation("apud", SqA, SqT),
    ],
    name="sekvi_brings_agent_to_theme",
)


# ---------- causal: voki_summons_theme ---------------------------------
#
# Calling relocates the THEME (the called party) to the agent's
# location. Mirror of sekvi — the agency is reversed: caller stays put,
# callee moves. Locomotion/posture/sleep gates are on theme (the one
# that has to move), not agent.

voki_summons_theme = rule(
    when=event("voki",
               agent=bind(VkA := var("A")),
               theme=bind(VkT := var("T"))),
    given=[
        rel("en", contained=VkA, container=bind(VkL := var("L"))),
        rel("en", contained=VkT, container=bind(VkO := var("O"))),
        ~rel("en", contained=VkT, container=VkL),
    ],
    then=[
        remove_relation("en", VkT, VkO),
        add_relation("apud", VkT, VkL),
        # Role-bound twin so the goal_index picks up voki as a
        # producer of apud(theme, agent). Same rationale as sekvi's
        # parallel add — VkL (agent's container, from given) has no
        # action-role mapping; this version uses only role vars.
        add_relation("apud", VkT, VkA),
    ],
    name="voki_summons_theme",
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
        ~rel("en", contained=VA, container=VD),
    ],
    then=[
        remove_relation("en", VA, VO),
        add_relation("apud", VA, VD),
    ],
    name="veturi_moves_agent",
)


# rajdi's relation transfer — same shape as veturi_moves_agent but
# fired on rajdi events, gated on the rider being en their current
# location (so the move is from origin → apud destination). Without
# this, rajdi was a no-op state-wise and so was absent from the
# goal_index's producer set for apud(agent, destination).
rajdi_moves_agent = rule(
    when=event("rajdi",
               agent=bind(RJA := var("A")),
               destination=bind(RJD := var("D"))),
    given=[
        rel("en", contained=RJA, container=bind(RJO := var("O"))),
        ~rel("en", contained=RJA, container=RJD),
    ],
    then=[
        remove_relation("en", RJA, RJO),
        add_relation("apud", RJA, RJD),
    ],
    name="rajdi_moves_agent",
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
                     & entity(flammability="brulebla")
                     & bind(N := var("N"))),
                max_steps=1),
        ~past_event("bruli", theme=N),
    ],
    then=emit("bruli", theme=N).changing(N, "presence", "manĝita"),
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
        # samloke (not direct `en`) so an actor sitting on a sofa or
        # standing on a balcony in the rain still gets wet — the
        # samloke chain rules close en/sur composition for us.
        rel("samloke", a=RL, b=bind(RX := var("X"))),
        # Anyone carrying an umbrella stays dry. `has_suffix` matches
        # the concept's lemma so this is a single concept-level guard
        # rather than a whole new slot.
        ~rel("havi", owner=RX, theme=entity(has_suffix="ombrelo")),
    ],
    then=emit("_wet", theme=RX).changing(RX, "wetness", "malseka"),
    name="rain_wets_contents",
)

rain_creates_puddle = rule(
    when=event("pluvi", location=bind(RPL := var("L"))),
    given=[
        # Outdoor places only — indoor rain would need a roof leak.
        entity(indoor_outdoor="ekstera") & bind(RPL),
    ],
    then=[
        create_entity(concept="flako", as_var=(RPF := var("F")), from_=RPL),
        emit("aperi", theme=RPF),
        add_relation("en", RPF, RPL),
        # Rain produces water inside the puddle. flako is the
        # depression (liquid_holder), akvo the water it holds.
        # Without this, the puddle would be an empty container,
        # and downstream "drink the puddle water" reasoning
        # would have nothing to act on.
        create_entity(concept="akvo", as_var=(RPW := var("W")), from_=RPF),
        add_relation("en", RPW, RPF),
    ],
    name="rain_creates_puddle",
)


# ---------- causal: skribi_creates_text -----------------------------------
#
# Writing produces marks-on-the-surface. Modeled as entity creation
# rather than a property toggle: the produced `skribaĵo` is a real
# thing that sits on the theme via `sur`, can later be referred to,
# and could be destroyed by viŝi. Symmetric with how rain_creates_puddle
# spawns a flako on a location.

skribi_creates_text = rule(
    when=event("skribi", theme=bind(SkT := var("T"))),
    then=[
        create_entity(concept="skribaĵo",
                      as_var=(SkS := var("S")), from_=SkT),
        emit("aperi", theme=SkS),
        add_relation("sur", SkS, SkT),
    ],
    name="skribi_creates_text",
)


# When the writer knows a scias-tuple, the produced skribaĵo records
# its content via 4-arity `priskribas(text, rel_type, subjekto,
# objekto)` — text inherits the same propositional shape that
# scias/rakonti/etc. use. Downstream `legi_extracts_scias` reads the
# tuple back out and adds it to the reader's scias. Split from
# skribi_creates_text so the base case (writing produces a blank
# text) still fires when the agent has no known scias.
skribi_records_scias = rule(
    when=event("skribi",
               agent=bind(SrA := var("A")),
               theme=bind(SrT := var("T"))),
    given=[
        rel("scias", knower=SrA,
            rel_type=bind(SrRT := var("RT")),
            subjekto=bind(SrSj := var("Sj")),
            objekto=bind(SrO := var("O"))),
    ],
    then=[
        create_entity(concept="skribaĵo",
                      as_var=(SrS := var("S")), from_=SrT),
        add_relation("priskribas", SrS, SrRT, SrSj, SrO),
    ],
    name="skribi_records_scias",
)


# ---------- causal: viŝi_destroys_skribaĵo -----------------------------
#
# Wiping is the symmetric inverse of writing: when viŝi targets a
# skribaĵo entity (text-on-surface), the entity is destroyed.
#
# Two constraints narrow this rule beyond "any wipe of any skribaĵo":
#   1. Theme must instantiate the `skribaĵo` concept exactly — wiping
#      a table is harmless narrative.
#   2. The wiping agent must have previously written on the surface
#      the skribaĵo lives on. You can only erase your own writing
#      (or text on a surface you've written on) — strangers don't
#      arrive and erase. Encoded via `past_event("skribi", ...)`.
#
# The surface is recovered via the `sur` relation that
# `skribi_creates_text` placed when synthesizing the skribaĵo.

viŝi_destroys_skribaĵo = rule(
    when=event("viŝi",
               agent=bind(VsA := var("A")),
               theme=entity(concept="skribaĵo") & bind(VsT := var("T"))),
    given=[
        rel("sur", contained=VsT, container=bind(VsP := var("P"))),
        past_event("skribi", agent=VsA, theme=VsP),
    ],
    then=destroy_entity(VsT),
    name="viŝi_destroys_skribaĵo",
)


# ---------- causal: person_slips_on_wet (Phase 3, updated) -------------
#
# Wet surfaces cause slips; sharp shards do not (stepping on broken
# glass would cut you, but that's a different event we don't model).
# Filter on `hazard="glita"` specifically — the aperi event has to
# bring a slippery thing into existence (a flako is the canonical
# example). The hazard's location is inferred from the cause's theme —
# the entity that fell or broke to produce the puddle.

person_slips_on_wet = rule(
    when=event("aperi",
               theme=entity(hazard="glita") & bind(H := var("H"))),
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


# ---------- causal: knowledge transfer ----------------------------------
#
# Knowledge is fact-grained. `scias(knower, rel_type, subjekto, objekto)`
# is a 4-arity relation: a knower knows that (rel_type, subjekto,
# objekto) holds — one specific propositional fact, not a generic
# "knows about". `scias_lokon(knower, located)` is the cheaper
# 2-arity sibling for "knows where X is".
#
# When an agent `vidi`s an entity X, they add one scias-tuple per
# relation X currently participates in — separate rule per relation
# kind / arg-position so the binding stays explicit. Add more rules
# here as new relations grow narrative weight (currently: en, sur,
# havi).
#
# `rakonti` transfers a specific scias-tuple from teller to recipient
# — pure information move; the tuple's components come pre-bound on
# the rakonti event itself.

vidi_learns_en = rule(
    when=event("vidi",
               agent=bind(VEA := var("A")),
               theme=bind(VET := var("T"))),
    given=[
        rel("en", contained=VET, container=bind(VEL := var("L"))),
    ],
    then=[
        add_relation("scias", VEA, "en", VET, VEL),
    ],
    name="vidi_learns_en",
)

flari_learns_en = rule(
    when=event("flari",
               agent=bind(FEA := var("A")),
               theme=bind(FET := var("T"))),
    given=[
        rel("en", contained=FET, container=bind(FEL := var("L"))),
    ],
    then=[
        add_relation("scias", FEA, "en", FET, FEL),
    ],
    name="flari_learns_en",
)

audi_learns_en = rule(
    when=event("aŭdi",
               agent=bind(AEA := var("A")),
               theme=bind(AET := var("T"))),
    given=[
        rel("en", contained=AET, container=bind(AEL := var("L"))),
    ],
    then=[
        add_relation("scias", AEA, "en", AET, AEL),
    ],
    name="audi_learns_en",
)

vidi_learns_sur = rule(
    when=event("vidi",
               agent=bind(VSA := var("A")),
               theme=bind(VST := var("T"))),
    given=[
        rel("sur", contained=VST, container=bind(VSL := var("L"))),
    ],
    then=[
        add_relation("scias", VSA, "sur", VST, VSL),
    ],
    name="vidi_learns_sur",
)


# Smell and sound also locate `sur`-placed targets. Mirrors
# `vidi_learns_sur` so the planner can establish scias_lokon for
# a target placed `sur` something without depending on illumination
# — vidi requires the agent be illuminated, which fails in indoor
# scenes without an active lamp. flari/audi only need samloke +
# the perceptual capacity (smell/hearing). Without these, an actor
# in a dark indoor room can never plan preni for a snack on a table.
flari_learns_sur = rule(
    when=event("flari",
               agent=bind(FSA := var("A")),
               theme=bind(FST := var("T"))),
    given=[
        rel("sur", contained=FST, container=bind(FSL := var("L"))),
    ],
    then=[
        add_relation("scias", FSA, "sur", FST, FSL),
    ],
    name="flari_learns_sur",
)


audi_learns_sur = rule(
    when=event("aŭdi",
               agent=bind(ASA := var("A")),
               theme=bind(AST := var("T"))),
    given=[
        rel("sur", contained=AST, container=bind(ASL := var("L"))),
    ],
    then=[
        add_relation("scias", ASA, "sur", AST, ASL),
    ],
    name="audi_learns_sur",
)

vidi_learns_havi_owner = rule(
    when=event("vidi",
               agent=bind(VHA := var("A")),
               theme=bind(VHT := var("T"))),
    given=[
        rel("havi", owner=bind(VHO := var("O")), theme=VHT),
    ],
    then=[
        add_relation("scias", VHA, "havi", VHO, VHT),
    ],
    name="vidi_learns_havi_owner",
)

# Vocalization broadcasts the speaker's location to nearby hearers.
# Same shape for krii, flustri, bleki, boji, miaŭi — they all model
# "agent makes sound, samloke + can_hear animates learn where the
# agent is". Built via a helper to keep the five definitions
# parallel and prevent drift.
def _vocal_announce_rule(verb_name):
    A = var(f"VC_{verb_name}_A")
    L = var(f"VC_{verb_name}_L")
    H = var(f"VC_{verb_name}_H")
    return rule(
        when=event(verb_name, agent=bind(A)),
        given=[
            rel("en", contained=A, container=bind(L)),
            entity(type="animate", can_hear="yes") & bind(H),
            rel("samloke", a=A, b=H),
        ],
        then=[
            add_relation("scias", H, "en", A, L),
        ],
        name=f"{verb_name}_announces_speaker",
    )


krii_announces_speaker = _vocal_announce_rule("krii")
flustri_announces_speaker = _vocal_announce_rule("flustri")
bleki_announces_speaker = _vocal_announce_rule("bleki")
boji_announces_speaker = _vocal_announce_rule("boji")
miaui_announces_speaker = _vocal_announce_rule("miaŭi")


# Playing an instrument is also a sound-broadcast: the player's
# location becomes known to samloke can_hear animates. Same shape
# as the vocal rules, just with `theme` (the instrument) bound too
# in the trigger event so it doesn't matter for the outgoing
# scias-tuple — what propagates is the player's location.
def _ludi_announce_rule():
    A = var("LU_A")
    T = var("LU_T")
    L = var("LU_L")
    H = var("LU_H")
    return rule(
        when=event("ludi", agent=bind(A), theme=bind(T)),
        given=[
            rel("en", contained=A, container=bind(L)),
            entity(type="animate", can_hear="yes") & bind(H),
            rel("samloke", a=A, b=H),
        ],
        then=[
            add_relation("scias", H, "en", A, L),
        ],
        name="ludi_announces_player",
    )

ludi_announces_player = _ludi_announce_rule()


# forgesi: no automatic trigger and no rule consequences right now —
# a scias-removal effect would need a negative-knowledge drive
# ("X wants to NOT know Y about Z") which we don't model. Action def
# kept (so the lemma stays usable for ambient simulation and future
# memory-management drives) but the DSL rule is gone.


rakonti_transfers_scias = rule(
    when=event("rakonti",
               agent=bind(RKA := var("A")),
               recipient=bind(RKR := var("R")),
               rel_type=bind(RKRT := var("RT")),
               theme=bind(RKT := var("T")),
               objekto=bind(RKO := var("O"))),
    # Transfer the specific scias(agent, rel_type, theme, objekto)
    # asserted by perception (and pre-grounded by
    # _ground_action_from_precondition) over to the recipient. The
    # realizer reads scias on the recipient post-event for the
    # ke-clause prose.
    then=add_relation("scias", RKR, RKRT, RKT, RKO),
    name="rakonti_transfers_scias",
)


# `demandi` is rakonti's inverse: the asker (agent) acquires the
# scias-tuple from someone who already knows it (recipient). Modeled
# as atomic Q&A — firing demandi adds scias(agent, ...), no separate
# "recipient replies" event needed for the planner's purposes.
demandi_extracts_scias = rule(
    when=event("demandi",
               agent=bind(DEA := var("A")),
               recipient=bind(DER := var("R")),
               rel_type=bind(DERT := var("RT")),
               theme=bind(DET := var("T")),
               objekto=bind(DEO := var("O"))),
    then=add_relation("scias", DEA, DERT, DET, DEO),
    name="demandi_extracts_scias",
)


# `respondi` mirrors rakonti structurally — agent transfers a
# scias-tuple to recipient. The conversational distinction (reply
# vs initiating tell) doesn't change the semantics for the engine;
# it gives the realizer a verb to pick when continuing a Q&A turn.
respondi_transfers_scias = rule(
    when=event("respondi",
               agent=bind(RPA := var("A")),
               recipient=bind(RPR := var("R")),
               rel_type=bind(RPRT := var("RT")),
               theme=bind(RPT := var("T")),
               objekto=bind(RPO := var("O"))),
    then=add_relation("scias", RPR, RPRT, RPT, RPO),
    name="respondi_transfers_scias",
)


# `montri` (show) is vidi-flavored from the recipient's side: the
# agent — already samloke with both theme and recipient — surfaces
# the theme's location and the recipient learns it. Same shape as
# vidi_learns_en, but scias_lokon/scias go to the recipient role
# instead of the agent. Lets "show, don't tell" transfer knowledge
# of physical things without requiring the agent to first know
# anything specific.
montri_shows_location = rule(
    when=event("montri",
               agent=bind(MNA := var("A")),
               theme=bind(MNT := var("T")),
               recipient=bind(MNR := var("R"))),
    given=[
        rel("en", contained=MNT, container=bind(MNL := var("L"))),
    ],
    then=[
        add_relation("scias", MNR, "en", MNT, MNL),
    ],
    name="montri_shows_location",
)


# `instrui` (teach) is rakonti's pedagogical sibling — same shape
# (agent transfers a scias-tuple they know to a recipient) and same
# semantics. Distinct lemma gives the realizer a verb choice for
# instructional contexts (instructor → student) vs narrative ones
# (storyteller → listener); the engine treats them identically.
instrui_transfers_scias = rule(
    when=event("instrui",
               agent=bind(ITA := var("A")),
               recipient=bind(ITR := var("R")),
               rel_type=bind(ITRT := var("RT")),
               theme=bind(ITT := var("T")),
               objekto=bind(ITO := var("O"))),
    then=add_relation("scias", ITR, ITRT, ITT, ITO),
    name="instrui_transfers_scias",
)


# `legi` (read) extracts a scias-tuple from a text. Asynchronous
# knowledge transfer: where rakonti requires the source to be
# physically present, legi only needs the reader to be samloke with
# the text. The text-to-content link comes from 4-arity
# `priskribas(text, rel_type, subjekto, objekto)`, set either at
# scene init (regression seed) or by skribi_records_scias when the
# writer's scias gets captured into the text.
legi_extracts_scias = rule(
    when=event("legi",
               agent=bind(LeA := var("A")),
               theme=bind(LeT := var("T"))),
    given=[
        rel("priskribas", text=LeT,
            rel_type=bind(LeRT := var("RT")),
            subjekto=bind(LeSj := var("Sj")),
            objekto=bind(LeO := var("O"))),
    ],
    then=add_relation("scias", LeA, LeRT, LeSj, LeO),
    name="legi_extracts_scias",
)


# `mezuri` (measure) is a perception event for a property the
# instrument reads (termometro→temperature, pesilo→maso, …). Emits
# `scias_propon(agent, theme, slot)` — the general "knows property
# of" predicate. Parallel to scias_lokon (knows location of) and
# scias (knows relation between). The slot value is read from the
# instrument's `mezuras` concept-field, so any future measuring tool
# auto-extends scias_propon coverage by declaring its mezuras.
mezuri_learns_dimension = rule(
    when=event("mezuri",
               agent=bind(MEA := var("A")),
               theme=bind(MET := var("T")),
               instrument=bind(MEI := var("I"))),
    given=[
        # Instrument tells us WHICH slot is measured (termometro→
        # temperature, pesilo→maso, …). Theme must model that slot,
        # otherwise the measurement is vacuous (no value to learn).
        # concept_models_slot_check covers declared props + pervasive
        # + derivation-lifted, same predicate the planner enforces.
        has_concept_field(MEI, "mezuras", MES := var("S")),
        concept_models_slot_check(MET, MES),
    ],
    then=add_relation("scias_propon", MEA, MET, MES),
    name="mezuri_learns_dimension",
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
          | entity(made_of="wicker")
          | entity(made_of="plant"))
         & bind(T_w := var("T")),
    implies=property(T_w, "flammability", "brulebla"),
    name="flammability_from_material",
)


# ---------- derivation: mezuri-instruments default to measuring volumeno
#
# The autoderived `mezurilo` (from `mezuri.derives_instrument=true`)
# carries `functional_signature=mezuri` but no semantics about WHICH
# slot it reads off the theme. This default fills in
# `mezuras=volumeno` for any concept that holds the mezuri signature,
# so the canonical ruler measures size. Concepts that need a
# different reading (termometro→temperature, pesilo→maso) can declare
# their own `mezuras` value — asserted-wins-on-scalar blocks this
# default for them, same as every other type-keyed derivation
# default. The mezuras vocab IS the set of real slot names: scias_propon
# emits `(agent, theme, <slot>)` from this directly, and slot-kind
# validation in `validate_relation` requires the value to be in
# lex.slots.

mezuri_instruments_default_grandeco = derive(
    when=entity(functional_signature="mezuri") & bind(M_inst := var("M")),
    implies=property(M_inst, "mezuras", "volumeno"),
    name="mezuri_instruments_default_volumeno",
)


# ---------- derivation: denseco-from-made_of -----------------------------
#
# Density defaults from material, in g/cm³. Asserted-wins on scalar so
# a concept can override (e.g. a hollow object would declare lower
# denseco directly). Numbers are typical bulk densities; precision
# beyond one decimal isn't meaningful for the symbolic narrative use
# case.

denseco_from_wood = derive(
    when=entity(made_of="wood") & bind(T_dw := var("T")),
    implies=property(T_dw, "denseco", "0.5"),
    name="denseco_from_wood",
)
denseco_from_metal = derive(
    when=entity(made_of="metal") & bind(T_dm := var("T")),
    implies=property(T_dm, "denseco", "7.8"),
    name="denseco_from_metal",
)
denseco_from_paper = derive(
    when=entity(made_of="paper") & bind(T_dp := var("T")),
    implies=property(T_dp, "denseco", "0.8"),
    name="denseco_from_paper",
)
denseco_from_stone = derive(
    when=entity(made_of="stone") & bind(T_ds := var("T")),
    implies=property(T_ds, "denseco", "2.5"),
    name="denseco_from_stone",
)
denseco_from_fabric = derive(
    when=entity(made_of="fabric") & bind(T_df := var("T")),
    implies=property(T_df, "denseco", "0.3"),
    name="denseco_from_fabric",
)
denseco_from_plant = derive(
    when=entity(made_of="plant") & bind(T_dpl := var("T")),
    implies=property(T_dpl, "denseco", "0.5"),
    name="denseco_from_plant",
)
denseco_from_wicker = derive(
    when=entity(made_of="wicker") & bind(T_dwk := var("T")),
    implies=property(T_dwk, "denseco", "0.3"),
    name="denseco_from_wicker",
)
denseco_from_meat = derive(
    when=entity(made_of="meat") & bind(T_dme := var("T")),
    implies=property(T_dme, "denseco", "1.0"),
    name="denseco_from_meat",
)
# Food and drink default to water-equivalent density. Catches concepts
# whose `category=["manĝaĵo"]` / `["trinkaĵo"]` (set by the -aĵ- affix
# autoderivation on manĝi/trinki) inherit this default.
denseco_default_manĝaĵo = derive(
    when=entity(category="manĝaĵo") & bind(T_dm := var("T")),
    implies=property(T_dm, "denseco", "1.0"),
    name="denseco_default_manĝaĵo",
)
denseco_default_trinkaĵo = derive(
    when=entity(category="trinkaĵo") & bind(T_dt := var("T")),
    implies=property(T_dt, "denseco", "1.0"),
    name="denseco_default_trinkaĵo",
)


# ---------- derivation: meat is edible ---------------------------------
#
# Things made of meat are edible. The `meat_is_edible` derivation ties
# `made_of=meat` to `edibility=manĝebla`, but live animals no longer
# auto-derive `made_of=meat` — that earlier blanket derivation made
# every cat, fish, and bee instantly edible to a hungry agent ("Maria
# manĝis la simion"). Now meat exists only on entities that are
# explicitly meat: the `viando` substance concept, plus any future
# processed-animal-product concepts. Killing an animal and obtaining
# meat is a separate (multi-step) chain to be modeled later.

animate_is_solid = derive(
    when=entity(type="animate") & bind(T_anim := var("T")),
    implies=property(T_anim, "state_of_matter", "solida"),
    name="animate_is_solid",
)


# All persons walk and swim by default. Two derivations because each
# `derive()` carries one `PropertyImplication`; the bake mechanism
# appends to multi-valued slots (locomotion is scalar=False), so both
# values land. Authored persons that need to differ (e.g. an injured
# character who can't swim) can override by asserting locomotion
# explicitly — asserted wins for scalar slots; for multi-valued ones
# the bake won't drop existing entries either.

# Persons share the `parts` vocabulary with animals. Two derivations
# give them feet and hands; locomotion=walk then arises through the
# anatomy chain (parts=piedo → has_paws_can_walk → locomotion=walk),
# which the bake's fixed-point loop resolves in a single load.
# Swimming stays as its own derivation — humans swim by skill, not
# anatomy, so there's no body-part to point at.

person_can_swim = derive(
    when=entity(type="person") & bind(T_ps := var("T")),
    implies=property(T_ps, "locomotion", "swim"),
    name="person_can_swim",
)


# Anatomy → ability. Body parts (the `parts` slot, multi-valued) imply
# what locomotion an animal can perform. The bake's append-on-non-scalar
# behavior lets multiple of these fire on the same animal — birdo
# (piedo + flugilo) gets locomotion=[walk, fly], rano (piedo + naĝilo)
# gets [walk, swim]. Animals without any of the relevant parts
# (e.g. serpento) keep whatever locomotion is asserted directly.
#
# This is meronymy as the source of affordances: instead of declaring
# "what kato can do", declare "what kato is made of" and let the
# anatomy imply ability. Adding a new motility (climb? jump?) means
# adding a part (manoj? kruroj?) and a one-line derivation.

has_paws_can_walk = derive(
    when=entity() & bind(T_paw := var("T")),
    given=[
        rel("havas_parton",
            tuto=T_paw,
            parto=bind(P_paw := var("P"))),
        entity(concept="piedo") & bind(P_paw),
    ],
    implies=property(T_paw, "locomotion", "walk"),
    name="has_paws_can_walk",
)

has_wings_can_fly = derive(
    when=entity() & bind(T_wing := var("T")),
    given=[
        rel("havas_parton",
            tuto=T_wing,
            parto=bind(P_wing := var("P"))),
        entity(concept="flugilo") & bind(P_wing),
    ],
    implies=property(T_wing, "locomotion", "fly"),
    name="has_wings_can_fly",
)

has_fins_can_swim = derive(
    when=entity() & bind(T_fin := var("T")),
    given=[
        rel("havas_parton",
            tuto=T_fin,
            parto=bind(P_fin := var("P"))),
        entity(concept="naĝilo") & bind(P_fin),
    ],
    implies=property(T_fin, "locomotion", "swim"),
    name="has_fins_can_swim",
)

# Nose → smell capability. Same shape as has_hands_can_use_tools.
# Persons get nazo via the new body-part wiring; quadruped mammals
# also get nazo. Birds/fish/insects/snakes lack nazo → can't be flari
# agents → smell-based knowledge transfer is restricted to nosed
# animates.
has_nose_can_smell = derive(
    when=entity() & bind(T_smell := var("T")),
    given=[
        rel("havas_parton",
            tuto=T_smell,
            parto=bind(P_smell := var("P"))),
        entity(concept="nazo") & bind(P_smell),
    ],
    implies=property(T_smell, "can_smell", "yes"),
    name="has_nose_can_smell",
)

# Ear → hear capability. Mirrors has_nose_can_smell.
has_ear_can_hear = derive(
    when=entity() & bind(T_hear := var("T")),
    given=[
        rel("havas_parton",
            tuto=T_hear,
            parto=bind(P_hear := var("P"))),
        entity(concept="orelo") & bind(P_hear),
    ],
    implies=property(T_hear, "can_hear", "yes"),
    name="has_ear_can_hear",
)


# A location is a body of water if some likva-state thing is part-of
# or en it. Two derivations to cover both routes: lakes/seas/rivers
# declare `parts: [{"concept": "akvo"}]` (auto-instantiated by the
# sampler — the water is "built in"), while a piscino or barelo with
# water poured in gets the same affordance via `en`. Same shape, same
# implies; the only difference is which relation surfaces the link
# between the location and the water inside it.

location_water_body_via_part = derive(
    when=entity(type="location") & bind(LwbL := var("L")),
    given=[
        rel("havas_parton",
            tuto=LwbL,
            parto=entity(state_of_matter="likva")),
    ],
    implies=property(LwbL, "water_body", "yes"),
    name="location_water_body_via_part",
)


location_water_body_via_en = derive(
    when=entity(type="location") & bind(LweL := var("L")),
    given=[
        rel("en",
            contained=entity(state_of_matter="likva"),
            container=LweL),
    ],
    implies=property(LweL, "water_body", "yes"),
    name="location_water_body_via_en",
)


# Habitat surfaces: a location's terrain is whatever its parts
# afford. Vehicles declare a matching `terrain` value intrinsically
# (aŭto/biciklo=land, trajno=rail, ŝipo=water). When MatchPrecondition
# lands, veturi will gate instrument.terrain ⊆ destination.terrain;
# until then these derivations supply the data side and let the
# vehicle seeder bias placement.

location_terrain_land_via_part = derive(
    when=entity(type="location") & bind(LtlL := var("L")),
    given=[
        rel("havas_parton",
            tuto=LtlL,
            parto=entity(concept="vojo")),
    ],
    implies=property(LtlL, "terrain", "land"),
    name="location_terrain_land_via_part",
)


location_terrain_rail_via_part = derive(
    when=entity(type="location") & bind(LtrL := var("L")),
    given=[
        rel("havas_parton",
            tuto=LtrL,
            parto=entity(concept="relo")),
    ],
    implies=property(LtrL, "terrain", "rail"),
    name="location_terrain_rail_via_part",
)


location_terrain_water_via_part = derive(
    when=entity(type="location") & bind(LtwL := var("L")),
    given=[
        rel("havas_parton",
            tuto=LtwL,
            parto=entity(state_of_matter="likva")),
    ],
    implies=property(LtwL, "terrain", "water"),
    name="location_terrain_water_via_part",
)


# Indoor places carry an `indoor` terrain so eniri's MatchPrecondition
# (agent.terrain ∩ destination.terrain) admits walkers — kuirejo,
# salono, etc. don't need explicit parts to be enterable.
location_terrain_indoor_via_indoor_outdoor = derive(
    when=entity(type="location", indoor_outdoor="interna") & bind(LtiL := var("L")),
    implies=property(LtiL, "terrain", "indoor"),
    name="location_terrain_indoor_via_indoor_outdoor",
)


# Every location has air — outdoor sky, indoor breathable space.
# Lets pure fliers (papilio, abelo with terrain=[air] only) reach
# indoor destinations as well as outdoor ones. Aviadilo intentionally
# does NOT carry air terrain (it's terrain=[land] now) — without an
# airport/runway concept, treating planes as land vehicles is the
# cleanest mechanical match; the "per aviadilo" narrative survives.
location_has_air_terrain = derive(
    when=entity(type="location") & bind(LtaL := var("L")),
    implies=property(LtaL, "terrain", "air"),
    name="location_has_air_terrain",
)


# Animate terrain comes from locomotion: walkers go on land and into
# buildings, swimmers go in water, fliers in air (and can land too),
# slitherers on land. The terrain slot is non-scalar so an animate
# with multiple locomotions accumulates terrain values from each
# derivation. eniri's MatchPrecondition then admits exactly the
# destinations the agent's habitats cover.

animate_terrain_land_from_walk = derive(
    when=entity(type="animate", locomotion="walk") & bind(AtwL := var("A")),
    implies=property(AtwL, "terrain", "land"),
    name="animate_terrain_land_from_walk",
)

animate_terrain_indoor_from_walk = derive(
    when=entity(type="animate", locomotion="walk") & bind(AtwI := var("A")),
    implies=property(AtwI, "terrain", "indoor"),
    name="animate_terrain_indoor_from_walk",
)

animate_terrain_water_from_swim = derive(
    when=entity(type="animate", locomotion="swim") & bind(AtsW := var("A")),
    implies=property(AtsW, "terrain", "water"),
    name="animate_terrain_water_from_swim",
)

animate_terrain_air_from_fly = derive(
    when=entity(type="animate", locomotion="fly") & bind(AtfA := var("A")),
    implies=property(AtfA, "terrain", "air"),
    name="animate_terrain_air_from_fly",
)

animate_terrain_land_from_slither = derive(
    when=entity(type="animate", locomotion="slither") & bind(AtsL := var("A")),
    implies=property(AtsL, "terrain", "land"),
    name="animate_terrain_land_from_slither",
)


# An entity inside a body of water is in_water. Built on top of
# water_body: the part-vs-en distinction is already absorbed there,
# so this derivation just chains "T en L" + "L is a water_body".
# Same shape as agent_illuminated (entity en a luma location →
# illuminated) — single free var L which the planner can subgoal.

entity_in_water_from_water_body = derive(
    when=entity(type="physical") & bind(IwT := var("T")),
    given=[
        rel("en", contained=IwT, container=bind(IwL := var("L"))),
        entity(water_body="yes") & bind(IwL),
    ],
    implies=property(IwT, "in_water", "yes"),
    name="entity_in_water_from_water_body",
)


# A water_body location IS water for floatation purposes: derive the
# physical properties of water onto it so density-based rules
# (floats_when_lighter_than_liquid) can compare against a known
# numeric denseco. Water's denseco is 1.0 g/cm³ (the universal
# baseline; substances with denseco<1.0 float). The state_of_matter
# isn't strictly needed for floating but keeps the semantic
# consistency: a body of water is liquid.
water_body_has_water_properties = derive(
    when=entity(water_body="yes") & bind(WBL := var("L")),
    implies=[
        property(WBL, "denseco", "1.0"),
        property(WBL, "state_of_matter", "likva"),
    ],
    name="water_body_has_water_properties",
)


# Enterability is derived from structural facts, not asserted:
#  - any location is enterable (open or built — you walk into a
#    field, a forest, or a room; eniri admits both)
#  - any concept with a `pordo` part is enterable (rooms, doored
#    vehicles like aŭto/buso/aviadilo)
#  - a vehicle WITHOUT a pordo part is non-enterable (biciklo,
#    ĉevalo, skutilo, rafto): forces use of surgrimpi/rajdi
#
# eniri.theme and veturi.instrument require enterable=yes;
# surgrimpi.theme and rajdi.instrument require enterable=no.
# Together this picks the right verb pair without per-vehicle
# tagging: schema reads the answer off `parts` membership.
location_is_enterable = derive(
    when=entity(type="location") & bind(LIEL := var("L")),
    implies=property(LIEL, "enterable", "yes"),
    name="location_is_enterable",
)

doored_is_enterable = derive(
    when=entity(type="physical") & bind(DIEH := var("H")),
    given=[
        rel("havas_parton", tuto=DIEH,
            parto=bind(DIEP := var("P"))),
        entity(concept="pordo") & bind(DIEP),
    ],
    implies=property(DIEH, "enterable", "yes"),
    name="doored_is_enterable",
)

# NotPattern over a parameterized rel: matches when there's no
# havas_parton(V, *) whose parto is a pordo concept. Two-clause
# composition mirrors doored_is_enterable's positive form so the
# negation evaluator sees the same structure.
doorless_vehicle_is_not_enterable = derive(
    when=entity(is_vehicle="yes") & bind(DVNEV := var("V")),
    given=[
        ~(rel("havas_parton", tuto=DVNEV,
              parto=entity(concept="pordo"))),
    ],
    implies=property(DVNEV, "enterable", "no"),
    name="doorless_vehicle_is_not_enterable",
)


# Person concepts inherit canonical human parts. Authored persons
# (persono, knabo, amiko, ...) declare these directly; this rule
# materializes them onto AFFIX-DERIVED person concepts (kuiristo,
# kantisto, ...) at bake time so they too qualify as tool-using
# agents via has_hands_can_use_tools below. PartImplications are
# bake-only (parts are static structural metadata).

person_has_human_parts = derive(
    when=entity(type="person") & bind(PHHP := var("P")),
    implies=[
        part(PHHP, "piedo"),
        part(PHHP, "mano"),
        part(PHHP, "kapo"),
        part(PHHP, "okulo"),
    ],
    name="person_has_human_parts",
)


# Runtime "is this entity currently a sub-component of another?"
# marker. Fires whenever the trace asserts `havas_parton(H, P)`:
# P (the parto) gets `is_part = yes` in derived state.
#
# Used by verbs whose theme shouldn't be a part-of-something —
# fali/aperi/ĵeti/porti/detrui/rompiĝi/bruli/timi all declare
# `theme.properties: {is_part: ["no"]}`. Body parts of animates
# pick this up at scene setup (the seeder asserts
# `havas_parton(person, fingo)` for every human part), and
# constructables pick it up at fari-time (the engine asserts
# `havas_parton(lito, ligno)` once the bed is built — after that,
# the wood can't be independently carried). Schema-driven, no
# trace-inspecting filter logic outside the derivation framework.
#
# `is_part` is marked `derived: true` in slots.jsonl so
# EntityInstance.from_concept skips populating the unmarked
# default — otherwise the asserted "no" would block this
# derivation under scalar-wins. The state-emission machinery
# (_state_facts in forward_planner) falls back to slot.unmarked
# at fact-emit time for derived slots without a derived value.
entity_is_part_if_attached = derive(
    when=rel("havas_parton",
             tuto=var("EIPA_H"),
             parto=bind(EIPA_P := var("P"))),
    implies=property(EIPA_P, "is_part", "yes"),
    name="entity_is_part_if_attached",
)


# Indoor locations have a door. Mirrors person_has_human_parts —
# materialized at bake time onto every indoor concept, including
# affix-derived ones like kuirejo, lavejo, dormejo. Combined with
# host_openness_*_from_pordo, this gates eniri to indoor places
# behind a malfermi step (and behind malŝlosi if the door's seruro
# is locked).
indoor_location_has_pordo = derive(
    when=entity(type="location", indoor_outdoor="interna")
         & bind(ILP := var("L")),
    implies=part(ILP, "pordo"),
    name="indoor_location_has_pordo",
)


# Tool-use capability. Any entity with hands (mano) can use tools.
# Currently this picks up persons (persono.parts includes mano) and
# the apes (simio/gorilo/ĉimpanzo, declared with mano in parts).
# Other animals lack mano → can't use tools → can't be agents of
# tool-using verbs. Extending to other body-part-based capabilities
# (e.g. trunk → can_use_tools for elephants) is a one-line addition.

has_hands_can_use_tools = derive(
    when=entity() & bind(T_tool := var("T")),
    given=[
        rel("havas_parton",
            tuto=T_tool,
            parto=bind(P_tool := var("P"))),
        entity(concept="mano") & bind(P_tool),
    ],
    implies=property(T_tool, "can_use_tools", "yes"),
    name="has_hands_can_use_tools",
)


# Broad transient-state slots get an initial value on every concept of
# the relevant type via these derivations. The bake materializes the
# default; the sampler's `_randomize_state` then overrides it with a
# uniform pick from the slot's vocabulary at each instance creation.
# So the derivation's value isn't really a "default" — it's an opt-in
# marker that says "this slot is meaningful for entities of this type,
# please vary it at instance time."

animate_has_hunger = derive(
    when=entity(type="animate") & bind(T_h := var("T")),
    implies=property(T_h, "hunger", "sata"),
    name="animate_has_hunger",
)

animate_has_thirst = derive(
    when=entity(type="animate") & bind(T_th := var("T")),
    implies=property(T_th, "thirst", "satigita"),
    name="animate_has_thirst",
)

animate_has_sleep_state = derive(
    when=entity(type="animate") & bind(T_sl := var("T")),
    implies=property(T_sl, "sleep_state", "vekita"),
    name="animate_has_sleep_state",
)


# ---------- posture: derived from spatial context ----------------------
#
# Posture used to be `varies: true` (uniformly randomized per instance),
# which made roughly 2/3 of agents start out sitting or lying for no
# scene-grounded reason — every locomotion plan then had to insert
# `stari` to satisfy the `posture: staranta` precondition on iri/eniri/
# kuri. Replaced with two derivations: a specific one that fires when
# the agent is `sur` a sittable artifact (chair, sofa) → sidanta, and a
# default that catches everyone else → staranta. First-write-wins on
# scalar slots means the specific derivation must be listed first; the
# default then skips for agents already seated.
#
# When a seeder explicitly places an agent on a chair, the resulting
# scene reads "Maria sidis sur la seĝo… ŝi staris kaj iris…" — the
# stari now has narrative motivation (she got up from the chair).

animate_lying_when_on_lieable = derive(
    when=entity(type="animate") & bind(T_lie := var("T")),
    given=[
        rel("sur", contained=T_lie,
            container=entity(lieable="yes") & bind(B_lie := var("B"))),
    ],
    implies=[
        property(T_lie, "posture", "kuŝanta"),
        property(T_lie, "sleep_state", "dormanta"),
    ],
    name="animate_lying_when_on_lieable",
)

animate_sitting_when_on_sittable = derive(
    when=entity(type="animate") & bind(T_sit := var("T")),
    given=[
        rel("sur", contained=T_sit,
            container=entity(sittable="yes") & bind(S_sit := var("S"))),
    ],
    implies=property(T_sit, "posture", "sidanta"),
    name="animate_sitting_when_on_sittable",
)

# Animate `en` a water_body location is swimming. Listed BEFORE
# `animate_default_standing` so first-write-wins gives `naĝanta`,
# the default never overrides it. The animate-only gate lives in
# the `when` (entity type=animate) and the water_body gate in the
# `given` — no per-pose render-time gating needed; the realizer
# just reads the agent's posture.
animate_swimming_when_in_water_body = derive(
    when=entity(type="animate") & bind(T_swim := var("T")),
    given=[
        rel("en", contained=T_swim,
            container=entity(water_body="yes") & bind(L_swim := var("L"))),
    ],
    implies=property(T_swim, "posture", "naĝanta"),
    name="animate_swimming_when_in_water_body",
)


# Floating derivation: an entity en a liquid container whose denseco
# is strictly less than the liquid's denseco emerges sur the liquid
# and is no longer en it. Schema-derived from the denseco slot — no
# per-concept tagging needed. Dense things (denseco >= liquid) stay
# en, modeled as "submerged": en a liquid means below or within; sur
# means on top.
#
# Uses Compare with VarProp to reach across the binding into the
# container's denseco — same numeric_args_compare evaluator that
# Relation.arg_compare uses, exposed for derivation pattern matching.
# The not_relation("en", T, L) implication hides the asserted en in
# effective state so downstream consumers see the swap cleanly.
floats_when_lighter_than_liquid = derive(
    when=entity(state_of_matter="likva") & bind(L_float := var("L")),
    given=[
        # Arg order is significant: the compiler iterates arg_patterns
        # in declaration order, so container=L_float must come before
        # contained=... so L_float is bound when the Compare on the
        # contained side references VarProp(L_float, "denseco").
        rel("en",
            container=L_float,
            contained=(
                entity(denseco=Compare("<", VarProp(L_float, "denseco")))
                & bind(T_float := var("T")))),
    ],
    implies=[
        relation("sur", T_float, L_float),
        not_relation("en", T_float, L_float),
        # Passive float pose. Listed AFTER the swim derivation in
        # RUNTIME_DERIVATIONS, so animates already-in-water get
        # naĝanta (active swim) via first-write-wins on the scalar
        # posture slot; non-animate floaters land here with flosanta.
        property(T_float, "posture", "flosanta"),
    ],
    name="floats_when_lighter_than_liquid",
)


# Containers can impose a posture on their contents — the
# `imposes_pose` slot. This derivation propagates that for `penda`,
# the only currently-used value not already covered by the
# sittable/lieable affordance pattern (those have their own
# specific derivations). Fruits sur a pomarbo (imposes_pose=penda)
# get posture=penda, rendering as "pendas sur la pomarbo".
# The animate-only constraints from the swimming/sitting/lying
# derivations don't apply here: a piece of fruit hanging is fine.
container_imposes_penda_on_contents = derive(
    when=entity() & bind(C_imp := var("C")),
    given=[
        rel("sur", contained=C_imp,
            container=entity(imposes_pose="penda") & bind(H_imp := var("H"))),
    ],
    implies=property(C_imp, "posture", "penda"),
    name="container_imposes_penda_on_contents",
)

animate_default_standing = derive(
    when=entity(type="animate") & bind(T_st := var("T")),
    implies=property(T_st, "posture", "staranta"),
    name="animate_default_standing",
)

animate_default_awake = derive(
    when=entity(type="animate") & bind(T_aw := var("T")),
    implies=property(T_aw, "sleep_state", "vekita"),
    name="animate_default_awake",
)

physical_has_cleanliness = derive(
    when=entity(type="physical") & bind(T_cl := var("T")),
    implies=property(T_cl, "cleanliness", "pura"),
    name="physical_has_cleanliness",
)

physical_has_temperature = derive(
    when=entity(type="physical") & bind(T_te := var("T")),
    implies=property(T_te, "temperature", "varma"),
    name="physical_has_temperature",
)

physical_has_wetness = derive(
    when=entity(type="physical") & bind(T_we := var("T")),
    implies=property(T_we, "wetness", "seka"),
    name="physical_has_wetness",
)

# Fragile entities default to integrity=tuta. Runtime-only because
# `integrity` is `varies: true` (varies-slots are skipped at bake-
# time so per-instance state — broken glass on the floor, intact
# cup on the shelf — can be authored or set via events). Lets
# verbs like rompiĝi gate `role.theme.properties: {integrity:
# tuta}` and reject fragile-but-already-broken targets without
# requiring every fragile concept to declare integrity in
# concepts.jsonl. After rompiĝi fires (effect: integrity=rompita
# via property_changes), the asserted value wins on the scalar
# slot, suppressing this derivation for that entity instance.
fragile_default_integrity_tuta = derive(
    when=entity(fragility="fragila") & bind(T_fi := var("T")),
    implies=property(T_fi, "integrity", "tuta"),
    name="fragile_default_integrity_tuta",
)


# Every physical entity defaults to presence=ĉeesta (present).
# Lets verbs whose effect consumes the entity (bruli sets
# presence=manĝita) gate on `presence: ["ĉeesta"]` without
# requiring every concept to declare presence in concepts.jsonl.
# Same lifecycle as `fragile_default_integrity_tuta`: after the
# consuming verb fires, asserted-wins on the scalar slot
# suppresses this derivation, and the role-filter sees the
# asserted manĝita and rejects subsequent attempts.
physical_default_presence_ceesta = derive(
    when=entity(type="physical") & bind(T_pr := var("T")),
    implies=property(T_pr, "presence", "ĉeesta"),
    name="physical_default_presence_ceesta",
)

meat_is_edible = derive(
    when=entity(made_of="meat") & bind(T_m := var("T")),
    implies=property(T_m, "edibility", "manĝebla"),
    name="meat_is_edible",
)


# Edible things emit smell. Foods/drinks tagged manĝebla become
# smellable, so a nosed animate samloke with bread/cheese/coffee can
# learn of its presence via flari without needing line-of-sight.
edible_emits_smell = derive(
    when=entity(edibility="manĝebla") & bind(T_es := var("T")),
    implies=property(T_es, "emits_smell", "yes"),
    name="edible_emits_smell",
)


# Flowers emit smell. Lets a nosed animate near rozo/lilio/sunfloro/etc.
# learn of their presence via flari, paralleling the edible chain. The
# bouquet-construct (florkrono) inherits via parts when its floro
# component is samloke with the agent.
floro_emits_smell = derive(
    when=entity(category="floro") & bind(T_fs := var("T")),
    implies=property(T_fs, "emits_smell", "yes"),
    name="floro_emits_smell",
)


# Things made of glass break into vitropecetoj when they break. Lets
# `broken_fragile_creates_shards` find the shard concept via the
# bake-materialized `transforms_on_break` rather than per-concept
# manual tagging. New glass artifacts (lenses, screens, mirrors)
# only need `made_of=glass`; the shard outcome falls out.
glass_breaks_to_shards = derive(
    when=entity(made_of="glass") & bind(T_gb := var("T")),
    implies=property(T_gb, "transforms_on_break", "vitropecetoj"),
    name="glass_breaks_to_shards",
)


# Animates emit sound (footsteps, breathing, vocalizations). Lets
# eared animates learn of someone else's presence via aŭdi.
animate_emits_sound = derive(
    when=entity(type="animate") & bind(T_as := var("T")),
    implies=property(T_as, "emits_sound", "yes"),
    name="animate_emits_sound",
)


# Vocal repertoire. Every animal can `bleki` (generic cry); dogs can
# also `boji`, cats `miaŭi`. These derive into the multi-valued
# `vocal_call` slot, which the verb's agent role checks. Adding
# species-specific calls (muĝi for cattle, hii for horses, etc.) is
# a one-line addition each. Persons aren't animals → don't get
# vocal_call → use krii/flustri instead, which are animate-typed.
animal_can_bleki = derive(
    when=entity(type="animal") & bind(T_blek := var("T")),
    implies=property(T_blek, "vocal_call", "bleki"),
    name="animal_can_bleki",
)

dog_can_boji = derive(
    when=entity(concept="hundo") & bind(T_boj := var("T")),
    implies=property(T_boj, "vocal_call", "boji"),
    name="dog_can_boji",
)

cat_can_miaui = derive(
    when=entity(concept="kato") & bind(T_miau := var("T")),
    implies=property(T_miau, "vocal_call", "miaŭi"),
    name="cat_can_miaui",
)


# Vehicles with a running engine emit sound. Mirrors
# vehicle_powered_from_active_motoro: keys off the host artifact
# having a motoro part whose power_state is aktiva. Parked vehicles
# (motoro neaktiva) and unmotorized vehicles (biciklo — no motoro)
# are silent. Runtime derivation since power_state varies.
vehicle_emits_sound = derive(
    when=entity(type="artifact") & bind(VES := var("V")),
    given=[
        rel("havas_parton", tuto=VES, parto=bind(VESM := var("M"))),
        entity(concept="motoro", power_state="aktiva") & bind(VESM),
    ],
    implies=property(VES, "emits_sound", "yes"),
    name="vehicle_emits_sound",
)


# ---------- derivations: lighting (lamps, indoor vs outdoor) -----------
#
# Outdoor locations are luma by default; indoor locations need an
# `aktiva` lamp present to be luma, otherwise they're malluma. An
# agent in a luma location is `illuminated`. Vidi/montri then gate on
# `illuminated=yes`, so dark-room scenes naturally chain through
# `ŝalti(lamp)` before any visual interaction.

# Outdoor light depends on the trace-wide `mondo` singleton's
# `tempo_de_tago`. Day/morning/evening leave the location luma;
# night flips it to malluma. Mutually exclusive on the night
# condition so first-write-wins on `lit_state` doesn't cross-fire.
outdoor_luma_during_day = derive(
    when=entity(type="location", indoor_outdoor="ekstera") & bind(OIL := var("L")),
    given=[
        ~(entity(concept="mondo", tempo_de_tago="nokto")),
    ],
    implies=property(OIL, "lit_state", "luma"),
    name="outdoor_luma_during_day",
)


outdoor_dark_at_night = derive(
    when=entity(type="location", indoor_outdoor="ekstera") & bind(ODN := var("L")),
    given=[
        entity(concept="mondo", tempo_de_tago="nokto"),
    ],
    implies=property(ODN, "lit_state", "malluma"),
    name="outdoor_dark_at_night",
)

# Light propagates through co-location: a lamp counts as lighting any
# location (indoor room OR outdoor space) it's `samloke` with, which
# by the en/sur chain rules covers direct (lampo en koridoro),
# surface-mounted (lampo sur tablo en koridoro), and any deeper
# nesting. The location_lit rule fires BEFORE outdoor_dark_at_night
# in RUNTIME_DERIVATIONS so a torch in a park at night beats the
# default night darkness — first-write-wins on the scalar slot.
location_lit_by_active_lamp = derive(
    when=entity(type="location") & bind(ILL := var("L")),
    given=[
        rel("samloke", a=ILL, b=bind(ILD := var("D"))),
        entity(power_state="aktiva", lights_when_on="yes") & bind(ILD),
    ],
    implies=property(ILL, "lit_state", "luma"),
    name="location_lit_by_active_lamp",
)

# Mirror semantics: a room is dark iff no active lamp is co-located
# with it (under the same samloke closure). Without the matching
# update, a `sur`-mounted lamp would make the room luma via the lit
# rule AND malluma via the dark rule — first-write-wins on scalar
# state would leave it inconsistent depending on derivation order.
indoor_dark_without_active_lamp = derive(
    when=entity(type="location", indoor_outdoor="interna") & bind(IDL := var("L")),
    given=[
        ~rel("samloke",
             a=IDL,
             b=entity(power_state="aktiva", lights_when_on="yes")),
    ],
    implies=property(IDL, "lit_state", "malluma"),
    name="indoor_dark_without_active_lamp",
)

# Agent is illuminated when co-located with a luma location. Was
# `en` only; switched to samloke so an actor sitting on a sofa
# (sur sofo en koridoro, no direct en(actor, koridoro)) is still
# illuminated by the room's light.
agent_illuminated = derive(
    when=entity(type="animate") & bind(AIA := var("A")),
    given=[
        rel("samloke", a=AIA, b=bind(AIL := var("L"))),
        entity(type="location", lit_state="luma") & bind(AIL),
    ],
    implies=property(AIA, "illuminated", "yes"),
    name="agent_illuminated",
)


# A vehicle's `power_state` lifts from its `motoro` part. Mirrors the
# host_lock_state_*_from_seruro pattern. Pre-condition for veturi —
# without an aktiva engine the vehicle can't carry the agent
# anywhere, so the planner must subgoal ŝalti(motoro).

vehicle_powered_from_active_motoro = derive(
    when=entity(type="artifact") & bind(VPMD := var("D")),
    given=[
        rel("havas_parton", tuto=VPMD, parto=bind(VPMM := var("M"))),
        entity(concept="motoro", power_state="aktiva") & bind(VPMM),
    ],
    implies=property(VPMD, "power_state", "aktiva"),
    name="vehicle_powered_from_active_motoro",
)

vehicle_unpowered_from_inactive_motoro = derive(
    when=entity(type="artifact") & bind(VUMD := var("D")),
    given=[
        rel("havas_parton", tuto=VUMD, parto=bind(VUMM := var("M"))),
        entity(concept="motoro", power_state="neaktiva") & bind(VUMM),
    ],
    implies=property(VUMD, "power_state", "neaktiva"),
    name="vehicle_unpowered_from_inactive_motoro",
)

# `motorized=yes` lifts from a motoro part. Marker that gates the
# veturi.IfPropertyPrecondition so only engine-bearing vehicles need
# their power_state to be aktiva — biciklo (no motoro) bypasses the
# check entirely.
vehicle_motorized_from_motoro = derive(
    when=entity(type="artifact") & bind(VMD := var("D")),
    given=[
        rel("havas_parton", tuto=VMD, parto=bind(VMM := var("M"))),
        entity(concept="motoro") & bind(VMM),
    ],
    implies=property(VMD, "motorized", "yes"),
    name="vehicle_motorized_from_motoro",
)


# ---------- derivation: host's lock_state lifts from its seruro --------
#
# A host with a `seruro` part takes its lock_state from the lock's
# state. Two derivations because PropertyImplication is per-value
# (no "bind a Var from a property value" mechanism — same constraint
# as everywhere else). Adding more lock states later means more
# derivations, but the value vocabulary is small.

host_lock_state_locked_from_seruro = derive(
    when=entity(type="artifact") & bind(HLLD := var("D")),
    given=[
        rel("havas_parton",
            tuto=HLLD,
            parto=bind(HLLS := var("S"))),
        entity(concept="seruro", lock_state="ŝlosita") & bind(HLLS),
    ],
    implies=property(HLLD, "lock_state", "ŝlosita"),
    name="host_lock_state_locked_from_seruro",
)


host_lock_state_unlocked_from_seruro = derive(
    when=entity(type="artifact") & bind(HLUD := var("D")),
    given=[
        rel("havas_parton",
            tuto=HLUD,
            parto=bind(HLUS := var("S"))),
        entity(concept="seruro", lock_state="malŝlosita") & bind(HLUS),
    ],
    implies=property(HLUD, "lock_state", "malŝlosita"),
    name="host_lock_state_unlocked_from_seruro",
)


# Lock-capability lifts from seruro to its host: a thing with a lock
# IS lock-capable. Without this, only the seruro itself counts as
# lock_capable=yes — pordo (which has the seruro as a part) wouldn't
# qualify as a ŝlosi/malŝlosi target. Mirrors the lock_state lifts.

host_lock_capable_from_seruro = derive(
    when=entity(type="artifact") & bind(HLCD := var("D")),
    given=[
        rel("havas_parton",
            tuto=HLCD,
            parto=bind(HLCS := var("S"))),
        entity(concept="seruro") & bind(HLCS),
    ],
    implies=property(HLCD, "lock_capable", "yes"),
    name="host_lock_capable_from_seruro",
)


# ---------- derivation: host's openness lifts from its pordo -----------
#
# A host (location OR artifact) with a `pordo` part takes its openness
# from the door's openness. Used by the eniri / veturi gates so a
# closed-door location/vehicle requires malfermi(pordo) before entry.
# Doorless hosts (kampo, lago, biciklo, ...) have no openness derived,
# so the gate vacuously passes for them.

host_openness_closed_from_pordo = derive(
    when=entity() & bind(HOCD := var("D")),
    given=[
        rel("havas_parton",
            tuto=HOCD,
            parto=bind(HOCP := var("P"))),
        entity(concept="pordo", openness="fermita") & bind(HOCP),
    ],
    implies=property(HOCD, "openness", "fermita"),
    name="host_openness_closed_from_pordo",
)


host_openness_open_from_pordo = derive(
    when=entity() & bind(HOOD := var("D")),
    given=[
        rel("havas_parton",
            tuto=HOOD,
            parto=bind(HOOP := var("P"))),
        entity(concept="pordo", openness="malfermita") & bind(HOOP),
    ],
    implies=property(HOOD, "openness", "malfermita"),
    name="host_openness_open_from_pordo",
)


# scias_lokon(K, X) is derived from scias(K, RT, X, _) whenever RT is
# a *locative* relation — one whose two args name a "contained"
# thing and its "container". en/sur match; havi, apud, samloke don't
# (havi is ownership, apud is adjacency, samloke is co-location). The
# set of locative rels is read from the lex at derivation-build time
# so adding e.g. `super` later auto-extends scias_lokon coverage
# without touching this file. One derivation per locative rel because
# the relaxed-graph planner indexes by distinct rule-effects entries
# and the forward planner's literal-pattern grounding doesn't handle
# OR across rel_type values.


def _is_locative_relation(rel_def) -> bool:
    """Schema-derived locative check: arity-2 relation whose arg_names
    are the canonical container shape. The schema author signals
    "this places its first arg" by using those names."""
    return (rel_def.arity == 2
            and tuple(rel_def.arg_names) == ("contained", "container"))


def make_scias_lokon_derivations(lex) -> list:
    """Build one DSL derivation per locative relation in `lex`. Used
    by callers that pass the runtime derivation list to the engine or
    planner — they prepend / extend with these so scias_lokon is
    derivable for every locative rel_type the lex declares."""
    out: list = []
    for rname, rel_def in lex.relations.items():
        if not _is_locative_relation(rel_def):
            continue
        # Distinct Var instances per derivation so engine bindings
        # don't cross-pollute when both derivations match.
        K = var(f"_SLK_{rname}")
        X = var(f"_SLX_{rname}")
        L = var(f"_SLL_{rname}")
        out.append(derive(
            when=rel("scias",
                     knower=bind(K),
                     rel_type=rname,
                     subjekto=bind(X),
                     objekto=bind(L)),
            implies=relation("scias_lokon", K, X),
            name=f"scias_lokon_via_scias_{rname}",
        ))
    return out


def runtime_derivations_for(lex) -> list:
    """RUNTIME_DERIVATIONS + lex-dependent dynamically-generated
    derivations (currently: scias_lokon per locative rel). Single
    helper so callers don't separately remember to add them."""
    return list(RUNTIME_DERIVATIONS) + make_scias_lokon_derivations(lex)


# ---------- derivation: parts inherit samloke from their host ----------
#
# A host is trivially samloke with its own parts (Maria with her own
# piedo). And anyone samloke with the host is samloke with the parts
# too — the door's lock is wherever the door is, so anyone in the
# room with the door is in the room with its lock. Two derivations,
# composable with the existing en-based samloke chain.

host_samloke_with_part = derive(
    when=rel("havas_parton",
             tuto=bind(HSWPH := var("H")),
             parto=bind(HSWPP := var("P"))),
    implies=relation("samloke", HSWPH, HSWPP),
    name="host_samloke_with_part",
)

samloke_propagates_through_artifact_parts = derive(
    # Iterate sparse side first: havas_parton (~30 facts) drives
    # iteration, then narrow to artifact hosts, then look up the
    # samloke pairs whose `b` is the host. Same closure as the
    # samloke-driven version (B and P are both bound from havas_parton,
    # and A from samloke), but cuts the call count by ~10× because
    # samloke pairs are hundreds in a typical scene.
    #
    # Only propagate through artifact hosts. Without this gate,
    # samloke would cascade across every animate's body parts
    # (samloke(maria, petro) → samloke(maria, petro_piedo) →
    # samloke(maria, petro_mano) → ...) — semantically wrong AND
    # non-terminating.
    when=rel("havas_parton",
             tuto=bind(SPTPB := var("B")),
             parto=bind(SPTPP := var("P"))),
    given=[
        entity(type="artifact") & bind(SPTPB),
        rel("samloke", a=bind(SPTPA := var("A")), b=SPTPB),
    ],
    implies=relation("samloke", SPTPA, SPTPP),
    name="samloke_propagates_through_artifact_parts",
)


# Same propagation for location hosts. Locations have static parts
# (vojo, relo, akvo, pordo) — anyone samloke with the location is
# samloke with those parts. Without this, malfermi(actor, kuirejo_pordo)
# can never satisfy its samloke(actor, pordo) precondition, since
# pordo isn't `en` anything (it's a part). Same iteration-flip as
# the artifact-parts variant — drive off havas_parton, not samloke.
samloke_propagates_through_location_parts = derive(
    when=rel("havas_parton",
             tuto=bind(SPLPB := var("B")),
             parto=bind(SPLPP := var("P"))),
    given=[
        entity(type="location") & bind(SPLPB),
        rel("samloke", a=bind(SPLPA := var("A")), b=SPLPB),
    ],
    implies=relation("samloke", SPLPA, SPLPP),
    name="samloke_propagates_through_location_parts",
)


# An entity in/at a location is samloke with the location itself.
# Without these, samloke(petro, kuirejo) is never derivable from
# `en(petro, kuirejo)` or `apud(petro, kuirejo)` — the existing
# samloke derivations all need a SHARED container between two
# entities. Lets the planner reach a location's parts (its pordo,
# its seruro) for malfermi/malŝlosi gating.

en_implies_samloke_with_container = derive(
    when=rel("en",
             contained=bind(EISA := var("A")),
             container=bind(EISL := var("L"))),
    implies=relation("samloke", EISA, EISL),
    name="en_implies_samloke_with_container",
)


# A carried object travels with its carrier — samloke with the owner.
# Without this, a planner that moves the carrier (kuzo iri kuirejo)
# breaks samloke(kuzo, viando) because viando's `en` relation pins
# it to wherever preni was performed; samloke would only re-derive
# via shared-container, which fails until the carrier and the carried
# happen to land in the same room. With it, a freshly-prena'd theme
# stays samloke through any number of locomotion steps — the planner
# can chain preni → iri → eniri → kuiri without the goal slipping.
# Mirrors host_samloke_with_part: ownership ≈ "in personal space".
havi_implies_samloke_with_carried = derive(
    when=rel("havi",
             owner=bind(HISCO := var("O")),
             theme=bind(HISCT := var("T"))),
    implies=relation("samloke", HISCO, HISCT),
    name="havi_implies_samloke_with_carried",
)


# `apud_implies_samloke_with_neighbor` was removed: it leaked
# location-to-location samloke through the seeder's sibling-room
# placement (e.g. forno en kuirejo, kuirejo apud maro → samloke(forno,
# anyone-in-maro) via samloke_chains_through_en). Justified originally
# by "iri-then-vidi" (actor at the doorstep sees into the room) but
# the realistic chain is iri→eniri→vidi anyway. Dropping it forces
# the planner to subgoal eniri before perceptual/instrument actions
# in another room — slightly longer chains, more grounded prose.


# ---------- derivation: samloke from shared `en` container -------------
#
# Two entities are `samloke` (in the same place) iff some container L
# holds both. Single derivation; engine's fixed-point loop is what makes
# samloke un-derive automatically when an actor walks away — no
# explicit retraction. The relation is symmetric (declared in
# relations.jsonl), so `RelPattern.search` will yield matches for both
# arg orderings; we only need to derive one ordering here.
#
# We DO derive the reflexive case samloke(X, X) for entities that are
# in some container — every entity is trivially in the same place as
# itself. Cluttery in the derived layer but correct, and the planner
# never asks "make X co-located with itself" so it doesn't matter.

shared_container_means_samloke = derive(
    when=rel("en",
             contained=bind(SLA := var("A")),
             container=bind(SLL := var("L"))),
    given=[
        rel("en",
            contained=bind(SLB := var("B")),
            container=SLL),
    ],
    implies=relation("samloke", SLA, SLB),
    name="shared_container_means_samloke",
)


# Samloke also derives from shared apud target. Two travelers who
# both arrived apud the lake are at the same place (the shore),
# even if neither is en the lake. Mirrors the en-based derivation.
shared_apud_means_samloke = derive(
    when=rel("apud",
             subject=bind(SLAA := var("A")),
             neighbor=bind(SLAN := var("N"))),
    given=[
        rel("apud",
            subject=bind(SLAB := var("B")),
            neighbor=SLAN),
    ],
    implies=relation("samloke", SLAA, SLAB),
    name="shared_apud_means_samloke",
)


# `mixed_en_apud_means_samloke` was removed alongside
# `apud_implies_samloke_with_neighbor`. Same leak: with
# `en(forno, kuirejo)` + `apud(kuirejo, maro)` + `en(Avino, maro)`
# this rule derives `samloke(Avino, kuirejo)`, which then chains
# through `samloke_chains_through_en` to make Avino samloke with
# the forno across rooms. Original motivation (door-gated rakonti
# between en-the-room + apud-at-doorstep persons) is thin enough
# that the cleaner narrative — eniri before interacting — wins.


# A entity is `samloke` with whatever it sits ON, mirror of
# `en_implies_samloke_with_container`. Without this, a glaso sur
# tablo doesn't derive samloke(glaso, tablo) — and the chain rules
# below have no seed to propagate. Keeps the same composition as
# the en variant so the closure of {en, sur} → samloke holds.
sur_implies_samloke_with_supporter = derive(
    when=rel("sur",
             contained=bind(SISA := var("A")),
             container=bind(SISB := var("B"))),
    implies=relation("samloke", SISA, SISB),
    name="sur_implies_samloke_with_supporter",
)


# Transitive samloke through nested `en`. Without this, the
# planner can't reach a liquid pre-placed in a vessel: lakto en
# botelo, botelo sur tablo, tablo en kuirejo → samloke(actor,
# lakto) wouldn't derive even when actor is in the kitchen,
# because the existing `shared_container` rule needs the two
# entities to share the SAME container directly.
#
# `A en B AND samloke(B, C)` → A is samloke with C — the engine's
# fixed-point loop closes the chain across arbitrary depth (lakto
# en botelo, samloke(botelo, tablo), samloke(tablo, kuirejo),
# samloke(kuirejo, actor) all derive in order).
samloke_chains_through_en = derive(
    when=rel("en",
             contained=bind(SCEA := var("A")),
             # Container must NOT be a location: chaining through a
             # location-container would propagate samloke across room
             # boundaries (forno en kuirejo, kuirejo en domo,
             # oficejo en domo → samloke(forno, anyone-in-oficejo)).
             # Restricting to non-location containers preserves the
             # intended use (lakto en botelo en tablo: artifact-only
             # transit) while making physical room-to-room separation
             # actually mean something.
             #
             # Closed containers also block transit: a juvelo en a
             # fermita kofro is isolated from anyone outside the kofro
             # until malfermi flips the openness. Lifts perception
             # (vidi → scias), reach (preni samloke check), and any
             # other samloke-gated verb without per-verb modification.
             container=(~entity(type="location") &
                        ~entity(openness="fermita") &
                        bind(SCEB := var("B")))),
    given=[
        rel("samloke", a=SCEB, b=bind(SCEC := var("C"))),
    ],
    implies=relation("samloke", SCEA, SCEC),
    name="samloke_chains_through_en",
)


# Mirror for `sur`. Together with `samloke_chains_through_en` and
# the `*_implies_samloke_with_*` seeds, this closes co-location
# under any en/sur path — co-location is what physical adjacency
# means in this model, and adjacency should compose. Same
# location-boundary restriction as the en variant: a thing sur a
# tablo en kuirejo is samloke with whoever's en kuirejo, but a
# thing sur a kuirejo (rare; conceptually "on top of the kitchen")
# wouldn't transit further.
samloke_chains_through_sur = derive(
    when=rel("sur",
             contained=bind(SCSA := var("A")),
             container=~entity(type="location") & bind(SCSB := var("B"))),
    given=[
        rel("samloke", a=SCSB, b=bind(SCSC := var("C"))),
    ],
    implies=relation("samloke", SCSA, SCSC),
    name="samloke_chains_through_sur",
)


# ---------- derivations: kin-relation contextual category labels -------
#
# Asserting `gepatro(P, C)` derives the surface-form labels on the
# parent and child, gendered by their concept's transitive category
# chain (viro → patro/filo, virino → patrino/filino). The labels are
# realizer-only — pattern matching against `entity(concept=...)` still
# sees the entity's immutable concept_lemma — but the back-reference
# alias pool gets enriched, so a knabo entity (category=viro) that
# happens to be a parent can be referred to as "la patro" alongside
# "la knabo" / "la viro".

gepatro_implies_patro = derive(
    when=rel("gepatro",
             parent=bind(GIPP := var("P")),
             child=bind(GIPC := var("C"))),
    given=[entity(category="viro") & bind(GIPP)],
    implies=category(GIPP, "patro"),
    name="gepatro_implies_patro",
)

gepatro_implies_patrino = derive(
    when=rel("gepatro",
             parent=bind(GIMP := var("P")),
             child=bind(GIMC := var("C"))),
    given=[entity(category="virino") & bind(GIMP)],
    implies=category(GIMP, "patrino"),
    name="gepatro_implies_patrino",
)

gepatro_implies_filo = derive(
    when=rel("gepatro",
             parent=bind(GIFP := var("P")),
             child=bind(GIFC := var("C"))),
    given=[entity(category="viro") & bind(GIFC)],
    implies=category(GIFC, "filo"),
    name="gepatro_implies_filo",
)

gepatro_implies_filino = derive(
    when=rel("gepatro",
             parent=bind(GINP := var("P")),
             child=bind(GINC := var("C"))),
    given=[entity(category="virino") & bind(GINC)],
    implies=category(GINC, "filino"),
    name="gepatro_implies_filino",
)


# Symmetric kin relations: a single male/female derivation each.
# The symmetric swap-yield in the engine fires bindings with both
# arg orderings, so labelling the `a` participant covers both.

frato_implies_frato = derive(
    when=rel("frato",
             a=bind(FRMA := var("A")),
             b=bind(FRMB := var("B"))),
    given=[entity(category="viro") & bind(FRMA)],
    implies=category(FRMA, "frato"),
    name="frato_implies_frato",
)

frato_implies_fratino = derive(
    when=rel("frato",
             a=bind(FRFA := var("A")),
             b=bind(FRFB := var("B"))),
    given=[entity(category="virino") & bind(FRFA)],
    implies=category(FRFA, "fratino"),
    name="frato_implies_fratino",
)

edzo_implies_edzo = derive(
    when=rel("edzo",
             a=bind(EDMA := var("A")),
             b=bind(EDMB := var("B"))),
    given=[entity(category="viro") & bind(EDMA)],
    implies=category(EDMA, "edzo"),
    name="edzo_implies_edzo",
)

edzo_implies_edzino = derive(
    when=rel("edzo",
             a=bind(EDFA := var("A")),
             b=bind(EDFB := var("B"))),
    given=[entity(category="virino") & bind(EDFA)],
    implies=category(EDFA, "edzino"),
    name="edzo_implies_edzino",
)

amiko_implies_amiko = derive(
    when=rel("amiko",
             a=bind(AMMA := var("A")),
             b=bind(AMMB := var("B"))),
    given=[entity(category="viro") & bind(AMMA)],
    implies=category(AMMA, "amiko"),
    name="amiko_implies_amiko",
)

amiko_implies_amikino = derive(
    when=rel("amiko",
             a=bind(AMFA := var("A")),
             b=bind(AMFB := var("B"))),
    given=[entity(category="virino") & bind(AMFA)],
    implies=category(AMFA, "amikino"),
    name="amiko_implies_amikino",
)

najbaro_implies_najbaro = derive(
    when=rel("najbaro",
             a=bind(NJMA := var("A")),
             b=bind(NJMB := var("B"))),
    given=[entity(category="viro") & bind(NJMA)],
    implies=category(NJMA, "najbaro"),
    name="najbaro_implies_najbaro",
)

najbaro_implies_najbarino = derive(
    when=rel("najbaro",
             a=bind(NJFA := var("A")),
             b=bind(NJFB := var("B"))),
    given=[entity(category="virino") & bind(NJFA)],
    implies=category(NJFA, "najbarino"),
    name="najbaro_implies_najbarino",
)


# Convenience bundle: every derivation the library ships with.
# Callers assemble explicitly (as they do for rules) — no hidden
# auto-registration. Pass to `run_dsl(..., derivations=...)`.
DEFAULT_DSL_DERIVATIONS = [
    flammability_from_material,
    mezuri_instruments_default_grandeco,
    denseco_from_wood,
    denseco_from_metal,
    denseco_from_paper,
    denseco_from_stone,
    denseco_from_fabric,
    denseco_from_plant,
    denseco_from_wicker,
    denseco_from_meat,
    denseco_default_manĝaĵo,
    denseco_default_trinkaĵo,
    meat_is_edible,
    animate_is_solid,
    person_can_swim,
    has_paws_can_walk,
    has_wings_can_fly,
    location_is_enterable,
    doored_is_enterable,
    doorless_vehicle_is_not_enterable,
    has_fins_can_swim,
    has_nose_can_smell,
    has_ear_can_hear,
    edible_emits_smell,
    floro_emits_smell,
    glass_breaks_to_shards,
    animate_emits_sound,
    vehicle_emits_sound,
    animal_can_bleki,
    dog_can_boji,
    cat_can_miaui,
    location_water_body_via_part,
    location_water_body_via_en,
    location_terrain_land_via_part,
    location_terrain_rail_via_part,
    location_terrain_water_via_part,
    location_terrain_indoor_via_indoor_outdoor,
    location_has_air_terrain,
    animate_terrain_land_from_walk,
    animate_terrain_indoor_from_walk,
    animate_terrain_water_from_swim,
    animate_terrain_air_from_fly,
    animate_terrain_land_from_slither,
    entity_in_water_from_water_body,
    person_has_human_parts,
    indoor_location_has_pordo,
    has_hands_can_use_tools,
    animate_has_hunger,
    animate_has_thirst,
    # animate_has_sleep_state moved to RUNTIME-only — bake would
    # materialize `vekita` onto every animate concept, preempting the
    # `lying_when_on_lieable` runtime override that sets dormanta.
    physical_has_cleanliness,
    physical_has_temperature,
    physical_has_wetness,
    shared_container_means_samloke,
    shared_apud_means_samloke,
    host_samloke_with_part,
    samloke_propagates_through_artifact_parts,
    samloke_propagates_through_location_parts,
    en_implies_samloke_with_container,
    havi_implies_samloke_with_carried,
    sur_implies_samloke_with_supporter,
    samloke_chains_through_en,
    samloke_chains_through_sur,
    host_lock_state_locked_from_seruro,
    host_lock_state_unlocked_from_seruro,
    host_lock_capable_from_seruro,
    host_openness_closed_from_pordo,
    host_openness_open_from_pordo,
    # Outdoor light is now conditional on mondo.tempo_de_tago — that's
    # a varies=true slot, so the rule is RUNTIME-only and removed from
    # the bake list. The runtime list below carries it.
    location_lit_by_active_lamp,
    indoor_dark_without_active_lamp,
    agent_illuminated,
    vehicle_powered_from_active_motoro,
    vehicle_unpowered_from_inactive_motoro,
    vehicle_motorized_from_motoro,
    gepatro_implies_patro,
    gepatro_implies_patrino,
    gepatro_implies_filo,
    gepatro_implies_filino,
    frato_implies_frato,
    frato_implies_fratino,
    edzo_implies_edzo,
    edzo_implies_edzino,
    amiko_implies_amiko,
    amiko_implies_amikino,
    najbaro_implies_najbaro,
    najbaro_implies_najbarino,
    # scias_lokon-derivations are NOT listed here: they're built per-
    # locative-rel at engine-call time via `runtime_derivations_for`,
    # since the set of locative rels lives in the lex (not in this
    # module). Bake skips them anyway — scias is dynamic state.
]


# Subset of DEFAULT_DSL_DERIVATIONS whose outputs depend ONLY on
# static state (entity_type, parts via havas_parton, non-varying
# slot values). These are materialized onto concept.properties at
# bake-time, so re-firing them at runtime is wasted work — the
# outputs are already in entity.properties when the entity is
# instantiated. The runtime engine can skip them.
#
# Classification rule: a derivation is RUNTIME if any of its
# patterns reference a `varies=true` slot value or a non-static
# relation (en, apud, samloke, havi, sur, priskribas, scias,
# scias_lokon, etc. — havas_parton is static). Everything else is
# concept-stable. Curated by hand here rather than auto-detected
# so the categorization is reviewable and stable across derivation
# additions; new derivations default to RUNTIME unless explicitly
# moved here.
RUNTIME_DERIVATIONS = [
    # `is_part` is a runtime-derived slot: fires for any entity
    # currently on the `parto` side of a havas_parton fact (body
    # parts of animates at scene setup, constructable ingredients
    # post-fari). See its definition for why it's runtime-only.
    entity_is_part_if_attached,
    # Fragile entities default to integrity=tuta. Runtime-only
    # because `integrity` is varies-true; the bake skips varies
    # slots so the default must be derived per-instance instead.
    # After rompiĝi fires (effect: integrity=rompita via
    # property_changes), asserted-wins suppresses this derivation
    # for that entity — subsequent rompiĝi attempts see integrity
    # =rompita in state and the role-filter rejects.
    fragile_default_integrity_tuta,
    # Physical entities default to presence=ĉeesta. Same
    # lifecycle as fragile_default_integrity_tuta — gates bruli/
    # manĝi-style consuming verbs against re-firing once the
    # asserted manĝita lands via property_changes.
    physical_default_presence_ceesta,
    # Posture / sleep_state derivations are runtime-only (NOT in
    # DEFAULT_DSL_DERIVATIONS): if the defaults were baked, every
    # animate concept's properties would carry posture=staranta /
    # sleep_state=vekita, and asserted-wins-on-scalar would block
    # context overrides (sittable → sidanta, lieable → kuŝanta +
    # dormanta) at trace-time. Order: most specific first
    # (lieable beats sittable beats default — the engine's first-
    # write-wins on derived state preserves the earlier value).
    animate_lying_when_on_lieable,
    animate_sitting_when_on_sittable,
    animate_swimming_when_in_water_body,
    water_body_has_water_properties,
    floats_when_lighter_than_liquid,
    container_imposes_penda_on_contents,
    animate_default_standing,
    animate_default_awake,
    location_water_body_via_en,
    entity_in_water_from_water_body,
    shared_container_means_samloke,
    shared_apud_means_samloke,
    samloke_propagates_through_artifact_parts,
    samloke_propagates_through_location_parts,
    en_implies_samloke_with_container,
    havi_implies_samloke_with_carried,
    sur_implies_samloke_with_supporter,
    samloke_chains_through_en,
    samloke_chains_through_sur,
    host_lock_state_locked_from_seruro,
    host_lock_state_unlocked_from_seruro,
    host_openness_closed_from_pordo,
    host_openness_open_from_pordo,
    # Lamp-lit fires BEFORE the outdoor day/night rules so an active
    # lamp wins over outdoor_dark_at_night (first-write-wins on the
    # scalar lit_state) — a torch in a park at night lights it.
    location_lit_by_active_lamp,
    outdoor_luma_during_day,
    outdoor_dark_at_night,
    indoor_dark_without_active_lamp,
    agent_illuminated,
    vehicle_powered_from_active_motoro,
    vehicle_unpowered_from_inactive_motoro,
    vehicle_emits_sound,
    gepatro_implies_patro,
    gepatro_implies_patrino,
    gepatro_implies_filo,
    gepatro_implies_filino,
    frato_implies_frato,
    frato_implies_fratino,
    edzo_implies_edzo,
    edzo_implies_edzino,
    amiko_implies_amiko,
    amiko_implies_amikino,
    najbaro_implies_najbaro,
    najbaro_implies_najbarino,
    # scias_lokon-derivations live in `runtime_derivations_for(lex)`,
    # not here — see the comment in DEFAULT_DSL_DERIVATIONS.
]


# Convenience bundle: all standalone rules, ordered to match the old
# engine's DEFAULT_RULES followed by its factory-produced rules. Same
# order means same firing sequence under the fixed-point loop, which
# is what the Phase-4 parity tests assert.
DEFAULT_DSL_RULES: list[Rule] = [
    fragile_falls_breaks,
    bati_breaks_fragile,
    vidi_learns_en,
    flari_learns_en,
    audi_learns_en,
    krii_announces_speaker,
    ludi_announces_player,
    flustri_announces_speaker,
    bleki_announces_speaker,
    boji_announces_speaker,
    miaui_announces_speaker,
    vidi_learns_sur,
    flari_learns_sur,
    audi_learns_sur,
    vidi_learns_havi_owner,
    rakonti_transfers_scias,
    demandi_extracts_scias,
    respondi_transfers_scias,
    montri_shows_location,
    instrui_transfers_scias,
    legi_extracts_scias,
    mezuri_learns_dimension,
    hungry_eats_sated,
    thirsty_drinks_quenched,
    manĝi_consumes_theme,
    trinki_consumes_theme,
    morti_destroys_self,
    mortigi_causes_morti,
    detrui_destroys_theme,
    container_falls_contents_fall,
    broken_container_releases_contents,
    person_slips_on_wet,
    rain_wets_contents,
    rain_creates_puddle,
    porti_drop_when_carrier_falls,
    fire_spreads_to_adjacent_flammables,
    preni_transfers_ownership,
    doni_transfers_ownership,
    peti_transfers_ownership,
    aĉeti_transfers,
    vendi_transfers,
    meti_places_in_location,
    planti_plants_theme,
    ami_creates_amas,
    timi_creates_timas,
    meti_places_on_surface,
    iri_moves_agent,
    veni_moves_agent,
    kuri_moves_agent,
    naĝi_moves_agent,
    flugi_moves_agent,
    eniri_enters_location,
    surgrimpi_mounts_vehicle,
    rajdi_moves_agent,
    sekvi_brings_agent_to_theme,
    voki_summons_theme,
    veturi_moves_agent,
    ĵeti_releases_possession,
    kapti_takes_possession,
    skribi_creates_text,
    skribi_records_scias,
    viŝi_destroys_skribaĵo,
    porti_establishes_carrying,
    # Previously factory-produced; now plain values after Phase 2.
    broken_fragile_creates_shards,
    wet_liquid_container_tips,
    verŝi_releases_possession,
    surmeti_dresses,
    demeti_undresses,
    plenigi_transfers_contents,
    malplenigi_releases_contents,
    verŝi_into_location_emits_fali,
    verŝi_into_container_transfers,
    fari_creates_constructable,
]
