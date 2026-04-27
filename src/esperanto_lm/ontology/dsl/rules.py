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
    add_relation, bind, caused_by, closure, create_entity, derive,
    destroy_entity, emit, entity, event, has_concept_field, part,
    past_event, property, rel, relation, remove_relation, rule, var,
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
# `preni` (take) covers both cases — same split as `kapti`:
#   - from_nobody: pick up an unowned item (e.g. a key on a shelf).
#                  Asserts havi(agent, theme); no removal needed.
#   - transfers:   take from a current owner. Removes old havi, adds new.
# Splitting keeps each case independently expressible and avoids the
# match-fails-silently bug that previously left `preni` unable to
# acquire unowned items.

preni_acquires_unowned = rule(
    when=event("preni",
               agent=bind(TUA := var("A")),
               theme=bind(TUT := var("T"))),
    given=[
        ~rel("havi", owner=bind(var("_any")), theme=TUT),
    ],
    then=add_relation("havi", TUA, TUT),
    name="preni_acquires_unowned",
)

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
    ],
    then=[
        remove_relation("en", VnA, VnO),
        add_relation("en", VnA, VnD),
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
    ],
    then=[
        remove_relation("en", KrA, KrO),
        add_relation("en", KrA, KrD),
    ],
    name="kuri_moves_agent",
)


naĝi_moves_agent = rule(
    when=event("naĝi",
               agent=bind(NgA := var("A")),
               destination=bind(NgD := var("D"))),
    given=[
        rel("en", contained=NgA, container=bind(NgO := var("O"))),
    ],
    then=[
        remove_relation("en", NgA, NgO),
        add_relation("en", NgA, NgD),
    ],
    name="naĝi_moves_agent",
)


flugi_moves_agent = rule(
    when=event("flugi",
               agent=bind(FgA := var("A")),
               destination=bind(FgD := var("D"))),
    given=[
        rel("en", contained=FgA, container=bind(FgO := var("O"))),
    ],
    then=[
        remove_relation("en", FgA, FgO),
        add_relation("en", FgA, FgD),
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
    ],
    then=[
        remove_relation("en", SqA, SqO),
        add_relation("en", SqA, SqL),
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
    ],
    then=[
        remove_relation("en", VkT, VkO),
        add_relation("en", VkT, VkL),
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
        rel("en", contained=bind(RX := var("X")), container=RL),
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


# When the writer already konas a fakto, the produced skribaĵo
# carries that fakto via `priskribas` — text records what the
# author knows. Downstream `legi_extracts_fakto` then propagates the
# fakto to readers. Split from skribi_creates_text so the base case
# (writing produces a blank text) still fires when the agent has
# nothing to record.
skribi_records_fakto = rule(
    when=event("skribi",
               agent=bind(SrA := var("A")),
               theme=bind(SrT := var("T"))),
    given=[
        rel("konas", knower=SrA, fakto=bind(SrF := var("F"))),
    ],
    then=[
        create_entity(concept="skribaĵo",
                      as_var=(SrS := var("S")), from_=SrT),
        add_relation("priskribas", SrS, SrF),
    ],
    name="skribi_records_fakto",
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
# Knowledge is fact-grained. A `fakto` entity captures one specific
# relation instance — `(pri_relacio, pri_subjekto, pri_objekto)` — and
# `konas(agent, fakto)` means the agent knows that specific fact. The
# fakto's id is composite-deterministic (`fakto_from_<rel>_<a>_<b>`)
# so the same fact creates the same entity across rule firings.
#
# When an agent `vidi`s an entity X, they learn one fakto per relation
# X currently participates in — separate rule per relation kind /
# arg-position so the binding stays explicit. Add more rules here as
# new relations grow narrative weight (currently: en, sur, havi).
#
# `rakonti` transfers knowledge of one specific fakto from teller to
# recipient — pure information move, no fakto creation.

vidi_learns_en = rule(
    when=event("vidi",
               agent=bind(VEA := var("A")),
               theme=bind(VET := var("T"))),
    given=[
        rel("en", contained=VET, container=bind(VEL := var("L"))),
    ],
    then=[
        create_entity(
            concept="fakto",
            as_var=(VEF := var("F")),
            id_parts=("en", VET, VEL),
            initial_properties={"pri_relacio": "en"},
        ),
        add_relation("subjekto", VEF, VET),
        add_relation("objekto", VEF, VEL),
        add_relation("konas", VEA, VEF),
    ],
    name="vidi_learns_en",
)

vidi_learns_sur = rule(
    when=event("vidi",
               agent=bind(VSA := var("A")),
               theme=bind(VST := var("T"))),
    given=[
        rel("sur", contained=VST, container=bind(VSL := var("L"))),
    ],
    then=[
        create_entity(
            concept="fakto",
            as_var=(VSF := var("F")),
            id_parts=("sur", VST, VSL),
            initial_properties={"pri_relacio": "sur"},
        ),
        add_relation("subjekto", VSF, VST),
        add_relation("objekto", VSF, VSL),
        add_relation("konas", VSA, VSF),
    ],
    name="vidi_learns_sur",
)

vidi_learns_havi_owner = rule(
    when=event("vidi",
               agent=bind(VHA := var("A")),
               theme=bind(VHT := var("T"))),
    given=[
        rel("havi", owner=bind(VHO := var("O")), theme=VHT),
    ],
    then=[
        create_entity(
            concept="fakto",
            as_var=(VHF := var("F")),
            id_parts=("havi", VHO, VHT),
            initial_properties={"pri_relacio": "havi"},
        ),
        add_relation("subjekto", VHF, VHO),
        add_relation("objekto", VHF, VHT),
        add_relation("konas", VHA, VHF),
    ],
    name="vidi_learns_havi_owner",
)

rakonti_transfers_fakto = rule(
    when=event("rakonti",
               agent=bind(RKA := var("A")),
               theme=bind(RKT := var("T")),
               recipient=bind(RKR := var("R"))),
    then=add_relation("konas", RKR, RKT),
    name="rakonti_transfers_fakto",
)


# `demandi` is rakonti's inverse: the asker (agent) acquires the
# fakto from someone who already knows it (recipient). Modeled as
# atomic Q&A — firing demandi adds konas(agent, fakto), no separate
# "recipient replies" event needed for the planner's purposes.
demandi_extracts_fakto = rule(
    when=event("demandi",
               agent=bind(DEA := var("A")),
               theme=bind(DET := var("T")),
               recipient=bind(DER := var("R"))),
    then=add_relation("konas", DEA, DET),
    name="demandi_extracts_fakto",
)


# `respondi` mirrors rakonti structurally — agent transfers a fakto
# to recipient. The conversational distinction (reply vs initiating
# tell) doesn't change the semantics for the engine; it gives the
# realizer a verb to pick when continuing a Q&A turn.
respondi_transfers_fakto = rule(
    when=event("respondi",
               agent=bind(RPA := var("A")),
               theme=bind(RPT := var("T")),
               recipient=bind(RPR := var("R"))),
    then=add_relation("konas", RPR, RPT),
    name="respondi_transfers_fakto",
)


# `montri` (show) is vidi-flavored from the recipient's side: the
# agent — already samloke with both theme and recipient — surfaces
# the theme's location as a fakto and the recipient learns it. Same
# create_entity shape as vidi_learns_en, but konas goes to the
# recipient role instead of the agent. Lets "show, don't tell"
# transfer knowledge of physical things without requiring the agent
# to first konas any fakto.
montri_shows_location = rule(
    when=event("montri",
               agent=bind(MNA := var("A")),
               theme=bind(MNT := var("T")),
               recipient=bind(MNR := var("R"))),
    given=[
        rel("en", contained=MNT, container=bind(MNL := var("L"))),
    ],
    then=[
        create_entity(
            concept="fakto",
            as_var=(MNF := var("F")),
            id_parts=("en", MNT, MNL),
            initial_properties={"pri_relacio": "en"},
        ),
        add_relation("subjekto", MNF, MNT),
        add_relation("objekto", MNF, MNL),
        add_relation("konas", MNR, MNF),
    ],
    name="montri_shows_location",
)


# `instrui` (teach) is rakonti's pedagogical sibling — same shape
# (agent transfers a fakto they konas to a recipient) and same
# semantics. Distinct lemma gives the realizer a verb choice for
# instructional contexts (instructor → student) vs narrative ones
# (storyteller → listener); the engine treats them identically.
instrui_transfers_fakto = rule(
    when=event("instrui",
               agent=bind(ITA := var("A")),
               theme=bind(ITT := var("T")),
               recipient=bind(ITR := var("R"))),
    then=add_relation("konas", ITR, ITT),
    name="instrui_transfers_fakto",
)


# `legi` (read) extracts a fakto from a text. Asynchronous knowledge
# transfer: where rakonti requires the source to be physically
# present, legi only needs the reader to be samloke with the text.
# The text-to-fakto link comes from `priskribas(text, fakto)`, set
# either at scene init (regression seed) or by skribi_creates_text
# when the writer konas a fakto they wanted to record.
legi_extracts_fakto = rule(
    when=event("legi",
               agent=bind(LeA := var("A")),
               theme=bind(LeT := var("T"))),
    given=[
        rel("priskribas", text=LeT, fakto=bind(LeF := var("F"))),
    ],
    then=add_relation("konas", LeA, LeF),
    name="legi_extracts_fakto",
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

animate_has_sleep_state = derive(
    when=entity(type="animate") & bind(T_sl := var("T")),
    implies=property(T_sl, "sleep_state", "vekita"),
    name="animate_has_sleep_state",
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

meat_is_edible = derive(
    when=entity(made_of="meat") & bind(T_m := var("T")),
    implies=property(T_m, "edibility", "manĝebla"),
    name="meat_is_edible",
)


# ---------- derivation: animates know facts about themselves -----------
#
# Lidia inherently knows that Lidia is in la maro — she IS Lidia, and
# she's the subject of that fact. Without these derivations the
# planner happily samples drives like "Sara wants Lidia to know that
# Lidia is in la maro" and dispatches a `vidi → rakonti` chain to
# tell Lidia what she already knows. Two derivations: one for facts
# where the animate is the pri_subjekto (containment, ownership), one
# for where they're pri_objekto (havi-from-the-thing's-perspective —
# rare for animals, but cheap to derive symmetrically).

animate_knows_self_subject = derive(
    when=entity(type="animate") & bind(SKSA := var("A")),
    given=[
        rel("subjekto",
            fakto=bind(SKSF := var("F")),
            entity=SKSA),
    ],
    implies=relation("konas", SKSA, SKSF),
    name="animate_knows_self_subject",
)

animate_knows_self_object = derive(
    when=entity(type="animate") & bind(SKOA := var("A")),
    given=[
        rel("objekto",
            fakto=bind(SKOF := var("F")),
            entity=SKOA),
    ],
    implies=relation("konas", SKOA, SKOF),
    name="animate_knows_self_object",
)


# ---------- derivations: lighting (lamps, indoor vs outdoor) -----------
#
# Outdoor locations are luma by default; indoor locations need an
# `aktiva` lamp present to be luma, otherwise they're malluma. An
# agent in a luma location is `illuminated`. Vidi/montri then gate on
# `illuminated=yes`, so dark-room scenes naturally chain through
# `ŝalti(lamp)` before any visual interaction.

outdoor_is_luma = derive(
    when=entity(type="location", indoor_outdoor="ekstera") & bind(OIL := var("L")),
    implies=property(OIL, "lit_state", "luma"),
    name="outdoor_is_luma",
)

indoor_lit_by_active_lamp = derive(
    when=entity(type="location", indoor_outdoor="interna") & bind(ILL := var("L")),
    given=[
        rel("en", contained=bind(ILD := var("D")), container=ILL),
        entity(power_state="aktiva", lights_when_on="yes") & bind(ILD),
    ],
    implies=property(ILL, "lit_state", "luma"),
    name="indoor_lit_by_active_lamp",
)

indoor_dark_without_active_lamp = derive(
    when=entity(type="location", indoor_outdoor="interna") & bind(IDL := var("L")),
    given=[
        ~rel("en",
             contained=entity(power_state="aktiva", lights_when_on="yes"),
             container=IDL),
    ],
    implies=property(IDL, "lit_state", "malluma"),
    name="indoor_dark_without_active_lamp",
)

agent_illuminated = derive(
    when=entity(type="animate") & bind(AIA := var("A")),
    given=[
        rel("en", contained=AIA, container=bind(AIL := var("L"))),
        entity(lit_state="luma") & bind(AIL),
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


# ---------- derivation: knowing a location-fakto means knowing where ----
#
# An agent who knows a fakto whose relation is `en` or `sur` (and the
# fakto's subjekto is some entity T) knows where T is. Two derivations,
# one per locative relation. Used as a precondition by verbs that
# require knowing the target's location (preni, kapti, veki, mortigi).

scias_lokon_via_en = derive(
    when=rel("konas",
             knower=bind(SLEK := var("K")),
             fakto=bind(SLEF := var("F"))),
    given=[
        entity(type="abstract", pri_relacio="en") & bind(SLEF),
        rel("subjekto", fakto=SLEF, entity=bind(SLET := var("T"))),
    ],
    implies=relation("scias_lokon", SLEK, SLET),
    name="scias_lokon_via_en",
)

scias_lokon_via_sur = derive(
    when=rel("konas",
             knower=bind(SLSK := var("K")),
             fakto=bind(SLSF := var("F"))),
    given=[
        entity(type="abstract", pri_relacio="sur") & bind(SLSF),
        rel("subjekto", fakto=SLSF, entity=bind(SLST := var("T"))),
    ],
    implies=relation("scias_lokon", SLSK, SLST),
    name="scias_lokon_via_sur",
)


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
    when=rel("samloke",
             a=bind(SPTPA := var("A")),
             b=bind(SPTPB := var("B"))),
    given=[
        # Only propagate through artifact hosts. Without this gate,
        # samloke cascades across every animate's body parts
        # (samloke(maria, petro) → samloke(maria, petro_piedo) →
        # samloke(maria, petro_mano) → ...) which doesn't terminate
        # in reasonable time. Body parts are samloke with their own
        # host (host_samloke_with_part) but not with arbitrary
        # observers — which is also more semantically right ("Maria
        # is in the same place as Petro's hand" is technically true
        # but not what we mean by samloke for verb planning).
        entity(type="artifact") & bind(SPTPB),
        rel("havas_parton",
            tuto=SPTPB,
            parto=bind(SPTPP := var("P"))),
    ],
    implies=relation("samloke", SPTPA, SPTPP),
    name="samloke_propagates_through_artifact_parts",
)


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


# Convenience bundle: every derivation the library ships with.
# Callers assemble explicitly (as they do for rules) — no hidden
# auto-registration. Pass to `run_dsl(..., derivations=...)`.
DEFAULT_DSL_DERIVATIONS = [
    flammability_from_material,
    meat_is_edible,
    animate_is_solid,
    person_can_swim,
    has_paws_can_walk,
    has_wings_can_fly,
    has_fins_can_swim,
    location_water_body_via_part,
    location_water_body_via_en,
    location_terrain_land_via_part,
    location_terrain_rail_via_part,
    location_terrain_water_via_part,
    entity_in_water_from_water_body,
    person_has_human_parts,
    has_hands_can_use_tools,
    animate_has_hunger,
    animate_has_sleep_state,
    physical_has_cleanliness,
    physical_has_temperature,
    physical_has_wetness,
    shared_container_means_samloke,
    host_samloke_with_part,
    samloke_propagates_through_artifact_parts,
    host_lock_state_locked_from_seruro,
    host_lock_state_unlocked_from_seruro,
    host_lock_capable_from_seruro,
    animate_knows_self_subject,
    animate_knows_self_object,
    scias_lokon_via_en,
    scias_lokon_via_sur,
    outdoor_is_luma,
    indoor_lit_by_active_lamp,
    indoor_dark_without_active_lamp,
    agent_illuminated,
    vehicle_powered_from_active_motoro,
    vehicle_unpowered_from_inactive_motoro,
    vehicle_motorized_from_motoro,
]


# Convenience bundle: all standalone rules, ordered to match the old
# engine's DEFAULT_RULES followed by its factory-produced rules. Same
# order means same firing sequence under the fixed-point loop, which
# is what the Phase-4 parity tests assert.
DEFAULT_DSL_RULES: list[Rule] = [
    fragile_falls_breaks,
    bati_breaks_fragile,
    vidi_learns_en,
    vidi_learns_sur,
    vidi_learns_havi_owner,
    rakonti_transfers_fakto,
    demandi_extracts_fakto,
    respondi_transfers_fakto,
    montri_shows_location,
    instrui_transfers_fakto,
    legi_extracts_fakto,
    hungry_eats_sated,
    manĝi_destroys_theme,
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
    preni_acquires_unowned,
    preni_transfers_ownership,
    doni_transfers_ownership,
    meti_places_in_location,
    meti_places_on_surface,
    iri_moves_agent,
    veni_moves_agent,
    kuri_moves_agent,
    naĝi_moves_agent,
    flugi_moves_agent,
    sekvi_brings_agent_to_theme,
    voki_summons_theme,
    veturi_moves_agent,
    ĵeti_releases_possession,
    kapti_takes_possession_from_nobody,
    kapti_takes_possession_from_owner,
    skribi_creates_text,
    skribi_records_fakto,
    viŝi_destroys_skribaĵo,
    porti_establishes_carrying,
    # Previously factory-produced; now plain values after Phase 2.
    broken_fragile_creates_shards,
    wet_liquid_container_tips,
]
