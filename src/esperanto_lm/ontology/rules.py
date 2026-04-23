"""Causal rules (event-calculus form).

Each rule is a callable with signature `rule(trace, t) -> list[Event]`.
The rule inspects state at position t via `trace.entities_at(t)` and
`trace.property_at(eid, prop, t)`, and returns events with
`property_changes` and (optionally) `creates`. The engine appends the
events and handles memoization — rules never mutate the trace directly.

Style: rules trigger on action lemma of a past event, match on property
patterns of the bound entities, and emit consequences. Memoization is
per event id — `caused_by` typically includes the triggering event, so
distinct causes produce distinct ids and allow the same rule to fire
multiple times on semantically different matches.

Rules that need lexicon access (e.g. `use_instrument`'s signature
resolution) are factory functions — the factory takes the lexicon and
returns a closure that matches the `Rule` signature. See
`make_use_instrument` below.
"""
from __future__ import annotations

from .causal import EntityInstance, Event, Trace, make_event
from .loader import Lexicon, resolve_signature


def _has_value(prop_value, target: str) -> bool:
    """Property values may be `list[str]` (when read from
    `entity.properties`) or scalar (when read from a prior event's
    `property_changes`). Membership-check both forms."""
    if prop_value is None:
        return False
    if isinstance(prop_value, list):
        return target in prop_value
    return prop_value == target


def _already_fired(trace: Trace, action: str, theme_id: str) -> bool:
    """True if any event in the trace already has this (action, theme).
    Used by cascade rules to prevent parallel redundant re-fires —
    e.g. fali(akvo) via container_falls_contents_fall AND via
    broken_container_releases_contents would produce two identical
    falls with different caused_by (hence different ids, escaping per-id
    memoization). One fall per entity per trace is the semantic we want.
    """
    for ev in trace.events:
        if ev.action == action and ev.roles.get("theme") == theme_id:
            return True
    return False


# ---------------------------- rules --------------------------------------

def fragile_falls_breaks(trace: Trace, t: int) -> list[Event]:
    """Any fali event on a fragile, not-yet-broken theme triggers a
    rompiĝi event that marks the theme as broken."""
    out: list[Event] = []
    for ev in trace.events[:t]:
        if ev.action != "fali":
            continue
        theme_id = ev.roles.get("theme")
        if not theme_id:
            continue
        fragility = trace.property_at(theme_id, "fragility", t)
        if not _has_value(fragility, "fragile"):
            continue
        integrity = trace.property_at(theme_id, "integrity", t)
        if _has_value(integrity, "broken") or _has_value(integrity, "severed"):
            continue
        out.append(make_event(
            action="rompiĝi",
            roles={"theme": theme_id},
            caused_by=[ev.id],
            property_changes={(theme_id, "integrity"): "broken"},
        ))
    return out


def hungry_eats_sated(trace: Trace, t: int) -> list[Event]:
    """If an agent was hungry at the time they fired a manĝi event,
    produce a satiĝi event that marks them as sated."""
    out: list[Event] = []
    for ev in trace.events[:t]:
        if ev.action != "manĝi":
            continue
        agent_id = ev.roles.get("agent")
        if not agent_id:
            continue
        event_pos = ev.trace_position
        if event_pos is None:
            try:
                event_pos = trace.events.index(ev)
            except ValueError:
                continue
        hunger = trace.property_at(agent_id, "hunger", event_pos)
        if not _has_value(hunger, "hungry"):
            continue
        cur_hunger = trace.property_at(agent_id, "hunger", t)
        if _has_value(cur_hunger, "sated"):
            continue
        out.append(make_event(
            action="satiĝi",
            roles={"theme": agent_id},
            caused_by=[ev.id],
            property_changes={(agent_id, "hunger"): "sated"},
        ))
    return out


def container_falls_contents_fall(trace: Trace, t: int) -> list[Event]:
    """For each fali on a container, every entity `en` or `sur` it also
    falls. Non-location-typed entities only (a person in the kitchen
    doesn't fall just because the kitchen is mentioned in a fali)."""
    out: list[Event] = []
    for ev in trace.events[:t]:
        if ev.action != "fali":
            continue
        container_id = ev.roles.get("theme")
        if not container_id:
            continue
        for rel in trace.relations:
            if rel.relation not in ("en", "sur"):
                continue
            if len(rel.args) != 2:
                continue
            if rel.args[1] != container_id:
                continue
            contained_id = rel.args[0]
            contained = trace.entities.get(contained_id)
            if contained is None or contained.entity_type == "location":
                continue
            if _already_fired(trace, "fali", contained_id):
                continue
            out.append(make_event(
                action="fali",
                roles={"theme": contained_id},
                caused_by=[ev.id],
            ))
    return out


def broken_container_releases_contents(trace: Trace, t: int) -> list[Event]:
    """When a container breaks (rompiĝi), anything in/on it falls. Sibling
    of container_falls_contents_fall; triggered by breaks instead of falls.
    Enables cascades like 'person hits glass with hammer → glass shatters
    → contents spill' without a preceding fall."""
    out: list[Event] = []
    for ev in trace.events[:t]:
        if ev.action != "rompiĝi":
            continue
        container_id = ev.roles.get("theme")
        if not container_id:
            continue
        for rel in trace.relations:
            if rel.relation not in ("en", "sur"):
                continue
            if len(rel.args) != 2:
                continue
            if rel.args[1] != container_id:
                continue
            contained_id = rel.args[0]
            contained = trace.entities.get(contained_id)
            if contained is None or contained.entity_type == "location":
                continue
            if _already_fired(trace, "fali", contained_id):
                continue
            out.append(make_event(
                action="fali",
                roles={"theme": contained_id},
                caused_by=[ev.id],
            ))
    return out


def make_use_instrument(lexicon: Lexicon):
    """Factory: returns a rule closure that resolves instrument signatures
    via the given lexicon.

    Done as a factory because the rule needs lexicon access but the
    `Rule` signature is `(trace, t) -> list[Event]` with no lexicon
    parameter. The synthesized event's `property_changes` are computed
    from the source verb's effect spec — same semantic as a direct
    mutation of the bound entities' state, but attached to the event
    itself so the event calculus can reconstruct it.
    """
    def use_instrument(trace: Trace, t: int) -> list[Event]:
        out: list[Event] = []
        for ev in trace.events[:t]:
            if ev.action != "uzi":
                continue
            instr_id = ev.roles.get("instrument")
            theme_id = ev.roles.get("theme")
            agent_id = ev.roles.get("agent")
            if not (instr_id and theme_id and agent_id):
                continue
            instr = trace.entity(instr_id)
            theme = trace.entity(theme_id)
            if instr is None or theme is None:
                continue
            instr_concept = lexicon.concepts.get(instr.concept_lemma)
            if instr_concept is None:
                continue
            source_verb = resolve_signature(lexicon, instr_concept)
            if source_verb is None:
                continue
            theme_role = next(
                (r for r in source_verb.roles if r.name == "theme"), None)
            if theme_role is None:
                continue
            if not lexicon.types.is_subtype(
                    theme.entity_type, theme_role.type):
                continue
            new_roles = {"agent": agent_id, "theme": theme_id}
            if any(r.name == "instrument" for r in source_verb.roles):
                new_roles["instrument"] = instr_id
            pc: dict[tuple[str, str], object] = {}
            for eff in source_verb.effects:
                tid = new_roles.get(eff.target_role)
                if tid is None:
                    continue
                pc[(tid, eff.property)] = eff.value
            out.append(make_event(
                action=source_verb.lemma,
                roles=new_roles,
                caused_by=[ev.id],
                property_changes=pc,
            ))
        return out

    use_instrument.__name__ = "use_instrument"
    return use_instrument


def _location_of(trace: Trace, eid: str) -> str | None:
    """Walk `en`/`sur` ancestors from entity `eid` and return the id of the
    first location-typed container found. Returns None if none exists.
    Used by hazard-propagation rules that need 'where did this thing
    end up' — the hazard entity (shards, puddle) itself has no
    explicit placement, so we infer it from the origin entity's chain.
    """
    seen: set[str] = set()
    current = eid
    while current and current not in seen:
        seen.add(current)
        ent = trace.entities.get(current)
        if ent is None:
            return None
        if ent.entity_type == "location":
            return current
        next_id: str | None = None
        for r in trace.relations:
            if r.relation in ("en", "sur") and len(r.args) == 2 \
                    and r.args[0] == current:
                next_id = r.args[1]
                break
        current = next_id
    return None


def make_broken_fragile_creates_shards(lexicon: Lexicon):
    """Factory: when a fragile entity breaks, spawn a hazard fragment
    entity (e.g. vitropecetoj) per the broken concept's
    `transforms_on_break` property. Emits an `aperi` event whose
    `creates` carries the new entity; the realizer introduces it via
    the appearance-line mechanism.
    """
    def broken_fragile_creates_shards(trace: Trace, t: int) -> list[Event]:
        out: list[Event] = []
        for ev in trace.events[:t]:
            if ev.action != "rompiĝi":
                continue
            theme_id = ev.roles.get("theme")
            if not theme_id:
                continue
            theme = trace.entities.get(theme_id)
            if theme is None:
                continue
            concept = lexicon.concepts.get(theme.concept_lemma)
            if concept is None:
                continue
            transforms = concept.properties.get("transforms_on_break")
            if not transforms:
                continue
            target_lemma = transforms[0]
            target_concept = lexicon.concepts.get(target_lemma)
            if target_concept is None:
                continue
            new_id = f"{target_lemma}_from_{theme_id}"
            if new_id in trace.entities:
                continue
            new_ent = EntityInstance(
                id=new_id,
                concept_lemma=target_lemma,
                entity_type=target_concept.entity_type,
                properties={k: list(v)
                            for k, v in target_concept.properties.items()},
            )
            out.append(make_event(
                action="aperi",
                roles={"theme": new_id},
                caused_by=[ev.id],
                creates=[new_ent],
            ))
        return out

    broken_fragile_creates_shards.__name__ = "broken_fragile_creates_shards"
    return broken_fragile_creates_shards


def make_wet_liquid_container_tips(lexicon: Lexicon):
    """Factory: when a liquid falls, spawn a puddle (flako) per its
    concept's `transforms_on_spill`. Mirrors broken_fragile_creates_shards
    but triggers on fali of a liquid rather than rompiĝi of a container.
    """
    def wet_liquid_container_tips(trace: Trace, t: int) -> list[Event]:
        out: list[Event] = []
        for ev in trace.events[:t]:
            if ev.action != "fali":
                continue
            theme_id = ev.roles.get("theme")
            if not theme_id:
                continue
            theme = trace.entities.get(theme_id)
            if theme is None:
                continue
            concept = lexicon.concepts.get(theme.concept_lemma)
            if concept is None:
                continue
            transforms = concept.properties.get("transforms_on_spill")
            if not transforms:
                continue
            target_lemma = transforms[0]
            target_concept = lexicon.concepts.get(target_lemma)
            if target_concept is None:
                continue
            new_id = f"{target_lemma}_from_{theme_id}"
            if new_id in trace.entities:
                continue
            new_ent = EntityInstance(
                id=new_id,
                concept_lemma=target_lemma,
                entity_type=target_concept.entity_type,
                properties={k: list(v)
                            for k, v in target_concept.properties.items()},
            )
            out.append(make_event(
                action="aperi",
                roles={"theme": new_id},
                caused_by=[ev.id],
                creates=[new_ent],
            ))
        return out

    wet_liquid_container_tips.__name__ = "wet_liquid_container_tips"
    return wet_liquid_container_tips


def person_walks_on_hazard_falls(trace: Trace, t: int) -> list[Event]:
    """When an `aperi` event brings a hazard entity into a location,
    any person `en` that same location falls. Location is inferred from
    the causing event's theme — the hazard itself has no explicit
    placement (it was born from the rule), but its origin does.
    """
    out: list[Event] = []
    for ev in trace.events[:t]:
        if ev.action != "aperi":
            continue
        hazard_id = ev.roles.get("theme")
        if not hazard_id:
            continue
        hazard = trace.entities.get(hazard_id)
        if hazard is None or not hazard.properties.get("hazard"):
            continue
        # Trace location via the cause's theme (the thing that broke/spilled).
        origin_id: str | None = None
        by_id = {e.id: e for e in trace.events}
        for cid in ev.caused_by:
            cause = by_id.get(cid)
            if cause is not None and cause.roles.get("theme"):
                origin_id = cause.roles["theme"]
                break
        if origin_id is None:
            continue
        loc_id = _location_of(trace, origin_id)
        if loc_id is None:
            continue
        for rel in trace.relations:
            if rel.relation != "en" or len(rel.args) != 2:
                continue
            if rel.args[1] != loc_id:
                continue
            person_id = rel.args[0]
            person = trace.entities.get(person_id)
            if person is None or person.entity_type != "person":
                continue
            if _already_fired(trace, "fali", person_id):
                continue
            out.append(make_event(
                action="fali",
                roles={"theme": person_id},
                caused_by=[ev.id],
            ))
    return out


def fire_spreads_to_adjacent_flammables(trace: Trace, t: int) -> list[Event]:
    """When a thing is burning, flammable entities in direct containment
    contact (en/sur, either direction) also catch fire. Skips locations
    (a room doesn't 'burn' in this model) and already-burning things.

    Iterates level-by-level because the fixed-point engine re-runs rules
    until nothing fires — contact-graph distance from the seed gives
    cascade depth. A chain of four flammables touching pairwise yields
    depth 3 from the seed.
    """
    out: list[Event] = []
    for ev in trace.events[:t]:
        if ev.action != "bruli":
            continue
        theme_id = ev.roles.get("theme")
        if not theme_id:
            continue
        neighbors: set[str] = set()
        for r in trace.relations:
            if r.relation not in ("en", "sur") or len(r.args) != 2:
                continue
            if r.args[0] == theme_id:
                neighbors.add(r.args[1])
            elif r.args[1] == theme_id:
                neighbors.add(r.args[0])
        for nb_id in neighbors:
            nb = trace.entities.get(nb_id)
            if nb is None or nb.entity_type == "location":
                continue
            if not _has_value(nb.properties.get("flammability"), "flammable"):
                continue
            if _already_fired(trace, "bruli", nb_id):
                continue
            out.append(make_event(
                action="bruli",
                roles={"theme": nb_id},
                caused_by=[ev.id],
                property_changes={(nb_id, "presence"): "consumed"},
            ))
    return out


def carried_fragile_falls_when_carrier_falls(
        trace: Trace, t: int) -> list[Event]:
    """When a person falls, anything fragile they `havi` also falls —
    the carrier drops it. Skips items already broken so we don't
    synthesize falls for rubble."""
    out: list[Event] = []
    for ev in trace.events[:t]:
        if ev.action != "fali":
            continue
        theme_id = ev.roles.get("theme")
        if not theme_id:
            continue
        theme = trace.entities.get(theme_id)
        if theme is None or theme.entity_type != "person":
            continue
        for rel in trace.relations:
            if rel.relation != "havi" or len(rel.args) != 2:
                continue
            if rel.args[0] != theme_id:
                continue
            carried_id = rel.args[1]
            carried = trace.entities.get(carried_id)
            if carried is None:
                continue
            if not _has_value(carried.properties.get("fragility"), "fragile"):
                continue
            integrity = trace.property_at(carried_id, "integrity", t)
            if _has_value(integrity, "broken") or \
                    _has_value(integrity, "severed"):
                continue
            if _already_fired(trace, "fali", carried_id):
                continue
            out.append(make_event(
                action="fali",
                roles={"theme": carried_id},
                caused_by=[ev.id],
            ))
    return out


# Default rule list (no-lexicon rules only). For full coverage including
# instrument-use and the two transforms_on_* rules, build the rule list
# manually:
#
#   rules = DEFAULT_RULES + [
#       make_use_instrument(lex),
#       make_broken_fragile_creates_shards(lex),
#       make_wet_liquid_container_tips(lex),
#   ]
DEFAULT_RULES = [
    fragile_falls_breaks,
    hungry_eats_sated,
    container_falls_contents_fall,
    broken_container_releases_contents,
    person_walks_on_hazard_falls,
    carried_fragile_falls_when_carrier_falls,
    fire_spreads_to_adjacent_flammables,
]
