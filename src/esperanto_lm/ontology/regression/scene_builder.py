"""SceneBuilder fluent DSL + scene-pref/auto-pose/ownership build hooks.

A SceneBuilder composes a regression scene by chaining
location/person/target/relation/fakto calls. Slot names are local
labels (e.g. "actor", "scene") that resolve to entity ids; pass slot
names anywhere a method wants an entity reference. `build()` returns
a `(trace, scene_id, drive)` triple ready for the planner, or `None`
on failure.

Three composable build-time enrichers run in `build()`:
  - scene preferences (lit indoor scenes, rainy outdoor scenes)
  - auto-pose (random `sur sittable/lieable` for persons)
  - ownership distribution (random `havi(npc, item)`)
Each can be disabled per-builder with the corresponding `no_*` method.
"""
from __future__ import annotations

from typing import Any, Optional

from ..causal import EntityInstance, Trace, effect_changes, make_event
from ..dsl import run_dsl
from ..dsl.rules import (
    DEFAULT_DSL_DERIVATIONS, DEFAULT_DSL_RULES, RUNTIME_DERIVATIONS,
)


# Background scene preferences. Two flavors:
#
#   ("state", predicate, slot, value, probability):
#       When the scene location matches `predicate`, target
#       slot=value. `_apply_scene_preferences` introspects derivations
#       to find what entity+property would make the goal derive and
#       scatters the entity in. Indoor lit_state=luma materializes a
#       lampo via `indoor_lit_by_active_lamp`.
#
#   ("event", predicate, action_lemma, probability):
#       When the scene location matches `predicate`, fire `action_lemma`
#       at scene-init with role bindings derived from scene context
#       (location=scene_id). Cascade rules fire (rain_creates_puddle,
#       rain_wets_contents) so consequences are visible in the
#       starting state. Restricted to actions whose only fillable
#       role is `location` — no agent, no theme — which keeps the
#       binding unambiguous and the event "ambient" rather than
#       attributable.
#
# Probabilities < 1.0 keep both cases (dark indoor, dry outdoor)
# represented; the regression dispatcher picks drives uniformly so
# stimulus-response chains (preni-via-darkness, fetch-umbrella-from-
# rain) emerge from drive × backdrop combinations naturally.
SCENE_PREFERENCES: list[tuple] = [
    ("state", {"indoor_outdoor": "interna"}, "lit_state", "luma", 0.7),
    ("event", {"indoor_outdoor": "ekstera"}, "pluvi", 0.10),
]


def _standalone_target_concepts(lex):
    """Concepts eligible to be standalone fakto-targets in regression
    scenes: physical, non-person, non-location, NOT a category stub
    (meblo, lignaĵo, …), AND not appearing as some other concept's
    part. The parts filter excludes body parts (haro, dento, fingro,
    brako, …) which would otherwise produce incoherent prose like
    'Dorso estis en salono'."""
    parts_of_something = {
        p.concept for c in lex.concepts.values() for p in (c.parts or [])
    }
    return [
        c.lemma for c in lex.concepts.values()
        if lex.types.is_subtype(c.entity_type, "physical")
        and c.entity_type != "person"
        and not lex.types.is_subtype(c.entity_type, "location")
        and c.lemma not in parts_of_something
        and not getattr(c, "is_category_stub", False)
    ]


class SceneBuilder:
    """Compose a regression scene by chaining location/person/target/
    relation/fakto calls. Slot names are local labels (e.g. "actor",
    "scene") that resolve to entity ids; pass slot names anywhere the
    builder wants an entity reference."""

    def __init__(self, lex, rng):
        self.lex = lex
        self.rng = rng
        self.t = Trace()
        self.slots: dict[str, str] = {}
        self._scene_slot: Optional[str] = None
        self._used_names: set[str] = set()
        self._failed = False
        self._drive: Optional[tuple] = None
        # Auto-pose: build() runs maybe_seat / maybe_recline on every
        # person in the scene. Seeders can disable via no_auto_pose()
        # when posture/sleep_state matters for the test.
        self._auto_pose = True
        # Scene prefs: build() materializes ambient entities to satisfy
        # SCENE_PREFERENCES (lamps in indoor rooms, etc.). Disable via
        # no_scene_prefs() for stimulus-response seeders that need the
        # opposite (e.g. dark-lampo wants malluma).
        self._scene_prefs = True
        # Ownership: build() randomly assigns havi to non-actor persons
        # so the world feels populated and possession drives surface
        # peti naturally (actor wants item NPC owns → ask). Disable
        # via no_ownership() for seeders that need actor-only havi
        # bindings (count drives where actor's stash matters).
        self._ownership = True

    def _fail(self):
        self._failed = True

    def _resolve(self, slot_or_id) -> Optional[str]:
        if isinstance(slot_or_id, str) and slot_or_id in self.slots:
            return self.slots[slot_or_id]
        return slot_or_id

    def _unique_id(self, base: str) -> str:
        if base not in self.t.entities:
            return base
        suffix = 1
        while f"{base}_{suffix}" in self.t.entities:
            suffix += 1
        return f"{base}_{suffix}"

    def _location_matches_terrain(self, loc_lemma, terrains) -> bool:
        c = self.lex.concepts[loc_lemma]
        for terr in terrains:
            if terr == "land" and any(p.concept == "vojo" for p in c.parts):
                return True
            if terr == "rail" and any(p.concept == "relo" for p in c.parts):
                return True
            if terr == "water" and any(p.concept == "akvo" for p in c.parts):
                return True
            if terr == "air" and "ekstera" in c.properties.get(
                    "indoor_outdoor", []):
                return True
        return False

    def _add(self, concept_lemma, eid):
        from ..sampler import _add_entity_randomized
        try:
            _add_entity_randomized(
                self.t, concept_lemma, self.lex, self.rng, entity_id=eid)
            return True
        except (KeyError, ValueError):
            return False

    def _place(self, eid, container_slot):
        if container_slot is None:
            return True
        container_id = self._resolve(container_slot)
        if container_id is None:
            return False
        try:
            self.t.assert_relation("en", (eid, container_id), self.lex)
            return True
        except (KeyError, ValueError):
            return False

    # ---------- placement ----------

    def location(self, slot, *, is_scene=False, different_from=None,
                 terrain_compatible_with=None):
        """Pick a location concept and add it. `different_from` excludes
        a previously-bound slot's id. `terrain_compatible_with` filters
        to locations whose declared parts/properties match the terrains
        of an already-placed vehicle."""
        if self._failed:
            return self
        candidates = [
            l for l, c in self.lex.concepts.items()
            if self.lex.types.is_subtype(c.entity_type, "location")
        ]
        if different_from is not None:
            other = self._resolve(different_from)
            candidates = [l for l in candidates if l != other]
        if terrain_compatible_with is not None:
            veh_eid = self._resolve(terrain_compatible_with)
            veh_ent = self.t.entities.get(veh_eid) if veh_eid else None
            if veh_ent is None:
                self._fail()
                return self
            terrains = veh_ent.properties.get("terrain", [])
            candidates = [
                l for l in candidates
                if self._location_matches_terrain(l, terrains)
            ]
        if not candidates:
            self._fail()
            return self
        lemma = self.rng.choice(candidates)
        if not self._add(lemma, lemma):
            self._fail()
            return self
        self.slots[slot] = lemma
        if is_scene:
            self._scene_slot = slot
        return self

    def _named_actor(self, slot, concept_pool, *, in_, name=None):
        from ..sampler import PERSON_NAMES
        if self._failed:
            return self
        if not concept_pool:
            self._fail()
            return self
        concept = self.rng.choice(concept_pool)
        if name is None:
            available = [n for n in PERSON_NAMES if n not in self._used_names]
            if not available:
                self._fail()
                return self
            name = self.rng.choice(available)
        self._used_names.add(name)
        if not self._add(concept, name):
            self._fail()
            return self
        self.slots[slot] = name
        if not self._place(name, in_):
            self._fail()
        return self

    def person(self, slot, *, in_=None):
        """Pick a person concept and bind it to `slot` with a name from
        `PERSON_NAMES`. Names are unique within a build."""
        persons = [
            c.lemma for c in self.lex.concepts.values()
            if c.entity_type == "person"
        ]
        return self._named_actor(slot, persons, in_=in_)

    def animal(self, slot, *, in_=None):
        """Pick a non-person animate concept; entity id is the lemma."""
        if self._failed:
            return self
        animals = [
            c.lemma for c in self.lex.concepts.values()
            if self.lex.types.is_subtype(c.entity_type, "animate")
            and not self.lex.types.is_subtype(c.entity_type, "person")
        ]
        if not animals:
            self._fail()
            return self
        concept = self.rng.choice(animals)
        eid = self._unique_id(concept)
        if not self._add(concept, eid):
            self._fail()
            return self
        self.slots[slot] = eid
        if not self._place(eid, in_):
            self._fail()
        return self

    def target(self, slot, *, in_=None, where=None, same_concept_as=None,
               concept=None):
        """Pick a `_standalone_target_concepts` concept (physical, non-
        person, non-location, not-a-part). `where=lambda concept: ...`
        narrows further (e.g. only foods, only readables).

        `same_concept_as=<other_slot>`: reuse the concept lemma of the
        entity bound to `<other_slot>` rather than picking randomly.
        Useful when a scene needs multiple stacks of the same concept
        (e.g. apples-stash + apples-stash-b for a count drive).

        `concept=<lemma>`: skip random selection entirely and use the
        literal concept lemma (e.g. concept="monero" for a coin stack)."""
        if self._failed:
            return self
        if concept is not None:
            if concept not in self.lex.concepts:
                self._fail()
                return self
        elif same_concept_as is not None:
            other_eid = self.slots.get(same_concept_as)
            other_ent = self.t.entities.get(other_eid) if other_eid else None
            if other_ent is None:
                self._fail()
                return self
            concept = other_ent.concept_lemma
        else:
            candidates = _standalone_target_concepts(self.lex)
            if where is not None:
                candidates = [c for c in candidates
                              if where(self.lex.concepts[c])]
            if not candidates:
                self._fail()
                return self
            concept = self.rng.choice(candidates)
        eid = self._unique_id(concept)
        if not self._add(concept, eid):
            self._fail()
            return self
        self.slots[slot] = eid
        if not self._place(eid, in_):
            self._fail()
        return self

    def _place_under(self, target_lemma, root_id, idx, depth=0):
        """Place `target_lemma` in trace with its container chain strictly
        rooted at `root_id`. Mirrors `_ensure_placed`'s containment-graph
        walk but constrains the candidate set to entities already nested
        under `root_id` (transitively via `en`/`sur`/`havas_parton`),
        materializing intermediate containers as needed.

        Cascade-emerged concepts (flako, skribaĵo, …) are excluded from
        materialization — pre-placing them preempts the cascade that
        would otherwise introduce them, and "actor walks to a fresh
        puddle of beer" reads as nonsense. Set is introspected from
        rules via `cascade_emerged_concepts` so it tracks new cascades.

        At `depth=0` we PREFER a direct placement of the target into
        the root location. The planner's `samloke` derivations don't
        propagate through nested `en`/`sur` (an actor in the room
        isn't samloke with a liquid-in-bottle-on-table), so a deep
        chain hides the target from preni. We materialize a vessel
        only if the root has no direct containment for the target —
        the universal liquid_holder requirement is the typical case.

        Used by `scatter(pressure="away")` so the chain doesn't leak
        back into the scene location — `_ensure_placed` scans all of
        `trace.entities` and would happily pick the scene as the
        container if it fit."""
        from ..containment import (
            containers_for, containment_relation_for,
            required_fact_violations,
        )
        if target_lemma in self.t.entities:
            return True

        nested = self._entities_under(root_id)
        transient = self._cascade_emerged_concepts()

        candidates: list[tuple[str, str]] = []
        for cid in nested:
            cent = self.t.entities.get(cid)
            if cent is None or cent.entity_type == "person":
                continue
            rel = containment_relation_for(
                cent.concept_lemma, target_lemma, idx, self.lex)
            if rel is None:
                continue
            if required_fact_violations(
                    cent.concept_lemma, target_lemma, rel, idx, self.lex):
                continue
            candidates.append((cid, rel))

        # Prefer materializing a non-location intermediate (akvo →
        # glaso → tablo → kuirejo, not akvo directly under the room).
        non_location_candidate = any(
            self.lex.concepts[self.t.entities[cid].concept_lemma].entity_type
                != "location"
            for cid, _ in candidates
            if self.t.entities[cid].concept_lemma in self.lex.concepts
        )
        if not non_location_candidate and depth < 2:
            possible = [
                (c, r) for c, r in containers_for(
                    target_lemma, idx, self.lex)
                if c in self.lex.concepts
                and self.lex.concepts[c].entity_type != "location"
                and c not in transient
                and not required_fact_violations(
                    c, target_lemma, r, idx, self.lex)
            ]
            self.rng.shuffle(possible)
            for container_lemma, rel in possible:
                if not self._place_under(
                        container_lemma, root_id, idx, depth + 1):
                    continue
                if container_lemma in self.t.entities:
                    candidates.append((container_lemma, rel))
                    break

        if not candidates:
            return False
        non_location = [
            (c, r) for c, r in candidates
            if self.t.entities[c].concept_lemma in self.lex.concepts
            and self.lex.concepts[
                self.t.entities[c].concept_lemma].entity_type != "location"
        ]
        pick = (self.rng.choice(non_location) if non_location
                else self.rng.choice(candidates))
        if not self._add(target_lemma, target_lemma):
            return False
        try:
            self.t.assert_relation(pick[1], (target_lemma, pick[0]), self.lex)
        except (KeyError, ValueError):
            return False
        return True

    def _cascade_emerged_concepts(self) -> frozenset[str]:
        """Concepts introduced by cascade rules; excluded as auto-
        materialized containers in `_place_under`. Cached on the class
        so repeated scatter() calls don't re-introspect the rule set."""
        cached = getattr(SceneBuilder, "_cascade_emerged_cache", None)
        if cached is None:
            from ..dsl.introspect import cascade_emerged_concepts
            cached = cascade_emerged_concepts(DEFAULT_DSL_RULES)
            SceneBuilder._cascade_emerged_cache = cached
        return cached

    def _entities_under(self, root_id) -> set[str]:
        """Set of entity_ids transitively nested under `root_id` via
        any containment-style relation (`en`, `sur`, `havas_parton`).
        Includes the root itself."""
        out = {root_id}
        edges: dict[str, list[str]] = {}
        for r in self.t.relations:
            if r.relation in ("en", "sur", "havas_parton") and len(r.args) == 2:
                edges.setdefault(r.args[1], []).append(r.args[0])
        frontier = [root_id]
        while frontier:
            cur = frontier.pop()
            for child in edges.get(cur, ()):
                if child not in out:
                    out.add(child)
                    frontier.append(child)
        return out

    def scatter(self, slot, *, concept=None, where=None,
                pressure="any"):
        """Place a target concept via the containment graph.

        Picks a concept by literal `concept` lemma, by `where` filter
        (predicate on `Concept`), or any standalone target if neither
        given. Then asks containment.jsonl to materialize a plausible
        container chain — e.g. picks a `kuirejo` → `tablo` → `glaso` →
        `akvo` nesting rather than dumping `akvo` directly into the
        scene location.

        `pressure` controls the chain root:
          - `"near"`: root the chain at the scene location. The target
            ends up samloke with the actor.
          - `"away"`: pick a NEW location distinct from the scene whose
            containment-reach includes a viable concept. Forces
            locomotion+retrieval planning.
          - `"any"` (default): coin flip between near and away.

        Bound entity_id matches the concept lemma — collisions are rare
        in regression scenes (one target per concept). Returns self for
        chaining; sets `_failed` if no viable concept/location exists."""
        if self._failed:
            return self
        if self._scene_slot is None:
            self._fail()
            return self
        from ..containment import reachable_from, resolve_containment

        # Candidate concepts.
        if concept is not None:
            if concept not in self.lex.concepts:
                self._fail()
                return self
            candidates = [concept]
        else:
            candidates = _standalone_target_concepts(self.lex)
            if where is not None:
                candidates = [c for c in candidates
                              if where(self.lex.concepts[c])]
        if not candidates:
            self._fail()
            return self

        scene_id = self.slots[self._scene_slot]
        idx = resolve_containment(self.lex)

        if pressure == "any":
            pressure = self.rng.choice(("near", "away"))

        if pressure == "near":
            roots_to_try = [scene_id]
        else:  # away
            locations = [
                l for l, c in self.lex.concepts.items()
                if self.lex.types.is_subtype(c.entity_type, "location")
                and l != scene_id and l not in self.t.entities
            ]
            self.rng.shuffle(locations)
            roots_to_try = locations

        # Try each candidate root until one materializes a placement.
        # `reachable_from` is necessary but not sufficient — the
        # cascade-emerged filter in `_place_under` rejects flako-only
        # chains, so a location can be "reachable" yet have no real
        # path. Iterating until something sticks handles that.
        for loc in roots_to_try:
            if pressure != "near":
                reach = reachable_from(loc, idx, self.lex)
                hits = [c for c in candidates if c in reach]
                if not hits:
                    continue
                if loc not in self.t.entities and not self._add(loc, loc):
                    continue
                local_candidates = hits
            else:
                local_candidates = [
                    c for c in candidates
                    if c in reachable_from(loc, idx, self.lex)
                ]
                if not local_candidates:
                    continue
            self.rng.shuffle(local_candidates)
            for chosen in local_candidates:
                if self._place_under(chosen, loc, idx):
                    self.slots[slot] = chosen
                    return self
        self._fail()
        return self

    def vehicle(self, slot, *, in_=None):
        """Pick a vehicle concept (is_vehicle=yes)."""
        if self._failed:
            return self
        vehicles = [
            c.lemma for c in self.lex.concepts.values()
            if "yes" in c.properties.get("is_vehicle", [])
        ]
        if not vehicles:
            self._fail()
            return self
        concept = self.rng.choice(vehicles)
        eid = self._unique_id(concept)
        if not self._add(concept, eid):
            self._fail()
            return self
        self.slots[slot] = eid
        if not self._place(eid, in_):
            self._fail()
        return self

    def readable(self, slot, *, in_=None):
        """Pick a readable artifact (readability=legebla)."""
        if self._failed:
            return self
        readables = [
            c.lemma for c in self.lex.concepts.values()
            if "legebla" in c.properties.get("readability", [])
        ]
        if not readables:
            self._fail()
            return self
        concept = self.rng.choice(readables)
        eid = self._unique_id(concept)
        if not self._add(concept, eid):
            self._fail()
            return self
        self.slots[slot] = eid
        if not self._place(eid, in_):
            self._fail()
        return self

    # ---------- relations ----------

    def relation(self, name, *slot_args):
        """Assert a generic relation between slot-bound entities."""
        if self._failed:
            return self
        args = tuple(self._resolve(s) for s in slot_args)
        if any(a is None for a in args):
            self._fail()
            return self
        try:
            self.t.assert_relation(name, args, self.lex)
        except (KeyError, ValueError):
            self._fail()
        return self

    def havi(self, owner_slot, theme_slot):
        return self.relation("havi", owner_slot, theme_slot)

    def _maybe_place_on(self, person_slot, *, probability, attribute):
        """Shared core for maybe_seat / maybe_recline. Picks a concept
        whose properties carry `attribute=yes`, materializes it at the
        person's location, and asserts sur(person, concept). Skips if
        the person is already `sur` something — the seat/recline calls
        can be stacked in any order; only the first to fire takes."""
        if self._failed:
            return self
        if self.rng.random() >= probability:
            return self
        person_eid = self._resolve(person_slot)
        if person_eid is None:
            return self
        # Skip if actor is already on something (so adjacent
        # maybe_seat / maybe_recline calls don't double-place).
        if any(r.relation == "sur" and len(r.args) == 2
                and r.args[0] == person_eid
                for r in self.t.relations):
            return self
        person_loc = None
        for r in self.t.relations:
            if (r.relation == "en" and len(r.args) == 2
                    and r.args[0] == person_eid):
                person_loc = r.args[1]
                break
        if person_loc is None:
            return self
        # Filter candidates by what containment.jsonl actually permits
        # in person's location — without this, an outdoor scene could
        # spawn `en(lito, vilaĝo)` because we'd pick lito (lieable=yes)
        # without checking that beds belong indoors.
        from ..containment import (
            containment_relations_for, required_fact_violations,
            resolve_containment,
        )
        idx = resolve_containment(self.lex)
        person_loc_lemma = self.t.entities[person_loc].concept_lemma
        candidates = [
            c.lemma for c in self.lex.concepts.values()
            if "yes" in c.properties.get(attribute, [])
            and "en" in containment_relations_for(
                person_loc_lemma, c.lemma, idx, self.lex)
            and not required_fact_violations(
                person_loc_lemma, c.lemma, "en", idx, self.lex)
        ]
        if not candidates:
            return self
        concept = self.rng.choice(candidates)
        eid = self._unique_id(concept)
        if not self._add(concept, eid):
            return self
        if not self._place(eid, person_loc):
            return self
        try:
            self.t.assert_relation("sur", (person_eid, eid), self.lex)
        except (KeyError, ValueError):
            self._fail()
        return self

    def maybe_seat(self, person_slot, *, probability=0.25):
        """With `probability`, place a sittable artifact at the person's
        location and assert sur(person, sittable). The runtime posture
        derivation then sets the person to sidanta — the planner's
        locomotion needs to insert `stari`, but with narrative motivation
        ("she got up from the chair") rather than the unprovenanced
        `vekiĝi → stari` opening."""
        return self._maybe_place_on(
            person_slot, probability=probability, attribute="sittable")

    def maybe_recline(self, person_slot, *, probability=0.10):
        """With `probability`, place a lieable artifact (lito, sofo,
        kuseno) and assert sur(person, lieable). The lying-derivation
        sets posture=kuŝanta + sleep_state=dormanta, motivating the
        `vekiĝi → stari` opening as a person waking from bed and
        getting up before walking. Probability lower than seating
        because lying-down framings should be uncommon."""
        return self._maybe_place_on(
            person_slot, probability=probability, attribute="lieable")

    def konas(self, knower_slot, fakto_slot):
        return self.relation("konas", knower_slot, fakto_slot)

    def priskribas(self, text_slot, fakto_slot):
        return self.relation("priskribas", text_slot, fakto_slot)

    def fakto(self, slot, *, about):
        """Pre-create a fakto entity. `about=(relation, target_slot,
        location_slot)` makes the id mirror what `vidi_learns_en` would
        synthesize, so the planner finds it by id rather than creating
        a duplicate."""
        if self._failed:
            return self
        rel_name, target_slot, loc_slot = about
        target_eid = self._resolve(target_slot)
        loc_eid = self._resolve(loc_slot)
        if target_eid is None or loc_eid is None:
            self._fail()
            return self
        fakto_concept = self.lex.concepts.get("fakto")
        if fakto_concept is None:
            self._fail()
            return self
        fakto_id = f"fakto_from_{rel_name}_{target_eid}_{loc_eid}"
        if fakto_id not in self.t.entities:
            self.t.entities[fakto_id] = EntityInstance(
                id=fakto_id, concept_lemma="fakto",
                entity_type=fakto_concept.entity_type,
                properties={"pri_relacio": [rel_name]},
            )
            try:
                self.t.assert_relation(
                    "subjekto", (fakto_id, target_eid), self.lex)
                self.t.assert_relation(
                    "objekto", (fakto_id, loc_eid), self.lex)
            except (KeyError, ValueError):
                self._fail()
                return self
        self.slots[slot] = fakto_id
        return self

    # ---------- drive + finalization ----------

    def drive(self, kind, **slots):
        """Build the drive tuple. Kwarg values are slot names that get
        resolved to entity ids; literal strings (e.g. a slot name)
        are looked up in `self.slots`."""
        if self._failed:
            return self
        resolved = {}
        for k, v in slots.items():
            r = self._resolve(v)
            if r is None:
                self._fail()
                return self
            resolved[k] = r
        if kind == "knowledge":
            self._drive = (
                "knowledge", resolved["actor"],
                resolved["knower"], resolved["fakto"])
        elif kind == "location":
            self._drive = ("location", resolved["actor"], resolved["loc"])
        elif kind == "possession":
            self._drive = ("possession", resolved["actor"], resolved["item"])
        elif kind == "wearing":
            self._drive = (
                "wearing", resolved["actor"], resolved["garment"])
        elif kind == "count":
            # `concept` may be a slot name (in which case we resolve
            # to the bound entity's concept_lemma) OR a literal
            # concept lemma string. `target` is the integer count goal.
            actor_eid = resolved["actor"]
            concept_arg = slots["concept"]
            if concept_arg in self.slots:
                eid = self.slots[concept_arg]
                ent = self.t.entities.get(eid)
                if ent is None:
                    self._fail()
                    return self
                concept_lemma = ent.concept_lemma
            else:
                concept_lemma = concept_arg
            try:
                tgt = int(slots["target"])
            except (TypeError, ValueError):
                self._fail()
                return self
            self._drive = (
                "count", actor_eid, concept_lemma, tgt)
        elif kind == "give_count":
            # Altruism count: planner-actor (donor) plans to bring
            # recipient's count of `concept` to `target`. Surfaces doni
            # via the planner's altruism preference; partial transfer
            # kicks in when donor's stash > target.
            donor_eid = resolved["donor"]
            recipient_eid = resolved["recipient"]
            concept_arg = slots["concept"]
            if concept_arg in self.slots:
                eid = self.slots[concept_arg]
                ent = self.t.entities.get(eid)
                if ent is None:
                    self._fail()
                    return self
                concept_lemma = ent.concept_lemma
            else:
                concept_lemma = concept_arg
            try:
                tgt = int(slots["target"])
            except (TypeError, ValueError):
                self._fail()
                return self
            self._drive = (
                "give_count", donor_eid, recipient_eid,
                concept_lemma, tgt)
        elif kind == "more_than":
            # Comparative count: actor wants strictly more units of
            # `concept` than `reference` has. Target resolves at plan
            # time to count_owned(reference) + 1.
            actor_eid = resolved["actor"]
            reference_eid = resolved["reference"]
            concept_arg = slots["concept"]
            if concept_arg in self.slots:
                eid = self.slots[concept_arg]
                ent = self.t.entities.get(eid)
                if ent is None:
                    self._fail()
                    return self
                concept_lemma = ent.concept_lemma
            else:
                concept_lemma = concept_arg
            self._drive = (
                "more_than", actor_eid, concept_lemma, reference_eid)
        elif kind == "self_slot":
            self._drive = (
                "self_slot", resolved["actor"],
                resolved["slot"], resolved["value"])
        elif kind == "entity_slot":
            self._drive = (
                "entity_slot", resolved["actor"], resolved["target"],
                resolved["slot"], resolved["value"])
        else:
            self._fail()
        return self

    def set(self, slot_name, **props):
        """Set scene-init property values on a placed entity. Useful
        for forcing a varies=true slot (e.g. agent.thirst=soifa)."""
        if self._failed:
            return self
        eid = self._resolve(slot_name)
        ent = self.t.entities.get(eid) if eid else None
        if ent is None:
            self._fail()
            return self
        for prop, val in props.items():
            ent.set_property(prop, val)
        return self

    def build(self):
        if self._failed or self._scene_slot is None or self._drive is None:
            return None
        scene_id = self.slots.get(self._scene_slot)
        if scene_id is None:
            return None
        if self._scene_prefs:
            self._apply_scene_preferences()
        if self._auto_pose:
            self._apply_auto_pose()
        if self._ownership:
            self._distribute_ownership()
        return self.t, scene_id, self._drive

    def no_auto_pose(self):
        """Disable the build-time pass that randomly seats/reclines
        persons in the scene. Useful for tests or seeders where exact
        posture/sleep_state matters."""
        self._auto_pose = False
        return self

    def no_scene_prefs(self):
        """Disable the build-time pass that materializes ambient
        entities to satisfy `SCENE_PREFERENCES` (lamps in indoor
        rooms, etc.). Useful for seeders that need the opposite
        backdrop (e.g. a stimulus-response setup that requires the
        un-preferred state)."""
        self._scene_prefs = False
        return self

    def no_ownership(self):
        """Disable the build-time pass that randomly assigns havi to
        non-actor persons. Useful for count drives where the actor's
        stash count is the variable being driven."""
        self._ownership = False
        return self

    def _distribute_ownership(self, *, probability=0.30):
        """For each non-actor person in the scene, with `probability`,
        assert havi(person, item) for an in-scene item co-located with
        them. Makes the world feel populated and surfaces peti
        naturally when an actor's possession drive happens to target
        an NPC-owned item.

        "What can be owned" lives in `relations.jsonl` — havi.theme's
        arg_excludes/arg_not_part reject locations, persons, abstracts,
        and body parts at assertion time. We just enumerate co-located
        candidates and let `assert_relation` filter; failed assertions
        no-op."""
        if self._failed or self._drive is None:
            return
        actor_eid = self._drive[1] if len(self._drive) > 1 else None
        owned: set[str] = {
            r.args[1] for r in self.t.relations
            if r.relation == "havi" and len(r.args) == 2
        }
        for person_eid, person_ent in list(self.t.entities.items()):
            if person_eid == actor_eid:
                continue
            if not self.lex.types.is_subtype(
                    person_ent.entity_type, "person"):
                continue
            if self.rng.random() >= probability:
                continue
            person_loc = next(
                (r.args[1] for r in self.t.relations
                 if r.relation == "en" and len(r.args) == 2
                 and r.args[0] == person_eid),
                None)
            if person_loc is None:
                continue
            co_located = [
                r.args[0] for r in self.t.relations
                if r.relation == "en" and len(r.args) == 2
                and r.args[1] == person_loc
                and r.args[0] != person_eid
                and r.args[0] != actor_eid
                and r.args[0] not in owned
            ]
            if not co_located:
                continue
            self.rng.shuffle(co_located)
            for chosen in co_located:
                try:
                    self.t.assert_relation(
                        "havi", (person_eid, chosen), self.lex)
                    owned.add(chosen)
                    break
                except (KeyError, ValueError):
                    continue

    def _apply_scene_preferences(self):
        """Walk SCENE_PREFERENCES; for each entry whose predicate matches
        the scene location, roll the probability and apply.

        State variant: introspect derivations to find what entity+property
        would make the goal slot derive, materialize via scatter. Lamps
        in indoor rooms fall out of `lit_state=luma` because
        `indoor_lit_by_active_lamp` is the derivation.

        Event variant: fire the named action with `location=scene_id`,
        let cascade rules apply (rain_creates_puddle, rain_wets_contents).
        Restricted to actions whose only fillable role is `location` —
        agent/theme would need entity binding we can't derive from
        scene context.

        Failed applications don't kill the scene — the drive is the
        scene's reason for existing; backdrop is best-effort."""
        scene_id = self.slots[self._scene_slot]
        scene_ent = self.t.entities.get(scene_id)
        if scene_ent is None:
            return
        scene_concept = self.lex.concepts.get(scene_ent.concept_lemma)
        if scene_concept is None:
            return
        for entry in SCENE_PREFERENCES:
            kind = entry[0]
            predicate = entry[1]
            if not all(v in scene_concept.properties.get(k, [])
                       for k, v in predicate.items()):
                continue
            probability = entry[-1]
            if self.rng.random() >= probability:
                continue
            if kind == "state":
                _, _, slot, target_value, _ = entry
                self._apply_state_pref(scene_concept, slot, target_value)
            elif kind == "event":
                _, _, action_lemma, _ = entry
                self._apply_event_pref(scene_id, action_lemma)

    def _apply_state_pref(self, scene_concept, slot, target_value):
        """Materialize an entity that makes scene.slot=target_value
        derive. See `background_satisfiers` for the introspection."""
        from ..dsl.introspect import background_satisfiers
        options = background_satisfiers(
            slot, target_value, scene_concept,
            DEFAULT_DSL_DERIVATIONS, self.lex)
        if not options:
            return
        sat = self.rng.choice(options)
        bg_slot = self._unique_id(f"_pref_{sat.concept}")
        self.scatter(bg_slot, concept=sat.concept, pressure="near")
        if self._failed:
            self._failed = False
            return
        if bg_slot in self.slots and sat.set_properties:
            self.set(bg_slot, **sat.set_properties)
            if self._failed:
                self._failed = False

    def _apply_event_pref(self, scene_id, action_lemma):
        """Fire `action_lemma` at scene-init with location bound to
        scene. Restricted to single-role `location`-only actions —
        the natural shape for ambient/atmospheric events (pluvi,
        future neĝi/brili). Effects + cascade rules fire so visible
        consequences (puddles from rain) are in the starting state."""
        action = self.lex.actions.get(action_lemma)
        if action is None:
            return
        role_names = {r.name for r in action.roles}
        if role_names != {"location"}:
            return
        roles = {"location": scene_id}
        try:
            ev = make_event(
                action_lemma, roles=roles,
                property_changes=effect_changes(action_lemma, roles, self.lex))
            self.t.events.append(ev)
            self.t._event_ids.add(ev.id)
            # Run the engine to fire cascade rules (rain_creates_puddle,
            # rain_wets_contents) so the post-event state is settled
            # before planning starts.
            run_dsl(self.t, DEFAULT_DSL_RULES,
                    DEFAULT_DSL_DERIVATIONS + RUNTIME_DERIVATIONS, self.lex)
        except (KeyError, ValueError):
            pass

    def _apply_auto_pose(self):
        """Composable scene enrichment: for each person in the scene,
        roll once for an `sur` placement on a sittable or lieable
        artifact. Probabilities are low (most persons stay standing)
        so locomotion chains usually don't pay the `stari` cost; when
        they do, the prior `sur` gives the rendered scene narrative
        motivation. No-op for persons who already have an `sur`
        relation (e.g. an explicit `.maybe_seat()` already fired)."""
        person_slots = [
            slot for slot, eid in self.slots.items()
            if (ent := self.t.entities.get(eid)) is not None
            and self.lex.types.is_subtype(ent.entity_type, "person")
        ]
        for slot in person_slots:
            # Per-person rolls. Recline is rarer than seat — lying
            # down + asleep is a stronger framing.
            self.maybe_recline(slot, probability=0.08)
            self.maybe_seat(slot, probability=0.20)


def scene(lex, rng) -> SceneBuilder:
    """Start a regression scene. See `SceneBuilder`."""
    return SceneBuilder(lex, rng)
