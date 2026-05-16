# Ontology: event-calculus model

The causal engine is **discrete-time event calculus**, not property mutation.
Entities are created once; their state at any position `t` is derived by
walking events backward through `property_changes`.

## Core invariants

1. `EntityInstance.properties` is the state at the moment the entity entered
   the trace (scene-init for `created_at_event=None`, or the state at
   creation for mid-trace entities). **It is never mutated after the engine
   starts running.** Scene-setup code (the sampler) may use `set_property`;
   rules must not.

2. All state transitions live in `Event.property_changes`, keyed by
   `(entity_id, prop_name)`. To read current state at position `t`, call
   `Trace.property_at(eid, prop, t)` — it walks events `[0..t-1]` backward
   and returns the most recent change, falling back to the entity's initial
   `properties`.

3. Rules have signature `(trace, t) -> list[Event]` and must be pure: they
   read state via `trace.entities_at(t)` and `trace.property_at(...)` and
   return events. They never mutate `trace` directly. Rules that need
   lexicon access are factory closures (see `make_use_instrument`).

4. Events are memoized by `event.id` (content hash of action+roles+causes).
   An event fires at most once per trace. To allow re-firing as state
   changes, rules must ensure distinguishing context — typically via
   `caused_by`, since the triggering event's id is part of the hash.

## Creating new entities

Rules can return events with `creates=[EntityInstance(...)]`. The engine
assigns `created_at_event` and registers the entity in `trace.entities`.
The entity is invisible to `entities_at(t)` for any `t <= created_at_event`
and visible thereafter. The realizer introduces created entities via an
appearance phrase (e.g. "Aperis vitropecetoj.") rather than pre-mentioning
them in scene setup.

## Intrinsic verb effects

Verb `Effect` specs (in `data/ontology/actions/*.json`) describe what a
verb does to its theme/patient. The engine does NOT apply these
automatically — seed events (from the sampler, or from tests) must bake
them in via `make_event(..., property_changes=effect_changes(action, roles, lex))`.
This keeps event semantics self-contained: an event's `property_changes`
is the full list of state transitions attributed to it.

## Planner

The **forward planner** in `agent/forward_planner.py` is the preferred
planner — h_FF-relaxed-graph heuristic with EHC + weighted A*. It's
the default in `scripts/bench_samplers.py` (96.8% yield, ~130 ms/scene
on the 500-scene goal_sampler bench). The legacy backward chainer in
`agent/planner.py` remains for comparison; opt back in with
`USE_BACKWARD=1`.

Reachability and grounding are shared between planner and sampler via
`dsl/introspect.concept_models_slot`: a slot is meaningful for a
concept if it's declared in `concept.properties`, the slot is
`pervasive`, or a derivation can populate it given the concept's
parts. The planner uses this to drop nonsense groundings
(`kuiri(actor, kafo)` — kafo doesn't declare `cooking_state`); the
sampler uses the same predicate as a cheap pre-filter so it bails
before the spawn + h_FF cycle.

`plan_for_goal(..., max_states=0)` is the reachability-only mode: do
the setup + heuristic and return `[]` iff the goal is reachable,
`None` if not. Used by samplers as a viability gate.

## What not to do

- Don't mutate `entity.properties` from a rule. The engine relies on
  `properties` being the immutable-after-creation initial state.
- Don't dedupe events by `(rule_name, arg_key, t)` — that misses cascading
  re-emissions at different t values of the same synthesized event id.
  Per-event-id memoization is correct.
- Don't add a post-engine "apply effects" pass. Effects belong on the
  event itself (see `effect_changes`).
