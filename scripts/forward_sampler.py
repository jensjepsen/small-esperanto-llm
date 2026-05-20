"""Forward-elaboration sampler: grow a trace by sampling applicable
rules from the current world state, biased toward narrative-coherent
structure. Companion to the goal-regression sampler in
`esperanto_lm.ontology.regression`; both pipelines coexist.

Stage status (per the design brief):
  ✅ 1. Causal-graph + reservoir bookkeeping over forward-firing.
  ✅ 2. Feature extraction + weighted scorer (uniform when weights = 0).
  ⬜ 3. Tuned weights.
  ⬜ 4. Non-forward moves (elaboration, companion, obstacle, goal, backward).
  ⬜ 5. Depth-distribution targeting.
  ⬜ 6. Per-trace weight randomization.

Usage (manual smoke test):
    uv run python scripts/forward_sampler.py --steps 12 --seed 0

The sampler reuses existing types (Trace, Event, EntityInstance,
RelationAssertion) and rule firing (run_dsl). New artifact types
(CausalEdge, ReservoirEntry, TraceArtifact) live here only.
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Make the in-repo package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from esperanto_lm.ontology import (
    Trace, effect_changes, load_lexicon, make_event, realize_trace,
)
from esperanto_lm.ontology.dsl import compute_derived_state, run_dsl
from esperanto_lm.ontology.dsl.introspect import state_modifying_verbs
from esperanto_lm.ontology.agent.planner import _entity_property_values
from esperanto_lm.ontology.dsl.rules import (
    DEFAULT_DSL_RULES, RUNTIME_DERIVATIONS,
)
from esperanto_lm.ontology.schemas import (
    Action, IfPropertyPrecondition, MatchPrecondition,
    NotPropertyPrecondition, NotRelationPrecondition,
    RelationPrecondition,
)


# =================== artifact types ====================

@dataclass(frozen=True)
class CausalEdge:
    """Records that `producer` event's effect on `fact` satisfied
    `consumer` event's precondition on the same fact. `producer=None`
    means the fact came from initial state (scene setup), not from a
    prior event."""
    producer: Optional[str]   # event id, or None for scene setup
    consumer: str             # event id
    fact: tuple                # ("rel", relation_name, args) | ("prop", eid, slot, value)


@dataclass
class ReservoirEntry:
    """A novel effect not yet consumed as a precondition. `established_at`
    is the trace-position of the producing event (None for scene setup).
    `decay_at` is the position past which the entry is dropped."""
    fact: tuple
    established_at: Optional[int]
    decay_at: int


@dataclass
class Goal:
    """A wanted state — either a relation that should hold, or a
    property an entity should have. Injected by the goal_injection
    move; consumed by the `cashes_goal` scoring feature, which boosts
    candidates whose effects make the goal fact true."""
    agent: str           # entity id whose goal this is (narrative subject)
    fact: tuple          # ("rel", relation, args) | ("prop", eid, slot, value)
    injected_at: int     # trace position where the goal was added


@dataclass
class TraceArtifact:
    """Output of `generate_trace` — the trace plus the bookkeeping the
    forward sampler tracks alongside it. `move_log` parallels
    `trace.events` (entry per fired step); `causal_graph` collects
    every (producer → consumer, fact) edge encountered; `depth_histogram`
    counts paths-of-length-N in the causal DAG."""
    trace: Trace
    final_state_relations: list   # list[RelationAssertion]
    causal_graph: list[CausalEdge]
    depth_histogram: dict[int, int]
    move_log: list[str]
    sampled_weights: dict[str, float]
    goals: list[Goal] = field(default_factory=list)


# =================== config ====================

@dataclass
class SamplerConfig:
    """Tunable knobs for the forward sampler. Defaults mirror the
    starting-config in the design brief — they exist so the sampler
    runs end-to-end on first try, not because they're tuned."""
    move_probs: dict[str, float] = field(default_factory=lambda: {
        "forward_step": 0.85,
        "goal_injection": 0.15,
        # backward_step intentionally omitted — retroactive scene-state
        # rewriting was flagged as the highest-risk move; deferred.
    })
    feature_weights: dict[str, float] = field(default_factory=lambda: {
        "introduces_new_entity": 2.0,
        "changes_location": 1.5,
        "multi_agent": 1.0,
        "object_arity": 0.5,
        "recently_used": -2.0,
        "entity_revisit": -1.0,
        "cashes_reservoir": 3.0,
        "reservoir_distance": 0.3,
        "cashes_goal": 5.0,   # bigger than reservoir-cashing — goals
                              # are the narrative anchor we're chasing
        # Cross-trace freshness: pull candidates toward verb-theme
        # combinations this worker hasn't seen often. Compensates
        # for the empty-by-default `cashes_goal` and the inertness
        # of `cashes_reservoir` on state/perception verbs. Strong
        # enough to redirect distribution without dominating
        # goal-cashing when goals exist.
        "cross_freshness": 2.0,
        # Verb-level freshness — coarser companion to
        # cross_freshness. Drives diversity even for intransitives
        # that the (verb, theme) histogram can't grade
        # (morti/sensoifiĝi/pluvi/krii). Smaller weight than
        # cross_freshness because the signal is coarser, but
        # together they push unused verbs (malglui, malpurigi,
        # boji, ŝalti, ...) into the active pool.
        "verb_freshness": 1.0,
    })
    temperature: float = 1.0
    reservoir_decay_steps: int = 8
    recency_window: int = 3
    max_trace_length: int = 12
    max_active_goals: int = 2
    # Depth-distribution targeting (brief, §5). At trace start we
    # sample a target max-depth; while the current causal graph's
    # max-path is below it, the chain-pressure features
    # (cashes_reservoir, reservoir_distance) get multiplied by
    # `chain_pressure_boost` to push toward longer chains. Once the
    # target is reached, those features are zeroed for the rest of
    # the trace so the trailing structure doesn't keep stretching.
    target_depth_dist: Optional[Callable[[random.Random], int]] = None
    chain_pressure_boost: float = 2.5


# =================== applicable-step enumeration ====================

def _role_filler_candidates(role_spec, trace, lex, exclude,
                            derived=None):
    """Yield entity ids whose entity_type and properties match the
    role spec. Skips ids in `exclude` (already filling another role
    of the same action). Pure check — no subgoaling.

    Property matching consults both `ent.properties` (instance
    state from concept declarations and engine events) and
    `derived.properties` (runtime derivation outputs like
    `fragile_default_integrity_tuta` setting integrity=tuta on
    fragile entities whose concept doesn't declare it). Either
    source can satisfy the role's expected values; the union of
    the two is checked against `expected_vals`.

    `slot.unmarked` is NOT consulted as a fallback for absent
    slots: that would phantom-match ŝalti(theme.power_state=
    neaktiva) against a key whose concept doesn't model
    power_state. The schema's intent is that absence = "doesn't
    model the slot" = no match.

    Negative gating (e.g. "theme must NOT be currently a part") is
    expressed as a `NotPropertyPrecondition` on the action and
    enforced in `_action_preconds_satisfied`, NOT here."""
    for eid, ent in trace.entities.items():
        if eid in exclude:
            continue
        if ent.destroyed_at_event is not None:
            continue
        if not lex.types.is_subtype(ent.entity_type, role_spec.type):
            continue
        if role_spec.properties:
            ok = True
            for slot, expected_vals in role_spec.properties.items():
                actual_vals = _entity_property_values(
                    ent, slot, trace=trace, derived=derived, lex=lex)
                if not any(v in actual_vals for v in expected_vals):
                    ok = False
                    break
            if not ok:
                continue
        yield eid


def _bind_roles(action: Action, trace, lex, rng, max_per_action=8,
                derived=None):
    """Yield role-binding dicts for `action`. Greedy random sampling:
    for each role, pick uniformly from the candidate filler set; reject
    bindings that put the same entity in two roles. Returns up to
    `max_per_action` distinct binding sets (caps combinatorics on
    actions with many roles)."""
    seen: set[tuple] = set()
    out = []
    attempts = 0
    while len(out) < max_per_action and attempts < max_per_action * 4:
        attempts += 1
        roles: dict[str, str] = {}
        ok = True
        for role_spec in action.roles:
            cands = list(_role_filler_candidates(
                role_spec, trace, lex, exclude=set(roles.values()),
                derived=derived))
            if not cands:
                ok = False
                break
            roles[role_spec.name] = rng.choice(cands)
        if not ok:
            continue
        if len(set(roles.values())) != len(roles):
            continue
        key = tuple(sorted(roles.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(roles)
    return out


def _has_relation(rel_name, args, trace, derived):
    """True if the relation holds in asserted state OR in derived state."""
    for r in trace.relations:
        if r.relation == rel_name and tuple(r.args) == tuple(args):
            return True
    return derived.has_relation(rel_name, tuple(args))


def _action_preconds_satisfied(action: Action, roles, trace, derived, lex):
    """All preconditions hold in the current trace + derived state.
    Pure check — no subgoaling, no plan fabrication."""
    for pc in action.preconditions:
        if isinstance(pc, RelationPrecondition):
            args = [roles.get(rn) for rn in pc.roles]
            if any(a is None for a in args):
                return False
            if not _has_relation(pc.rel, tuple(args), trace, derived):
                return False
        elif isinstance(pc, IfPropertyPrecondition):
            eid = roles.get(pc.role)
            if eid is None:
                return False
            ent = trace.entities.get(eid)
            if ent is None:
                return False
            cur = ent.properties.get(pc.if_property, [])
            if pc.if_value in cur:
                target = ent.properties.get(pc.then_property, [])
                if pc.then_value not in target:
                    return False
        elif isinstance(pc, MatchPrecondition):
            ent_a = trace.entities.get(roles.get(pc.role_a))
            ent_b = trace.entities.get(roles.get(pc.role_b))
            if ent_a is None or ent_b is None:
                return False
            vals_a = set(ent_a.properties.get(pc.slot_a, []))
            vals_b = set(ent_b.properties.get(pc.slot_b, []))
            if not (vals_a & vals_b):
                return False
        elif isinstance(pc, NotRelationPrecondition):
            # Reject if the relation currently holds for the given
            # role bindings (asserted or derived). Mirror of the
            # positive RelationPrecondition path.
            args = [roles.get(rn) for rn in pc.roles]
            if any(a is None for a in args):
                return False
            if _has_relation(pc.rel, tuple(args), trace, derived):
                return False
        elif isinstance(pc, NotPropertyPrecondition):
            # Reject if the role entity currently holds the
            # forbidden value. Consults `_entity_property_values`
            # which merges `trace.property_at` (event-driven
            # current state — catches `malŝalti` having flipped
            # power_state to neaktiva already), instance state,
            # and derived state. Absence in all three passes the
            # gate.
            eid = roles.get(pc.role)
            if eid is None:
                return False
            ent = trace.entities.get(eid)
            if ent is None:
                return False
            actual = _entity_property_values(
                ent, pc.property, trace=trace, derived=derived, lex=lex)
            if pc.value in actual:
                return False
    return True


def enumerate_applicable_steps(trace, lex, derived, rng,
                               *, rules=None, derivations=None):
    """Return list of (action_name, roles_dict) pairs whose
    preconditions are satisfied right now AND whose effects wouldn't
    be redundant in current state. Random binding-sampling caps
    combinatorics. The redundancy check is a per-verb effect-lookup
    (no simulation) — see `_is_redundant`."""
    no_ops = _no_op_verbs(lex, rules or DEFAULT_DSL_RULES)
    out = []
    for action in lex.actions.values():
        if not action.roles:
            continue
        if action.lemma in no_ops:
            continue
        # `cascade_only` verbs are reactive intransitives (morti,
        # rompiĝi, fali, satiĝi, sensoifiĝi, bruli) emitted by DSL
        # cascades, not picked as deliberate forward-step actions.
        if getattr(action, "cascade_only", False):
            continue
        for roles in _bind_roles(
                action, trace, lex, rng, derived=derived):
            if not _action_preconds_satisfied(
                    action, roles, trace, derived, lex):
                continue
            if _is_redundant(action.lemma, roles, trace, derived):
                continue
            out.append((action.lemma, roles))
    return out


# Cache: (id(lex), id(rules)) -> frozenset of verb lemmas with no
# downstream state effect. Computed by introspecting DSL rules and
# action.effects; see `state_modifying_verbs`.
_NO_OP_VERBS_CACHE: dict[tuple[int, int], frozenset[str]] = {}


# Cross-trace freshness reservoirs. Per-process histograms of
# (verb, theme_concept) AND bare-verb counts across all traces
# sampled in this worker so far. Two granularities:
#   - `_CROSS_TRACE_HIST`: keyed on (verb, theme_concept). Pushes
#     the sampler toward unseen verb-theme combinations (so a
#     saturated `preni(ŝlosilo)` opens room for `preni(termometro)`).
#     Inert for intransitives (no theme to key on).
#   - `_CROSS_TRACE_VERB_HIST`: bare verb counts. Pushes the
#     sampler toward underused verbs as a whole — covers
#     intransitives the (verb, theme) histogram can't touch
#     (morti, sensoifiĝi, pluvi, ...) and reins in over-used
#     transitives that have already exhausted their theme space.
# Both are per-worker (no cross-worker sync); the union of
# per-worker exploration still covers the verb × theme space.
_CROSS_TRACE_HIST: dict[tuple[str, str], int] = {}
_CROSS_TRACE_VERB_HIST: dict[str, int] = {}


def _cross_trace_freshness_key(action_name: str, roles: dict,
                                trace) -> tuple[str, str] | None:
    """The (verb, theme_concept) histogram key for one (action,
    roles) candidate. Returns None when the candidate has no theme
    role bound (intransitives with only an agent, e.g. krii/morti
    — those would all collapse to (action, None))."""
    theme_eid = roles.get("theme")
    if not theme_eid:
        return None
    ent = trace.entities.get(theme_eid)
    if ent is None:
        return None
    return (action_name, ent.concept_lemma)


def _update_cross_trace_histogram(trace) -> None:
    """Walk the just-completed trace's events and increment both
    cross-trace histograms. Called at the end of `generate_trace`."""
    for ev in trace.events:
        _CROSS_TRACE_VERB_HIST[ev.action] = (
            _CROSS_TRACE_VERB_HIST.get(ev.action, 0) + 1)
        key = _cross_trace_freshness_key(
            ev.action, dict(ev.roles or {}), trace)
        if key is not None:
            _CROSS_TRACE_HIST[key] = _CROSS_TRACE_HIST.get(key, 0) + 1


def _no_op_verbs(lex, rules) -> frozenset[str]:
    key = (id(lex), id(rules))
    cached = _NO_OP_VERBS_CACHE.get(key)
    if cached is not None:
        return cached
    smv = state_modifying_verbs(rules, lex)
    no_ops = frozenset(
        a.lemma for a in lex.actions.values() if a.lemma not in smv)
    _NO_OP_VERBS_CACHE[key] = no_ops
    return no_ops


# Predicate per state-modifying verb: returns True iff the action's
# effects are already in the current state ("redundant" — firing
# would be a no-op for THIS binding). Pure no-op verbs (kanti, ami,
# plori …) are dropped earlier in `enumerate_applicable_steps` via
# the introspected `_no_op_verbs` set, so they don't appear here.
def _is_redundant(action_name, roles, trace, derived):
    def has(rel, args):
        return _has_relation(rel, args, trace, derived)

    def prop(eid, slot):
        ent = trace.entities.get(eid)
        if ent is None:
            return None
        v = ent.properties.get(slot)
        return v[0] if v else None

    a = roles.get("agent")
    th = roles.get("theme")
    rc = roles.get("recipient")
    ins = roles.get("instrument")
    loc = roles.get("location")
    dest = roles.get("destination")

    # --- transfer / acquisition verbs ---
    if action_name in ("preni", "peti", "kapti"):
        return has("havi", (a, th))
    if action_name == "aĉeti":
        return has("havi", (a, th)) and has("havi", (rc, ins))
    if action_name == "vendi":
        return has("havi", (rc, th)) and has("havi", (a, ins))
    if action_name == "doni":
        return has("havi", (rc, th))
    # --- movement / placement ---
    if action_name == "meti":
        # meti adds en(theme, location) for type=location; sur otherwise.
        loc_ent = trace.entities.get(loc)
        if loc_ent is None:
            return False
        if loc_ent.entity_type == "location":
            return has("en", (th, loc))
        return has("sur", (th, loc))
    if action_name in ("iri", "veni", "kuri", "naĝi", "flugi", "veturi"):
        return has("apud", (a, dest)) or has("en", (a, dest))
    if action_name == "eniri":
        return has("en", (a, th))
    # --- state-toggling verbs ---
    if action_name == "malfermi":
        return prop(th, "openness") == "malfermita"
    if action_name == "fermi":
        return prop(th, "openness") == "fermita"
    if action_name == "ŝlosi":
        return prop(th, "lock_state") == "ŝlosita"
    if action_name == "malŝlosi":
        return prop(th, "lock_state") == "malŝlosita"
    if action_name == "ŝalti":
        return prop(th, "power_state") == "aktiva"
    if action_name == "malŝalti":
        return prop(th, "power_state") == "neaktiva"
    if action_name == "purigi":
        return prop(th, "cleanliness") == "pura"
    if action_name == "sekigi":
        return prop(th, "wetness") == "seka"
    if action_name == "akvumi":
        return prop(th, "water_state") == "akvumita"
    if action_name == "plenigi":
        return prop(th, "fullness") == "plena"
    if action_name == "malplenigi":
        return prop(th, "fullness") == "malplena"
    if action_name == "surmeti":
        return has("vestita", (a, th))
    if action_name == "demeti":
        return not has("vestita", (a, th))
    # --- posture / sleep ---
    if action_name == "stari":
        return prop(a, "posture") == "staranta"
    if action_name == "sidi":
        return prop(a, "posture") == "sidanta"
    if action_name == "kuŝi":
        return prop(a, "posture") == "kuŝanta"
    if action_name == "dormi":
        return prop(a, "sleep_state") == "dormanta"
    if action_name == "vekiĝi":
        return prop(a, "sleep_state") == "vekita"
    return False


# =================== feature extraction & scoring ====================

# Relations that are pure derivations from spatial/typing structure
# (samloke = shared container; scias_lokon = en-derived knowledge).
# These get re-derived on almost every state change, flooding the
# reservoir and making `cashes_reservoir` fire on any verb with a
# samloke precondition — i.e. most of them. Filter them out so the
# reservoir only carries facts that came from genuine narrative
# effects (havi changes from preni/doni, en changes from movement,
# openness/lock_state changes from open/lock verbs, etc.).
RESERVOIR_NOISE_RELATIONS = {"samloke", "scias_lokon"}


def _is_reservoir_worthy(fact) -> bool:
    """True iff a state-fact is worth tracking in the reservoir.
    Drops derivation-noise (samloke etc.) — see RESERVOIR_NOISE_RELATIONS."""
    if fact[0] == "rel" and fact[1] in RESERVOIR_NOISE_RELATIONS:
        return False
    return True


# Hardcoded action → fact-it-establishes table. Cheap alternative to
# fork-and-simulate per candidate. Each entry says "this action+role-
# binding pattern, when fired, asserts this kind of fact". The lookup
# is consulted once per (candidate, goal) pair to decide if the
# candidate satisfies the goal — no engine call.
#
# For relations:  ("havi", agent_role, theme_role) etc.
# Actions outside this table never directly satisfy a goal. (They can
# still indirectly help, but the cheap signal is just terminal-step
# cashing — the same semantics the brief specifies for cashes_goal.)
_GOAL_CASHING_TABLE = {
    "preni":  ("havi", "agent", "theme"),
    "peti":   ("havi", "agent", "theme"),
    "kapti":  ("havi", "agent", "theme"),
    "aĉeti":  ("havi", "agent", "theme"),
    "doni":   ("havi", "recipient", "theme"),
    "vendi":  ("havi", "recipient", "theme"),
    "eniri":  ("en", "agent", "theme"),
    "meti":   ("en", "theme", "location"),
}


def _candidate_satisfies_goal(action_name, roles, goals) -> bool:
    """True iff firing (action, roles) would directly establish any
    pending goal's relation. Cheap: pure dict lookup + role binding
    comparison; no simulation. Only catches terminal-step cashing —
    the candidate that COMPLETES the goal — which is the semantics
    the brief specifies for `cashes_goal`."""
    if not goals:
        return False
    schema = _GOAL_CASHING_TABLE.get(action_name)
    if schema is None:
        return False
    rel_name, role_a, role_b = schema
    a = roles.get(role_a)
    b = roles.get(role_b)
    if a is None or b is None:
        return False
    target = ("rel", rel_name, (a, b))
    return any(g.fact == target for g in goals)


def _extract_features(action_name, roles, trace, history, reservoir, lex,
                       *, goals=None):
    """Compute the symbolic feature dict for one (action, bindings)
    candidate. Cheap; no derivation closure."""
    action = lex.actions.get(action_name)
    if action is None:
        return {}
    role_eids = list(roles.values())
    feats: dict[str, float] = {}

    # introduces_new_entity: would the rule create a new entity?
    feats["introduces_new_entity"] = 0.0  # forward sampler doesn't synthesize entities yet

    # changes_location: action affects en/sur for its agent.
    feats["changes_location"] = float(
        action_name in {"iri", "veni", "kuri", "naĝi", "flugi", "veturi",
                        "eniri", "sekvi", "voki"})

    # multi_agent: more than one animate role bound.
    n_animate = sum(
        1 for eid in role_eids
        if (e := trace.entities.get(eid)) is not None
        and lex.types.is_subtype(e.entity_type, "animate"))
    feats["multi_agent"] = float(n_animate >= 2)

    # object_arity: number of role bindings (normalized).
    feats["object_arity"] = len(role_eids) / 4.0

    # recently_used: this action fired in the recency window?
    feats["recently_used"] = float(action_name in history.recent_actions)

    # entity_revisit: how many of the bound entities have been touched
    # in this trace (counts; capped).
    revisit = sum(history.entity_touches.get(eid, 0) for eid in role_eids)
    feats["entity_revisit"] = min(revisit, 5) / 5.0

    # cashes_reservoir: does any precondition match a current reservoir
    # entry's fact? Plus reservoir_distance for the cashed entry.
    cashed = False
    distance = 0
    for pc in action.preconditions:
        if isinstance(pc, RelationPrecondition):
            args = tuple(roles.get(rn) for rn in pc.roles)
            for entry in reservoir.entries:
                if entry.fact == ("rel", pc.rel, args):
                    cashed = True
                    if entry.established_at is not None:
                        distance = max(
                            distance,
                            len(trace.events) - entry.established_at)
                    break
            if cashed:
                break
    feats["cashes_reservoir"] = float(cashed)
    feats["reservoir_distance"] = float(distance) / 8.0

    # cashes_goal: cheap dict lookup against `_GOAL_CASHING_TABLE`.
    # No fork, no run_dsl — just "would this verb's effect type, with
    # these role bindings, match any goal's fact?".
    feats["cashes_goal"] = float(
        _candidate_satisfies_goal(action_name, roles, goals or []))

    # cross_freshness: how often have we sampled this (verb,
    # theme_concept) combo in prior traces this worker handled?
    # Reciprocal-of-1+count keeps the feature in [0, 1]: unseen
    # combos get 1.0, the 99th sample of preni(ŝlosilo) gets 0.01.
    # Intransitives or theme-less verbs get a key of None and skip
    # this feature (their variety is captured by `verb_freshness`).
    fresh_key = _cross_trace_freshness_key(action_name, roles, trace)
    if fresh_key is not None:
        seen = _CROSS_TRACE_HIST.get(fresh_key, 0)
        feats["cross_freshness"] = 1.0 / (1.0 + seen)
    else:
        feats["cross_freshness"] = 0.0

    # verb_freshness: bare-verb-level freshness. Same reciprocal
    # decay but on the verb-only histogram, so intransitives
    # (morti, sensoifiĝi, krii, ...) get a freshness signal too,
    # and over-used transitives that have exhausted their theme
    # space still see their `verb_freshness` collapsing toward 0.
    verb_seen = _CROSS_TRACE_VERB_HIST.get(action_name, 0)
    feats["verb_freshness"] = 1.0 / (1.0 + verb_seen)
    return feats


def _score(features, weights):
    return sum(weights.get(f, 0.0) * v for f, v in features.items())


def _softmax_pick(items, scores, temperature, rng):
    """Sample one index from items via softmax(scores / τ). At
    temperature=0 (or very small), ties go to first. With all-zero
    weights, falls back to uniform."""
    import math
    if not items:
        return None
    if temperature <= 0:
        best = max(range(len(scores)), key=lambda i: scores[i])
        return items[best]
    max_s = max(scores) if scores else 0.0
    weights = [math.exp((s - max_s) / temperature) for s in scores]
    total = sum(weights)
    if total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if r <= acc:
            return it
    return items[-1]


# =================== bookkeeping ====================

@dataclass
class Reservoir:
    entries: list[ReservoirEntry] = field(default_factory=list)

    def add(self, fact, established_at, decay_steps):
        self.entries.append(ReservoirEntry(
            fact=fact,
            established_at=established_at,
            decay_at=(established_at or 0) + decay_steps))

    def consume(self, fact):
        """Mark the first matching entry consumed — drops it from the
        reservoir. Returns the entry (or None)."""
        for i, e in enumerate(self.entries):
            if e.fact == fact:
                return self.entries.pop(i)
        return None

    def evict_old(self, current_pos):
        self.entries = [e for e in self.entries if e.decay_at > current_pos]

    def invalidate(self, fact):
        """Drop entries whose fact was negated."""
        self.entries = [e for e in self.entries if e.fact != fact]


@dataclass
class History:
    """Lightweight running summary used by feature extraction."""
    recent_actions: list[str] = field(default_factory=list)
    entity_touches: dict[str, int] = field(default_factory=dict)
    recency_window: int = 3

    def record(self, action_name, roles):
        self.recent_actions.append(action_name)
        if len(self.recent_actions) > self.recency_window:
            self.recent_actions = self.recent_actions[-self.recency_window:]
        for eid in roles.values():
            self.entity_touches[eid] = self.entity_touches.get(eid, 0) + 1


# =================== firing & novelty tracking ====================

def _state_facts(trace, derived):
    """Snapshot of (relation, args) tuples and (eid, slot, value)
    triples currently true. Used to diff before/after firing."""
    rels = {("rel", r.relation, tuple(r.args)) for r in trace.relations}
    rels.update(("rel", name, args) for (name, args) in derived.relations)
    props: set[tuple] = set()
    pos = len(trace.events)
    for eid, ent in trace.entities.items():
        for slot, vals in ent.properties.items():
            for v in vals:
                props.add(("prop", eid, slot, v))
        # also derived
    for (eid, slot), v in derived.properties.items():
        props.add(("prop", eid, slot, v))
    return rels | props


def _causal_edges_for_event(action, roles, trace, prior_facts, edges_out):
    """For each precondition of the firing action, find which prior
    fact in `prior_facts` satisfies it. Records the satisfying-side
    as the edge's `producer`. The producer event is whichever event
    most recently established the fact; we use the simpler proxy of
    the prior-state membership and look up the producer in the trace's
    event log."""
    consumer_id = trace.events[-1].id
    for pc in action.preconditions:
        if isinstance(pc, RelationPrecondition):
            args = tuple(roles.get(rn) for rn in pc.roles)
            fact = ("rel", pc.rel, args)
            if not _is_reservoir_worthy(fact):
                continue
            if fact in prior_facts:
                producer = _find_producer_event(trace, fact)
                edges_out.append(CausalEdge(
                    producer=producer, consumer=consumer_id, fact=fact))


def _find_producer_event(trace, fact):
    """Walk back through events to find the one that established
    `fact` (added the relation or set the property). Returns None if
    the fact existed at scene setup."""
    if fact[0] == "rel":
        _, rel_name, args = fact
        # No per-event audit log in Trace today — we approximate by
        # checking if the relation exists in setup_relations (None) or
        # was added by the most recent event whose role-values overlap
        # the relation args.
        for ev in reversed(trace.events[:-1]):
            ev_referents = set(v for v in ev.roles.values() if isinstance(v, str))
            if set(args) & ev_referents and rel_name in {"havi", "en", "sur", "konas", "apud"}:
                # Heuristic — the precise audit would need engine-level
                # tracking. Good enough for stage-1 instrumentation.
                return ev.id
    elif fact[0] == "prop":
        _, eid, slot, value = fact
        for ev in reversed(trace.events[:-1]):
            for (e, s), v in ev.property_changes.items():
                if e == eid and s == slot and v == value:
                    return ev.id
    return None


def _depth_histogram(edges, events):
    """Count paths of length k in the causal DAG. Iterative pass in
    chronological order — producers are always earlier than consumers,
    so the memo is populated before lookup."""
    by_consumer: dict[str, list[str]] = {}
    for e in edges:
        if e.producer is None:
            continue
        by_consumer.setdefault(e.consumer, []).append(e.producer)
    memo: dict[str, int] = {}
    hist: dict[int, int] = {}
    for ev in events:
        producers = by_consumer.get(ev.id, [])
        d = (1 + max((memo.get(p, 0) for p in producers), default=0)
             if producers else 0)
        memo[ev.id] = d
        hist[d] = hist.get(d, 0) + 1
    return hist


# =================== main loop ====================

def default_target_depth_dist(rng: random.Random) -> int:
    """Power-law-ish: most mass on 2-7, light tail to 12. Matches the
    brief's suggested shape — short chains common, long chains rare
    but present so the corpus has variety."""
    if rng.random() < 0.85:
        return rng.randint(2, 7)
    return rng.randint(8, 12)


def _current_max_depth(edges, events) -> int:
    """Longest causal path length anchored at any event. Iterative
    walk in event order — events are appended chronologically, so a
    consumer's producers are always earlier and already memoized
    when we get to the consumer. Avoids recursion-depth blowups
    on long chains."""
    if not edges:
        return 0
    by_consumer: dict[str, list[str]] = {}
    for e in edges:
        if e.producer is not None:
            by_consumer.setdefault(e.consumer, []).append(e.producer)
    memo: dict[str, int] = {}
    for ev in events:
        producers = by_consumer.get(ev.id, [])
        if not producers:
            memo[ev.id] = 0
        else:
            memo[ev.id] = 1 + max(memo.get(p, 0) for p in producers)
    return max(memo.values(), default=0)


def _depth_adjusted_weights(
    base: dict[str, float], current_depth: int, target_depth: int,
    boost: float,
) -> dict[str, float]:
    """Apply depth-targeting to the chain-pressure features. Below
    target depth, scale `cashes_reservoir` and `reservoir_distance`
    by `boost` to push toward chain formation. At/above target,
    zero them so subsequent steps don't keep over-stretching the
    chain."""
    out = dict(base)
    if current_depth >= target_depth:
        out["cashes_reservoir"] = 0.0
        out["reservoir_distance"] = 0.0
    else:
        out["cashes_reservoir"] = base.get("cashes_reservoir", 0.0) * boost
        out["reservoir_distance"] = base.get("reservoir_distance", 0.0) * boost
    return out


def _pick_move(config, n_active_goals, rng) -> str:
    """Sample a move from `config.move_probs`. Goal-injection skipped
    when at the active-goal cap so we don't pile up unsatisfiable
    wants. Forward-step always available as the fallback."""
    weights = []
    moves = []
    for move, prob in config.move_probs.items():
        if move == "goal_injection" and n_active_goals >= config.max_active_goals:
            continue
        moves.append(move)
        weights.append(prob)
    if not moves:
        return "forward_step"
    total = sum(weights)
    if total <= 0:
        return "forward_step"
    r = rng.random() * total
    acc = 0.0
    for m, w in zip(moves, weights):
        acc += w
        if r <= acc:
            return m
    return moves[-1]


def _propose_goal(trace, lex, rng) -> Optional[Goal]:
    """Pick an animate entity, pick a desirable target state it doesn't
    already have. Returns None if the scene has nothing to want.

    Goal flavors (uniform pick when both apply):
      - havi: target an item present in the scene the agent doesn't own.
      - location: target a room different from the agent's current `en`.
    Both are chain-rich (require traversal/interaction to satisfy)."""
    animate = [
        eid for eid, e in trace.entities.items()
        if e.destroyed_at_event is None
        and lex.types.is_subtype(e.entity_type, "animate")
    ]
    if not animate:
        return None
    rng.shuffle(animate)
    for agent_id in animate:
        # havi target: any non-owned, non-location, non-person entity.
        owned = {r.args[1] for r in trace.relations
                 if r.relation == "havi" and r.args[0] == agent_id}
        candidates = [
            eid for eid, e in trace.entities.items()
            if e.destroyed_at_event is None
            and not lex.types.is_subtype(e.entity_type, "location")
            and e.entity_type != "person"
            and eid not in owned
            and eid != agent_id
        ]
        # location target: any room the agent isn't currently in.
        cur_loc = next(
            (r.args[1] for r in trace.relations
             if r.relation == "en" and r.args[0] == agent_id), None)
        loc_targets = [
            eid for eid, e in trace.entities.items()
            if e.destroyed_at_event is None
            and lex.types.is_subtype(e.entity_type, "location")
            and eid != cur_loc
        ]
        flavors = []
        if candidates:
            flavors.append(("havi", candidates))
        if loc_targets:
            flavors.append(("loc", loc_targets))
        if not flavors:
            continue
        flavor, pool = rng.choice(flavors)
        target = rng.choice(pool)
        if flavor == "havi":
            return Goal(
                agent=agent_id,
                fact=("rel", "havi", (agent_id, target)),
                injected_at=len(trace.events))
        else:
            return Goal(
                agent=agent_id,
                fact=("rel", "en", (agent_id, target)),
                injected_at=len(trace.events))
    return None


def fire_step(action_name, roles, trace, lex, rules, derivations,
              reservoir, edges_out, history, decay_steps):
    """Execute one (action, roles) step: capture prior state, append
    the event, run derivation closure, attribute causal edges, update
    reservoir."""
    derived_before = compute_derived_state(trace, derivations, lex)
    prior_facts = _state_facts(trace, derived_before)

    ev = make_event(
        action_name, roles=roles,
        property_changes=effect_changes(action_name, roles, lex))
    trace.events.append(ev)
    trace._event_ids.add(ev.id)
    run_dsl(trace, rules, derivations, lex)

    derived_after = compute_derived_state(trace, derivations, lex)
    after_facts = _state_facts(trace, derived_after)
    novel = after_facts - prior_facts
    invalidated = prior_facts - after_facts

    # Reservoir bookkeeping. Skip derivation-noise facts (samloke,
    # scias_lokon) that get re-derived on almost every state change —
    # they would flood the reservoir and dilute the chain-cashing
    # signal across nearly every applicable verb.
    pos = len(trace.events)
    reservoir.evict_old(pos)
    for fact in invalidated:
        if _is_reservoir_worthy(fact):
            reservoir.invalidate(fact)
    for fact in novel:
        if _is_reservoir_worthy(fact):
            reservoir.add(fact, established_at=pos - 1,
                          decay_steps=decay_steps)

    # Causal-graph attribution: which precondition was satisfied by
    # which prior-state fact?
    action = lex.actions.get(action_name)
    if action is not None:
        _causal_edges_for_event(action, roles, trace, prior_facts, edges_out)
        # Consume reservoir entries that matched preconditions.
        for pc in action.preconditions:
            if isinstance(pc, RelationPrecondition):
                args = tuple(roles.get(rn) for rn in pc.roles)
                reservoir.consume(("rel", pc.rel, args))

    history.record(action_name, roles)


def generate_trace(seed_factory: Callable[[], tuple[Trace, str]],
                   config: SamplerConfig, rng: random.Random,
                   *, lex=None, rules=None, derivations=None) -> TraceArtifact:
    """Run the forward sampler from a seed state to `max_trace_length`
    steps (or until no applicable rule remains). `seed_factory` returns
    (initial_trace, scene_location_id)."""
    if lex is None:
        lex = load_lexicon()
    if rules is None:
        rules = list(DEFAULT_DSL_RULES)
    if derivations is None:
        derivations = list(RUNTIME_DERIVATIONS)

    trace, _scene_id = seed_factory()
    edges: list[CausalEdge] = []
    reservoir = Reservoir()
    history = History(recency_window=config.recency_window)
    move_log: list[str] = []
    goals: list[Goal] = []
    target_depth = (config.target_depth_dist or default_target_depth_dist)(rng)

    # NB: do NOT seed setup facts into the reservoir. The reservoir's
    # purpose is to flag "novel-effect-not-yet-consumed" so the
    # `cashes_reservoir` feature can reward picking a candidate whose
    # precondition is met by a recent event's effect — i.e. chain
    # formation. Seeding setup would make every candidate a casher
    # (since setup facts trivially satisfy most preconditions), washing
    # out the chain signal entirely.

    for step in range(config.max_trace_length):
        # Prune satisfied goals so cashes_goal stops boosting verbs
        # that no longer move the world toward anything.
        derived = compute_derived_state(trace, derivations, lex)
        cur_facts = _state_facts(trace, derived)
        goals = [g for g in goals if g.fact not in cur_facts]

        # Pick a move: forward_step (default) or goal_injection.
        # `backward_step` intentionally absent.
        move = _pick_move(config, len(goals), rng)

        if move == "goal_injection":
            new_goal = _propose_goal(trace, lex, rng)
            if new_goal is not None:
                goals.append(new_goal)
                move_log.append("goal_injection")
                continue
            # nothing to want — fall through to forward.

        # forward_step
        candidates = enumerate_applicable_steps(
            trace, lex, derived, rng,
            rules=rules, derivations=derivations)
        if not candidates:
            break
        # Depth-targeted weight schedule: while below target, push
        # chain-pressure features hard; once reached, zero them.
        cur_depth = _current_max_depth(edges, trace.events)
        weights = _depth_adjusted_weights(
            config.feature_weights, cur_depth, target_depth,
            config.chain_pressure_boost)
        scored = [
            _score(_extract_features(
                an, rs, trace, history, reservoir, lex, goals=goals),
                   weights)
            for (an, rs) in candidates
        ]
        pick = _softmax_pick(candidates, scored, config.temperature, rng)
        if pick is None:
            break
        action_name, roles = pick
        fire_step(action_name, roles, trace, lex, rules, derivations,
                  reservoir, edges, history, config.reservoir_decay_steps)
        move_log.append("forward_step")

    # Folder cross-trace freshness counter — every event this
    # trace produced increments the (verb, theme_concept)
    # histogram so the next trace from this worker sees these
    # combos as "saturated" and pulls toward fresher ones.
    _update_cross_trace_histogram(trace)

    return TraceArtifact(
        trace=trace,
        final_state_relations=list(trace.relations),
        causal_graph=edges,
        depth_histogram=_depth_histogram(edges, trace.events),
        move_log=move_log,
        sampled_weights=dict(config.feature_weights),
        goals=goals,
    )


# =================== seed factories ====================

def goal_regression_seed(lex, rng):
    """Initial state from the goal-first regression sampler (the
    one the production chain-sampling pipeline uses). Calls
    `regress_for_goal`, discards the drive, returns the rich
    pre-spawned scene as the forward sampler's starting state.

    Why this is the only seed factory we keep:
      - The spawner has already materialized 40-100 entities with
        proper containment, ownership, kin, weather, lighting,
        lockable items — the same scene-quality the goal-driven
        chains corpus runs against.
      - No bespoke per-seed hand-coding. One factory covers every
        scene shape the goal sampler produces; adding a new scene
        type or entity class flows through here automatically.
      - Forward chains running on goal-sampler scenes are directly
        comparable to goal-regression chains on the same scenes —
        the corpus difference is purely in the sampling strategy,
        not in scene composition.

    Returns None when the sampler can't produce a scene in
    `max_attempts` tries — caller should skip rather than
    fall back to a hand-built tableau."""
    from esperanto_lm.ontology.regression.goal_sampler import (
        regress_for_goal)
    for _ in range(8):
        sample = regress_for_goal(lex, rng, DEFAULT_DSL_RULES)
        if sample is None:
            continue
        t, scene_id, _drive = sample
        return t, scene_id
    return None


SEEDS = {
    "goal_regression": goal_regression_seed,
}


# ============== parallel worker ==============

# Worker-local state. Each forked Pool worker imports this module
# fresh, so these globals are populated once per worker by
# `_init_worker` and reused across the tasks that worker handles.
_WORKER_LEX = None
_WORKER_RULES = None
_WORKER_DERIVS = None
_WORKER_CONFIG = None
_WORKER_SEED_FN = None


def _init_worker():
    """Per-worker init. Loads the lex + rules + derivations once and
    snapshots the SamplerConfig + seed factory passed via env vars.
    Subsequent `_worker_task` calls reuse the loaded state."""
    import os as _os
    from pathlib import Path as _Path2
    here = _Path2(__file__).parent
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(here.parent / "src"))
    global _WORKER_LEX, _WORKER_RULES, _WORKER_DERIVS
    global _WORKER_CONFIG, _WORKER_SEED_FN
    _WORKER_LEX = load_lexicon()
    _WORKER_RULES = list(DEFAULT_DSL_RULES)
    _WORKER_DERIVS = list(RUNTIME_DERIVATIONS)
    _WORKER_CONFIG = SamplerConfig(
        max_trace_length=int(_os.environ.get("FWD_STEPS", "12")))
    _WORKER_SEED_FN = SEEDS[_os.environ.get(
        "FWD_SEED_FACTORY", "goal_regression")]


def _worker_task(args):
    """Process a batch of forward traces for one seed. Returns
    `(records, skipped_no_seed, skipped_empty)`. Records are JSONL-
    shaped dicts ready to write."""
    from esperanto_lm.ontology import realize_trace
    seed, n_traces = args
    rng = random.Random(seed)
    records = []
    skipped_no_seed = 0
    skipped_empty = 0
    attempt = 0
    while len(records) < n_traces:
        attempt += 1
        if attempt > n_traces * 4:
            break
        seed_val = _WORKER_SEED_FN(_WORKER_LEX, rng)
        if seed_val is None:
            skipped_no_seed += 1
            continue
        t, scene_id = seed_val
        setup_rels = list(t.relations)
        try:
            artifact = generate_trace(
                lambda t=t, s=scene_id: (t, s),
                _WORKER_CONFIG, rng,
                lex=_WORKER_LEX, rules=_WORKER_RULES,
                derivations=_WORKER_DERIVS)
        except Exception:
            continue
        if not artifact.trace.events:
            skipped_empty += 1
            continue
        try:
            prose = realize_trace(
                artifact.trace, _WORKER_LEX,
                setup_relations=setup_rels,
                scene_location_id=scene_id, rng=rng)
        except Exception:
            continue
        chain = " → ".join(e.action for e in artifact.trace.events)
        records.append({
            "status": "ok",
            "scene": scene_id,
            "chain": chain,
            "n_events": len(artifact.trace.events),
            "prose": prose,
        })
    return records, skipped_no_seed, skipped_empty


# =================== CLI ====================

def main():
    import gzip as _gzip
    import json as _json
    import os as _os
    import time as _time
    from multiprocessing import Pool, cpu_count
    from pathlib import Path as _Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12,
                    help="Max events per trace (SamplerConfig."
                         "max_trace_length)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Base seed; each task gets seed + task_index")
    ap.add_argument("--n-traces", type=int, default=3,
                    help="Total traces to attempt across all workers")
    ap.add_argument("--seed-factory", choices=list(SEEDS.keys()),
                    default="goal_regression")
    ap.add_argument("--out", type=str, default=None,
                    help="JSONL output path. Same record shape as "
                         "scripts/run_regression_parallel.py — "
                         "{status, scene, chain, n_events, prose}.")
    ap.add_argument("--workers", type=int, default=cpu_count(),
                    help="Number of worker processes (default: "
                         "cpu_count). Use 1 to disable the Pool and "
                         "run single-process for debugging.")
    ap.add_argument("--batch", type=int, default=20,
                    help="Traces per task (smaller = finer progress "
                         "granularity, more queue traffic)")
    ap.add_argument("--maxtasksperchild", type=int, default=25,
                    help="Recycle each worker after this many tasks. "
                         "Forward sampling allocates less than the "
                         "goal planner, so 25 is fine. 0 to disable.")
    args = ap.parse_args()

    # Pass config to workers via env so the initializer can pick
    # them up without pickling closures.
    _os.environ["FWD_STEPS"] = str(args.steps)
    _os.environ["FWD_SEED_FACTORY"] = args.seed_factory

    out_f = None
    if args.out:
        parent = _Path(args.out).parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except (FileExistsError, OSError):
            pass
        out_f = open(args.out, "w")

    prose_chunks: list = []
    chain_counts: dict = {}
    verb_counts: dict = {}
    emitted = 0
    skipped_no_seed = 0
    skipped_empty = 0
    t0 = _time.time()

    # Single-process path (for debugging or tiny n_traces):
    # also exposes per-trace prints when --out isn't set.
    if args.workers <= 1:
        _init_worker()
        n_tasks = (args.n_traces + args.batch - 1) // args.batch
        tasks = [(args.seed + i, args.batch) for i in range(n_tasks)]
        for task_idx, task in enumerate(tasks, start=1):
            records, ns, ne = _worker_task(task)
            skipped_no_seed += ns
            skipped_empty += ne
            for rec in records:
                if out_f is None:
                    print(f"=== trace {emitted} "
                          f"({rec['n_events']} events, "
                          f"scene={rec['scene']}) ===")
                    print(f"chain: {rec['chain']}")
                    print(f"prose: {rec['prose']}")
                    print()
                else:
                    out_f.write(
                        _json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                if rec.get("prose"):
                    prose_chunks.append(rec["prose"])
                chain_counts[rec["chain"]] = (
                    chain_counts.get(rec["chain"], 0) + 1)
                for v in rec["chain"].split(" → "):
                    verb_counts[v] = verb_counts.get(v, 0) + 1
                emitted += 1
                if emitted >= args.n_traces:
                    break
            # Per-task progress line — flushed so a background log
            # tail / cat sees current state without needing `tail -f`.
            elapsed = _time.time() - t0
            print(f"  {task_idx}/{n_tasks} tasks "
                  f"({emitted}/{args.n_traces} emitted) — "
                  f"{elapsed:.1f}s", flush=True)
            if emitted >= args.n_traces:
                break
    else:
        # Parallel path: pool of workers, imap_unordered streams
        # results as they arrive.
        n_tasks = (args.n_traces + args.batch - 1) // args.batch
        tasks = [(args.seed + i, args.batch) for i in range(n_tasks)]
        print(f"Spawning {args.workers} workers; {n_tasks} tasks "
              f"× {args.batch} traces each (~{args.n_traces} total)",
              flush=True)
        pool_kwargs = {
            "processes": args.workers,
            "initializer": _init_worker,
        }
        if args.maxtasksperchild > 0:
            pool_kwargs["maxtasksperchild"] = args.maxtasksperchild
        completed_tasks = 0
        try:
            with Pool(**pool_kwargs) as pool:
                for (batch_records, ns, ne) in pool.imap_unordered(
                        _worker_task, tasks):
                    completed_tasks += 1
                    skipped_no_seed += ns
                    skipped_empty += ne
                    for rec in batch_records:
                        if emitted >= args.n_traces:
                            break
                        if out_f is not None:
                            out_f.write(
                                _json.dumps(rec, ensure_ascii=False)
                                + "\n")
                            out_f.flush()
                        if rec.get("prose"):
                            prose_chunks.append(rec["prose"])
                        chain_counts[rec["chain"]] = (
                            chain_counts.get(rec["chain"], 0) + 1)
                        for v in rec["chain"].split(" → "):
                            verb_counts[v] = verb_counts.get(v, 0) + 1
                        emitted += 1
                    elapsed = _time.time() - t0
                    rate = emitted / max(elapsed, 1e-6)
                    eta_s = max(0, args.n_traces - emitted) / max(rate, 0.1)
                    print(f"  {completed_tasks}/{n_tasks} tasks "
                          f"({emitted}/{args.n_traces} emitted, "
                          f"{rate:.1f} traces/s) — "
                          f"{elapsed:.1f}s "
                          f"ETA {eta_s:.0f}s", flush=True)
                    if emitted >= args.n_traces:
                        pool.terminate()
                        break
        finally:
            if out_f is not None:
                out_f.close()

    if out_f is not None and not out_f.closed:
        out_f.close()

    elapsed = _time.time() - t0
    print(f"\n=== Done: {emitted} traces in {elapsed:.1f}s "
          f"({elapsed*1000/max(emitted,1):.0f}ms/trace; "
          f"{emitted/max(elapsed, 1e-6):.1f}/s) ===",
          flush=True)
    print(f"skipped: no_seed={skipped_no_seed}  "
          f"empty_trace={skipped_empty}", flush=True)
    print(f"unique chains: {len(chain_counts)}", flush=True)
    print(f"\nTop verbs:")
    for v, n in sorted(
            verb_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {v:<14} {n}")
    print(f"\nTop chains (showing 10):")
    for chain, n in sorted(
            chain_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {n:>4}×  {chain[:120]}")
    if prose_chunks:
        raw = "\n".join(prose_chunks).encode("utf-8")
        compressed = _gzip.compress(raw, compresslevel=9)
        ratio = len(compressed) / len(raw)
        print(f"\nProse gzip ratio:")
        print(f"  records w/ prose: {len(prose_chunks)}")
        print(f"  raw bytes:        {len(raw):>10,}")
        print(f"  gzipped (lvl 9):  {len(compressed):>10,}")
        print(f"  ratio:            {ratio:.3f}  "
              f"({len(raw)/max(len(compressed),1):.2f}× shrink)")
        print(f"  Gutenberg eo ref: 0.380  (2.63×) — natural prose")
    if args.out:
        print(f"\nWrote {emitted} records to {args.out}", flush=True)


if __name__ == "__main__":
    main()
