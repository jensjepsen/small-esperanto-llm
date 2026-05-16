"""Trace → Esperanto prose.

Three-stage pipeline:

  plan    trace → list[Message]      — what to say, one sentence each
  transform list[Message]            — aggregate/combine (currently:
                                        same-subject → kaj-coordination)
  render  list[Message] → str        — surface realization

The `realize_trace` entry point keeps the old signature for callers
and wraps the pipeline. Each stage is independently replaceable —
the full document planner replaces `plan_messages`, richer sentence
planning replaces `transform`, richer templates replace `render`.
"""
from __future__ import annotations

import random
from typing import Optional

from ..causal import RelationAssertion, Trace
from ..loader import Lexicon
from .messages import (
    AppearanceMessage,
    CoordinatedMessage,
    DestructionMessage,
    EventMessage,
    GroupedRelationMessage,
    Message,
    RelationAddedMessage,
    RelationMessage,
    RelationRemovedMessage,
    SceneGroundingMessage,
    SubordinatedMessage,
)
from .plan import (
    aggregate_relations,
    aggregate_same_subject,
    plan_messages,
    subordinate_creations,
)
from .render import (
    PRONOUN_OF_NAME, PRONOUN_RATE, RELATION_TEMPLATES, TENSES,
    inflect, past_tense, render_messages, to_accusative,
)


def realize_trace(
    trace: Trace, lexicon: Lexicon, *,
    scene_location_id: Optional[str] = None,
    rng: Optional[random.Random] = None,
    tense: Optional[str] = None,
    setup_relations: Optional[list[RelationAssertion]] = None,
) -> str:
    """Render the full trace as a paragraph of Esperanto.

    Same signature as the pre-rebuild version. Internally runs the
    plan → transform → render pipeline; `rng`, `tense`, and
    `setup_relations` flow through unchanged.

    Argument notes:
      `setup_relations` — snapshot of `trace.relations` taken before
        running the engine. Rules that modify relations (preni, doni,
        iri) leave `trace.relations` in the post-trace state; pass the
        snapshot for correct scene-setup rendering. Also enables
        relation-change narration (Maria ne plu havis la libron).

        Required when `trace.events` is non-empty: without it the
        preamble walks post-trace relations and would narrate
        rule-added havi as if it held at scene start
        ("Kuzo havis sandviĉon" before fari fires). Static traces
        (no events, scene-description only) may omit it.

      `rng` — per-trace variation (templates, connectives, pronouns,
        tense). None makes the realizer fully deterministic.
      `tense` — explicit override ('is' past, 'as' present).
    """
    if setup_relations is None and trace.events:
        raise ValueError(
            "realize_trace: setup_relations is required when "
            "trace.events is non-empty. Take a snapshot before "
            "running the engine: `setup_rels = list(t.relations)` "
            "after scene setup, before run_dsl / event seeding. "
            "Pass as `setup_relations=setup_rels`. Without it the "
            "preamble would describe post-event ownership as if it "
            "held at scene start.")
    # Compute derived state once for the whole render pass. The render
    # consults it for context-dependent surface forms — e.g. an animate
    # `en` a water_body has posture=naĝanta via the
    # `animate_swimming_when_in_water_body` derivation, and the
    # realizer renders "Lidia naĝas en la lago" rather than the bland
    # "Lidia estas en la lago". No optional fallback: the truth of an
    # entity's posture is the derived state; rendering without it would
    # produce inconsistent prose depending on caller plumbing.
    from ..dsl import compute_derived_state
    from ..dsl.rules import DEFAULT_DSL_DERIVATIONS, RUNTIME_DERIVATIONS
    all_derivations = list(DEFAULT_DSL_DERIVATIONS) + list(RUNTIME_DERIVATIONS)
    derived = compute_derived_state(trace, all_derivations, lexicon)
    # Setup-time derived state: a clone of the trace with relations
    # rewound to the pre-event snapshot and events dropped. The
    # renderer uses this for setup-phase messages so derived
    # posture/category reflects how the world stood before any
    # events fired — without it, post-event en(najbaro, rivero)
    # back-propagated posture=naĝanta into the setup preamble's
    # en(najbaro, oficejo) line. Only meaningful when the caller
    # passed setup_relations and events ran; otherwise the two
    # snapshots are equivalent and we reuse `derived`.
    setup_derived = derived
    if setup_relations is not None and trace.events:
        setup_trace = trace.fork()
        setup_trace.relations = list(setup_relations)
        setup_trace.events = []
        setup_trace._event_ids = set()
        setup_trace._current_props = {}
        setup_trace._current_props_version = 0
        setup_trace._parts_index = {
            r.args[1] for r in setup_relations
            if r.relation == "havas_parton" and len(r.args) >= 2}
        setup_derived = compute_derived_state(
            setup_trace, all_derivations, lexicon)
    messages = plan_messages(
        trace, lexicon,
        scene_location_id=scene_location_id,
        setup_relations=setup_relations)
    messages = aggregate_relations(messages)
    messages = aggregate_same_subject(messages, lexicon)
    messages = subordinate_creations(messages)
    return render_messages(
        messages, trace, lexicon,
        scene_location_id=scene_location_id, rng=rng, tense=tense,
        derived=derived, setup_derived=setup_derived)


__all__ = [
    # public API
    "realize_trace",
    # message IR (for callers who want to plug into the pipeline)
    "Message",
    "SceneGroundingMessage", "RelationMessage", "EventMessage",
    "AppearanceMessage", "RelationRemovedMessage", "RelationAddedMessage",
    "DestructionMessage", "CoordinatedMessage", "SubordinatedMessage",
    "GroupedRelationMessage",
    # pipeline stages (for library consumers customizing a stage)
    "plan_messages", "aggregate_relations", "aggregate_same_subject",
    "subordinate_creations", "render_messages",
    # morphology helpers re-exported for the test suite's convenience
    "inflect", "past_tense", "to_accusative",
    "RELATION_TEMPLATES", "PRONOUN_OF_NAME", "PRONOUN_RATE", "TENSES",
]
