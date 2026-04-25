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
      `rng` — per-trace variation (templates, connectives, pronouns,
        tense). None makes the realizer fully deterministic.
      `tense` — explicit override ('is' past, 'as' present).
    """
    messages = plan_messages(
        trace, lexicon,
        scene_location_id=scene_location_id,
        setup_relations=setup_relations)
    messages = aggregate_relations(messages)
    messages = aggregate_same_subject(messages, lexicon)
    messages = subordinate_creations(messages)
    return render_messages(
        messages, trace, lexicon,
        scene_location_id=scene_location_id, rng=rng, tense=tense)


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
