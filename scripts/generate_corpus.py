"""Generate a multi-scenario (trace, prose) corpus with summary stats.

Each JSONL line:
  {
    "trace":  {entities, relations, events},
    "prose":  "<rendered Esperanto>",
    "metadata": {
      sample_seed, scene, recipe, persons, pruned_persons,
      n_events, n_synthesized, synthesized_actions, rules_fired,
      iterations_to_fixed_point,
      prose_word_count, distinct_content_words,
    }
  }

After emission, prints summary statistics to stdout: trace-length
distribution, rule-firing counts, vocabulary coverage, scenario
distribution.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from esperanto_lm.ontology import (
    Trace,
    load_lexicon,
    prune_unused_persons,
    realize_trace,
    sample_scene,
)
from esperanto_lm.ontology.dsl import run_dsl
from esperanto_lm.ontology.dsl.rules import (
    DEFAULT_DSL_DERIVATIONS,
    DEFAULT_DSL_RULES,
    make_use_instrument_rules,
)


SCENES_DEFAULT = ["kuirejo", "laborejo", "ĝardeno",
                  "manĝejo", "salono", "oficejo"]


# Small function-word blocklist for "distinct content words" approximation.
# Not linguistically rigorous — just excludes the most common non-content
# tokens so the counts reflect nouns/verbs/adjectives.
_FUNCTION_WORDS = frozenset({
    "la", "en", "sur", "per", "al", "de", "kaj", "tial", "sekve",
    "pro", "tio", "kun", "sen", "ĉe",
    "estis", "estas", "estos", "estus",
    "kuŝis", "kuŝas", "kuŝos",
    "havis", "havas", "havos", "tenis", "tenas", "tenos",
    "ili", "li", "ŝi", "ĝi", "mi", "vi", "ni",
    "lin", "ŝin", "ĝin",
})


# Synthesized action lemma → rule name that produced it. For `fali` we
# disambiguate by inspecting the cause event.
_SIG_VERBS = {"tranĉi", "ŝlosi", "purigi", "najli", "malŝlosi"}


def _attribute_rule(event, trace) -> str:
    """Infer which rule fired to produce this synthesized event."""
    by_id = {e.id: e for e in trace.events}
    if event.action == "rompiĝi":
        return "fragile_falls_breaks"
    if event.action == "satiĝi":
        return "hungry_eats_sated"
    if event.action == "bruli":
        return "fire_spreads_to_adjacent_flammables"
    if event.action == "aperi":
        for cid in event.caused_by:
            cause = by_id.get(cid)
            if cause is None:
                continue
            if cause.action == "rompiĝi":
                return "broken_fragile_creates_shards"
            if cause.action == "fali":
                return "wet_liquid_container_tips"
        return "aperi:unknown"
    if event.action == "fali":
        for cid in event.caused_by:
            cause = by_id.get(cid)
            if cause is None:
                continue
            if cause.action == "aperi":
                return "person_slips_on_wet"
            if cause.action == "fali":
                theme = trace.entities.get(event.roles.get("theme", ""))
                if theme is not None and theme.entity_type != "person":
                    return "container_falls_contents_fall"
                return "carried_thing_falls_when_carrier_falls"
            if cause.action == "rompiĝi":
                return "broken_container_releases_contents"
        return "fali:unknown"
    if event.action in _SIG_VERBS:
        return "use_instrument"
    return f"unknown:{event.action}"


def _word_stats(prose: str) -> tuple[int, int]:
    """Return (word_count, distinct_content_word_count)."""
    # Strip punctuation and lower.
    cleaned = prose.lower()
    for ch in ".,;:!?—–-":
        cleaned = cleaned.replace(ch, " ")
    tokens = [t for t in cleaned.split() if t]
    content = {t for t in tokens if t not in _FUNCTION_WORDS}
    return len(tokens), len(content)


def trace_to_dict(trace: Trace) -> dict:
    return {
        "entities": [
            {
                "id": e.id,
                "concept": e.concept_lemma,
                "entity_type": e.entity_type,
                "properties": e.properties,
            }
            for e in trace.entities.values()
        ],
        "relations": [
            {"relation": r.relation, "args": list(r.args)}
            for r in trace.relations
        ],
        "events": [
            {
                "id": ev.id,
                "action": ev.action,
                "roles": ev.roles,
                "caused_by": list(ev.caused_by),
            }
            for ev in trace.events
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("data/causal_corpus/corpus.jsonl"))
    ap.add_argument("--scenes", type=str,
                    default=",".join(SCENES_DEFAULT),
                    help="Comma-separated scene concept lemmas.")
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    lex = load_lexicon()
    outer_rng = random.Random(args.seed)
    # Full DSL rule set: base library plus one rule per instrument-
    # capable verb (tranĉi, ŝlosi, purigi, najli). Derivations are the
    # compositional layer — currently just flammability-from-material.
    rules = DEFAULT_DSL_RULES + make_use_instrument_rules(lex)
    derivations = DEFAULT_DSL_DERIVATIONS

    # Aggregates for summary stats
    scene_counts: Counter = Counter()
    recipe_counts: Counter = Counter()
    event_count_hist: Counter = Counter()
    rules_fired: Counter = Counter()
    concepts_seen: set[str] = set()
    skipped = 0

    with open(args.out, "w") as f:
        for i in range(args.n):
            sample_seed = outer_rng.randrange(1 << 31)
            scene = outer_rng.choice(scenes)
            rng = random.Random(sample_seed)

            try:
                trace, info = sample_scene(lex, rng, scene=scene)
            except (ValueError, KeyError, RuntimeError) as e:
                skipped += 1
                continue

            seed_event_ids = {ev.id for ev in trace.events}
            # Snapshot the initial-state relations BEFORE the engine
            # runs — rules that swap relations (preni/doni/iri/...)
            # mutate trace.relations in place, so the realizer needs
            # the original to render the scene setup correctly AND to
            # detect change narration ("Maria ne plu havas la libron").
            setup_relations = trace.snapshot_relations()
            iters = run_dsl(trace, rules, derivations, lex)

            synthesized = [
                ev for ev in trace.events if ev.id not in seed_event_ids]
            synth_actions = [ev.action for ev in synthesized]

            # Attribute each synthesized event to a rule.
            for ev in synthesized:
                rules_fired[_attribute_rule(ev, trace)] += 1

            pruned = prune_unused_persons(trace)

            prose = realize_trace(
                trace, lex, scene_location_id=info.scene_location_id,
                setup_relations=setup_relations,
                rng=rng,
            )

            word_count, distinct = _word_stats(prose)

            # Track concept coverage (by entity concept_lemma)
            for ent in trace.entities.values():
                concepts_seen.add(ent.concept_lemma)

            scene_counts[scene] += 1
            recipe_counts[info.recipe] += 1
            event_count_hist[len(trace.events)] += 1

            record = {
                "trace": trace_to_dict(trace),
                "prose": prose,
                "metadata": {
                    "sample_seed": sample_seed,
                    "scene": scene,
                    "recipe": info.recipe,
                    "persons": info.persons,
                    "pruned_persons": pruned,
                    "n_events": len(trace.events),
                    "n_synthesized": len(synthesized),
                    "synthesized_actions": synth_actions,
                    "iterations_to_fixed_point": iters,
                    "prose_word_count": word_count,
                    "distinct_content_words": distinct,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---------- summary ----------
    total = sum(scene_counts.values())
    print(f"\nWrote {total}/{args.n} traces → {args.out}")
    if skipped:
        print(f"  ({skipped} skipped due to sampler errors)")

    print("\n== scene distribution ==")
    for s, n in sorted(scene_counts.items(), key=lambda x: -x[1]):
        pct = 100 * n / max(total, 1)
        print(f"  {s:<10} {n:>4}  ({pct:4.1f}%)")

    print("\n== trace length (events) ==")
    for length in sorted(event_count_hist):
        n = event_count_hist[length]
        bar = "█" * min(40, int(40 * n / max(event_count_hist.values())))
        print(f"  {length} events: {n:>4}  {bar}")

    print("\n== rules fired ==")
    for rule, n in rules_fired.most_common():
        print(f"  {rule:<35} {n:>5}")

    print("\n== recipe distribution (top 20) ==")
    for rec, n in recipe_counts.most_common(20):
        print(f"  {rec:<35} {n:>4}")

    print("\n== vocabulary coverage ==")
    all_concepts = set(lex.concepts.keys())
    unseen = all_concepts - concepts_seen
    print(f"  concepts appearing: {len(concepts_seen)}/{len(all_concepts)}")
    if unseen:
        print(f"  NEVER sampled: {sorted(unseen)}")

    print("\n== recipe-family distribution ==")
    family_counts: Counter = Counter()
    for rec, n in recipe_counts.items():
        # Family: first prefix before _ for use_/open_/drop_/water_/plant_;
        # or the animate_eats_X pattern.
        if "_eats_" in rec:
            family_counts["eat"] += n
        else:
            family = rec.split("_", 1)[0]
            family_counts[family] += n
    for fam, n in family_counts.most_common():
        pct = 100 * n / max(total, 1)
        print(f"  {fam:<20} {n:>4}  ({pct:4.1f}%)")


if __name__ == "__main__":
    main()
