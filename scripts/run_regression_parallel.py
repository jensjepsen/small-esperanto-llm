"""Parallel regression sampler. Spawns a worker pool; each worker
loads the lexicon + rules + derivations ONCE at startup (in module
globals) and processes seed-batched scene tasks. The big read-only
state never gets pickled — only small task tuples and result records
cross the process boundary.

Usage:
    uv run python scripts/run_regression_parallel.py \\
        --scenes 1000 --workers 8 --out runs/regression_1k.jsonl

Throughput: with W workers, scales near-linearly until cores saturate.
On an 8-core box, 1000 scenes runs in ~60-80s instead of ~500s
single-threaded.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Worker-local state. Each forked process imports this module fresh,
# so these globals are populated once per worker by `_init_worker`.
_LEX = None
_RULES = None
_DERIVATIONS = None


def _init_worker():
    """Per-worker init: load lexicon and rules. Called once when the
    Pool spawns the worker. Stores in module globals so subsequent
    task calls reuse the loaded state.

    Adds `src/` (for the esperanto_lm package) to sys.path. Needed
    when launched with `--no-project` (e.g. PyPy runs that skip the
    torch dependency), since uv doesn't auto-add the project's
    source path in that mode."""
    here = Path(__file__).parent
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(here.parent / "src"))
    from esperanto_lm.ontology import load_lexicon
    from esperanto_lm.ontology.dsl.rules import (
        DEFAULT_DSL_RULES, RUNTIME_DERIVATIONS,
    )
    global _LEX, _RULES, _DERIVATIONS
    _LEX = load_lexicon()
    _RULES = list(DEFAULT_DSL_RULES)
    _DERIVATIONS = list(RUNTIME_DERIVATIONS)


def _json_safe(value):
    """Recursively convert non-JSON-encodable atoms in a failure
    reason. Currently catches Ellipsis (which can appear as a
    wildcard slot value in derivation patterns the planner walks).
    Tuples → lists; everything else passed through."""
    if value is Ellipsis:
        return "..."
    if isinstance(value, (tuple, list)):
        return [_json_safe(x) for x in value]
    return value


def _ref(eid: str, lex) -> str:
    """Surface form for an entity id (nominative) when narrating
    failure tails. Convention:
      - eid in lex.concepts (a concept lemma, e.g. "kravato",
        "ŝtrumpo", "patro") → "la <eid>" (definite article).
      - eid not in lex.concepts (a person name like "petro" or a
        suffixed derived id like "kravato_theme1") → capitalize the
        first character. Person names render naturally; derived
        ids look slightly off but are rare."""
    if not eid:
        return eid
    if eid in lex.concepts:
        return f"la {eid}"
    return eid.capitalize()


def _ref_acc(eid: str, lex) -> str:
    """Accusative form: nominative + 'n' suffix per Esperanto."""
    nom = _ref(eid, lex)
    if not nom:
        return nom
    return f"{nom}n"


def _is_animate_eid(eid: str, lex) -> bool:
    """True if the entity (looked up by id, with concept-lemma
    fallback for derived ids like 'kravato_theme1') is animate per
    the type spine. Person names (not in lex.concepts) heuristic-
    treated as animate too — lowercase identifiers in the regression
    sampler are persons by convention."""
    concept = lex.concepts.get(eid)
    if concept is None:
        # Derived id: try to peel the concept lemma off the prefix.
        prefix = eid.split("_", 1)[0]
        concept = lex.concepts.get(prefix)
    if concept is None:
        # Person names (anna, petro, ...) aren't in concepts but the
        # regression sampler always names persons; treat as animate.
        return True
    return lex.types.is_subtype(concept.entity_type, "animate")


def _failure_why_phrase(reason, lex) -> str:
    """Map a planner failure_reason tuple to a contextual Esperanto
    "because Z couldn't happen" clause. Falls back to the bland
    'ne sukcesis' when the reason isn't recognized. The clauses
    don't include leading 'sed' — caller adds that to compose with
    a 'volis Y, sed [why]' sentence shape.

    Animate-vs-non-animate first-arg distinction matters for
    movement-style relations (en, samloke, apud): "Pavel ne povis
    veni al la domo" (animate Pavel goes places) reads naturally,
    but "la pomo ne povis veni al la kuirejo" (the apple goes
    places??) reads as nonsense. For non-animates we either flip to
    a passive ("ne povis esti alportita" — could not be brought) or
    fall back bland for relations whose passive is awkward."""
    if not reason:
        return "ne sukcesis"
    kind = reason[0]
    if kind == "budget":
        return "ne sukcesis"
    if kind == "relation":
        rel_name = reason[1]
        args = reason[2] if len(reason) > 2 else ()
        if rel_name == "havi" and len(args) == 2:
            return (f"{_ref(args[0], lex)} ne povis akiri "
                    f"{_ref_acc(args[1], lex)}")
        if rel_name == "samloke" and len(args) == 2:
            if not _is_animate_eid(args[0], lex):
                return "ne sukcesis"
            return (f"{_ref(args[0], lex)} ne povis atingi "
                    f"{_ref_acc(args[1], lex)}")
        if rel_name == "konas" and len(args) == 2:
            return f"{_ref(args[0], lex)} ne ricevis la informon"
        if rel_name == "scias_lokon" and len(args) == 2:
            return (f"{_ref(args[0], lex)} ne sciis kie estis "
                    f"{_ref(args[1], lex)}")
        if rel_name == "vestita" and len(args) == 2:
            return (f"{_ref(args[0], lex)} ne povis surmeti "
                    f"{_ref_acc(args[1], lex)}")
        if rel_name == "en" and len(args) == 2:
            if _is_animate_eid(args[0], lex):
                return (f"{_ref(args[0], lex)} ne povis veni al "
                        f"{_ref(args[1], lex)}")
            # Non-animate theme: passive — nobody could move it
            # there. Honest about the structure (a theme doesn't
            # "go" anywhere on its own) without surfacing planner
            # internals.
            return (f"{_ref(args[0], lex)} ne povis esti alportita al "
                    f"{_ref(args[1], lex)}")
        if rel_name == "apud" and len(args) == 2:
            if not _is_animate_eid(args[0], lex):
                return "ne sukcesis"
            return (f"{_ref(args[0], lex)} ne povis aliri "
                    f"{_ref_acc(args[1], lex)}")
        return "ne sukcesis"
    if kind == "property":
        eid = reason[1]
        slot = reason[2]
        value = reason[3] if len(reason) > 3 else ""
        ref = _ref(eid, lex)
        if slot == "openness":
            if value == "malfermita":
                return f"{ref} restis fermita"
            if value == "fermita":
                return f"{ref} restis malfermita"
        if slot == "lock_state":
            if value == "malŝlosita":
                return f"{ref} restis ŝlosita"
            if value == "ŝlosita":
                return f"{ref} restis malŝlosita"
        if slot == "cleanliness" and value == "pura":
            return f"{ref} restis malpura"
        if slot == "wetness" and value == "seka":
            return f"{ref} restis malseka"
        if slot == "hunger" and value == "sata":
            return f"{ref} restis malsata"
        if slot == "thirst" and value == "satigita":
            return f"{ref} restis soifa"
        if slot == "sleep_state":
            if value == "dormanta":
                return f"{ref} ne povis ekdormi"
            if value == "vekita":
                return f"{ref} ne vekiĝis"
        if slot == "fullness":
            if value == "plena":
                return f"{ref} restis malplena"
            if value == "malplena":
                return f"{ref} restis plena"
        if slot == "planted_state" and value == "plantita":
            return f"{ref} restis ne plantita"
        if slot == "attachment" and value == "fiksita":
            return f"{ref} restis loza"
        if slot == "temperature":
            return f"{ref} ne fariĝis {value}"
        if slot == "power_state":
            if value == "aktiva":
                return f"{ref} restis neaktiva"
            if value == "neaktiva":
                return f"{ref} restis aktiva"
        # Slot the planner tried to flip but isn't narratively
        # meaningful (locomotion, emits_sound, indoor_outdoor,
        # is_currency, water_body, can_hear, etc.) — these are
        # intrinsic concept traits that the planner explores via
        # alternate-path subgoals before giving up. Fall back to
        # bland "ne sukcesis" rather than emit an opaque internal
        # detail like "la ringo.emits_sound restis ne yes."
        return "ne sukcesis"
    if kind == "count":
        actor, concept_lemma, target = reason[1], reason[2], reason[3]
        return (f"{_ref(actor, lex)} ne povis akiri sufiĉe "
                f"da {concept_lemma}")
    return "ne sukcesis"


def _drive_attempt_phrase(drive, lex, failure_reason=None):
    """Compose a "X wanted Y, but Z" sentence for a failed drive.
    Drive-shape switch builds the "X wanted Y" prefix; the why-clause
    comes from `_failure_why_phrase(reason)` so the obstacle is named
    when the planner identified a leaf it gave up on (deeper than the
    top-level drive). Bland fallback "ne sukcesis" when no reason is
    available or the reason isn't templated."""
    kind = drive[0]
    why = _failure_why_phrase(failure_reason, lex)
    if kind == "self_slot":
        actor, slot, value = drive[1], drive[2], drive[3]
        return (f"{_ref(actor, lex)} volis fariĝi {value}, sed {why}.")
    if kind == "entity_slot":
        actor, target, slot, value = drive[1], drive[2], drive[3], drive[4]
        return (f"{_ref(actor, lex)} volis ke "
                f"{_ref(target, lex)} estu {value}, sed {why}.")
    if kind == "location":
        actor, loc = drive[1], drive[2]
        return f"{_ref(actor, lex)} volis iri al {_ref(loc, lex)}, sed {why}."
    if kind == "possession":
        actor, item = drive[1], drive[2]
        return (f"{_ref(actor, lex)} volis havi "
                f"{_ref_acc(item, lex)}, sed {why}.")
    if kind == "knowledge":
        actor, knower, fakto = drive[1], drive[2], drive[3]
        return (f"{_ref(actor, lex)} volis ke {_ref(knower, lex)} sciu, "
                f"sed {why}.")
    if kind == "wearing":
        actor, garment = drive[1], drive[2]
        return (f"{_ref(actor, lex)} volis surmeti "
                f"{_ref_acc(garment, lex)}, sed {why}.")
    if kind == "count":
        actor, concept_lemma, target = drive[1], drive[2], drive[3]
        return (f"{_ref(actor, lex)} volis havi {target} {concept_lemma}jn, "
                f"sed {why}.")
    if kind == "give_count":
        donor, recipient, concept_lemma, target = drive[1:5]
        return (f"{_ref(donor, lex)} volis ke {_ref(recipient, lex)} havu "
                f"{target} {concept_lemma}jn, sed {why}.")
    if kind == "more_than":
        actor, concept_lemma, ref = drive[1], drive[2], drive[3]
        return (f"{_ref(actor, lex)} volis havi pli da {concept_lemma} "
                f"ol {_ref(ref, lex)}, sed {why}.")
    return None


def _worker_task(args):
    """Process a batch of scenes for one seed. Returns a list of
    JSONL-shaped record dicts. Must be at module level (not nested or
    a closure) so multiprocessing can pickle it."""
    seed, n_scenes = args

    # Defer imports until the worker is initialized.
    import os
    from esperanto_lm.ontology import realize_trace
    from esperanto_lm.ontology.agent.dispatcher import (
        _drive_summary, plan_for_drive,
    )
    from esperanto_lm.ontology.agent.planner import (
        _step_to_event, get_planner_failure_reason,
    )
    from esperanto_lm.ontology.agent.forward_planner import plan_for_goal
    from esperanto_lm.ontology.dsl import run_dsl
    from esperanto_lm.ontology.regression import sample_regression_scene
    from esperanto_lm.ontology.regression.goal_sampler import regress_for_goal

    # Forward planner + goal sampler are the default — same path as
    # bench_samplers.py. Set USE_BACKWARD=1 to opt back into the
    # legacy verb-sampler + backward chainer.
    use_forward = os.environ.get("USE_BACKWARD") != "1"
    rng = random.Random(seed)
    out = []

    for _ in range(n_scenes):
        if use_forward:
            sample = regress_for_goal(_LEX, rng, _RULES)
        else:
            sample = sample_regression_scene(_LEX, rng, rules=_RULES)
        if sample is None:
            continue
        t, scene_id, drive = sample
        kind = drive[0]
        setup = t.snapshot_relations()
        plan_exc: Exception | None = None
        try:
            from esperanto_lm.ontology.regression.spawner import make_spawner
            spawner = make_spawner(scene_id, _LEX, rng)
            if use_forward:
                plan = plan_for_goal(
                    drive, t, _LEX, _RULES, _DERIVATIONS,
                    max_states=int(os.environ.get("MAX_STATES", "1200")),
                    max_plan_length=int(
                        os.environ.get("MAX_PLAN_LENGTH", "16")),
                    entity_resolver=spawner, rng=rng,
                    exclude_verbs=getattr(
                        t, "_planner_exclude_verbs", None))
            else:
                plan = plan_for_drive(
                    drive, t, _LEX, _RULES, _DERIVATIONS, rng=rng,
                    entity_resolver=spawner)
        except Exception as e:
            plan = None
            plan_exc = e
        failure_reason = get_planner_failure_reason() if not plan else None
        if not plan:
            # Failed-plan record: emit setup state + drive so callers
            # can inspect what kinds of goals the planner couldn't
            # satisfy. Setup-only prose renders the scene as a static
            # tableau (entities + relations) without events. Useful as
            # a starting point for non-happy-path narration; for now
            # just visibility.
            try:
                prose = realize_trace(
                    t, _LEX, setup_relations=setup,
                    scene_location_id=scene_id, rng=rng)
            except Exception:
                prose = ""
            tail = _drive_attempt_phrase(
                drive, _LEX, failure_reason=failure_reason)
            if tail:
                # Sentence-initial capital — "la X" → "La X" since the
                # tail begins a fresh sentence after the setup prose.
                tail = tail[:1].upper() + tail[1:]
                prose = f"{prose} {tail}".strip()
            out.append({
                "status": "failed",
                "scene": scene_id,
                "drive": {"kind": kind, "spec": list(drive[1:])},
                "drive_summary": _drive_summary(drive),
                "chain": "",
                "n_events": 0,
                "prose": prose,
                "entities": sorted(
                    {(eid, ent.concept_lemma)
                     for eid, ent in t.entities.items()}),
                "failure_reason": (
                    [_json_safe(x) for x in failure_reason]
                    if failure_reason else None),
                "failure_exception": (
                    type(plan_exc).__name__ if plan_exc else None),
            })
            continue
        try:
            for step in plan:
                event = _step_to_event(step, _LEX)
                t.events.append(event)
                run_dsl(t, _RULES, _DERIVATIONS, _LEX)
            prose = realize_trace(
                t, _LEX, setup_relations=setup,
                scene_location_id=scene_id, rng=rng)
        except Exception:
            continue
        chain = " → ".join(ev.action for ev in t.events)
        out.append({
            "status": "ok",
            "scene": scene_id,
            "drive": {"kind": kind, "spec": list(drive[1:])},
            "drive_summary": _drive_summary(drive),
            "chain": chain,
            "n_events": len(t.events),
            "prose": prose,
        })

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", type=int, default=200,
                   help="Total scenes to attempt across all workers")
    p.add_argument("--workers", type=int, default=cpu_count(),
                   help="Number of worker processes (default: cpu_count)")
    p.add_argument("--batch", type=int, default=20,
                   help="Scenes per task (smaller = better load balance, "
                        "more queue traffic)")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; each task gets seed + task_index")
    p.add_argument("--out", type=str, default=None,
                   help="JSONL output path (default: print summary only)")
    args = p.parse_args()

    n_tasks = (args.scenes + args.batch - 1) // args.batch
    tasks = [(args.seed + i, args.batch) for i in range(n_tasks)]
    print(f"Spawning {args.workers} workers; {n_tasks} tasks "
          f"× {args.batch} scenes each (~{args.scenes} total)")

    start = time.time()
    records = []
    chain_counts: dict[str, int] = {}
    verb_counts: dict[str, int] = {}
    drive_kind_counts: dict[str, int] = {}
    completed_tasks = 0

    with Pool(processes=args.workers, initializer=_init_worker) as pool:
        for batch_records in pool.imap_unordered(_worker_task, tasks):
            completed_tasks += 1
            for rec in batch_records:
                records.append(rec)
                chain_counts[rec["chain"]] = (
                    chain_counts.get(rec["chain"], 0) + 1)
                drive_kind_counts[rec["drive"]["kind"]] = (
                    drive_kind_counts.get(rec["drive"]["kind"], 0) + 1)
                for v in rec["chain"].split(" → "):
                    verb_counts[v] = verb_counts.get(v, 0) + 1
            if completed_tasks % max(1, n_tasks // 20) == 0:
                elapsed = time.time() - start
                print(f"  {completed_tasks}/{n_tasks} tasks done "
                      f"({len(records)} fired) — {elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"\n=== Done: {len(records)} fired scenes in {elapsed:.1f}s "
          f"({elapsed*1000/max(len(records),1):.0f}ms/scene; "
          f"{len(records)/elapsed:.1f}/s) ===")
    print(f"\nDrive kinds: {drive_kind_counts}")
    print(f"\nTop verbs:")
    for v in sorted(verb_counts, key=lambda x: -verb_counts[x])[:15]:
        print(f"  {v:<14} {verb_counts[v]}")
    print(f"\nTop chains (showing 10):")
    for chain, n in sorted(
            chain_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {n:>4}×  {chain}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
