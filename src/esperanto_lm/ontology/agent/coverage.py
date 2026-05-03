"""Coverage harnesses: bulk-run the planner across N sampled scenes,
report per-verb / per-chain stats and prose samples.

Two flavors:
  - `run_coverage_regression`: scenes from the regression sampler
    (`ontology.regression.sample_regression_scene`); each scene is
    constructed backward from a chosen drive.
  - `run_coverage`: scenes from the random drive sampler (`sample_scene`
    + `sample_drive`); each scene is constructed forward, then
    augmented to support the picked drive.

Used by `scripts/agent_drive_coverage.py` and
`scripts/run_regression_parallel.py`.
"""
from __future__ import annotations

import random

from .. import realize_trace
from ..dsl import run_dsl
from ..regression import sample_regression_scene
from .dispatcher import _drive_summary, plan_for_drive
from .drive_sampler import (
    _build_property_writability, _build_relation_writability,
    augment_scene_for_drive, sample_drive, sample_scene,
)
from .planner import _step_to_event


def run_coverage_regression(lex, rules, derivations, *, n_scenes=50, seed=0,
                             verbose_samples=8, save_jsonl=None):
    """Companion to run_coverage that uses goal-regression to build
    scenes. Same plan/fire/realize pipeline; different scene source."""
    rng = random.Random(seed)
    fired_records = []
    idle = 0
    failed = 0
    verb_counts: dict[str, int] = {}
    chain_counts: dict[str, int] = {}
    drive_kind_counts = {"sampled": {}, "fired": {}}
    jsonl_records = []

    for _ in range(n_scenes):
        sample = sample_regression_scene(lex, rng, rules=rules)
        if sample is None:
            failed += 1
            continue
        t, scene_id, drive = sample
        kind = drive[0]
        drive_kind_counts["sampled"][kind] = (
            drive_kind_counts["sampled"].get(kind, 0) + 1)
        setup = t.snapshot_relations()
        try:
            plan = plan_for_drive(drive, t, lex, rules, derivations, rng=rng)
        except Exception:
            failed += 1
            continue
        if not plan:
            idle += 1
            continue
        try:
            for step in plan:
                event = _step_to_event(step, lex)
                t.events.append(event)
                run_dsl(t, rules, derivations, lex)
        except Exception:
            failed += 1
            continue
        try:
            prose = realize_trace(
                t, lex, setup_relations=setup,
                scene_location_id=scene_id)
        except Exception:
            prose = "<render failed>"
        chain = " → ".join(ev.action for ev in t.events)
        chain_counts[chain] = chain_counts.get(chain, 0) + 1
        for ev in t.events:
            verb_counts[ev.action] = verb_counts.get(ev.action, 0) + 1
        drive_kind_counts["fired"][kind] = (
            drive_kind_counts["fired"].get(kind, 0) + 1)
        fired_records.append((drive, chain, prose))
        jsonl_records.append({
            "scene": scene_id,
            "drive": {"kind": kind, "spec": list(drive[1:])},
            "drive_summary": _drive_summary(drive),
            "chain": chain,
            "n_events": len(t.events),
            "prose": prose,
        })

    print(f"\n========== regression coverage ({n_scenes} scenes) ==========")
    print(f"  fired: {len(fired_records)}   "
          f"idle: {idle}   failed: {failed}")
    print(f"\n  Verb counts (over all events fired):")
    for verb in sorted(verb_counts, key=lambda v: -verb_counts[v]):
        print(f"    {verb:<15} {verb_counts[verb]:>3}")
    print(f"\n  Plan-chain distribution:")
    for chain, n in sorted(chain_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"    {n:>3}×  {chain}")
    print(f"\n  Sample prose ({min(verbose_samples, len(fired_records))} "
          f"of {len(fired_records)}):")
    for drive, chain, prose in rng.sample(
            fired_records, min(verbose_samples, len(fired_records))):
        print(f"    [{chain}]  drive: {_drive_summary(drive)}")
        print(f"      prose: {prose}")
    if save_jsonl:
        import json
        from pathlib import Path
        Path(save_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(save_jsonl, "w") as f:
            for rec in jsonl_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n  Saved {len(jsonl_records)} records to {save_jsonl}")


def run_coverage(lex, rules, derivations, *, n_scenes=50, seed=0,
                 verbose_samples=8, save_jsonl=None):
    """Sample N scenes, sample one drive per scene, dispatch the drive
    to its planner entry, fire the resulting plan if any. Direct
    dispatch (not run_simulation) means we resolve ONLY the sampled
    drive — no opportunistic resolution of whatever else random init
    left unsatisfied.

    Filtering rule: keep scenes where the planner returned a non-empty
    plan AND firing it produced ≥1 trace event. Idle = no plan found
    for the sampled drive (can happen when scene lacks the props the
    drive's chain needs)."""
    rng = random.Random(seed)
    fired_records = []  # (drive, chain, prose)
    idle = 0
    failed = 0
    verb_counts: dict[str, int] = {}
    chain_counts: dict[str, int] = {}
    drive_kind_counts: dict[str, int] = {"sampled": {}, "fired": {}}
    drive_kind_counts["sampled"] = {}
    drive_kind_counts["fired"] = {}
    jsonl_records = []

    # Build writability caches once — they're a pure function of the
    # lexicon + rules + derivations, so reused across all scenes.
    property_writable = _build_property_writability(lex, rules, derivations)
    relation_writable = _build_relation_writability(lex, rules, derivations)

    for _ in range(n_scenes):
        try:
            t, scene_id, persons = sample_scene(lex, rng)
        except Exception:
            failed += 1
            continue
        drive = sample_drive(
            t, lex, rng, derivations=derivations, rules=rules,
            property_writable=property_writable,
            relation_writable=relation_writable)
        if drive is None:
            idle += 1
            continue
        # Goal-aware augmentation: place props the drive's chain
        # needs but blind sampling didn't put in scene.
        try:
            augment_scene_for_drive(t, drive, lex, rng, scene_id)
        except Exception:
            pass  # augmentation is best-effort; never let it block
        kind = drive[0]
        drive_kind_counts["sampled"][kind] = (
            drive_kind_counts["sampled"].get(kind, 0) + 1)
        setup = t.snapshot_relations()
        try:
            plan = plan_for_drive(drive, t, lex, rules, derivations, rng=rng)
        except Exception:
            failed += 1
            continue
        if not plan:
            idle += 1
            continue
        try:
            for step in plan:
                event = _step_to_event(step, lex)
                t.events.append(event)
                run_dsl(t, rules, derivations, lex)
        except Exception:
            failed += 1
            continue
        try:
            prose = realize_trace(
                t, lex, setup_relations=setup,
                scene_location_id=scene_id)
        except Exception:
            prose = "<render failed>"
        chain = " → ".join(ev.action for ev in t.events)
        chain_counts[chain] = chain_counts.get(chain, 0) + 1
        for ev in t.events:
            verb_counts[ev.action] = verb_counts.get(ev.action, 0) + 1
        drive_kind_counts["fired"][kind] = (
            drive_kind_counts["fired"].get(kind, 0) + 1)
        fired_records.append((drive, chain, prose))
        jsonl_records.append({
            "scene": scene_id,
            "drive": {"kind": kind, "spec": list(drive[1:])},
            "drive_summary": _drive_summary(drive),
            "chain": chain,
            "n_events": len(t.events),
            "prose": prose,
        })

    print(f"\n========== coverage run ({n_scenes} scenes) ==========")
    print(f"  fired: {len(fired_records)}   "
          f"idle: {idle}   failed: {failed}")
    print(f"\n  Drive kind: sampled vs fired (success rate):")
    all_kinds = sorted(set(drive_kind_counts["sampled"]) |
                       set(drive_kind_counts["fired"]))
    for kind in all_kinds:
        s = drive_kind_counts["sampled"].get(kind, 0)
        f = drive_kind_counts["fired"].get(kind, 0)
        rate = f"{100*f/s:.0f}%" if s else "—"
        print(f"    {kind:15s} sampled={s:3d}  fired={f:3d}  ({rate})")
    print(f"\n  Verb counts (over all events fired):")
    for v, c in sorted(verb_counts.items(), key=lambda x: -x[1]):
        print(f"    {v:15s} {c}")
    print(f"\n  Plan-chain distribution:")
    for ch, c in sorted(chain_counts.items(), key=lambda x: -x[1]):
        print(f"    {c:3d}×  {ch}")
    print(f"\n  Sample prose ({verbose_samples} of {len(fired_records)}):")
    rng2 = random.Random(seed + 1)
    samples = rng2.sample(
        fired_records, min(verbose_samples, len(fired_records)))
    for drive, chain, prose in samples:
        print(f"    [{chain}]  drive: {_drive_summary(drive)}")
        print(f"      prose: {prose}")
    if save_jsonl:
        import json
        with open(save_jsonl, "w") as f:
            for rec in jsonl_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n  Saved {len(jsonl_records)} records to {save_jsonl}")

