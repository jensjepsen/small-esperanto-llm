"""Audit how often Q/A generators ask about facts the prose didn't disclose.

Reconstructs traces from the regression-sampler JSONL, re-realizes them
to get (prose, sentence_facts) via the new disclosure tracking, then
runs the existing Q/A generators. For each Q/A, checks whether the
answer string appears in the prose (loose disclosure proxy) and tallies
per template.

No filtering happens; the script just reports the distribution so we
can see how often the model is asked about undisclosed propositions.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# generate_icl_from_traces imports use this path layout
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from esperanto_lm.ontology import load_lexicon, realize_trace
from esperanto_lm.ontology.causal import EntityInstance, Event, RelationAssertion, Trace

import generate_icl_from_traces as gen


def _rebuild_trace(rec: dict) -> tuple[Trace, list[RelationAssertion], str | None]:
    entities = {}
    for ent in rec["entities"]:
        entities[ent["eid"]] = EntityInstance(
            id=ent["eid"], concept_lemma=ent["concept"],
            entity_type=ent["type"],
            properties=ent["properties"],
            created_at_event=ent.get("created_at_event"))
    events = []
    for ev in rec["events"]:
        roles = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in ev["roles"].items()
        }
        pc = {}
        for k, v in ev.get("property_changes", {}).items():
            eid, slot = k.split("|", 1)
            pc[(eid, slot)] = v
        events.append(Event(
            id=ev["id"], action=ev["action"], roles=roles,
            caused_by=tuple(ev.get("caused_by", [])),
            property_changes=pc))
    setup = [
        RelationAssertion(relation=r["relation"], args=tuple(r["args"]))
        for r in rec.get("setup_relations", [])
    ]
    final = [
        RelationAssertion(relation=r["relation"], args=tuple(r["args"]))
        for r in rec.get("final_relations", [])
    ]
    trace = Trace(entities=entities, relations=final, events=events)
    return trace, setup, rec.get("scene")


def _normalize_for_match(s: str) -> str:
    import re
    s = s.lower().strip().rstrip(".,;:!?")
    # Drop leading article/preposition
    for p in ("la ", "en la ", "en ", "sur la ", "sur ", "al la ", "al ",
             "per la ", "per ", "de la ", "de ", "tra la ", "tra ",
             "apud la ", "apud ", "el la ", "el ", "ĉe la ", "ĉe ",
             "sub la ", "sub ", "por la ", "por ", "ĉar "):
        if s.startswith(p):
            s = s[len(p):]
            break
    # Strip accusative endings
    s = re.sub(r"\b(\w+)ojn\b", r"\1oj", s)
    s = re.sub(r"\b(\w+)on\b", r"\1o", s)
    return s


def _answer_in_prose(answer: str, prose: str) -> bool:
    a = _normalize_for_match(answer)
    p = _normalize_for_match(prose)
    if not a:
        return False
    # Allow individual content words to anchor the match
    words = [w for w in a.split() if len(w) > 2]
    if not words:
        return a in p
    return all(w in p for w in words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--n", type=int, default=500,
                    help="Number of traces to audit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lex = load_lexicon()
    rng = random.Random(args.seed)

    by_template: dict[str, Counter] = defaultdict(lambda: Counter())
    samples_undisclosed: dict[str, list] = defaultdict(list)
    n_done = 0
    n_records = 0
    type_counts: dict[str, int] = {}

    with open(args.inp) as f:
        for line in f:
            if n_done >= args.n:
                break
            n_records += 1
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            try:
                trace, setup, scene_id = _rebuild_trace(rec)
                rng_r = random.Random(n_records)
                prose, sentence_facts = realize_trace(
                    trace, lex, setup_relations=setup,
                    scene_location_id=scene_id, rng=rng_r,
                    definition_p=0.3, return_facts=True)
            except Exception as e:
                continue
            rec_for_gen = dict(rec)
            rec_for_gen["prose"] = prose
            qas = gen.generate_qas_for_trace(
                rec_for_gen, rng, max_per_trace=8,
                type_counts=type_counts)
            for qa in qas:
                template = gen._qa_type_key(qa["q"])
                disclosed = _answer_in_prose(qa["a"], prose)
                by_template[template]["total"] += 1
                by_template[template]["disclosed" if disclosed else "undisclosed"] += 1
                if not disclosed and len(samples_undisclosed[template]) < 3:
                    samples_undisclosed[template].append(
                        {"q": qa["q"], "a": qa["a"], "prose": prose[:200]})
            n_done += 1

    print(f"\nAudited {n_done} traces\n")
    print(f"{'Template':<20s} {'Total':>7s} {'Disc':>7s} {'Undisc':>7s} {'%U':>6s}")
    print("-" * 60)
    tot = sum(c["total"] for c in by_template.values())
    tot_u = sum(c["undisclosed"] for c in by_template.values())
    for tpl in sorted(by_template, key=lambda k: -by_template[k]["total"]):
        c = by_template[tpl]
        pct = 100 * c["undisclosed"] / max(c["total"], 1)
        print(f"{tpl:<20s} {c['total']:>7d} {c['disclosed']:>7d} {c['undisclosed']:>7d} {pct:>5.1f}%")
    print("-" * 60)
    print(f"{'TOTAL':<20s} {tot:>7d} {tot - tot_u:>7d} {tot_u:>7d} "
          f"{100*tot_u/max(tot,1):>5.1f}%")

    print("\nSample undisclosed Q/A (per template):")
    for tpl in sorted(samples_undisclosed):
        print(f"\n=== {tpl} ===")
        for s in samples_undisclosed[tpl]:
            print(f"  Q: {s['q']}")
            print(f"  A: {s['a']}")
            print(f"  Prose: {s['prose']}...")


if __name__ == "__main__":
    main()
