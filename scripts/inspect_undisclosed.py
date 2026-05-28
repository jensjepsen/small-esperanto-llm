"""Sample undisclosed Q/A pairs and show their prose + missing facts."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_icl_from_traces as gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--n-per-template", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    samples: dict[str, list] = defaultdict(list)
    type_counts: dict[str, int] = {}

    all_concepts: set[str] = set()
    with open(args.inp) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            for ent in rec.get("entities", []):
                if ent["type"] in ("location", "abstract"):
                    continue
                if "_" in ent["eid"] and ent["eid"] != ent["concept"]:
                    continue
                all_concepts.add(ent["concept"])
    all_concepts_frozen = frozenset(all_concepts)

    with open(args.inp) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            prose = rec.get("prose")
            if not prose:
                continue
            disclosed_facts = gen._flatten_facts(rec.get("sentence_facts"))
            qas = gen.generate_qas_for_trace(
                rec, rng, max_per_trace=8,
                all_concepts=all_concepts_frozen,
                type_counts=type_counts)
            for qa in qas:
                if qa.get("disclosed") is False:
                    template = gen._qa_type_key(qa["q"])
                    if len(samples[template]) < args.n_per_template:
                        missing = []
                        for pat in qa.get("requires", []):
                            if not any(gen._fact_matches(pat, f) for f in disclosed_facts):
                                missing.append(pat)
                        samples[template].append({
                            "q": qa["q"],
                            "a": qa["a"],
                            "missing": missing,
                            "prose": prose,
                        })

    for tpl in sorted(samples):
        print(f"=== {tpl} ({len(samples[tpl])} samples) ===")
        for s in samples[tpl]:
            print(f"  Q: {s['q']}")
            print(f"  A: {s['a']}")
            print(f"  Missing: {s['missing']}")
            print(f"  Prose: {s['prose'][:350]}...")
            print()


if __name__ == "__main__":
    main()
