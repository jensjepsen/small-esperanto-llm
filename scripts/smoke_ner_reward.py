"""Fast self-check for the NER reward surface.

Run after ANY edit to rl_rewards.py. Twice in one session a slice-based edit
silently deleted parse_ner and the module-level regexes it depends on, leaving
reward_ner raising NameError — which would only have surfaced on the first
rollout of a GPU run.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from esperanto_lm.rl_rewards import (reward_ner, ner_prompt, parse_ner,
                                     ner_schema_score, ner_match_counts)

G = lambda *e: json.dumps({"ents": [list(x) for x in e],
                           "buckets": ["person", "org", "sted", "dato"]},
                          ensure_ascii=False)
checks = [
    ("perfect", '{"person":["Mette"],"organisation":[],"sted":["Danmark"],"dato":[]}',
     G(("mette", "person"), ("danmark", "sted")), 1.0),
    ("abstain ok", '{"person":[],"organisation":[],"sted":[],"dato":[]}',
     G(), 1.0),
    ("abstain wrong", '{"person":[],"organisation":[],"sted":[],"dato":[]}',
     G(("mette", "person")), 0.0),
    ("no json", "ingen json her", G(("mette", "person")), 0.0),
]
bad = 0
for name, comp, gold, want in checks:
    got = reward_ner(comp, gold)
    ok = abs(got - want) < 0.02
    bad += not ok
    print(f"  {name:<16}{got:>7.4f}  want {want:.2f}  {'ok' if ok else 'FAIL'}")
assert parse_ner('{"person":["A"]}') == [("a", "person")], "parse_ner broken"
assert "{t}" in ner_prompt(("person",)), "ner_prompt lost its slot"
assert ner_match_counts([("a", "person")], {("a", "person")}) == (1, 0, 0)
print("FAIL" if bad else "all NER reward smoke checks pass")
sys.exit(1 if bad else 0)
