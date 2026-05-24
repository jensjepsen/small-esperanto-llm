"""Generate ICL Q/A pairs from rich regression traces.

Input: a JSONL produced by `run_regression_parallel.py` with the
extended schema (events + entities + setup_relations alongside
prose).

Output: an SFT JSONL where each line is

    {"messages":
        [{"role": "user",    "content": "<prose>\\n\\nDemando: <Q>"},
         {"role": "assistant", "content": "<A>"}]}

Each trace yields multiple Q/A pairs covering several question
templates. Questions are grounded in the trace's actual events
and entity properties, so answers are always factually correct
relative to the rendered prose.

Templates:

  - intrinsic property      "Kia estis la koloro de la X?"
                            answered from entity.properties.

  - first/last action       "Kio okazis unue/laste?"
                            answered from events[0/-1].

  - action attribution      "Kiu prenis la Y?" / "Kion la X manĝis?"
                            answered from events[i].roles.

  - state change            "Post X-i la pordon, kia estis ĝia stato?"
                            answered from events[i].property_changes.

  - location at start       "Kie estis la Y komence?"
                            answered from setup_relations.

  - sequencing              "Kio okazis post la prenado?"
                            answered by walking events forward.

Usage:
    python scripts/generate_icl_from_traces.py \\
        --in  data/causal_corpus/sample_10000_rich_v27.jsonl \\
        --out data/causal_corpus/sample_10000_icl_v27.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# Esperanto question/answer phrasings — kept compact; the SFT
# trainer will tokenize these as ordinary text.

CARDINALS_EO = [
    "nul", "unu", "du", "tri", "kvar", "kvin",
    "ses", "sep", "ok", "naŭ", "dek",
]


def _past(verb: str) -> str:
    """Esperanto past-tense form. Strip infinitive -i, add -is.
    Naive — assumes regular verbs, which is all our action lemmas."""
    if verb.endswith("i"):
        return verb[:-1] + "is"
    return verb + "is"


def _noun_acc(noun: str) -> str:
    """Accusative ending: -o → -on, -oj → -ojn. Leaves names/
    pronouns alone (caller is responsible)."""
    if noun.endswith("oj"):
        return noun + "n"
    if noun.endswith("o"):
        return noun + "n"
    return noun


def _name(eid: str, entities: dict) -> str:
    """Surface form for an entity: concept lemma. Capitalize for
    person types (proper nouns). Falls back to eid."""
    ent = entities.get(eid)
    if ent is None:
        return eid
    lemma = ent["concept"]
    if ent["type"] == "person":
        return lemma.capitalize()
    return lemma


def _acc(noun: str) -> str:
    """Accusative ending for an Esperanto noun. Naive: -o → -on,
    -oj → -ojn. Leaves names alone."""
    if noun and noun[0].isupper():
        # Proper noun — apply -n directly
        return noun + "n" if not noun.endswith("n") else noun
    if noun.endswith("oj"):
        return noun + "n"
    if noun.endswith("o"):
        return noun + "n"
    return noun


def _q_intrinsic_property(rec: dict, rng: random.Random) -> list[dict]:
    """For each entity with a notable observable property (color,
    posture, openness), emit a Q/A. Skips body parts (eid contains
    '_'-suffix substrings) to keep questions about top-level items.
    """
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    interesting_slots = ["koloro", "posture", "openness",
                         "fullness", "lock_state", "power_state",
                         "cleanliness"]
    for ent in rec["entities"]:
        if "_" in ent["eid"] and ent["eid"] != ent["concept"]:
            continue  # body part / sub-component
        if ent["type"] in ("location", "abstract"):
            continue
        for slot in interesting_slots:
            vals = ent["properties"].get(slot)
            if not vals:
                continue
            val = vals[0]
            name = _name(ent["eid"], entities)
            if slot == "koloro":
                q = f"Kia estis la koloro de la {ent['concept']}?"
                a = val + "a"  # adjective form
            elif slot == "posture":
                q = f"En kia pozicio estis la {ent['concept']}?"
                a = val
            elif slot == "openness":
                q = f"Ĉu la {ent['concept']} estis malfermita aŭ fermita?"
                a = val
            elif slot == "fullness":
                q = f"Ĉu la {ent['concept']} estis plena aŭ malplena?"
                a = val
            elif slot == "lock_state":
                q = f"Ĉu la {ent['concept']} estis ŝlosita aŭ malŝlosita?"
                a = val
            elif slot == "power_state":
                q = f"Ĉu la {ent['concept']} estis aktiva aŭ neaktiva?"
                a = val
            elif slot == "cleanliness":
                q = f"Ĉu la {ent['concept']} estis pura aŭ malpura?"
                a = val
            else:
                continue
            out.append({"q": q, "a": a})
    return out


def _q_first_last(rec: dict, rng: random.Random) -> list[dict]:
    """First and last verb in the event sequence."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []

    def describe(ev):
        a = ev["roles"].get("agent")
        if a is None:
            return f"{ev['action']}"
        agent_name = _name(a, entities)
        theme = ev["roles"].get("theme")
        if theme is None:
            return f"{agent_name} {_past(ev['action'])}"
        # theme may be a list (fari.parts) — collapse
        if isinstance(theme, list):
            theme_name = ", ".join(_name(t, entities) for t in theme)
        else:
            theme_name = _name(theme, entities)
        return f"{agent_name} {_past(ev['action'])} {theme_name}"

    out.append({
        "q": "Kio okazis unue en la rakonto?",
        "a": describe(events[0]) + ".",
    })
    if len(events) > 1:
        out.append({
            "q": "Kio okazis laste en la rakonto?",
            "a": describe(events[-1]) + ".",
        })
    return out


def _q_action_attribution(rec: dict, rng: random.Random) -> list[dict]:
    """Per content event, ask "who did X to Y" and / or
    "what did Z do"."""
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    interesting_verbs = {"preni", "doni", "manĝi", "trinki", "verŝi",
                         "ĵeti", "malfermi", "fermi", "ŝlosi",
                         "malŝlosi", "ŝalti", "malŝalti", "fari",
                         "kuiri", "boli", "akvumi", "planti", "meti",
                         "porti"}
    for ev in events:
        if ev["action"] not in interesting_verbs:
            continue
        agent = ev["roles"].get("agent")
        theme = ev["roles"].get("theme")
        if agent is None or theme is None:
            continue
        if isinstance(theme, list):
            continue  # list themes (fari.parts) handled elsewhere
        agent_ent = entities.get(agent)
        theme_ent = entities.get(theme)
        if agent_ent is None or theme_ent is None:
            continue
        agent_name = _name(agent, entities)
        theme_name = _name(theme, entities)
        # "Who did X-i the Y?" (theme is accusative in Esperanto)
        out.append({
            "q": (f"Kiu {_past(ev['action'])} la "
                  f"{_noun_acc(theme_ent['concept'])}?"),
            "a": agent_name + ".",
        })
        # "What did Z X-i?"
        if agent_ent["type"] == "person":
            out.append({
                "q": f"Kion {agent_name} {_past(ev['action'])}?",
                "a": "la " + _noun_acc(theme_ent["concept"]) + ".",
            })
    return out


def _q_state_change(rec: dict, rng: random.Random) -> list[dict]:
    """Property-change questions: "After X did Y, what state was Z in?"
    """
    events = rec.get("events", [])
    if not events:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for ev in events:
        if not ev.get("property_changes"):
            continue
        for key, new_val in ev["property_changes"].items():
            if "|" not in key:
                continue
            eid, slot = key.split("|", 1)
            ent = entities.get(eid)
            if ent is None:
                continue
            # Skip slots we don't have natural Esperanto phrasings for
            if slot not in ("openness", "fullness", "lock_state",
                            "power_state", "cleanliness", "posture",
                            "wetness", "temperature", "presence"):
                continue
            verb = ev["action"]
            theme = ev["roles"].get("theme")
            if theme and isinstance(theme, str):
                theme_ent_pc = entities.get(theme)
                if theme_ent_pc is None:
                    continue
                theme_name = _noun_acc(theme_ent_pc["concept"])
                q = (f"Post kiam la aganto {_past(verb)} la "
                     f"{theme_name}, kia estis la stato de la "
                     f"{ent['concept']}?")
            else:
                q = (f"Kio okazis al la {ent['concept']} post la "
                     f"ago?")
            out.append({"q": q, "a": str(new_val)})
    return out


def _q_location_at_start(rec: dict, rng: random.Random) -> list[dict]:
    """"Where was X at the start?" — from setup_relations."""
    setup = rec.get("setup_relations", [])
    if not setup:
        return []
    entities = {e["eid"]: e for e in rec["entities"]}
    out = []
    for r in setup:
        if r["relation"] != "en":
            continue
        if len(r["args"]) != 2:
            continue
        contained, container = r["args"]
        c_ent = entities.get(contained)
        co_ent = entities.get(container)
        if c_ent is None or co_ent is None:
            continue
        if c_ent["type"] in ("location",):
            continue
        if "_" in contained and contained != c_ent["concept"]:
            continue  # body part
        out.append({
            "q": f"Kie estis la {c_ent['concept']} komence?",
            "a": f"En la {co_ent['concept']}.",
        })
    return out


# Registry of question generators.
GENERATORS = [
    _q_intrinsic_property,
    _q_first_last,
    _q_action_attribution,
    _q_state_change,
    _q_location_at_start,
]


def generate_qas_for_trace(
    rec: dict, rng: random.Random, max_per_trace: int = 4,
) -> list[dict]:
    """Yield up to max_per_trace Q/A pairs sampled across generators.
    Skipping empty generators; sampled uniformly so question types
    stay balanced."""
    candidates: list[dict] = []
    for gen in GENERATORS:
        candidates.extend(gen(rec, rng))
    if not candidates:
        return []
    rng.shuffle(candidates)
    seen_qs: set = set()
    picked: list[dict] = []
    for qa in candidates:
        if qa["q"] in seen_qs:
            continue
        seen_qs.add(qa["q"])
        picked.append(qa)
        if len(picked) >= max_per_trace:
            break
    return picked


def format_sft_record(prose: str, qa: dict) -> dict:
    """Wrap a (prose, Q, A) triple into the SFT conversation format
    `train_sft.py` expects."""
    return {
        "messages": [
            {"role": "user",
             "content": f"{prose}\n\nDemando: {qa['q']}"},
            {"role": "assistant", "content": qa["a"]},
        ]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-per-trace", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    n_traces = 0
    n_qas = 0
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            prose = rec.get("prose")
            if not prose:
                continue
            qas = generate_qas_for_trace(
                rec, rng, max_per_trace=args.max_per_trace)
            for qa in qas:
                fout.write(json.dumps(
                    format_sft_record(prose, qa),
                    ensure_ascii=False) + "\n")
                n_qas += 1
            n_traces += 1
    print(f"Wrote {n_qas} Q/A pairs from {n_traces} traces to {args.out}")


if __name__ == "__main__":
    main()
