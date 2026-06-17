"""Compose semantically coherent event chains from ConceptNet facts.

Each chain is an ORDERED LIST OF TRIPLES (rel, head, tail) — no prose.
Chain TEMPLATES define the logical structure; ConceptNet provides the
concrete facts. Surface rendering into Esperanto happens downstream.

Chain templates implemented:
    weather_mitigation : activity -> weather state -> effect -> mitigation
    prerequisite_chain : activity -> prereq -> prereq-of-prereq (depth 2-3)
    causal_chain       : state -> effect -> effect-of-effect (depth 2-3)
    tool_for_goal      : goal -> tool that achieves it
    location_activity  : activity -> location + related facts
    property_cause     : thing -> its property -> what the property causes
"""

import argparse
import gzip
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

CN_DUMP   = Path("data/conceptnet/conceptnet-assertions-5.7.0.csv.gz")
CN_INDEX  = Path("data/conceptnet/index.pkl")

# ---- Portability signal for filtering artifact slots --------------------
PORTABLE_POS_SUB = (
    "garment", "clothing", "clothes", "apparel", "outerwear", "overgarment",
    "device", "tool", "instrument", "implement", "utensil",
    "accessory", "headgear", "headdress", "footwear", "eyewear",
    "glasses", "spectacles",
    "lamp", "torch", "light source",
    "protection", "protective covering", "protective equipment",
    "shelter providing artifact",
)
PORTABLE_BLOCK_SUB = (
    "area of land", "area ", "land", "ground", "surface", "place",
    "location", "structure", "building", "region", "field", "enclosure",
    "facility", "lawn",
    "phenomenon", "anatomical", "body part", "chemical reaction",
    "combustion", "beverage", "liquid",
    "person", "human", "animal", "plant", "organism",
    "vehicle", "aircraft", "ship",
)

# Weather-like states for the weather_mitigation template.
CONDITION_NOUNS = {
    "rain":     {"rain"},
    "sun":      {"sun", "sunlight", "sunburn", "uv"},
    "darkness": {"dark", "darkness", "night"},
}
COUNTER_VERBS = {
    "rain":     {"dry", "shelter", "protect", "keep", "protection",
                 "keeping", "drying", "shielding", "sheltering"},
    "sun":      {"shade", "shield", "protect", "block", "prevent", "cover",
                 "blocking", "shielding", "protecting", "covering",
                 "preventing", "shading"},
    "darkness": {"see", "light", "illuminate", "seeing", "lighting",
                 "illuminating", "lit"},
}

MOTION_VERBS = {
    "go", "walk", "drive", "ride", "run", "jog", "hike", "travel",
    "cycle", "bike", "stroll", "visit", "shop", "fish",
}


# ---- ConceptNet index: parse once, cache --------------------------------

KEEP_REL = {
    "/r/IsA", "/r/MotivatedByGoal", "/r/Causes",
    "/r/UsedFor", "/r/CapableOf", "/r/HasPrerequisite",
    "/r/AtLocation", "/r/HasProperty", "/r/HasSubevent",
    "/r/PartOf", "/r/MadeOf", "/r/HasA", "/r/Desires",
    "/r/CausesDesire",
}

def build_index(dump_path: Path, out_path: Path) -> dict:
    """Parse the raw dump once and persist indexes by relation."""
    idx = defaultdict(lambda: defaultdict(list))
    isa_parents  = defaultdict(set)
    isa_children = defaultdict(set)

    with gzip.open(dump_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5: continue
            _, rel, start, end, meta = parts
            if rel not in KEEP_REL: continue
            if not (start.startswith("/c/en/") and end.startswith("/c/en/")):
                continue
            try: w = json.loads(meta).get("weight", 1.0)
            except Exception: w = 1.0
            s = start.split("/")[3].replace("_", " ")
            e = end.split("/")[3].replace("_", " ")
            name = rel.split("/")[-1]
            if rel == "/r/IsA":
                isa_parents[s].add(e)
                isa_children[e].add(s)
            else:
                idx[name][s].append((e, w))

    # Convert to plain dicts for pickle
    out = {
        "isa_parents":  dict(isa_parents),
        "isa_children": dict(isa_children),
        "by_rel": {rel: dict(heads) for rel, heads in idx.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    return out


def load_index() -> dict:
    if CN_INDEX.exists():
        with open(CN_INDEX, "rb") as f:
            return pickle.load(f)
    print(f"[build] parsing {CN_DUMP} (one-time) ...")
    return build_index(CN_DUMP, CN_INDEX)


# ---- Validators ---------------------------------------------------------

def is_portable(artifact: str, isa_parents: dict) -> bool:
    """Parent-voting: >=2 POS parents AND POS > BLOCK."""
    parents = isa_parents.get(artifact, set())
    if not parents: return False
    pos = block = 0
    for p in parents:
        pl = p.lower()
        if any(s in pl for s in PORTABLE_POS_SUB):   pos   += 1
        if any(s in pl for s in PORTABLE_BLOCK_SUB): block += 1
    return pos >= 2 and pos > block


def is_clean_phrase(s: str, max_tokens: int = 5) -> bool:
    """Reject ConceptNet tails that are stranded fragments / noise."""
    if not s: return False
    t = s.split()
    if len(t) > max_tokens: return False
    if s.endswith((" from", " to", " with", " of", " in", " on", " at")):
        return False
    if t[0] in {"doesn't", "didn't", "won't", "don't", "needed", "yourself"}:
        return False
    return True


def activity_is_motion(head: str) -> bool:
    toks = head.split()
    return bool(toks) and toks[0] in MOTION_VERBS


def mitigation_fits(usage: str, condition: str) -> bool:
    ul = usage.lower().strip()
    if not ul: return False
    first_words = ul.split()[:2]
    return (any(n in ul for n in CONDITION_NOUNS[condition]) and
            any(w in COUNTER_VERBS[condition] for w in first_words))


# ---- Chain templates -----------------------------------------------------
# Each template is a function (idx, rng) -> list[triple] | None.
# A triple is a dict {"rel": str, "head": str, "tail": str}.

def tpl_weather_mitigation(idx, rng):
    """Activity with a weather state that the actor mitigates via an artifact.

    Triples:
        (MotivatedByGoal, activity, goal)
        (Causes,          state,    effect)
        (UsedFor,         artifact, usage)
    """
    by = idx["by_rel"]; ip = idx["isa_parents"]
    # 1) Pick an activity
    motivated = [(h, g, w) for h, tails in by.get("MotivatedByGoal", {}).items()
                 for g, w in tails
                 if w >= 2.0 and activity_is_motion(h)
                 and is_clean_phrase(h, 4) and is_clean_phrase(g, 5)]
    if not motivated: return None
    h, g, _ = rng.choice(motivated)

    # 2) Pick a weather condition + an effect
    cond_options = [c for c in CONDITION_NOUNS if c in by.get("Causes", {})]
    cond = rng.choice(cond_options)
    effect_pool = [(e, w) for e, w in by["Causes"][cond] if is_clean_phrase(e, 4)]
    if not effect_pool: return None
    effect = rng.choices(effect_pool, weights=[w for _, w in effect_pool])[0][0]

    # 3) Pick a mitigation artifact whose UsedFor tail fits the condition
    candidates = []
    for art, tails in by.get("UsedFor", {}).items():
        if not is_portable(art, ip): continue
        for tail, w in tails:
            if mitigation_fits(tail, cond):
                candidates.append((art, tail, w))
    if not candidates: return None
    art, usage, _ = rng.choices(candidates, weights=[w for *_, w in candidates])[0]

    return [
        {"rel": "MotivatedByGoal", "head": h,    "tail": g},
        {"rel": "Causes",          "head": cond, "tail": effect},
        {"rel": "UsedFor",         "head": art,  "tail": usage},
    ]


def tpl_prerequisite_chain(idx, rng, depth=2):
    """Activity with a chain of prerequisites.

    Triples (depth=2):
        (MotivatedByGoal,   activity, goal)
        (HasPrerequisite,   activity, prereq1)
        (HasPrerequisite,   prereq1,  prereq2)   # optional
    """
    by = idx["by_rel"]
    mot = by.get("MotivatedByGoal", {})
    pre = by.get("HasPrerequisite", {})
    # Pick an activity that has both a goal and a prereq
    heads = [h for h in mot if h in pre]
    if not heads: return None
    rng.shuffle(heads)
    for h in heads:
        goals = [(g, w) for g, w in mot[h] if w >= 2.0 and is_clean_phrase(g, 5)]
        prereqs = [(p, w) for p, w in pre[h] if w >= 2.0 and is_clean_phrase(p, 5)]
        if not goals or not prereqs: continue
        g = rng.choices(goals, weights=[w for _, w in goals])[0][0]
        p1 = rng.choices(prereqs, weights=[w for _, w in prereqs])[0][0]
        triples = [
            {"rel": "MotivatedByGoal", "head": h,  "tail": g},
            {"rel": "HasPrerequisite", "head": h,  "tail": p1},
        ]
        # try to extend
        if depth >= 2 and p1 in pre:
            p2s = [(p, w) for p, w in pre[p1] if w >= 2.0 and is_clean_phrase(p, 5)]
            if p2s:
                p2 = rng.choices(p2s, weights=[w for _, w in p2s])[0][0]
                triples.append({"rel": "HasPrerequisite", "head": p1, "tail": p2})
        return triples
    return None


def tpl_causal_chain(idx, rng, depth=2):
    """Multi-hop Causes chain.

    Triples:
        (Causes, s,  e1)
        (Causes, e1, e2)   # optional
    """
    by = idx["by_rel"]
    causes = by.get("Causes", {})
    # Seeds: any Causes head with a clean downstream chain
    heads = [h for h in causes if is_clean_phrase(h, 4)]
    rng.shuffle(heads)
    for s in heads:
        nxt1 = [(e, w) for e, w in causes[s]
                if is_clean_phrase(e, 4) and e != s]
        if not nxt1: continue
        e1 = rng.choices(nxt1, weights=[w for _, w in nxt1])[0][0]
        triples = [{"rel": "Causes", "head": s, "tail": e1}]
        if depth >= 2 and e1 in causes:
            nxt2 = [(e, w) for e, w in causes[e1]
                    if is_clean_phrase(e, 4) and e not in (s, e1)]
            if nxt2:
                e2 = rng.choices(nxt2, weights=[w for _, w in nxt2])[0][0]
                triples.append({"rel": "Causes", "head": e1, "tail": e2})
        return triples
    return None


def tpl_tool_for_goal(idx, rng):
    """An artifact that CapableOf or UsedFor an action matching a goal.

    Triples:
        (MotivatedByGoal, activity, goal)
        (UsedFor|CapableOf, tool,   action-that-resembles-goal)
    """
    by = idx["by_rel"]; ip = idx["isa_parents"]
    mot = by.get("MotivatedByGoal", {})
    # Flatten activities with decent weight
    flat = [(h, g, w) for h, tails in mot.items() for g, w in tails
            if w >= 2.0 and is_clean_phrase(h, 4) and is_clean_phrase(g, 5)]
    if not flat: return None
    rng.shuffle(flat)
    for h, g, _ in flat:
        # Look for a portable tool whose UsedFor tail overlaps the goal tokens
        g_tokens = set(g.lower().split())
        if not g_tokens: continue
        hits = []
        for rel_name in ("UsedFor", "CapableOf"):
            for art, tails in by.get(rel_name, {}).items():
                if not is_portable(art, ip): continue
                for tail, w in tails:
                    if not is_clean_phrase(tail, 5): continue
                    t_tokens = set(tail.lower().split())
                    # Content-word overlap
                    shared = g_tokens & t_tokens
                    shared -= {"a","an","the","to","of","for","in","on","at","and"}
                    if len(shared) >= 1:
                        hits.append((rel_name, art, tail, w, len(shared)))
        if not hits: continue
        hits.sort(key=lambda x: (-x[4], -x[3]))
        rel_name, art, tail, _, _ = hits[0]
        return [
            {"rel": "MotivatedByGoal", "head": h,    "tail": g},
            {"rel": rel_name,           "head": art, "tail": tail},
        ]
    return None


def tpl_location_activity(idx, rng):
    """An activity and where it typically happens.

    Triples:
        (AtLocation, activity, location)
        (MotivatedByGoal, activity, goal)   # if known
    """
    by = idx["by_rel"]
    atloc = by.get("AtLocation", {})
    mot = by.get("MotivatedByGoal", {})
    heads = [h for h in atloc if is_clean_phrase(h, 4)]
    rng.shuffle(heads)
    for h in heads:
        locs = [(l, w) for l, w in atloc[h] if is_clean_phrase(l, 4)]
        if not locs: continue
        loc = rng.choices(locs, weights=[w for _, w in locs])[0][0]
        triples = [{"rel": "AtLocation", "head": h, "tail": loc}]
        if h in mot:
            goals = [(g, w) for g, w in mot[h] if w >= 2.0 and is_clean_phrase(g, 5)]
            if goals:
                g = rng.choices(goals, weights=[w for _, w in goals])[0][0]
                triples.append({"rel": "MotivatedByGoal", "head": h, "tail": g})
        return triples
    return None


def tpl_capability_location(idx, rng):
    """Agent/tool, its location, and what it can do.

    Triples:
        (AtLocation, X, place)
        (CapableOf,  X, action)
    """
    by = idx["by_rel"]
    cap = by.get("CapableOf", {})
    atloc = by.get("AtLocation", {})
    candidates = [h for h in cap if h in atloc and is_clean_phrase(h, 3)]
    rng.shuffle(candidates)
    for h in candidates:
        locs = [(l, w) for l, w in atloc[h] if is_clean_phrase(l, 4)]
        acts = [(a, w) for a, w in cap[h]
                if w >= 2.0 and is_clean_phrase(a, 5) and a != h]
        if not locs or not acts: continue
        loc = rng.choices(locs, weights=[w for _, w in locs])[0][0]
        act = rng.choices(acts, weights=[w for _, w in acts])[0][0]
        return [
            {"rel": "AtLocation", "head": h, "tail": loc},
            {"rel": "CapableOf",  "head": h, "tail": act},
        ]
    return None


def tpl_capability_consequence(idx, rng):
    """Capability chained to causal consequence.

    Triples:
        (CapableOf, X, action)
        (Causes,    action, effect)
    """
    by = idx["by_rel"]
    cap = by.get("CapableOf", {})
    causes = by.get("Causes", {})
    # Flatten (head, action, w) where action is also a Causes head
    pairs = []
    for h, lst in cap.items():
        if not is_clean_phrase(h, 3): continue
        for act, w in lst:
            if w < 2.0: continue
            if act == h: continue  # drop self-loops
            if act in causes and is_clean_phrase(act, 4):
                pairs.append((h, act, w))
    if not pairs: return None
    rng.shuffle(pairs)
    for h, act, _ in pairs:
        effs = [(e, w) for e, w in causes[act]
                if w >= 1.5 and is_clean_phrase(e, 4) and e != act and e != h]
        if not effs: continue
        eff = rng.choices(effs, weights=[w for _, w in effs])[0][0]
        return [
            {"rel": "CapableOf", "head": h,   "tail": act},
            {"rel": "Causes",    "head": act, "tail": eff},
        ]
    return None


def tpl_tool_function_dual(idx, rng):
    """Artifact confirmed as a tool via both CapableOf and UsedFor.

    Triples:
        (UsedFor,   tool, usage)
        (CapableOf, tool, action)
    """
    by = idx["by_rel"]; ip = idx["isa_parents"]
    cap = by.get("CapableOf", {})
    usedfor = by.get("UsedFor", {})
    candidates = [h for h in cap if h in usedfor and is_portable(h, ip)]
    rng.shuffle(candidates)
    for h in candidates:
        uses = [(t, w) for t, w in usedfor[h]
                if w >= 1.5 and is_clean_phrase(t, 5)]
        acts = [(t, w) for t, w in cap[h]
                if w >= 1.5 and is_clean_phrase(t, 5) and t != h]
        if not uses or not acts: continue
        usage = rng.choices(uses, weights=[w for _, w in uses])[0][0]
        action = rng.choices(acts, weights=[w for _, w in acts])[0][0]
        if usage == action: continue
        return [
            {"rel": "UsedFor",   "head": h, "tail": usage},
            {"rel": "CapableOf", "head": h, "tail": action},
        ]
    return None


def tpl_property_cause(idx, rng):
    """A thing has a property, and that property causes something.

    Triples:
        (HasProperty, thing,    property)
        (Causes,      property, effect)   # when property is a state
    """
    by = idx["by_rel"]
    hp = by.get("HasProperty", {})
    causes = by.get("Causes", {})
    heads = [h for h in hp if is_clean_phrase(h, 3)]
    rng.shuffle(heads)
    for h in heads:
        props = [(p, w) for p, w in hp[h]
                 if is_clean_phrase(p, 3) and p in causes]
        if not props: continue
        prop = rng.choices(props, weights=[w for _, w in props])[0][0]
        effs = [(e, w) for e, w in causes[prop] if is_clean_phrase(e, 4)]
        if not effs: continue
        eff = rng.choices(effs, weights=[w for _, w in effs])[0][0]
        return [
            {"rel": "HasProperty", "head": h,    "tail": prop},
            {"rel": "Causes",      "head": prop, "tail": eff},
        ]
    return None


TEMPLATES = {
    "weather_mitigation":     tpl_weather_mitigation,
    "prerequisite_chain":     tpl_prerequisite_chain,
    "causal_chain":           tpl_causal_chain,
    "tool_for_goal":          tpl_tool_for_goal,
    "location_activity":      tpl_location_activity,
    "property_cause":         tpl_property_cause,
    "capability_location":    tpl_capability_location,
    "capability_consequence": tpl_capability_consequence,
    "tool_function_dual":     tpl_tool_function_dual,
}


# ---- Output -------------------------------------------------------------

def fmt_chain(name: str, triples: list) -> str:
    lines = [f"[{name}]"]
    for t in triples:
        lines.append(f"  {t['rel']:<18}  {t['head']:<34}  {t['tail']}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=24)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--template", default=None,
                   help="restrict to one template (default: cycle all)")
    p.add_argument("--jsonl", type=Path, default=None,
                   help="also write chains as JSONL to this path")
    args = p.parse_args()

    if args.rebuild_index and CN_INDEX.exists():
        CN_INDEX.unlink()

    idx = load_index()
    rng = random.Random(args.seed)

    templates = [args.template] if args.template else list(TEMPLATES)
    if any(t not in TEMPLATES for t in templates):
        raise SystemExit(f"unknown template; valid: {list(TEMPLATES)}")

    jsonl_fh = open(args.jsonl, "w") if args.jsonl else None
    seen = set(); emitted = 0; tries = 0
    while emitted < args.n and tries < args.n * 30:
        tries += 1
        name = templates[tries % len(templates)]
        triples = TEMPLATES[name](idx, rng)
        if not triples: continue
        key = (name, tuple((t["rel"], t["head"], t["tail"]) for t in triples))
        if key in seen: continue
        seen.add(key)
        emitted += 1
        print(fmt_chain(name, triples))
        print()
        if jsonl_fh:
            jsonl_fh.write(json.dumps({"template": name, "triples": triples}) + "\n")
    if jsonl_fh: jsonl_fh.close()


if __name__ == "__main__":
    main()
