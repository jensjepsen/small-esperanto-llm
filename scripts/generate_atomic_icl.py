"""Generate ICL SFT examples from translated ATOMIC commonsense triples.

Each example shows K (head, tail) demonstrations under some relation, then asks
the model to fill the (K+1)-th tail. Two modes:

- pure-ICL (relation implicit): shots share a relation but it's not named; model
  must infer the relation from the examples.
- labeled: the relation gloss appears in the format (e.g., "Petro kuras →
  intencas: ?"); easier, still valuable for format-following with explicit labels.

Input:  data/atomic_eo/atomic_eo.jsonl (from translate_atomic.py)
Output: data/sft/sft_atomic_icl.jsonl (matches train_sft.py format)
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


# ---- Relation glosses -------------------------------------------------
# Override the glosses baked into atomic_eo.jsonl. Two variants:
#   LABEL_GLOSS:       short, for inline formats like "X → gloss: Y"
#   SENTENCE_CONNECTOR: natural Esperanto for prose-style "X. connector Y."

LABEL_GLOSS = {
    "xIntent":    "intencas",
    "xAttr":      "estas",
    "xWant":      "poste deziras",
    "xNeed":      "antaŭe bezonis",
    "xEffect":    "rezulte",
    "xReact":     "sentas",
    "xReason":    "ĉar",
    "oWant":      "alia deziras",
    "oReact":     "alia sentas",
    "oEffect":    "al alia okazas",
    "isBefore":   "antaŭe okazis",
    "isAfter":    "poste okazas",
    "HasSubEvent": "inkluzivas",
    "Causes":     "kaŭzas",
    "HinderedBy": "malhelpata de",
    "isFilledBy": "plenigeble per",
    "AtLocation": "troviĝas en",
    "ObjectUse":  "uzata por",
    "CapableOf":  "kapablas",
    "HasProperty": "havas econ",
    "MadeUpOf":   "konsistas el",
    "Desires":    "deziras",
    "NotDesires": "ne deziras",
}

# Natural connectors for sentence-chain renderer — "X. {connector} Y."
SENTENCE_CONNECTOR = {
    "xIntent":    "Intencas:",
    "xAttr":      "Estas:",
    "xWant":      "Poste deziras:",
    "xNeed":      "Antaŭe bezonis:",
    "xEffect":    "Rezulte:",
    "xReact":     "Sentas:",
    "xReason":    "Ĉar:",
    "oWant":      "Alia persono deziras:",
    "oReact":     "Alia persono sentas:",
    "oEffect":    "Rezulte al alia:",
    "isBefore":   "Antaŭ tio okazas:",
    "isAfter":    "Post tio okazas:",
    "HasSubEvent": "Inkluzivas:",
    "Causes":     "Kaŭzas:",
    "HinderedBy": "Sed:",
    "isFilledBy": "Plenigeble per:",
    "AtLocation": "Troviĝas en:",
    "ObjectUse":  "Uzata por:",
    "CapableOf":  "Kapablas:",
    "HasProperty": "Havas la econ:",
    "MadeUpOf":   "Konsistas el:",
    "Desires":    "Deziras:",
    "NotDesires": "Ne deziras:",
}


# ---- Renderers --------------------------------------------------------
# Each returns (user_body, assistant_template). {ans} replaced by the tail.

# --- Pure-ICL renderers (relation NOT in format) ---

def r_arrow_plain(shots, query, rel_eo=None):
    lines = [f"{h} → {t}" for h, t in shots] + [f"{query} →"]
    return "\n".join(lines), ""

def r_eq_plain(shots, query, rel_eo=None):
    lines = [f"{h} = {t}" for h, t in shots] + [f"{query} ="]
    return "\n".join(lines), ""

def r_colon(shots, query, rel_eo=None):
    lines = [f"{h}: {t}" for h, t in shots] + [f"{query}:"]
    return "\n".join(lines), ""

def r_dash(shots, query, rel_eo=None):
    lines = [f"{h} — {t}" for h, t in shots] + [f"{query} —"]
    return "\n".join(lines), ""

def r_numbered(shots, query, rel_eo=None):
    lines = [f"{i+1}. {h} → {t}" for i, (h, t) in enumerate(shots)]
    lines.append(f"{len(shots)+1}. {query} →")
    return "\n".join(lines), ""

def r_bullet(shots, query, rel_eo=None):
    lines = [f"- {h}: {t}" for h, t in shots] + [f"- {query}:"]
    return "\n".join(lines), ""

def r_bracket(shots, query, rel_eo=None):
    lines = [f"[A] {h} [B] {t}" for h, t in shots] + [f"[A] {query}"]
    return "\n".join(lines), "[B] {ans}"

def r_qa_tag(shots, query, rel_eo=None):
    lines = [f"<Q>{h}</Q> <A>{t}</A>" for h, t in shots] + [f"<Q>{query}</Q>"]
    return "\n".join(lines), "<A>{ans}</A>"


# --- Labeled renderers (relation gloss appears in format) ---

def r_label_arrow(shots, query, rel_eo):
    lines = [f"{h} → {rel_eo}: {t}" for h, t in shots] + [f"{query} → {rel_eo}:"]
    return "\n".join(lines), ""

def r_label_sentence(shots, query, rel_eo, connector=None):
    """Natural sentence-chain form: 'X. Connector: Y.'"""
    c = connector or rel_eo
    lines = [f"{h}. {c} {t}." for h, t in shots] + [f"{query}. {c}"]
    return "\n".join(lines), ""

def r_label_triple(shots, query, rel_eo):
    lines = [f"<{h}, {rel_eo}, {t}>" for h, t in shots] + [f"<{query}, {rel_eo}, ?>"]
    return "\n".join(lines), ""

def r_label_if(shots, query, rel_eo):
    """Works nicely for PersonX heads: 'Se X, tiam [rel_eo] Y'."""
    lines = [f"Se: {h}. Do: {rel_eo} {t}." for h, t in shots]
    lines.append(f"Se: {query}. Do: {rel_eo}")
    return "\n".join(lines), ""

def r_label_table(shots, query, rel_eo):
    head = f"| Situacio | {rel_eo.capitalize()} |\n| --- | --- |"
    lines = [head] + [f"| {h} | {t} |" for h, t in shots] + [f"| {query} |"]
    return "\n".join(lines), ""

def r_label_yaml(shots, query, rel_eo):
    key = rel_eo.replace(" ", "_")
    lines = [f"- situacio: {h}\n  {key}: {t}" for h, t in shots]
    lines.append(f"- situacio: {query}\n  {key}:")
    return "\n".join(lines), ""


PURE_RENDERERS = [r_arrow_plain, r_eq_plain, r_colon, r_dash, r_numbered,
                  r_bullet, r_bracket, r_qa_tag]
LABELED_RENDERERS = [r_label_arrow, r_label_sentence, r_label_triple,
                     r_label_if, r_label_table, r_label_yaml]


# ---- Preambles --------------------------------------------------------

PREAMBLES = [
    "", "", "", "", "",  # often none
    "Plenigu la mankon:\n",
    "Sekvante la ekzemplojn:\n",
    "Komplete la lastan linion:\n",
    "Daŭrigu la ŝablonon:\n",
    "Identigu la rilaton kaj plenumu:\n",
    "Laŭ la sama logiko:\n",
    "Same kiel supre:\n",
    "Ekzemploj de la sama rilato:\n",
    "Studu kaj kompletigu:\n",
    "Imitu la ŝablonon:\n",
    "Observu la rilaton inter la paroj:\n",
    "Uzante la saman rilaton kiel la ekzemploj:\n",
]


# ---- Main generator ---------------------------------------------------

def make_example(by_relation, rng, labeled_prob=0.40, shared_prob=0.20):
    """Sample one ICL example from ATOMIC-EO triples."""
    # Pick a relation (weighted by triple count, sqrt to not over-dominate)
    rels = list(by_relation.keys())
    weights = [len(by_relation[r]) ** 0.5 for r in rels]
    rel = rng.choices(rels, weights=weights, k=1)[0]
    rel_eo = by_relation[rel]["_gloss"]
    connector = by_relation[rel]["_connector"]
    pairs = by_relation[rel]["pairs"]

    K = rng.choice([2, 3, 3, 3, 4, 4, 5])
    if len(pairs) < K + 1:
        return None

    # Optionally pick shared-tail mode: sample K+1 pairs that share the same tail
    use_shared = rng.random() < shared_prob
    if use_shared:
        tail_groups = by_relation[rel]["tail_groups"]
        viable = [t for t, hs in tail_groups.items() if len(hs) >= K + 1]
        if viable:
            tail = rng.choice(viable)
            heads_sample = rng.sample(tail_groups[tail], K + 1)
            sample = [(h, tail) for h in heads_sample]
        else:
            use_shared = False

    if not use_shared:
        sample = rng.sample(pairs, K + 1)

    shots = sample[:-1]
    query, answer = sample[-1]

    # Pick renderer — labeled vs pure-ICL
    if rng.random() < labeled_prob:
        renderer = rng.choice(LABELED_RENDERERS)
    else:
        renderer = rng.choice(PURE_RENDERERS)

    # r_label_sentence takes an extra natural connector
    if renderer is r_label_sentence:
        user_body, ans_template = renderer(shots, query, rel_eo, connector=connector)
    else:
        user_body, ans_template = renderer(shots, query, rel_eo)
    preamble = rng.choice(PREAMBLES)
    user_msg = preamble + user_body
    assistant_msg = ans_template.format(ans=answer) if ans_template else answer

    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def _clean(s: str) -> str:
    """Strip trailing punctuation and whitespace — prevents double-period artifacts."""
    return s.strip().rstrip(".,;:").strip()


def load_index(path: Path) -> dict:
    """Build: relation → {pairs: [(head, tail)], tail_groups: {tail: [heads]}, _gloss: str, _connector: str}."""
    by_rel: dict[str, dict] = defaultdict(lambda: {
        "pairs": [], "tail_groups": defaultdict(list),
        "_gloss": "", "_connector": "",
    })
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            h, r, t = _clean(d["head"]), d["relation"], _clean(d["tail"])
            if not h or not t or h.lower() == "neniu" or t.lower() == "neniu":
                continue
            # Filter obvious untranslated leaks (Person X / PersonY / lone x)
            low = (h + " " + t).lower()
            if re.search(r"\bperson\s*[xy]\b", low) or re.search(r"\bx\b", t.lower()):
                continue
            by_rel[r]["pairs"].append((h, t))
            by_rel[r]["tail_groups"][t].append(h)
    # Dedupe pairs, attach cleaner glosses
    for r in by_rel:
        by_rel[r]["pairs"] = list({(h, t) for h, t in by_rel[r]["pairs"]})
        by_rel[r]["_gloss"] = LABEL_GLOSS.get(r, r)
        by_rel[r]["_connector"] = SENTENCE_CONNECTOR.get(r, LABEL_GLOSS.get(r, r) + ":")
    return dict(by_rel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/atomic_eo/atomic_eo.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/sft/sft_atomic_icl.jsonl"))
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labeled-prob", type=float, default=0.40,
                        help="Fraction of examples that include the Esperanto relation gloss")
    parser.add_argument("--shared-prob", type=float, default=0.20,
                        help="Fraction of examples using shared-tail mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print 8 sample examples instead of writing")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    by_rel = load_index(args.input)
    total = sum(len(v["pairs"]) for v in by_rel.values())
    print(f"Loaded {total:,} triples across {len(by_rel)} relations")
    for r, v in sorted(by_rel.items(), key=lambda x: -len(x[1]["pairs"])):
        print(f"  {r:15s} {len(v['pairs']):>5} pairs  gloss={v['_gloss']!r}")

    rng = random.Random(args.seed)

    if args.dry_run:
        print(f"\n--- 8 sample examples ---\n")
        shown = 0
        while shown < 8:
            ex = make_example(by_rel, rng, args.labeled_prob, args.shared_prob)
            if ex is None:
                continue
            print(f"=== Example {shown+1} ===\nUSER:\n{ex['messages'][0]['content']}\n\nASSISTANT:\n{ex['messages'][1]['content']}\n")
            shown += 1
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "w") as f:
        while written < args.n:
            ex = make_example(by_rel, rng, args.labeled_prob, args.shared_prob)
            if ex is None:
                continue
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1
            if written % 2000 == 0:
                print(f"  written {written:,}/{args.n:,}")
    print(f"\nWrote {written:,} ICL examples → {args.out}")


if __name__ == "__main__":
    main()
