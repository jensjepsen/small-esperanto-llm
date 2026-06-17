"""Generate multi-step reasoning chains from ATOMIC commonsense triples.

Chains 3–5 related triples about the same head into a step-by-step answer with
"Pripensu paŝon post paŝo. Unue... Due... Trie... La respondo estas..." structure.
This is the same shape as the arithmetic CoT format the model already learned,
applied to commonsense/causal reasoning instead of math.

Output: data/sft/sft_atomic_chains.jsonl
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


# ---- Per-relation sentence templates -----------------------------------
# {tail} substituted at render time; mood/tense matches typical narrative flow.

REL_SENTENCES = {
    "xNeed":     ["antaŭ tio li devis {tail}",
                  "antaŭe li bezonis {tail}",
                  "li devis {tail} unue"],
    "xIntent":   ["lia intenco estis {tail}",
                  "li volis {tail}",
                  "li planis {tail}"],
    "xEffect":   ["rezulte {tail}",
                  "kiel rezulto, {tail}",
                  "tio kondukas al jeno: {tail}"],
    "xReact":    ["li sentas sin {tail}",
                  "li sentas {tail}",
                  "lia reago estas {tail}"],
    "xWant":     ["poste li volas {tail}",
                  "sekve li deziras {tail}",
                  "fine li volas {tail}"],
    "xAttr":     ["oni povus priskribi lin kiel {tail}",
                  "li aperas kiel {tail}"],
    "xReason":   ["la kialo estas: {tail}",
                  "li tion faras ĉar {tail}"],
    "oWant":     ["alia persono volas {tail}",
                  "Maria volas {tail}"],
    "oReact":    ["alia persono sentas {tail}",
                  "Maria sentas {tail}"],
    "oEffect":   ["al alia persono okazas: {tail}",
                  "Maria spertas: {tail}"],
    "isBefore":  ["antaŭ tio okazis: {tail}"],
    "isAfter":   ["post tio okazas: {tail}"],
    "HinderedBy": ["tamen tion povus malhelpi: {tail}",
                   "sed povus iri malbone: {tail}"],
    "Causes":    ["tio kaŭzas: {tail}",
                  "rezulte: {tail}"],
    "HasSubEvent": ["parto de tio inkluzivas: {tail}",
                    "tio implikas: {tail}"],
    "xCausesOf":  ["tio okazigas: {tail}"],
}

# Canonical narrative ordering — earlier relations come first in the chain.
RELATION_ORDER = {
    "xNeed":      0,    # before-state: what was needed
    "isBefore":   1,    # what came before in the timeline
    "xIntent":    2,    # subject's motivation
    "xReason":    3,    # the cause
    "Causes":     4,    # general cause-effect
    "HasSubEvent":5,    # what's part of the event
    "xEffect":    6,    # consequence on subject
    "oEffect":    7,    # consequence on others
    "xReact":     8,    # subject's emotion
    "oReact":     9,    # others' emotions
    "xAttr":     10,    # subject's attribute (description)
    "xWant":     11,    # subject's next desire
    "oWant":     12,    # others' wants
    "isAfter":   13,    # what came after in the timeline
    "HinderedBy":14,    # caveat / obstacle (often last)
}

CONNECTORS = ["Unue,", "Due,", "Trie,", "Plue,", "Krome,", "Fine,", "Lastvice,"]

QUESTION_TEMPLATES_INDICATIVE = [
    "Kio okazas, kiam {head}?",
    "Priskribu paŝon post paŝo: {head}.",
    "Klarigu la situacion: {head}.",
    "Kio kaj kial okazas, dum {head}?",
    "Detale priskribu: {head}.",
    "{head}. Kio okazas, paŝon post paŝo?",
]
QUESTION_TEMPLATES_COUNTERFACTUAL = [
    "Kio okazus, se {head_cond}?",
    "Imagu ke {head_cond}. Kio okazus paŝon post paŝo?",
    "Se {head_cond}, kio sekvas?",
]

ANSWER_OPENINGS = [
    "Pripensu paŝon post paŝo.",
    "Ni analizu ĉi tion paŝon post paŝo.",
    "Ekzamenu la situacion paŝon post paŝo.",
    "Pripensu la sekvojn unu post la alia.",
    "Ni rigardu paŝon post paŝo.",
]

CONCLUSION_TEMPLATES = [
    "La respondo estas: la situacio havas plurajn sekvojn kaj kuntekstojn.",
    "La respondo estas: tio estas plurfaca evento kun pluraj efikoj.",
    "La respondo estas: estas multaj aspektoj rilataj al ĉi tio.",
    "Konklude, la situacio havas profundajn implikiĝojn.",
    "Tial, la evento influas multajn aspektojn.",
    "La respondo estas: ĉi tio kondukas al konsiderindaj sekvoj.",
]


# ---- Head wrapping (copied from generate_atomic_qa.py for self-containment) ----

_VERB_INF = re.compile(r"\w{2,}[iu]$", re.UNICODE)
_NOT_VERBS = {"tri", "ni", "vi", "li", "ŝi", "ili", "ĉi", "mi",
              "tro", "ĉiu", "neniu", "kiu", "iu", "plu", "ju", "ĝu"}

def _is_verb_head(h: str) -> bool:
    parts = h.strip().split()
    if not parts:
        return False
    first = parts[0].lower()
    if first in _NOT_VERBS:
        return False
    return bool(_VERB_INF.fullmatch(first))

def _conjugate(h: str, ending: str) -> str:
    parts = h.strip().split()
    if not parts:
        return h
    first = parts[0]
    if first.endswith(("i", "u")):
        parts[0] = first[:-1] + ending
    return " ".join(parts)

def head_present(h: str) -> str:
    """For indicative question: 'migri' → 'iu migras'; 'Petro kuras' stays."""
    if _is_verb_head(h):
        return "iu " + _conjugate(h, "as")
    return h

def head_conditional(h: str) -> str:
    """For counterfactual: 'migri' → 'iu migrus'; 'Petro kuras' → 'Petro kurus'."""
    if _is_verb_head(h):
        return "iu " + _conjugate(h, "us")
    parts = h.split()
    for i, p in enumerate(parts):
        low = p.lower()
        if low.endswith(("as", "is", "os")) and len(p) > 3:
            parts[i] = p[:-2] + "us"
            break
    return " ".join(parts)


# ---- Index ------------------------------------------------------------

def load_index(path: Path) -> dict[str, dict[str, list[str]]]:
    """Build: head → relation → [tails], filtered to chain-relevant relations."""
    by_head: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            h, r, t = d["head"].strip().rstrip(".,;:"), d["relation"], d["tail"].strip().rstrip(".,;:")
            if not h or not t or h.lower() == "neniu" or t.lower() == "neniu":
                continue
            if r not in REL_SENTENCES:
                continue
            low = (h + " " + t).lower()
            if re.search(r"\bperson\s*[xy]\b", low) or re.search(r"\bx\b", t.lower()):
                continue
            by_head[h][r].append(t)
    return {h: dict(rs) for h, rs in by_head.items()}


# ---- Chain generation -------------------------------------------------

def make_chain(head: str, relations: dict[str, list[str]], rng,
               counterfactual_prob: float = 0.20) -> dict | None:
    """Sample 3–5 relations about `head`, render as a step-by-step answer."""
    available = list(relations.keys())
    if len(available) < 3:
        return None

    n_steps = rng.randint(3, min(5, len(available)))
    chosen = rng.sample(available, n_steps)
    # Sort in canonical narrative order
    chosen.sort(key=lambda r: RELATION_ORDER.get(r, 99))

    # Render steps
    steps = []
    for i, rel in enumerate(chosen):
        tail = rng.choice(relations[rel])
        sentence_template = rng.choice(REL_SENTENCES[rel])
        sentence = sentence_template.format(tail=tail)
        connector = CONNECTORS[i] if i < len(CONNECTORS) else CONNECTORS[-1]
        steps.append(f"{connector} {sentence}.")

    # Pick question format (indicative vs counterfactual)
    use_cf = rng.random() < counterfactual_prob
    if use_cf:
        q_template = rng.choice(QUESTION_TEMPLATES_COUNTERFACTUAL)
        question = q_template.format(head_cond=head_conditional(head))
    else:
        q_template = rng.choice(QUESTION_TEMPLATES_INDICATIVE)
        question = q_template.format(head=head_present(head))

    opening = rng.choice(ANSWER_OPENINGS)
    conclusion = rng.choice(CONCLUSION_TEMPLATES)
    answer = f"{opening} {' '.join(steps)} {conclusion}"

    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/atomic_eo/atomic_eo.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/sft/sft_atomic_chains.jsonl"))
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--counterfactual-prob", type=float, default=0.20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    by_head = load_index(args.input)
    eligible = [h for h, rs in by_head.items() if len(rs) >= 3]
    print(f"Loaded {len(by_head):,} heads, {len(eligible):,} with ≥3 chain-relations")

    rng = random.Random(args.seed)

    if args.dry_run:
        print("\n--- 4 sample chains ---\n")
        shown = 0
        tries = 0
        while shown < 4 and tries < 50:
            head = rng.choice(eligible)
            chain = make_chain(head, by_head[head], rng, args.counterfactual_prob)
            tries += 1
            if not chain:
                continue
            shown += 1
            tag = " [CF]" if "se " in chain["messages"][0]["content"].lower() else ""
            print(f"=== Chain {shown}{tag} (head: {head}) ===")
            print(f"USER: {chain['messages'][0]['content']}")
            print(f"ASST: {chain['messages'][1]['content']}")
            print()
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "w") as f:
        while written < args.n:
            head = rng.choice(eligible)
            chain = make_chain(head, by_head[head], rng, args.counterfactual_prob)
            if not chain:
                continue
            f.write(json.dumps(chain, ensure_ascii=False) + "\n")
            written += 1
            if written % 2000 == 0:
                print(f"  written {written:,}/{args.n:,}")
    print(f"\nWrote {written:,} reasoning chains → {args.out}")


if __name__ == "__main__":
    main()
