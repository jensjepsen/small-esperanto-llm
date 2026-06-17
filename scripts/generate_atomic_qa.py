"""Generate multi-turn commonsense QA SFT from translated ATOMIC triples.

Each conversation drills down on a single ATOMIC head (event or object) across
multiple relations. The first user turn establishes context; each follow-up
asks about a different relation, so the model learns to carry context and
produce coherent commonsense chains.

Example conversation:
  user:       Petro komencas komercon. Kion li intencas?
  assistant:  Gajni monon.
  user:       Kion li bezonis antaŭe?
  assistant:  Esplori la merkaton.
  user:       Kion li sentas?
  assistant:  Optimismon.

Input:  data/atomic_eo/atomic_eo.jsonl
Output: data/sft/sft_atomic_qa.jsonl (train_sft.py format)
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


# ---- Per-relation question templates ----------------------------------
# Multiple variants; random pick per turn adds diversity.
# {h} is the head (used only in FIRST turn context-setting).

FIRST_TURN_QUESTIONS = {
    # Social / PersonX events
    "xIntent":   ["{h}. Kion li intencas?", "{h}. Kial li tion faras?", "{h}. Kun kia intenco?"],
    "xReason":   ["{h}. Kial li tion faras?", "{h}. Kio estas la kialo?"],
    "xWant":     ["{h}. Kion li deziras poste?", "{h}. Kion li volas fari sekve?"],
    "xNeed":     ["{h}. Kion li bezonis antaŭe?", "{h}. Kion li devis fari unue?"],
    "xEffect":   ["{h}. Kio okazas al li?", "{h}. Kio estas la rezulto por li?"],
    "xReact":    ["{h}. Kion li sentas?", "{h}. Kiel li reagas?"],
    "xAttr":     ["{h}. Kia li estas?", "{h}. Kiel vi priskribus lin?"],
    "oWant":     ["{h}. Kion la alia persono deziras?", "{h}. Kion Maria volas?"],
    "oReact":    ["{h}. Kion la alia persono sentas?", "{h}. Kion Maria sentas?"],
    "oEffect":   ["{h}. Kio okazas al la alia persono?", "{h}. Kio okazas al Maria?"],
    # Event / temporal / causal
    "isBefore":  ["{h}. Kio okazis antaŭ tio?", "{h}. Kio estis antaŭe?"],
    "isAfter":   ["{h}. Kio okazis antaŭe, kondukante al tio?",
                  "Kio okazis antaŭ: {h}?"],
    "HinderedBy": ["{h}. Kio povus malhelpi tion?",
                   "{h}. Kio povus iri malbone?"],
    "HasSubEvent": ["{h}. Kion tio inkluzivas?", "{h}. Kio estas parto de tio?"],
    "Causes":    ["Kio okazas, kiam okazas {h}?",
                  "{h}. Kio estas la rezulto?"],
    "isFilledBy": ["{h}. Kia aĵo plenigas la mankon?",
                   "{h}. Kio povas plenigi la lokon?"],
    # Physical / object — {h_acc} = accusative form when verb demands it
    "AtLocation": ["Kie oni trovas {h_acc}?", "Kie troviĝas {h}?"],
    "ObjectUse":  ["Por kio oni uzas {h_acc}?", "Kion oni faras per {h}?"],
    "CapableOf":  ["Kion {h} povas fari?", "Kion {h} kapablas?"],
    "HasProperty": ["Kia estas {h}?", "Kian econ havas {h}?"],
    "MadeUpOf":   ["El kio konsistas {h}?", "El kio estas farita {h}?"],
    "Desires":    ["Kion {h} deziras?", "Kion {h} volas?"],
    "NotDesires": ["Kion {h} ne deziras?", "Kion {h} ne volas?"],
}

# Counterfactual / conditional variants — use "se" + -us mood.
# Only for event-style heads (PersonX sentences or verb infinitives).
# {h_cond} = head in conditional form: "Petro kuras" → "Petro kurus",
#                                       "migri"       → "iu migrus"
COUNTERFACTUAL_QUESTIONS = {
    "xIntent":   ["Se {h_cond}, kial li tion farus?",
                  "Kial iu {h_cond_bare}? Kun kia intenco?"],
    "xWant":     ["Se {h_cond}, kion li volus poste?",
                  "Se {h_cond}, kion li dezirus sekve?"],
    "xNeed":     ["Se {h_cond}, kion li devus fari antaŭe?",
                  "Kion iu devus fari, por ke {h_cond}?"],
    "xEffect":   ["Se {h_cond}, kio okazus al li?",
                  "Kio okazus al iu, se {h_cond}?"],
    "xReact":    ["Se {h_cond}, kion li sentus?",
                  "Kion li sentus, se {h_cond}?"],
    "xAttr":     ["Kia li estus, se {h_cond}?",
                  "Se {h_cond}, kiel ni priskribus lin?"],
    "xReason":   ["Kial iu {h_cond_bare}?",
                  "Se iu {h_cond_bare}, kio estas la kialo?"],
    "oWant":     ["Se {h_cond}, kion Maria volus?",
                  "Kion la alia persono dezirus, se {h_cond}?"],
    "oReact":    ["Se {h_cond}, kion Maria sentus?",
                  "Kion Maria sentus, se {h_cond}?"],
    "oEffect":   ["Se {h_cond}, kio okazus al Maria?",
                  "Se {h_cond}, kio okazus al la alia persono?"],
    "isBefore":  ["Se {h_cond}, kio okazus poste?"],
    "isAfter":   ["Kio devus okazi unue, por ke {h_cond}?"],
    "HinderedBy": ["Kio povus malhelpi, se {h_cond}?",
                   "Se {h_cond}, kio povus iri malbone?"],
    "HasSubEvent": ["Se {h_cond}, kiujn paŝojn li farus?"],
    "Causes":    ["Kio okazus, se {h_cond}?",
                  "Se {h_cond}, kio estus la rezulto?"],
}


# Follow-up questions — reference the head implicitly, not by name.
FOLLOWUP_QUESTIONS = {
    "xIntent":   ["Kion li intencas?", "Kial li tion faras?", "Kun kia intenco?"],
    "xReason":   ["Kial?", "Kio estas la kialo?"],
    "xWant":     ["Kion li deziras poste?", "Kion li volas fari sekve?"],
    "xNeed":     ["Kion li bezonis antaŭe?", "Kion li devis fari unue?"],
    "xEffect":   ["Kio okazas al li?", "Kion li spertas?"],
    "xReact":    ["Kion li sentas?", "Kiel li reagas?"],
    "xAttr":     ["Kia li estas?", "Kiel priskribi lin?"],
    "oWant":     ["Kaj Maria?", "Kion la alia persono deziras?", "Kion Maria volas?"],
    "oReact":    ["Kion Maria sentas?", "Kion la alia persono sentas?"],
    "oEffect":   ["Kio okazas al Maria?", "Kio okazas al la alia persono?"],
    "isBefore":  ["Kio okazis antaŭ tio?", "Kio estis antaŭe?"],
    "isAfter":   ["Kio okazis antaŭ tio?", "Kio estas la antaŭhistorio?"],
    "HinderedBy": ["Kio povus malhelpi?", "Kio povus iri malbone?"],
    "HasSubEvent": ["Kion tio inkluzivas?", "Kio estas parto de tio?"],
    "Causes":    ["Kio estas la rezulto?", "Kio sekvas?"],
    "isFilledBy": ["Kio povas plenigi la mankon?", "Per kio?"],
    "AtLocation": ["Kie oni trovas ĝin?", "Kie ĝi troviĝas?"],
    "ObjectUse":  ["Por kio oni uzas ĝin?", "Kion oni faras per ĝi?"],
    "CapableOf":  ["Kion ĝi kapablas?", "Kion ĝi povas fari?"],
    "HasProperty": ["Kia ĝi estas?", "Kian econ havas?"],
    "MadeUpOf":   ["El kio konsistas?", "El kio ĝi estas farita?"],
    "Desires":    ["Kion ĝi deziras?"],
    "NotDesires": ["Kion ĝi ne deziras?"],
}


def _clean(s: str) -> str:
    return s.strip().rstrip(".,;:").strip()


_NOM_ENDING = re.compile(r"(\w*)(oj|aj|o|a)$", re.UNICODE)
_VERB_INF = re.compile(r"\w{2,}[iu]$", re.UNICODE)  # stem ≥ 2 chars + -i/-u ending
_NOT_VERBS = {  # Esperanto words ending in -i/-u that aren't verbs
    "tri", "ni", "vi", "li", "ŝi", "ili", "ĉi", "mi",
    "tro", "ĉiu", "neniu", "kiu", "iu",
    "plu", "ju", "ĝu",
}


def is_verb_head(h: str) -> bool:
    """True if the head's first word is a bare verb infinitive (-i) or imperative (-u)."""
    parts = h.strip().split()
    if not parts:
        return False
    first = parts[0].lower()
    if first in _NOT_VERBS:
        return False
    return bool(_VERB_INF.fullmatch(first))


def conjugate_present(h: str) -> str:
    """Replace first-word verb ending -i or -u with -as (3rd-person present).

    'migri' → 'migras'
    'aŭskulti muzikon' → 'aŭskultas muzikon'
    'rilaksu' → 'rilaksas'
    """
    parts = h.strip().split()
    if not parts:
        return h
    first = parts[0]
    if first.endswith(("i", "u")):
        parts[0] = first[:-1] + "as"
    return " ".join(parts)


def wrap_verb_head(h: str) -> str:
    """Turn a verb-infinitive head into a subject-ed sentence: 'migri' → 'Iu migras'."""
    return "Iu " + conjugate_present(h)


def _cond(word: str) -> str:
    """Convert a verb to conditional -us form. 'kuras' → 'kurus', 'migri' → 'migrus'."""
    if word.endswith(("as", "is", "os")):
        return word[:-2] + "us"
    if word.endswith(("i", "u")):
        return word[:-1] + "us"
    return word


def to_conditional(h: str) -> str:
    """Head as full conditional clause.

    'Petro kuras'       → 'Petro kurus'
    'migri'             → 'iu migrus'
    'aŭskulti muzikon'  → 'iu aŭskultus muzikon'
    Only transforms the first verb found; subsequent verbs/nouns left alone.
    """
    parts = h.strip().split()
    if not parts:
        return h
    if is_verb_head(h):
        parts[0] = _cond(parts[0])
        return "iu " + " ".join(parts)
    # Non-verb-inf head: find the main verb (-as/-is/-os) and make it conditional
    for i, p in enumerate(parts):
        low = p.lower()
        if low.endswith(("as", "is", "os")) and len(p) > 3:
            parts[i] = _cond(p)
            break
    return " ".join(parts)


def to_conditional_bare(h: str) -> str:
    """Conditional form without subject prefix. For templates that already have 'iu'.
    'migri'            → 'migrus'
    'aŭskulti muzikon' → 'aŭskultus muzikon'
    """
    parts = h.strip().split()
    if not parts:
        return h
    if is_verb_head(h):
        parts[0] = _cond(parts[0])
        return " ".join(parts)
    # Non-verb-inf head: convert main verb
    for i, p in enumerate(parts):
        low = p.lower()
        if low.endswith(("as", "is", "os")) and len(p) > 3:
            parts[i] = _cond(p)
            break
    return " ".join(parts)


def is_event_head(h: str) -> bool:
    """Eligible for counterfactual framing: PersonX-style events or verb infinitives.
    Object/noun heads (lakto, ombrelo) are not good counterfactual candidates."""
    if is_verb_head(h):
        return True
    # PersonX events start with a capital proper noun; detect by heuristic
    parts = h.strip().split()
    if not parts:
        return False
    first = parts[0]
    # Capitalized first word + contains a conjugated verb (-as/-is/-os) → likely event
    if first[:1].isupper():
        return any(p.lower().endswith(("as", "is", "os")) and len(p) > 3 for p in parts)
    return False


_ESP_PREPS = {
    "de", "en", "al", "kun", "sur", "sub", "pri", "per", "por", "ĉe",
    "el", "da", "ekster", "inter", "kontraŭ", "tra", "trans",
    "apud", "antaŭ", "post", "ĝis", "ĉirkaŭ", "sen",
}


_ADJ_ENDING = re.compile(r"\w+aj?$", re.UNICODE)


def _can_decline(w: str) -> bool:
    if w.endswith(("on", "an", "ojn", "ajn")):  # already accusative
        return False
    return bool(_NOM_ENDING.fullmatch(w))


def _is_adjective(w: str) -> bool:
    return bool(_ADJ_ENDING.fullmatch(w)) and w.lower() not in {"la"}


def _inflect_with_agreement(parts: list[str], idx: int) -> None:
    """In-place: mark parts[idx] accusative and agree any adjectives before it."""
    parts[idx] = parts[idx] + "n"
    j = idx - 1
    while j >= 0 and _is_adjective(parts[j]):
        parts[j] = parts[j] + "n"
        j -= 1


def accusative(s: str) -> str:
    """Add -n to the head noun of a phrase, with adjective agreement.

    'lakto'                 → 'lakton'
    'la granda arbo'        → 'la grandan arbon'     (adjective agrees)
    'la ĉefurbo de Francio' → 'la ĉefurbon de Francio' (prep object stays nominative)
    'amiko en la urbo'      → 'amikon en la urbo'
    """
    s = s.rstrip()
    if not s:
        return s
    parts = s.split()

    # Find first preposition — if any, inflect the word immediately before it.
    for i in range(1, len(parts)):
        if parts[i].lower() in _ESP_PREPS:
            if _can_decline(parts[i - 1]):
                _inflect_with_agreement(parts, i - 1)
            return " ".join(parts)

    # No preposition — inflect the last word
    if _can_decline(parts[-1]):
        _inflect_with_agreement(parts, len(parts) - 1)
    return " ".join(parts)


def load_by_head(path: Path) -> dict[str, dict[str, list[str]]]:
    """Load triples, grouped: head → relation → [tails]."""
    by_head: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            h, r, t = _clean(d["head"]), d["relation"], _clean(d["tail"])
            if not h or not t or h.lower() == "neniu" or t.lower() == "neniu":
                continue
            if r not in FIRST_TURN_QUESTIONS:
                continue
            low = (h + " " + t).lower()
            if re.search(r"\bperson\s*[xy]\b", low) or re.search(r"\bx\b", t.lower()):
                continue
            by_head[h][r].append(t)
    return {h: dict(rs) for h, rs in by_head.items()}


def make_conversation(head: str, relations: dict[str, list[str]], rng,
                      min_turns: int = 2, max_turns: int = 5,
                      counterfactual_prob: float = 0.0) -> dict:
    """Build a multi-turn conversation drilling down on one head.

    With probability counterfactual_prob, frame the first turn as a counterfactual
    "Se X, kio farus Y?" — teaches the model to parse conditional mood. Only used
    when the head is event-shaped and the first relation has a counterfactual template.
    """
    rels = list(relations.keys())
    if len(rels) < min_turns:
        return None
    n_turns = rng.randint(min_turns, min(max_turns, len(rels)))
    chosen_rels = rng.sample(rels, n_turns)

    messages = []
    for i, rel in enumerate(chosen_rels):
        tail = rng.choice(relations[rel])
        if i == 0:
            use_cf = (
                rng.random() < counterfactual_prob
                and is_event_head(head)
                and rel in COUNTERFACTUAL_QUESTIONS
            )
            if use_cf:
                template = rng.choice(COUNTERFACTUAL_QUESTIONS[rel])
                q = template.format(
                    h_cond=to_conditional(head),
                    h_cond_bare=to_conditional_bare(head),
                )
            else:
                template = rng.choice(FIRST_TURN_QUESTIONS[rel])
                h_for_template = wrap_verb_head(head) if is_verb_head(head) else head
                q = template.format(h=h_for_template, h_acc=accusative(h_for_template))
        else:
            q = rng.choice(FOLLOWUP_QUESTIONS[rel])
        a = tail[0].upper() + tail[1:] if tail else tail
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a + "."})
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/atomic_eo/atomic_eo.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/sft/sft_atomic_qa.jsonl"))
    parser.add_argument("--n", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--counterfactual-prob", type=float, default=0.25,
                        help="Fraction of examples framed as counterfactuals (Se X, kio farus Y?)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    by_head = load_by_head(args.input)
    print(f"Loaded {len(by_head):,} distinct heads")
    # Count heads with enough relations
    eligible = [h for h, rs in by_head.items() if len(rs) >= args.min_turns]
    print(f"Eligible heads (≥{args.min_turns} relations): {len(eligible):,}")

    rng = random.Random(args.seed)

    if args.dry_run:
        print("\n--- 8 sample conversations ---\n")
        shown = 0
        tries = 0
        while shown < 8 and tries < 80:
            head = rng.choice(eligible)
            conv = make_conversation(head, by_head[head], rng, args.min_turns, args.max_turns,
                                     counterfactual_prob=args.counterfactual_prob)
            tries += 1
            if not conv:
                continue
            shown += 1
            is_cf = "se " in conv["messages"][0]["content"].lower()
            tag = " [CF]" if is_cf else ""
            print(f"=== Conversation {shown}{tag} (head: {head}) ===")
            for m in conv["messages"]:
                role = "USER" if m["role"] == "user" else "ASST"
                print(f"  {role}: {m['content']}")
            print()
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "w") as f:
        while written < args.n:
            head = rng.choice(eligible)
            conv = make_conversation(head, by_head[head], rng, args.min_turns, args.max_turns,
                                     counterfactual_prob=args.counterfactual_prob)
            if not conv:
                continue
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            written += 1
            if written % 2500 == 0:
                print(f"  written {written:,}/{args.n:,}")
    print(f"\nWrote {written:,} multi-turn QA conversations → {args.out}")


if __name__ == "__main__":
    main()
