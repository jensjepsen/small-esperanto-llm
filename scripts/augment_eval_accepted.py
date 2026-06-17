"""Augment a handcrafted eval jsonl with `accepted_answers` —
a curated list of equally-valid surface forms per Q/A pair.

The model's answer-variation (full-sentence wrappers, with/without "la",
"Estis X." vs bare X) is the rule rather than the exception in trained
output. Strict-equality eval over-penalizes these. This script generates
a small set of obvious alternates per record so a wrapped-but-correct
answer no longer counts as a failure.

Polarity-flipped variants are NOT generated (gold "fermita" never
accepts "malfermita") — only surface-form alternates.

Usage:
  python scripts/augment_eval_accepted.py \
    --in data/causal_corpus/eval_handcrafted_v30.jsonl \
    --out data/causal_corpus/eval_handcrafted_v31.jsonl
"""
import argparse
import json
import re
from pathlib import Path


def _strip_la(noun_phrase: str) -> str:
    p = noun_phrase.strip()
    if p.lower().startswith("la "):
        return p[3:].strip()
    return p


def _add_la(noun_phrase: str) -> str:
    p = noun_phrase.strip()
    if not p.lower().startswith("la "):
        return f"la {p}"
    return p


def _capitalize(s: str) -> str:
    if not s:
        return s
    return s[0].upper() + s[1:]


def _is_simple_noun_acc(s: str) -> bool:
    """Bare accusative noun like 'lakton', 'sandviĉon', 'la fenestron'."""
    t = _strip_la(s.rstrip(".")).strip()
    return bool(re.match(r"^[a-zĉĝĵĥŝŭ]+n$", t))


def _is_state_word(s: str) -> bool:
    """Adjective-like state value: ruĝa, fermita, plena, aktiva..."""
    t = s.strip().rstrip(".").lower()
    return bool(re.match(r"^[a-zĉĝĵĥŝŭ]+a$", t))


def _is_cardinal(s: str) -> bool:
    t = s.strip().rstrip(".").lower()
    return t in {"unu", "du", "tri", "kvar", "kvin", "ses", "sep",
                 "ok", "naŭ", "dek", "dudek", "tridek", "kvardek",
                 "kvindek", "sesdek", "sepdek", "okdek", "naŭdek",
                 "cent"}


def _is_loc_phrase(s: str) -> bool:
    """Phrases like 'Sur la breto', 'En la salono', 'Al la lago'."""
    t = s.strip().rstrip(".")
    return bool(re.match(
        r"^(En|Sur|Sub|Apud|Ĉe|Al|Tra)\s+(la\s+)?",
        t, re.IGNORECASE))


def _is_proper_name(s: str) -> bool:
    """Single capitalized token, no preposition / verb prefix."""
    t = s.strip().rstrip(".")
    if " " in t:
        return False
    return bool(re.match(r"^[A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+$", t))


def expand_accepted(gold: str, question: str) -> list[str]:
    """Compute a deduplicated list of accepted answer forms for `gold`."""
    out: list[str] = [gold]
    g = gold.rstrip(".").strip()

    # Bare noun-acc → also accept stripped/added "la", lowercased.
    if _is_simple_noun_acc(g):
        bare = _strip_la(g)
        out.extend([f"{bare}.", f"la {bare}.", f"La {bare}.", f"la {bare}"])

    # State word → wrappers
    if _is_state_word(g):
        out.extend([
            f"Estis {g}.",
            f"Estas {g}.",
            f"La X estas {g}.",   # generic placeholder — won't match anyway
            f"{g}.",
        ])

    # Cardinal numbers → accept "Estis N X-oj." too (matcher already
    # strips estis prefix but adding explicit forms helps)
    if _is_cardinal(g):
        out.extend([
            f"{g}.",
            _capitalize(g),
            _capitalize(g) + ".",
            f"Estis {g}.",
        ])

    # Location phrase → accept lowercase + with/without trailing dot
    if _is_loc_phrase(g):
        low = g.lower()
        out.extend([low, low + ".", g + "."])

    # Proper name → accept "Estis Name." and "Name." capitalized
    if _is_proper_name(g):
        out.extend([f"{g}.", f"Estis {g}.", g.lower(), g.lower() + "."])

    # Always include with/without trailing period
    if gold.endswith("."):
        out.append(gold[:-1])
    else:
        out.append(gold + ".")

    # Dedupe preserving order
    seen: set = set()
    deduped: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    n = 0
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            gold = rec["messages"][1]["content"]
            q = rec["messages"][0]["content"].split("Demando:")[-1].strip()
            accepted = expand_accepted(gold, q)
            rec["accepted_answers"] = accepted
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"Augmented {n} records → {args.out}")


if __name__ == "__main__":
    main()
