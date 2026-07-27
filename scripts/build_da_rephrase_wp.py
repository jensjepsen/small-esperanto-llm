"""Build a rephrase-instruction SFT dataset from danish-word-problems-reworded-v1.

Each source row has (q_orig, q_new) — templated math problem vs natural-language
rewrite. We emit two training rows per source row:
  * Forward (orig → new):  "instruction + q_orig" → q_new    (compact → narrative)
  * Reverse (new → orig):  "instruction + q_new"  → q_orig   (narrative → compact)

Instruction pool is intentionally varied: tone (formal/polite/casual/imperative/
question), verb (omskriv/omformulér/sig/skriv om/parafrasér/formuler), framing
(direct/roleplay/meta/constraint), and directional hints (længere/kortere/mere
narrativt/mere formelt/mindre teknisk). Neutral templates work either direction.

Usage:
  uv run python scripts/build_da_rephrase_wp.py --out data/da_rephrase_wp_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


# ── Forward: orig (compact/formal) → new (narrative/concrete) ────────────────
# Use these when we want the assistant to ADD context/narrative.
FORWARD_TEMPLATES = [
    "Skriv følgende matematikopgave om, så den lyder som noget fra en lærebog: {q}",
    "Omformulér denne opgave med mere fortællende sprog: {q}",
    "Kan du gøre denne opgave mere levende og konkret?\n\n{q}",
    "Skriv opgaven om, så den handler om en dagligdags situation.\n\nOpgave: {q}",
    "Præsentér følgende opgave som en lille historie: {q}",
    "Gør denne opgave mere naturlig at læse: {q}",
    "Omskriv med almindeligt hverdagssprog:\n{q}",
    "Fortæl denne opgave som om du fortalte den til en ven — {q}",
    "Formulér den her mere flydende og mindre teknisk: {q}",
    "Skriv om, så det ikke lyder som en direkte formel.\n{q}",
    "Skriv følgende opgave om, så den ligner et virkeligt problem: {q}",
    "Prøv at formulere det med mere kød på: {q}",
    "Giv opgaven et mere hverdagsagtigt præg — {q}",
    "Skriv en version, der er lettere at forestille sig: {q}",
    "Klæd opgaven på med lidt kontekst: {q}",
    "Fortæl det som et lille scenarie: {q}",
    "Formulér som en tekstopgave med scenarie: {q}",
    # Roleplay
    "Du er en lærer. Skriv følgende opgave om, så dine elever bedre kan relatere til den: {q}",
    "Skriv denne opgave som en journalist ville formulere den i en avis: {q}",
    "Forestil dig, at du forklarer opgaven til en 10-årig — skriv den om: {q}",
    "Du er forfatter til en matematikbog for udskolingen. Omformulér denne opgave: {q}",
    "Skriv opgaven om, som om den stammede fra virkeligheden — ikke fra en lærebog: {q}",
    # Meta / question form
    "Hvordan kunne man præsentere følgende opgave på en mere naturlig måde?\n{q}",
    "Er der en bedre, mere fortællende måde at formulere følgende opgave på?\n{q}",
    # Constraint-flavored (still fits the mapping)
    "Skriv opgaven om — gerne længere og med mere kontekst: {q}",
    "Formulér opgaven med mindst én dagligdags situation: {q}",
]

# ── Reverse: new (narrative/concrete) → orig (compact/formal) ────────────────
REVERSE_TEMPLATES = [
    "Skriv denne opgave så kort og direkte som muligt: {q}",
    "Fjern al 'pynt' — hvad er opgaven i sin reneste form?\n{q}",
    "Omformulér til en klassisk matematikopgave uden narrativ: {q}",
    "Reducér til det essentielle: {q}",
    "Skriv opgaven som en formel tekstopgave: {q}",
    "Gør denne opgave mere kompakt og præcis: {q}",
    "Omskriv i det mest matematisk-tekniske sprog: {q}",
    "Fjern kontekst og narrativ; hvad er selve opgaven?\n{q}",
    "Formulér i den mest direkte form: {q}",
    "Skær unødvendige detaljer væk: {q}",
    "Sig det samme, bare kortere og mere teknisk: {q}",
    "Fokusér på det matematiske indhold; drop det narrative: {q}",
    "Formulér som en simpel opgave uden historie: {q}",
    "Skriv en strippet version af opgaven: {q}",
    "Ryd op i formuleringen og gør den præcis: {q}",
    "Skriv om til en klassisk lærebogsformulering: {q}",
    "Reducér til blot spørgsmålet og tallene: {q}",
    "Kort og formelt, tak — {q}",
    "Skriv en mere komprimeret version: {q}",
    # Roleplay
    "Du er lærer og skal skrive opgaven ind i en test. Formulér den kort og præcist: {q}",
    "Du er matematikbogens redaktør — skriv opgaven mere kompakt: {q}",
    "Som eksamensopgave, hvordan ville denne opgave så være formuleret? {q}",
    # Meta / question form
    "Hvordan ville en matematikbog formulere følgende problem kort?\n{q}",
    "Kan du give den essentielle, formelle version af følgende opgave?\n{q}",
    # Constraint-flavored
    "Formulér opgaven i så få ord som muligt: {q}",
    "Skriv opgaven på under 25 ord: {q}",
]

# ── Neutral: same-content-different-wording, direction-agnostic ──────────────
# Usable in either direction — samples add texture without hinting.
NEUTRAL_TEMPLATES = [
    "Sig følgende på en anden måde: {q}",
    "Skriv om med andre ord: {q}",
    "Omformulér: {q}",
    "Kan du give mig en anden version af følgende?\n\n{q}",
    "Skriv følgende igen — men anderledes: {q}",
    "Hvad er en anden måde at formulere det på?\n{q}",
    "Formulér følgende med andre ord: {q}",
    "Prøv at sige det her på en helt anden måde: {q}",
    "Kan du parafrasere følgende?\n{q}",
    "Omskriv: {q}",
    "Giv en alternativ formulering af følgende opgave: {q}",
    "Skriv opgaven om, uden at ændre betydningen: {q}",
    "Sig det med andre ord, men bevar indholdet: {q}",
    "Formulér følgende matematikopgave anderledes — samme mening, ny form: {q}",
]

FORWARD_POOL = FORWARD_TEMPLATES + NEUTRAL_TEMPLATES
REVERSE_POOL = REVERSE_TEMPLATES + NEUTRAL_TEMPLATES


def build_rows(ds, rng: random.Random) -> list[dict]:
    out = []
    skipped = 0
    for r in ds:
        if r.get("status") != "ok":
            skipped += 1
            continue
        qo, qn = r["q_orig"].strip(), r["q_new"].strip()
        if not qo or not qn or qo == qn:
            skipped += 1
            continue
        # Forward: orig → new
        fwd_tpl = rng.choice(FORWARD_POOL)
        out.append({
            "messages": [
                {"role": "user", "content": fwd_tpl.format(q=qo)},
                {"role": "assistant", "content": qn},
            ],
            "direction": "orig_to_new",
            "orig_idx": r.get("orig_idx"),
        })
        # Reverse: new → orig
        rev_tpl = rng.choice(REVERSE_POOL)
        out.append({
            "messages": [
                {"role": "user", "content": rev_tpl.format(q=qn)},
                {"role": "assistant", "content": qo},
            ],
            "direction": "new_to_orig",
            "orig_idx": r.get("orig_idx"),
        })
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo", default="jensjepsen/danish-word-problems-reworded-v1")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"loading {args.repo}…", flush=True)
    ds = load_dataset(args.repo, split="train")
    print(f"source rows: {len(ds):,}")

    rows, skipped = build_rows(ds, rng)
    rng.shuffle(rows)
    print(f"built {len(rows):,} rephrase rows  (skipped {skipped:,} malformed/degenerate)")
    print(f"  forward templates: {len(FORWARD_TEMPLATES)} + {len(NEUTRAL_TEMPLATES)} neutral")
    print(f"  reverse templates: {len(REVERSE_TEMPLATES)} + {len(NEUTRAL_TEMPLATES)} neutral")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}")

    # Preview 3 samples of each direction
    print("\n=== 3 forward samples ===")
    fwds = [r for r in rows if r["direction"] == "orig_to_new"][:3]
    for r in fwds:
        print(f"\nU: {r['messages'][0]['content']}")
        print(f"A: {r['messages'][1]['content']}")
    print("\n=== 3 reverse samples ===")
    revs = [r for r in rows if r["direction"] == "new_to_orig"][:3]
    for r in revs:
        print(f"\nU: {r['messages'][0]['content']}")
        print(f"A: {r['messages'][1]['content']}")


if __name__ == "__main__":
    main()
