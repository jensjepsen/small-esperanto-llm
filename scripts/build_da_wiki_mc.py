"""Build a Danish MC dataset from wiki closed-QA sources for teaching
label emission (letters A-D and numbers 1-4).

Source(s): jensjepsen/danish-wiki-closedqa-v1 and -stem-v1 (same schema).

For each source row (q, a):
  - Sample K-1 distractor answers, preferring same orig_title so
    distractors are topically plausible (falls back to random).
  - Assemble K options, shuffle position.
  - Label with letters (A/B/C/D) or numbers (1/2/3/4) — random per row.
  - Wrap in one of N instruction templates.
  - Emit messages-format row where assistant reply is just the label.

Rationale: DA v10-v16 all show cit-MC bias — model picks a positional
default (D or A) rather than reasoning. Teaching explicit label emission
from OUR OWN wiki-QA content should transfer without contaminating the
EuroEval knowledge benchmarks (which come from citizen-tests, not wiki).

Usage:
    python scripts/build_da_wiki_mc.py \\
        --output-repo jensjepsen/danish-wiki-mc-letters-v1 \\
        --k 4 --seed 42

Writes ~210k rows (one per source row, random letter-or-number labeling).
"""
from __future__ import annotations

import argparse
import random
import string
from collections import defaultdict

from datasets import Dataset, load_dataset

SOURCES = [
    "jensjepsen/danish-wiki-closedqa-v1",
    "jensjepsen/danish-wiki-closedqa-stem-v1",
]

# 20 instruction templates — mix of neutral and constraint-forcing.
LETTER_TEMPLATES = [
    "{q}\n\n{options}\n\nSvar med bogstavet på det korrekte svar.",
    "{q}\n\n{options}\n\nHvilket bogstav er korrekt?",
    "Besvar spørgsmålet ved at vælge det korrekte bogstav.\n\n{q}\n\n{options}",
    "Vælg det korrekte svar:\n\n{q}\n\n{options}\n\nSvar kun med et bogstav.",
    "{q}\n\nMuligheder:\n{options}\n\nSvar:",
    "Hvilken af følgende er korrekt?\n\n{q}\n\n{options}",
    "Spørgsmål: {q}\n\n{options}\n\nDet korrekte bogstav er:",
    "{q}\n\n{options}\n\nSkriv bogstavet der svarer til det rigtige valg.",
    "Læs spørgsmålet og vælg det bedste svar.\n\n{q}\n\n{options}",
    "{options}\n\n{q}\n\nSvar med det korrekte bogstav.",
]

NUMBER_TEMPLATES = [
    "{q}\n\n{options}\n\nSvar med tallet på det korrekte svar.",
    "{q}\n\n{options}\n\nHvilket tal er korrekt?",
    "Besvar spørgsmålet ved at vælge det korrekte tal.\n\n{q}\n\n{options}",
    "Vælg det korrekte svar:\n\n{q}\n\n{options}\n\nSvar kun med et tal.",
    "{q}\n\nMuligheder:\n{options}\n\nSvar:",
    "Hvilken af følgende er korrekt?\n\n{q}\n\n{options}",
    "Spørgsmål: {q}\n\n{options}\n\nDet korrekte tal er:",
    "{q}\n\n{options}\n\nSkriv tallet der svarer til det rigtige valg.",
    "Læs spørgsmålet og vælg det bedste svar.\n\n{q}\n\n{options}",
    "{options}\n\n{q}\n\nSvar med det korrekte tal.",
]


DELIMITERS = [
    "{lab})",    # A) foo
    "{lab}.",    # A. foo
    "{lab}:",    # A: foo
    "({lab})",   # (A) foo
    "[{lab}]",   # [A] foo
    "{lab} -",   # A - foo
]


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def load_all_rows():
    """Concatenate rows from every source with schema (q, a, orig_title)."""
    rows = []
    for src in SOURCES:
        print(f"loading {src}", flush=True)
        ds = load_dataset(src, split="train")
        for r in ds:
            q = r.get("q") or r.get("question")
            a = r.get("a") or r.get("answer")
            title = r.get("orig_title") or ""
            if not q or not a:
                continue
            rows.append({"q": q.strip(), "a": a.strip(),
                         "title": title.strip()})
    print(f"total source rows: {len(rows):,}", flush=True)
    return rows


def build_title_index(rows):
    idx = defaultdict(list)
    for i, r in enumerate(rows):
        if r["title"]:
            idx[r["title"]].append(i)
    return idx


def sample_distractors(row_ix, rows, title_idx, k, rng, max_tries=50):
    """Sample k unique distractor answers, preferring same-title."""
    correct = norm(rows[row_ix]["a"])
    title = rows[row_ix]["title"]
    seen = {correct}
    out = []

    # First pass: try same-title candidates
    same_title = [i for i in title_idx.get(title, []) if i != row_ix]
    rng.shuffle(same_title)
    for cand in same_title:
        if len(out) >= k:
            break
        cand_a = rows[cand]["a"]
        if norm(cand_a) in seen:
            continue
        seen.add(norm(cand_a))
        out.append(cand_a)

    # Second pass: random fill
    tries = 0
    while len(out) < k and tries < max_tries:
        tries += 1
        cand = rng.randrange(len(rows))
        if cand == row_ix:
            continue
        cand_a = rows[cand]["a"]
        if norm(cand_a) in seen:
            continue
        seen.add(norm(cand_a))
        out.append(cand_a)

    return out if len(out) == k else None


def build_mc(row_ix, rows, title_idx, rng, use_letters,
             p_lowercase=0.3, p_shuffle_display=0.3):
    # 2-5 options total (1-4 distractors), uniform.
    k = rng.randint(2, 5)
    distractors = sample_distractors(row_ix, rows, title_idx, k - 1, rng)
    if distractors is None:
        return None
    correct = rows[row_ix]["a"]
    q = rows[row_ix]["q"]

    # Assign correct+distractors to labeled slots (position of correct is random).
    options = [correct] + distractors
    rng.shuffle(options)
    correct_pos = options.index(correct)

    if use_letters:
        labels = list(string.ascii_uppercase[:k])
        if rng.random() < p_lowercase:
            labels = [lab.lower() for lab in labels]
    else:
        # numeric labels — no case; leave as-is.
        labels = [str(i + 1) for i in range(k)]

    delim = rng.choice(DELIMITERS)
    lines = [f"{delim.format(lab=lab)} {opt}"
             for lab, opt in zip(labels, options)]

    # Sometimes shuffle the DISPLAY order of the labeled lines. The label
    # is bound to the option via the prefix, so the correct-label answer
    # stays valid regardless of display order. Teaches robustness to
    # non-monotone option lists.
    if rng.random() < p_shuffle_display:
        rng.shuffle(lines)

    options_str = "\n".join(lines)

    templates = LETTER_TEMPLATES if use_letters else NUMBER_TEMPLATES
    template = rng.choice(templates)
    prompt = template.format(q=q, options=options_str)
    answer = labels[correct_pos]

    return {"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-repo", required=True,
                    help="HF dataset repo id, e.g. jensjepsen/danish-wiki-mc-letters-v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap rows (debug)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print a few samples, don't push")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_all_rows()
    title_idx = build_title_index(rows)
    print(f"unique titles: {len(title_idx):,}", flush=True)

    n_iter = min(args.limit, len(rows)) if args.limit else len(rows)

    out = []
    for i in range(n_iter):
        use_letters = rng.random() < 0.5
        row = build_mc(i, rows, title_idx, rng, use_letters)
        if row is not None:
            out.append(row)
        if (i + 1) % 20000 == 0:
            print(f"  built {i+1:,}/{n_iter:,}", flush=True)

    print(f"built {len(out):,} MC rows", flush=True)

    if args.dry_run:
        print("\n=== SAMPLES (evenly spaced across build) ===")
        stride = max(1, len(out) // 20)
        for r in out[::stride][:20]:
            print("-" * 60)
            print("USER:")
            print(r["messages"][0]["content"])
            print(f"ASSISTANT: {r['messages'][1]['content']}")
        return

    ds = Dataset.from_list(out)
    print(f"pushing to {args.output_repo} ...", flush=True)
    ds.push_to_hub(args.output_repo, private=False)
    print(f"done → https://huggingface.co/datasets/{args.output_repo}",
          flush=True)


if __name__ == "__main__":
    main()
