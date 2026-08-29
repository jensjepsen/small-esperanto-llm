"""Adapter: wraps `sample_mixture` from word_problems_da_procedural.py to
emit the same output schema as wp_compose_da.py.

Output schema (matches wp_compose_da.py):
    {question, answer, chain_lines, final, recipe="mixture",
     n_steps, direction="forward"}

Usage:
    uv run python scripts/wp_mixture_da.py --count 100 > mixture.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from word_problems_da_procedural import (
    fix_possessives,
    sample_mixture,
)


def _extract_chain_lines(chain_da: str) -> list[str]:
    """Extract `LHS = RHS` arithmetic expressions from the mixture chain.

    Mixture Steps use the `render_prose` convention: `{pre} {expr} = {result}. {post}`.
    We regex-scan each line for arithmetic-only `LHS = RHS` fragments that
    match the wp_compose chain_lines format.
    """
    import re
    lines_out: list[str] = []
    # Each source line looks like: "Mængden ... : 250 * 10 / 100 = 25. altså ..."
    # The arithmetic sits between the last `:` (or start) and the `.` after `= NNN`.
    # A safer pass: iterate line-by-line, find each occurrence of `<expr> = <num>`
    # where <expr> uses only digits/whitespace/`+ - * / ( )` and <num> is a
    # decimal / integer.
    pat = re.compile(
        r"(?<![A-Za-zæøåÆØÅ])"                      # not preceded by a letter
        r"([\d\s\(\)\+\-\*/\.]+?)"                    # LHS
        r"\s*=\s*"
        r"(-?\d+(?:\.\d+)?)"                          # numeric RHS
        r"(?=[\.\s,;]|$)"
    )
    for src_line in chain_da.split("\n"):
        if src_line.startswith("####"):
            continue
        for m in pat.finditer(src_line):
            lhs = m.group(1).strip().rstrip(":").strip()
            rhs = m.group(2).strip()
            # LHS must contain at least one operator to be a real arith line
            if not any(c in lhs for c in "+-*/"):
                continue
            lines_out.append(f"{lhs} = {rhs}")
    return lines_out


def _answer_from_chain(chain_da: str, final: str) -> str:
    """Join chain_da lines (excluding trailing #### N) with spaces + append."""
    parts: list[str] = []
    for line in chain_da.split("\n"):
        s = line.strip()
        if not s or s.startswith("####"):
            continue
        parts.append(s)
    return " ".join(parts) + f" #### {final}"


def make_row(rng: random.Random) -> dict:
    p = sample_mixture(rng)
    p.question_da = fix_possessives(p.question_da)
    p.chain_da = fix_possessives(p.chain_da)
    chain_lines = _extract_chain_lines(p.chain_da)
    answer = _answer_from_chain(p.chain_da, p.answer)
    return {
        "question": p.question_da,
        "answer": answer,
        "chain_lines": chain_lines,
        "final": p.answer,
        "recipe": "mixture",
        "n_steps": len(chain_lines),
        "direction": "forward",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    for _ in range(args.count):
        print(json.dumps(make_row(rng), ensure_ascii=False))


if __name__ == "__main__":
    main()
