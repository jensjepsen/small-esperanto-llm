"""Convert GSM8K-eo CoT answers to funcall format.

GSM8K answers contain inline `<<X OP Y=Z>>` markers. This converter:
  1. Extracts each call in order
  2. Identifies which operands equal a prior call's result; replaces
     with `#N` back-references (so the model learns proper chaining)
  3. Emits the assistant message as just the `[[X OP Y]]` sequence
  4. Keeps the original user prompt unchanged (full natural Eo)

Each output record is the same SFT format as `generate_funcall_arith`:
  {messages: [{user,...}, {assistant, "[[a OP b]] [[#1 OP c]] ..."}],
   category: "gsm8k_funcall",
   expected_calls: ["[[a OP b]]", ...],
   n_steps: int,
   final_answer: str}

Usage:
    uv run python scripts/convert_gsm8k_to_funcall.py \\
        --in data/sft/gsm8k/train.jsonl \\
        --out data/sft/gsm8k_funcall_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Capture `<<expr=result>>`. expr can have multiple ops in rare cases
# (e.g. `<<5+3-1=7>>`); for those we record the call verbatim and skip
# back-ref substitution for that one (too brittle to parse precedence).
_GSM_CALL_RE = re.compile(r"<<\s*([^=>]+?)\s*=\s*([^>]+?)\s*>>")
# Simple two-operand check: lhs OP rhs where each operand is a number
# (possibly decimal, possibly negative) and OP is +/-/*//.
_SIMPLE_EXPR = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*$"
)
_FINAL_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def convert_answer(cot_text):
    """Parse a CoT answer; return (call_strings, final_answer, n_skipped).

    Returns (None, ...) if the answer has no extractable calls.
    `n_skipped` counts calls with non-simple expressions (e.g.
    multi-op `<<5+3-1>>`) that were emitted verbatim without back-ref
    substitution.
    """
    raw_calls = _GSM_CALL_RE.findall(cot_text)
    if not raw_calls:
        return None, None, 0

    # Step 1: parse each call into (op, lhs, rhs, result) when simple,
    # else (None, full_expr, None, result) for raw-emit cases.
    parsed = []
    n_skipped = 0
    for expr, result in raw_calls:
        m = _SIMPLE_EXPR.match(expr)
        if m:
            lhs, op, rhs = m.group(1), m.group(2), m.group(3)
            parsed.append((op, lhs, rhs, result.strip()))
        else:
            parsed.append((None, expr.strip(), None, result.strip()))
            n_skipped += 1

    # Step 2: substitute back-refs. A prior result Z (as a string) at
    # index k becomes #{k+1} when it appears as a later operand. Use
    # most-recent prior occurrence on collision.
    result_to_idx = {}
    out_calls = []
    for i, (op, lhs, rhs, result) in enumerate(parsed):
        if op is None:
            # Raw-emit: wrap whole expr in [[]] verbatim
            out_calls.append(f"[[{lhs}]]")
            result_to_idx[result] = i
            continue
        # Try back-ref substitution for each operand
        if lhs in result_to_idx:
            lhs = f"#{result_to_idx[lhs] + 1}"
        if rhs in result_to_idx:
            rhs = f"#{result_to_idx[rhs] + 1}"
        out_calls.append(f"[[{lhs}{op}{rhs}]]")
        result_to_idx[result] = i

    # Final answer (#### N)
    fm = _FINAL_RE.search(cot_text)
    final = fm.group(1) if fm else parsed[-1][3]

    return out_calls, final, n_skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-prefix", action="store_true",
                    help="don't add the 'Solvu paŝon...' prefix to user msg")
    args = ap.parse_args()

    PREFIX = ("Solvu paŝon post paŝo per kalkulado. "
              "Skribu ĉiun paŝon kiel [[esprimo]]:\n\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = n_no_calls = total_skipped = 0
    step_counts = Counter()

    with args.inp.open() as fin, args.out.open("w") as fout:
        for line in fin:
            n_in += 1
            rec = json.loads(line)
            msgs = rec["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
            calls, final, n_skipped = convert_answer(assistant)
            if calls is None:
                n_no_calls += 1
                continue
            total_skipped += n_skipped
            step_counts[len(calls)] += 1
            user_msg = user if args.no_prefix else f"{PREFIX}{user}"
            fout.write(json.dumps({
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": " ".join(calls)},
                ],
                "category": "gsm8k_funcall",
                "expected_calls": calls,
                "n_steps": len(calls),
                "final_answer": final,
            }, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"in:  {n_in}")
    print(f"out: {n_out}  (no-calls dropped: {n_no_calls})")
    print(f"non-simple calls emitted verbatim: {total_skipped}")
    print("step-count distribution:")
    for k in sorted(step_counts):
        print(f"  {k}-step: {step_counts[k]}")


if __name__ == "__main__":
    main()
