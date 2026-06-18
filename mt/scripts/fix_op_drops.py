"""Fix v5b operator-drop artifacts in translate_traces.py output.

v5b sometimes drops the operator in arithmetic expressions, turning
'5 * 0.1 = 0.5' into '5 0.1 = 0.5'. This script scans each row's a_eo
for "A B = C" patterns and inserts the right operator, preferring the
EN-side expression when available and falling back to whichever single
operator makes the equation true.

Writes a new JSONL with `a_eo` rewritten. Original is unchanged.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Full literal: A op B = C  (op already present)
FULL = re.compile(
    r'(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)'
)
# Drop pattern: A B = C  (whitespace separating numbers, no operator)
DROP = re.compile(
    r'(?<![\d.])'                         # not preceded by digit/dot
    r'(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)'  # A B
    r'\s*=\s*'
    r'(\d+(?:\.\d+)?)'                    # = C
    r'(?!\d)'                             # not followed by another digit
)


def en_expressions(en_text: str) -> dict[tuple[str, str, str], str]:
    """Extract EN literal expressions, keyed by (a, b, c) → op."""
    out = {}
    en_text = en_text.replace(',', '.')
    for m in FULL.finditer(en_text):
        a, op, b, c = m.groups()
        out[(a, b, c)] = op
    return out


def infer_op(a: str, b: str, c: str) -> str | None:
    """Find the unique operator that makes a op b == c. None if 0 or >1 work."""
    try:
        a_f, b_f, c_f = float(a), float(b), float(c)
    except ValueError:
        return None
    candidates = []
    for op in '+-*/':
        try:
            if op == '+': v = a_f + b_f
            elif op == '-': v = a_f - b_f
            elif op == '*': v = a_f * b_f
            else:
                if b_f == 0: continue
                v = a_f / b_f
            if abs(v - c_f) < 0.01 * max(1, abs(c_f)):
                candidates.append(op)
        except ZeroDivisionError:
            continue
    return candidates[0] if len(candidates) == 1 else None


def fix_drops(eo_text: str, en_ops: dict[tuple[str, str, str], str]) -> tuple[str, int]:
    """Insert operators into drop patterns in EO. Returns (fixed_text, n_fixed)."""
    # Normalize decimal-comma → decimal-dot for matching, but rebuild with
    # the original characters so we don't change the rest of the prose.
    n = 0
    result = []
    last = 0
    eo_dot = eo_text.replace(',', '.')

    for m in DROP.finditer(eo_dot):
        a, b, c = m.groups()
        # Prefer EN-side operator if known; else infer from arithmetic.
        op = en_ops.get((a, b, c)) or infer_op(a, b, c)
        if op is None:
            continue  # ambiguous; leave as-is
        result.append(eo_text[last:m.start()])
        # Reconstruct the span: A op B = C, preserving spaces
        # Use the same spacing style as the surrounding text
        result.append(f"{a} {op} {b} = {c}")
        last = m.end()
        n += 1

    if n == 0:
        return eo_text, 0
    result.append(eo_text[last:])
    return ''.join(result), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--report-every', type=int, default=5000)
    args = ap.parse_args()

    n_rows = n_skipped = n_touched = n_ops_total = 0
    with args.input.open() as fin, args.output.open('w') as fout:
        for line in fin:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip the one known corrupt line from disk-full crash
            n_rows += 1
            if r.get('skipped'):
                n_skipped += 1
                fout.write(json.dumps(r, ensure_ascii=False) + '\n')
                continue
            en_text = r.get('a_en_clean', r.get('a_en', ''))
            en_ops = en_expressions(en_text)
            fixed, n_fix = fix_drops(r['a_eo'], en_ops)
            if n_fix:
                n_touched += 1
                n_ops_total += n_fix
                r['a_eo'] = fixed
                r['_op_drops_fixed'] = n_fix
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
            if n_rows % args.report_every == 0:
                print(f"  {n_rows} rows  ({n_touched} touched, {n_ops_total} ops fixed)",
                      flush=True)

    print(f"=== done ===")
    print(f"  total rows         {n_rows}")
    print(f"  skipped (passthrough) {n_skipped}")
    print(f"  rows touched       {n_touched}")
    print(f"  ops fixed          {n_ops_total}")


if __name__ == '__main__':
    main()
