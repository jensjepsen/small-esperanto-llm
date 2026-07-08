"""Extract EN↔EO parallel pairs from the Gemini-translated orca-math JSONL.

The translator output has the schema:
    {orig_idx, en_question, en_answer, eo_translation, reject_reason}

where ``eo_translation`` contains BOTH the question and answer joined with
Esperanto section markers ``DEMANDO:`` / ``RESPONDO:``. This script splits
those apart and emits two rows per source row:

    {en: <Q_en>,   eo: <Q_eo>,   src: "orca_math:q"}
    {en: <A_en>,   eo: <A_eo>,   src: "orca_math:a"}

Rows where the split fails (missing ``RESPONDO:``, imbalanced lengths, etc.)
are logged to ``--rejects`` for inspection.

Usage:
    uv run python scripts/extract_orcamath_parallel.py \\
        --in  /mnt/data2/orca_math_eo_20k.jsonl \\
        --out /mnt/data2/orca_math_parallel.jsonl \\
        --rejects /mnt/data2/orca_math_parallel_rejects.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Match the "DEMANDO:" (or variants) and "RESPONDO:" section headers we
# instructed Gemini to preserve. Tolerate leading whitespace, ASCII/Unicode
# colons, and stray blank lines.
_QA_SPLIT = re.compile(
    r"^\s*(?:DEMANDO|DEMANDOJ|Q)[:.]\s*\n(.*?)\n\s*(?:RESPONDO|RESPONDOJ|A)[:.]\s*\n(.*)\Z",
    re.S | re.M | re.I,
)


def split_qa(eo_text: str) -> tuple[str, str] | None:
    """Try to split the Gemini output into (question, answer). Returns None on
    failure."""
    if not eo_text:
        return None
    m = _QA_SPLIT.search(eo_text)
    if not m:
        return None
    q = m.group(1).strip()
    a = m.group(2).strip()
    if not q or not a:
        return None
    return q, a


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rejects", type=Path, default=None)
    ap.add_argument("--max-len-ratio", type=float, default=3.0,
                    help="Reject if len(eo)/len(en) or its reciprocal exceeds "
                         "this per Q or A. Default 3.0 (very lenient).")
    args = ap.parse_args()

    n_rows = 0
    n_pairs = 0
    n_rej_upstream = 0
    n_rej_split = 0
    n_rej_ratio = 0

    with args.in_path.open() as fin, args.out.open("w") as fout, (
        args.rejects.open("w") if args.rejects else _nullcontext()
    ) as frej:
        for line in fin:
            r = json.loads(line)
            n_rows += 1
            # Skip upstream rejects (Gemini error, filter failure)
            if r.get("reject_reason") is not None:
                n_rej_upstream += 1
                if frej:
                    frej.write(json.dumps(
                        {**r, "extract_reject": "upstream:" + r["reject_reason"]},
                        ensure_ascii=False,
                    ) + "\n")
                continue

            pair = split_qa(r["eo_translation"])
            if pair is None:
                n_rej_split += 1
                if frej:
                    frej.write(json.dumps(
                        {**r, "extract_reject": "split_failed"},
                        ensure_ascii=False,
                    ) + "\n")
                continue

            q_eo, a_eo = pair
            en_q = r["en_question"].strip()
            en_a = r["en_answer"].strip()

            # Length-ratio sanity
            def ratio(a: str, b: str) -> float:
                if not a or not b:
                    return float("inf")
                return max(len(a), len(b)) / min(len(a), len(b))

            if ratio(en_q, q_eo) > args.max_len_ratio or \
               ratio(en_a, a_eo) > args.max_len_ratio:
                n_rej_ratio += 1
                if frej:
                    frej.write(json.dumps(
                        {**r, "extract_reject": "len_ratio",
                         "q_eo": q_eo, "a_eo": a_eo},
                        ensure_ascii=False,
                    ) + "\n")
                continue

            fout.write(json.dumps(
                {"en": en_q, "eo": q_eo, "src": "orca_math:q",
                 "orig_idx": r["orig_idx"]},
                ensure_ascii=False,
            ) + "\n")
            fout.write(json.dumps(
                {"en": en_a, "eo": a_eo, "src": "orca_math:a",
                 "orig_idx": r["orig_idx"]},
                ensure_ascii=False,
            ) + "\n")
            n_pairs += 2

    print(f"input rows:      {n_rows:,}")
    print(f"  upstream rej:  {n_rej_upstream:,}")
    print(f"  split failed:  {n_rej_split:,}")
    print(f"  ratio rej:     {n_rej_ratio:,}")
    print(f"parallel pairs:  {n_pairs:,}  → {args.out}")


class _nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
