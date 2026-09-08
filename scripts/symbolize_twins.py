"""Stage 6: emit a symbolized twin of every answered row.

Runs between the answer stage and the push, and is a stage rather than a flag
on the push for the same reason the pedagogy filters had to move out of it:
the push re-rendering its own corpus is how filters came to apply to sft.jsonl
and to nothing that reached the Hub. The push should upload and split. This
transforms.

WHY TWINS. `eval_unseen_tools` withholds the tool NAME but keeps the parameter
vocabulary -- `location`, `cuisine`, `song_title` appear on hundreds of other
tools in training -- so it measures name substitution, not schema
comprehension. Measured on v41 step-30224, the same five rows with only the
key strings changed:

    real parameter names   right-tool 5/5   mean argF1 1.000
    symbolized names       right-tool 4/5   mean argF1 0.600

~40 argF1 points of `tool_unseen` is name recall. The twin trains the other
half of that: with a per-row random symbol there is nothing to recall, so the
Danish description is the only route from question to slot.

The ability is already partially there rather than absent -- 3 of 5 map
symbols correctly from the description alone, and at k=32 sampling 4 of 5 rows
reach argF1 1.000 -- so this consolidates something intermittent.

ORDER. After the answers, never before: gen_tool_answer_turns keys its cache
on (call, question, returns-fingerprint), so renaming argument keys upstream
would miss every cached answer and re-buy them.

    uv run python scripts/symbolize_twins.py \
        --in  $OUT/sft_answered.jsonl \
        --out $OUT/sft_answered_twins.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_toolmind_sft import symbolize_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--show", type=int, default=2)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.src.open() if l.strip()]
    out, why = [], Counter()
    for r in rows:
        out.append(r)                       # the original, untouched
        got = symbolize_params(r["messages"], r["idx"],
                               r.get("answer_relevance"))
        if got is None:
            why["unsymbolizable"] += 1
            continue
        msgs, rel = got
        twin = {"idx": r["idx"], "messages": msgs, "sym": True}
        if rel is not None:
            twin["answer_relevance"] = rel
        out.append(twin)
        why["twinned"] += 1

    with args.out.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows):,} rows in -> {len(out):,} out "
          f"({dict(why)})", flush=True)
    print(f"-> {args.out}", flush=True)

    for r in out:
        if not r.get("sym") or args.show <= 0:
            continue
        args.show -= 1
        head = r["messages"][0]["content"]
        cat = head.split("Værktøjer:\n", 1)[1][:200]
        call = next((m["content"] for m in r["messages"]
                     if m["role"] == "tool_call"), "-")
        print(f"\n  idx {r['idx']}\n    catalogue: {cat}\n    call: {call[:120]}")


if __name__ == "__main__":
    main()
