"""Render the translated ToolMind rows into the trainer's chat format.

The `<think>` wrapper is DROPPED and its text becomes ordinary assistant
content. The tags are an artefact of how ToolMind was generated, not something
we want a 400M Danish model to emit: keeping them would make the model produce
a literal `<think>` string it was never taught to close reliably (16 of 6,486
assistant turns in the source are missing their closing tag), and would put
reasoning behind a marker that nothing downstream parses. Reasoning that flows
straight into a tool call is exactly what `format_conversation` already
encodes -- `<|assistant|> reasoning <|tool_call|>{...}<|/tool_call|> <|end|>`
as one autoregressive burst -- so the wrapper adds nothing.

Role mapping into the trainer's protocol:

    user            -> user            (catalogue prepended to the FIRST one)
    assistant text  -> assistant       (think tags stripped)
    assistant calls -> tool_call, one per call
    tool            -> tool_result     (masked from the loss by the trainer)

Usage:
  python scripts/render_toolmind_sft.py --in scratch/toolmind_da_v2 --n 5
  python scripts/render_toolmind_sft.py --in scratch/toolmind_da_v2 --out sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sft_packed import (  # noqa: E402
    ASSISTANT_TOKEN, SPECIAL_TOKENS, TOOL_CALL_OPEN, TOOL_RESULT_CLOSE,
    TOOL_RESULT_OPEN, USER_TOKEN, _build_label_masker,
    _build_preprocess_and_tokenize, format_conversation,
)

# The translator rendered the tag in five different ways -- <think> 4,346,
# <taenk> 1,619, <tanke> 85, <taenke> 9, <taenker> 5 -- because it is inside
# text being translated. Match the family, not the literal. Tolerates unclosed
# tags too (16 rows have an opening tag and no close).
THINK = re.compile(r"</?\s*(?:think|thought|t\u00e6nk\w*|tanke\w*)\s*>", re.I)
# The per-row request numbers lines as "1. [kind] text", and the model echoes
# the hint back -- TRANSLATED, which the English-only stripper in the
# translator never caught: [svar] 1,408, [taenk] 1,219, plus variants. Strip a
# leading bracket tag; real assistant text does not open with one.
LEAD_TAG = re.compile(r"^\s*\[[^\]\n]{1,14}\]\s*")
CATALOG_LABEL = "Værktøjer"


def strip_think(text: str) -> str:
    out = LEAD_TAG.sub("", text or "")
    out = THINK.sub("", out)
    return LEAD_TAG.sub("", out).strip()   # tag can sit inside the think block


def to_messages(row) -> list[dict] | None:
    tools = [t.get("function") for t in row.get("tools", [])
             if isinstance(t, dict) and t.get("function")]
    catalog = json.dumps(tools, ensure_ascii=False)
    msgs: list[dict] = []
    for m in row.get("conversations", []):
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            body = strip_think(content)
            if not msgs:                      # catalogue rides the first turn
                body = f"{CATALOG_LABEL}:\n{catalog}\n\n{body}"
            msgs.append({"role": "user", "content": body})
        elif role == "assistant":
            body = strip_think(content)
            calls = m.get("tool_calls") or []
            # ALWAYS open a model turn with <|assistant|>, even when there is
            # no reasoning to put in it. Every inference path prompts with
            # f"{USER}{q}{END}{ASST}", so the model only ever generates after
            # an <|assistant|> marker. Emitting a bare <|tool_call|> straight
            # after <|user|> -- which 1,797 of 7,834 calls did, 22.9% -- trains
            # a context that never occurs at inference.
            if body or calls:
                msgs.append({"role": "assistant", "content": body})
            for tc in calls:
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    continue
                msgs.append({"role": "tool_call",
                             "content": json.dumps(fn, ensure_ascii=False)})
        elif role == "tool":
            msgs.append({"role": "tool_result", "content": content})
        else:
            return None                       # unknown role: drop the row
    if not any(m["role"] in ("assistant", "tool_call") for m in msgs):
        return None                           # nothing for the model to learn
    if msgs and msgs[0]["role"] != "user":
        return None                           # must open with the catalogue
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path,
                    default=Path("scratch/toolmind_da_v2"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n", type=int, default=0, help="print N rendered rows")
    ap.add_argument("--clean-only", action="store_true",
                    help="render only rows the gate passed, read from "
                         "gate_verdicts.jsonl. Without it EVERY translated row "
                         "is rendered, failures included.")
    ap.add_argument("--tokenizer",
                    default="jensjepsen/danish-lm-400m-sft-v34-mid")
    ap.add_argument("--subfolder", default="step-30240-agg-0.264")
    args = ap.parse_args()

    recs = [json.loads(l) for l in (args.src / "translated.jsonl").open()
            if l.strip()]
    if args.clean_only:
        vp = args.src / "gate_verdicts.jsonl"
        if not vp.exists():
            raise SystemExit(f"--clean-only needs {vp}; run --gate-only first")
        verdicts = {}
        for line in vp.open():
            v = json.loads(line)
            verdicts[v["idx"]] = v["bad"]
        before = len(recs)
        recs = [r for r in recs if not verdicts.get(r.get("idx"), ["unknown"])]
        print(f"clean-only: {len(recs):,} of {before:,} rows passed the gate",
              flush=True)
    rows = [r["da"] for r in recs]
    print(f"loaded {len(rows):,} translated rows", flush=True)

    rendered, drops = [], Counter()
    for r in rows:
        m = to_messages(r)
        if m is None:
            drops["unrenderable"] += 1
            continue
        rendered.append({"messages": m})
    print(f"rendered {len(rendered):,}  dropped {dict(drops) or 0}", flush=True)

    # VERIFY AGAINST THE TRAINER, not against an idea of it. Every row must
    # tokenise, produce a label mask, and train on something -- and no user or
    # tool-result text may appear among the trained tokens.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                        subfolder=args.subfolder)
    unk = tok.unk_token_id
    tid = tok.convert_tokens_to_ids
    mask = _build_label_masker(tid(ASSISTANT_TOKEN), tid(TOOL_RESULT_OPEN),
                               tid(TOOL_RESULT_CLOSE), unk,
                               user_id=tid(USER_TOKEN),
                               tool_call_id=tid(TOOL_CALL_OPEN))
    tk = _build_preprocess_and_tokenize(tok, SPECIAL_TOKENS, 8048,
                                        morpheme_preprocess=False)
    bad = Counter()
    check = rendered[:400]
    for row in check:
        text = format_conversation(row["messages"])
        ids = tk(text)["input_ids"]
        labels = mask(ids)
        if labels is None:
            bad["no-model-turn"] += 1
            continue
        if all(l == -100 for l in labels):
            bad["nothing-trained"] += 1
            continue
        # POSITIONAL, not textual. Checking whether the user's words appear
        # among the trained tokens flags correct behaviour: the model quotes
        # the user in its reasoning ("brugeren spurgte om...") and copies the
        # user's phrasing into arguments ({"content": "Jeg vil gerne..."}).
        # Both are exactly what it should learn. The real invariant is that
        # the <|user|> and <|tool_result|> SPANS carry no loss.
        u_id, tr_id = tid(USER_TOKEN), tid(TOOL_RESULT_OPEN)
        a_id, tc_id = tid(ASSISTANT_TOKEN), tid(TOOL_CALL_OPEN)
        in_world = False
        for t, l in zip(ids, labels):
            if t in (a_id, tc_id):
                in_world = False
            elif t in (u_id, tr_id):
                in_world = True
            if in_world and l != -100:
                bad["world-span-trained"] += 1
                break
        if "<think>" in text or "</think>" in text:
            bad["think-tag-survived"] += 1
    print(f"\ntrainer check over {len(check)} rows: "
          f"{'CLEAN' if not bad else dict(bad)}")

    if args.n:
        for row in rendered[:args.n]:
            print("\n" + "=" * 72)
            for m in row["messages"]:
                print(f"  [{m['role']:<11}] {m['content'][:220]}")

    if args.out:
        with args.out.open("w") as f:
            for row in rendered:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n-> {args.out}  ({len(rendered):,} rows)")


if __name__ == "__main__":
    main()
