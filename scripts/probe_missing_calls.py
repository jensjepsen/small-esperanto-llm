"""What does the model emit when it fails to emit a tool call?

`emitted-a-call` fell from 95.9% to 85.2% on seen tools and 90.8% to 84.8% on
unseen between v39's step 7,554 and 11,331, on both splits at once, while
training loss improved. That is not tool selection getting worse -- roughly 15%
of prompts stopped producing anything parseable as a call.

The hypothesis this tests: v5 trains two continuations for the <|assistant|>
slot -- emit a call immediately (the turn is empty), or write answer prose
(after a tool_result). v4's reasoning text disambiguated which was coming.
Without it the model must choose with no textual cue, so some fraction starts
prose where a call belongs.

If that is right, the non-call outputs are fluent Danish answers attempting the
user's question directly. If they are truncations, refusals, or malformed JSON,
the hypothesis is wrong and the cause is something else.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sft_packed import format_conversation  # noqa: E402

ASST = "<|assistant|>"
CALL_RE = re.compile(r"<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|$)", re.S)
CALL_FALLBACK = re.compile(r"(\{\s*\"name\"\s*:.*)", re.S)


def build_items(repo, split, limit):
    """Same cut the downstream eval makes, so the numbers are comparable."""
    ds = load_dataset(repo, "sft", split=split)
    items = []
    for r in ds:
        msgs = r["messages"]
        at = next((i for i, m in enumerate(msgs)
                   if m["role"] == "tool_call"), None)
        if at is None:
            continue
        start = at
        if start and msgs[start - 1]["role"] == "assistant":
            start -= 1
        if start == 0:
            continue
        try:
            gold = json.loads(msgs[at]["content"])
        except Exception:
            continue
        items.append((format_conversation(msgs[:start]), gold))
        if len(items) >= limit:
            break
    return items


def classify(out):
    """Why no call? The categories are the competing explanations."""
    if CALL_RE.search(out) or CALL_FALLBACK.search(out):
        return "call"
    s = out.strip()
    if not s:
        return "empty"
    if s.startswith("{") or '"name"' in s[:80]:
        return "malformed-json"
    if "<|tool_call|>" in s:
        return "marker-no-payload"
    if len(s.split()) >= 3:
        return "prose"           # the hypothesis
    return "short-fragment"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/data2/ckpts/v39_11331")
    ap.add_argument("--repo", default="jensjepsen/danish-tool-dialogues-v5")
    ap.add_argument("--split", default="eval_seen_tools")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    print(f"ckpt {args.ckpt}\nrepo {args.repo}:{args.split}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.float16).cuda().eval()
    eos = [i for i in (tok.eos_token_id,
                       tok.convert_tokens_to_ids("<|end|>")) if i is not None]

    items = build_items(args.repo, args.split, args.n)
    prompts = [f"{q} {ASST}" for q, _ in items]
    print(f"{len(items)} prompts\n", flush=True)

    kinds, misses = Counter(), []
    for i in range(0, len(prompts), args.bs):
        chunk = prompts[i:i + args.bs]
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left",
                  truncation=True, max_length=2048).to(model.device)
        enc.pop("token_type_ids", None)
        gen = model.generate(**enc, max_new_tokens=200, do_sample=False,
                             eos_token_id=eos,
                             pad_token_id=tok.pad_token_id or 0)
        for j in range(len(chunk)):
            out = tok.decode(gen[j][enc["input_ids"].shape[1]:],
                             skip_special_tokens=False)
            k = classify(out)
            kinds[k] += 1
            if k != "call" and len(misses) < 40:
                misses.append((k, items[i + j][1], out))
        print(f"  {min(i+args.bs, len(prompts))}/{len(prompts)}", flush=True)

    n = sum(kinds.values())
    print(f"\n{'outcome':<20}{'n':>6}{'share':>9}")
    for k, c in kinds.most_common():
        print(f"{k:<20}{c:>6}{100*c/n:>8.1f}%")

    print(f"\n{'='*74}\nNON-CALL OUTPUTS")
    for k, gold, out in misses[:args.show]:
        print("-" * 74)
        print(f"  [{k}]  gold call: {gold.get('name')}")
        print(f"  {out.strip()[:320]}")


if __name__ == "__main__":
    main()
