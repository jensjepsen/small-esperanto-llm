"""Does the reasoning turn earn its share of the tool corpus?

78.7% of the trained tokens in the tool data are the reasoning that precedes a
call; the calls themselves are 17.8% and the grounded answers 3.5%. The case
for keeping that spend is that the reasoning is the immediate context which
produces the arguments -- read the user's text, work out which value maps to
which field, then emit JSON. This measures that directly on a trained
checkpoint by taking the reasoning away at inference:

  free   -- generate from <|assistant|>, the model reasons and then calls
  forced -- prefill <|assistant|> <|tool_call|>, straight to JSON

Same prompts, same gold, same scoring as the downstream tool_seen eval. If
argF1 holds up under `forced`, the reasoning is not doing the work its token
share implies.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sft_packed import format_conversation  # noqa: E402

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
CALL_OPEN = "<|tool_call|>"
CALL_RE = re.compile(r"<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|$)", re.S)
CALL_FALLBACK = re.compile(r"(\{\s*\"name\"\s*:.*)", re.S)


def pair_f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if not tp:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def build_items(repo: str, split: str, limit: int):
    """Prompt stops before the assistant turn preceding the call -- same cut
    the downstream eval makes, so the numbers are comparable to the run."""
    ds = load_dataset(repo, "sft", split=split)
    items = []
    for r in ds:
        msgs = r["messages"]
        call_at = next((i for i, m in enumerate(msgs)
                        if m["role"] == "tool_call"), None)
        if call_at is None:
            continue
        start = call_at
        if start and msgs[start - 1]["role"] == "assistant":
            start -= 1
        if start == 0:
            continue
        try:
            gold = json.loads(msgs[call_at]["content"])
        except Exception:
            continue
        items.append((format_conversation(msgs[:start]), gold))
        if len(items) >= limit:
            break
    return items


@torch.no_grad()
def generate(model, tok, prompts, max_new, bs):
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  padding_side="left", truncation=True,
                  max_length=2048).to(model.device)
        enc.pop("token_type_ids", None)   # tokenizer emits it; the model has no use for it
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=False))
        print(f"    {min(i + bs, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def score(outs, items, prepend_marker: bool):
    """prepend_marker: under `forced` the open tag is in the prompt, not the
    generation, so the marker regex would never fire on the raw output."""
    n_call = n_name = 0
    f1s = []
    rows = []
    for out, (_, gold) in zip(outs, items):
        text = (CALL_OPEN + out) if prepend_marker else out
        m = CALL_RE.search(text) or CALL_FALLBACK.search(text)
        if not m:
            f1s.append(0.0)
            rows.append((0.0, None))
            continue
        try:
            got, _ = json.JSONDecoder().raw_decode(m.group(1).strip())
        except Exception:
            f1s.append(0.0)
            rows.append((0.0, None))
            continue
        n_call += 1
        if not isinstance(got, dict) or got.get("name") != gold.get("name"):
            f1s.append(0.0)
            rows.append((0.0, got))
            continue
        n_name += 1
        ga, pa = gold.get("arguments") or {}, got.get("arguments") or {}
        if not isinstance(pa, dict):
            f1s.append(0.0)
            rows.append((0.0, got))
            continue
        key = lambda d: {(k, json.dumps(v, sort_keys=True, ensure_ascii=False))  # noqa: E731
                         for k, v in d.items()}
        f = pair_f1(key(pa), key(ga))
        f1s.append(f)
        rows.append((f, got))
    n = len(items)
    return {"call": n_call / n, "name": n_name / n,
            "f1": sum(f1s) / n, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/data2/ckpts/v38_33993")
    ap.add_argument("--repo", default="jensjepsen/danish-tool-dialogues-v4")
    ap.add_argument("--split", default="eval_seen_tools")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    print(f"loading {args.ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.float16).cuda().eval()

    items = build_items(args.repo, args.split, args.n)
    print(f"{len(items)} items from {args.repo}:{args.split}\n", flush=True)

    base = [f"{q} {ASST}" for q, _ in items]
    conds = {
        "free   (model reasons, then calls)": (base, 512, False),
        "forced (prefilled straight to call)": (
            [p + f" {CALL_OPEN}" for p in base], 160, True),
    }
    res = {}
    for label, (prompts, max_new, prep) in conds.items():
        print(f"  {label}", flush=True)
        res[label] = score(generate(model, tok, prompts, max_new, args.bs),
                           items, prep)

    print(f"\n{'condition':<38} {'call':>7} {'tool':>7} {'argF1':>7}")
    for label, r in res.items():
        print(f"{label:<38} {100*r['call']:>6.1f}% {100*r['name']:>6.1f}% "
              f"{100*r['f1']:>6.1f}%")
    a, b = list(res.values())
    print(f"\nargF1 delta (forced - free): {100*(b['f1'] - a['f1']):+.1f}pp")

    moved = [(i, x, y) for i, ((x, _), (y, _))
             in enumerate(zip(a["rows"], b["rows"])) if abs(x - y) > 0.01]
    print(f"rows whose score moved     : {len(moved)}/{len(items)}"
          f"  ({sum(1 for _, x, y in moved if y > x)} better forced, "
          f"{sum(1 for _, x, y in moved if y < x)} worse)")


if __name__ == "__main__":
    main()
