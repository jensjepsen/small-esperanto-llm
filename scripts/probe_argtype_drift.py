"""How many argF1 misses are type-only -- right value, wrong JSON type?

`_tool_score` builds argument pairs as (key, json.dumps(value)), so a call
emitting {"floor": "4"} against gold {"floor": 4} scores that argument WRONG
even though it is semantically right. v39's step-11331 probe emitted exactly
that, having emitted {"floor": 4} at step 7,554.

If a meaningful share of the eval's lost pairs are type-only, then the argF1
drop across both splits is partly a serialisation artefact rather than the
model getting worse at calling -- and the metric, not the model, is what moved.

Reports argF1 as scored, plus a type-insensitive argF1 for the same
generations, so the gap IS the artefact.
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


def pair_f1(pred, gold):
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if not tp:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def strict(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def loose(v):
    """Type-insensitive: 4, 4.0 and "4" all compare equal; case-folded text."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(int(v)) if float(v).is_integer() else str(float(v))
    if isinstance(v, str):
        s = v.strip()
        try:
            f = float(s.replace(",", "."))
            return str(int(f)) if f.is_integer() else str(f)
        except ValueError:
            return s.lower()
    return strict(v)


def build_items(repo, split, limit):
    ds = load_dataset(repo, "sft", split=split)
    items = []
    for r in ds:
        msgs = r["messages"]
        at = next((i for i, m in enumerate(msgs)
                   if m["role"] == "tool_call"), None)
        if at is None:
            continue
        start = at - 1 if at and msgs[at - 1]["role"] == "assistant" else at
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/data2/ckpts/v39_11331")
    ap.add_argument("--repo", default="jensjepsen/danish-tool-dialogues-v5")
    ap.add_argument("--splits", nargs="+",
                    default=["eval_seen_tools", "eval_unseen_tools"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    print(f"ckpt {args.ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.float16).cuda().eval()
    eos = [i for i in (tok.eos_token_id,
                       tok.convert_tokens_to_ids("<|end|>")) if i is not None]

    for split in args.splits:
        items = build_items(args.repo, split, args.n)
        prompts = [f"{q} {ASST}" for q, _ in items]
        strict_f1, loose_f1 = [], []
        n_call = n_name = 0
        typeonly = Counter()
        examples = []
        for i in range(0, len(prompts), args.bs):
            chunk = prompts[i:i + args.bs]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      padding_side="left", truncation=True,
                      max_length=2048).to(model.device)
            enc.pop("token_type_ids", None)
            gen = model.generate(**enc, max_new_tokens=200, do_sample=False,
                                 eos_token_id=eos,
                                 pad_token_id=tok.pad_token_id or 0)
            for j in range(len(chunk)):
                out = tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                 skip_special_tokens=False)
                gold = items[i + j][1]
                m = CALL_RE.search(out) or CALL_FALLBACK.search(out)
                if not m:
                    strict_f1.append(0.0), loose_f1.append(0.0)
                    continue
                try:
                    got, _ = json.JSONDecoder().raw_decode(m.group(1).strip())
                except Exception:
                    strict_f1.append(0.0), loose_f1.append(0.0)
                    continue
                n_call += 1
                if not isinstance(got, dict) or got.get("name") != gold.get("name"):
                    strict_f1.append(0.0), loose_f1.append(0.0)
                    continue
                n_name += 1
                ga = gold.get("arguments") or {}
                pa = got.get("arguments") if isinstance(
                    got.get("arguments"), dict) else {}
                strict_f1.append(pair_f1({(k, strict(v)) for k, v in pa.items()},
                                         {(k, strict(v)) for k, v in ga.items()}))
                loose_f1.append(pair_f1({(k, loose(v)) for k, v in pa.items()},
                                        {(k, loose(v)) for k, v in ga.items()}))
                for k, gv in ga.items():
                    if k in pa and strict(pa[k]) != strict(gv) \
                            and loose(pa[k]) == loose(gv):
                        typeonly[f"{type(gv).__name__} <- {type(pa[k]).__name__}"] += 1
                        if len(examples) < args.show:
                            examples.append((gold.get("name"), k, gv, pa[k]))
            print(f"  {split} {min(i+args.bs, len(prompts))}/{len(prompts)}",
                  flush=True)
        n = len(items)
        mean = lambda v: sum(v) / len(v) if v else 0.0  # noqa: E731
        print(f"\n=== {split}  (n={n}) ===")
        print(f"  emitted-a-call        {100*n_call/n:.1f}%")
        print(f"  right-tool            {100*n_name/n:.1f}%")
        print(f"  argF1 as scored       {100*mean(strict_f1):.1f}%")
        print(f"  argF1 type-insensitive{100*mean(loose_f1):>6.1f}%"
              f"   (+{100*(mean(loose_f1)-mean(strict_f1)):.1f}pp)")
        print(f"  type-only mismatches  {sum(typeonly.values()):,}")
        for k, c in typeonly.most_common():
            print(f"     gold {k:<28} {c:,}")
        for name, k, gv, pv in examples:
            print(f"     {name}.{k}: gold {gv!r} got {pv!r}")
        print(flush=True)


if __name__ == "__main__":
    main()
