"""Generate every turn of a held-out tool split, for reading.

The downstream eval reports a number per split. This writes the actual
generations beside their gold, so a score can be traced to the rows that
produced it -- which is how "argF1 62" turned out to be mostly id-format and
slot-assignment errors rather than the model failing to understand the task.

Two generations per row, kept separate because they fail independently:

    CALL    prompt stops before the assistant turn that precedes the call,
            exactly as DownstreamEvaluator._tool_items builds it
    ANSWER  the row's OWN tool_result is fed back, so a wrong answer cannot
            be blamed on a wrong call

Both are BATCHED with left padding -- the same arrangement the eval uses.
Generating one at a time took ~4 s/row, which is 90 minutes for an 807-row
split and the reason this was five rows at a time before.

    uv run --no-sync python scripts/gen_unseen_turns.py \
        --ckpt jensjepsen/danish-lm-400m-sft-toolmix-mid \
        --subfolder step-1816-agg-0.806 \
        --repo jensjepsen/danish-tool-dialogues-proc-v1 \
        --split eval_unseen_tools --out scratch/unseen_turns
"""
from __future__ import annotations

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

CALL = re.compile(r"<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|$)", re.S)


def argf1(pred, gold):
    """Pair-F1 over (key, value), the same shape the eval scores."""
    def pairs(d):
        return {(k, json.dumps(v, ensure_ascii=False, sort_keys=True))
                for k, v in (d or {}).items()}
    p, g = pairs(pred), pairs(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if not tp:
        return 0.0
    pr, rc = tp / len(p), tp / len(g)
    return 2 * pr * rc / (pr + rc)


def build(model, tok, prompts, max_new, bs, eos, label):
    """Batched greedy generation. Left padding, restored afterwards."""
    prev_side, prev_pad = tok.padding_side, tok.pad_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    out = []
    try:
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False,
                      return_token_type_ids=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=max_new, do_sample=False, num_beams=1,
                    eos_token_id=eos,
                    pad_token_id=tok.pad_token_id or eos[0],
                    repetition_penalty=1.1)
            plen = enc["input_ids"].shape[1]
            for row in gen:
                out.append(tok.decode(row[plen:], skip_special_tokens=False))
            print(f"  [{label}] {min(i + bs, len(prompts)):,}/{len(prompts):,}",
                  flush=True)
    finally:
        tok.padding_side, tok.pad_token = prev_side, prev_pad
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--split", default="eval_unseen_tools")
    ap.add_argument("--config", default="sft")
    ap.add_argument("--n", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-call", type=int, default=320)
    ap.add_argument("--max-new-answer", type=int, default=160)
    ap.add_argument("--out", type=Path, required=True,
                    help="directory; writes turns.jsonl and turns.txt")
    args = ap.parse_args()

    kw = {"subfolder": args.subfolder} if args.subfolder else {}
    print(f"ckpt {args.ckpt} [{args.subfolder}]", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt, **kw)
    # fp32: the masters are fp32 and this is read for behaviour, so a
    # load-time downcast would sit between the weights and what is inspected.
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.float32, **kw).cuda().eval()
    eos = [i for i in (tok.eos_token_id,
                       tok.convert_tokens_to_ids("<|end|>")) if i is not None]

    ds = load_dataset(args.repo, args.config, split=args.split)
    rows = []
    for r in ds:
        msgs = r["messages"]
        call_at = next((i for i, m in enumerate(msgs)
                        if m["role"] == "tool_call"), None)
        if call_at is None:
            continue
        start = call_at - 1 if (call_at and msgs[call_at - 1]["role"]
                                == "assistant") else call_at
        if start == 0:
            continue
        try:
            gold = json.loads(msgs[call_at]["content"])
        except Exception:
            continue
        rows.append({"msgs": msgs, "call_at": call_at, "start": start,
                     "gold": gold})
        if args.n and len(rows) >= args.n:
            break
    print(f"{len(rows):,} rows from {args.repo}:{args.config}:{args.split}",
          flush=True)

    gens = build(model, tok,
                 [format_conversation(r["msgs"][:r["start"]]) + " <|assistant|>"
                  for r in rows],
                 args.max_new_call, args.batch_size, eos, "call")

    # Answers are generated for every row that HAS a result, using the gold
    # call rather than the generated one: the point is to score the answer
    # turn on its own, and prompting it with a call the model got wrong would
    # confound the two failures this script exists to separate.
    idx = [i for i, r in enumerate(rows)
           if r["call_at"] + 1 < len(r["msgs"])
           and r["msgs"][r["call_at"] + 1]["role"] == "tool_result"]
    answers = build(model, tok,
                    [format_conversation(rows[i]["msgs"][:rows[i]["call_at"] + 2])
                     + " <|assistant|>" for i in idx],
                    args.max_new_answer, args.batch_size, eos, "answer")
    amap = dict(zip(idx, answers))

    args.out.mkdir(parents=True, exist_ok=True)
    n_right = n_parsed = 0
    f1_sum = 0.0
    with (args.out / "turns.jsonl").open("w") as fj, \
         (args.out / "turns.txt").open("w") as ft:
        for i, (r, raw) in enumerate(zip(rows, gens)):
            m = CALL.search(raw)
            body = (m.group(1) if m else raw).strip()
            try:
                got, _ = json.JSONDecoder().raw_decode(body)
            except Exception:
                got = None
            right = bool(got) and got.get("name") == r["gold"].get("name")
            f1 = argf1(got.get("arguments"), r["gold"].get("arguments")) \
                if right else 0.0
            n_parsed += got is not None
            n_right += right
            f1_sum += f1

            head = r["msgs"][0]["content"]
            cat = head.split("Værktøjer:\n", 1)[1] if "Værktøjer:\n" in head \
                else head
            try:
                tools, end = json.JSONDecoder().raw_decode(cat)
                names, question = [t.get("name") for t in tools], cat[end:].strip()
            except Exception:
                names, question = [], head
            ans = (amap.get(i) or "").replace("<|end|>", "").strip()
            gold_ans = ""
            if i in amap and r["call_at"] + 2 < len(r["msgs"]):
                gold_ans = (r["msgs"][r["call_at"] + 2].get("content") or "").strip()
            res = r["msgs"][r["call_at"] + 1]["content"] if i in amap else ""

            fj.write(json.dumps({
                "i": i, "catalogue": names, "question": question,
                "gold_call": r["gold"], "gen_call": got, "raw_call": body,
                "right_tool": right, "argf1": f1,
                "tool_result": res, "gold_answer": gold_ans,
                "gen_answer": ans}, ensure_ascii=False) + "\n")

            ft.write("=" * 92 + "\n")
            ft.write(f"### {i}   right-tool {'YES' if right else 'NO'}   "
                     f"argF1 {f1:.2f}   catalogue {names}\n")
            for m2 in r["msgs"][1:r["start"]]:
                ft.write(f"    [{m2['role']:<11}] {(m2.get('content') or '')[:220]}\n")
            ft.write(f"USER: {question[:400]}\n")
            ft.write(f"GOLD CALL: {json.dumps(r['gold'], ensure_ascii=False)}\n")
            ft.write(f"GEN  CALL: {body[:300]}\n")
            if res:
                ft.write(f"RESULT   : {res[:300]}\n")
                ft.write(f"GOLD ANS : {gold_ans[:300]}\n")
                ft.write(f"GEN  ANS : {ans[:300]}\n")
            ft.write("\n")

    n = max(len(rows), 1)
    print(f"\nrows {len(rows):,}   parsed {n_parsed / n * 100:.1f}%   "
          f"right-tool {n_right / n * 100:.1f}%   argF1 {f1_sum / n * 100:.1f}%")
    print(f"-> {args.out}/turns.jsonl  and  turns.txt", flush=True)


if __name__ == "__main__":
    main()
