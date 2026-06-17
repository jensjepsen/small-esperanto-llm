"""Pass@1 eval for LFM2.5-350M on the English-translated ICL handcrafted set."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch


def normalize(s: str) -> str:
    """Lenient English match: case-fold, strip punctuation, drop leading articles
    and common prepositions, collapse whitespace."""
    s = s.strip().lower()
    s = re.sub(r"[“”\"']", "", s)            # quotes
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    s = s.rstrip(".,;:!?")
    s = re.sub(r"\s+", " ", s)
    # Strip leading articles + prepositions like "in the", "on the", "to the"…
    for prefix in ("in the ", "on the ", "at the ", "from the ", "to the ",
                   "with the ", "by the ", "of the ",
                   "in a ", "on a ", "at a ", "from a ", "to a ", "with a ",
                   "the ", "a ", "an "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.strip()


def matches(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return False
    return p == g or g in p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--eval", default="data/causal_corpus/eval_handcrafted_v31_en.jsonl")
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default="mt/runs/lfm25_icl_en_pass1.jsonl")
    ap.add_argument("--hf-cache", default="/mnt/data/hf_cache")
    ap.add_argument("--no-system-prompt", action="store_true",
                    help="Skip the steering system prompt — see raw model behavior")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_cache)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} on cuda fp16…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
    print(f"  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M  GPU mem={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    rows = [json.loads(l) for l in open(args.eval)]
    N = len(rows) if not args.n else min(args.n, len(rows))

    SYS_MSG = ("You answer reading-comprehension questions about a short story. "
               "Reply with ONLY the answer (a word or short phrase). No explanation.")

    prompts = []
    accepteds = []
    for i in range(N):
        msgs = []
        if not args.no_system_prompt:
            msgs.append({"role": "system", "content": SYS_MSG})
        msgs.append({"role": "user", "content": rows[i]["messages"][0]["content"]})
        prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
        accs = list(rows[i].get("accepted_answers") or [])
        gold = rows[i]["messages"][1]["content"]
        if gold not in accs:
            accs = [gold] + accs
        accepteds.append(accs)

    correct = seen = 0
    t0 = time.perf_counter()
    n_gen_tokens = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fout:
        for bs in range(0, N, args.batch_size):
            batch_p = prompts[bs : bs + args.batch_size]
            batch_a = accepteds[bs : bs + args.batch_size]
            enc = tok(batch_p, return_tensors="pt", padding=True, truncation=False).to("cuda")
            with torch.no_grad():
                out = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            in_len = enc["input_ids"].shape[1]
            gen_tokens = out[:, in_len:]
            n_gen_tokens += int((gen_tokens != tok.pad_token_id).sum().item())
            texts = tok.batch_decode(gen_tokens, skip_special_tokens=True)
            for j, (txt, accs) in enumerate(zip(texts, batch_a)):
                ok = any(matches(txt, a) for a in accs)
                correct += ok
                seen += 1
                fout.write(json.dumps({"i": bs + j, "pred": txt, "accepted": accs, "ok": ok}) + "\n")
            dt = time.perf_counter() - t0
            print(f"  batch {bs//args.batch_size + 1}/{(N+args.batch_size-1)//args.batch_size}  "
                  f"{seen}/{N}  pass@1={correct} ({100*correct/seen:.1f}%)  "
                  f"{n_gen_tokens/dt:.0f} tok/s  elapsed={dt:.0f}s", flush=True)

    print(f"\n=== final: pass@1 = {correct}/{N} = {100*correct/N:.2f}% in {dt:.0f}s ({n_gen_tokens/dt:.0f} tok/s) ===")


if __name__ == "__main__":
    main()
