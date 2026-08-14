"""ARC-DA eval on alexandrainst/m_arc:da:test — 4-5 option MC via
generation + letter parse (same style as cit_mc).

Usage:
    uv run python scripts/eval_arc_da.py --ckpt HF_ID [--n 1167]
"""
from __future__ import annotations
import argparse, re, time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
LETTER_RE = re.compile(r"\b([ABCDE])\b")


def score_cont(model, tok, prompt, cont):
    """Length-normalized log P(cont | prompt)."""
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False,
                     return_token_type_ids=False).input_ids.cuda()
    full_ids = tok(prompt + cont, return_tensors="pt", add_special_tokens=False,
                   return_token_type_ids=False).input_ids.cuda()
    p_len = prompt_ids.shape[1]
    n_cont = full_ids.shape[1] - p_len
    if n_cont <= 0: return -float("inf")
    with torch.no_grad():
        logits = model(full_ids).logits
    cont_logits = logits[0, p_len - 1 : -1, :].float()
    cont_targets = full_ids[0, p_len:]
    log_probs = F.log_softmax(cont_logits, dim=-1)
    tok_logp = log_probs.gather(1, cont_targets.unsqueeze(1)).squeeze(1)
    return tok_logp.sum().item() / n_cont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32","fp16","bf16"])
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--report-every", type=int, default=50)
    ap.add_argument("--mode", default="chat-mc", choices=["chat-mc","raw-logp"],
                    help="chat-mc: chat-wrapped MC letter generation + parse. "
                         "raw-logp: score each option as continuation of "
                         "'question\\nAnswer: ', pick highest length-norm log P.")
    args = ap.parse_args()

    print(f"loading {args.ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=dtype).cuda().eval()
    end_id = tok.convert_tokens_to_ids(END)
    eos_ids = [tok.eos_token_id] + ([end_id] if end_id != tok.unk_token_id else [])

    ds = load_dataset("alexandrainst/m_arc", "da", split="test")
    if args.n: ds = ds.select(range(min(args.n, len(ds))))
    n = len(ds)
    print(f"  {n} rows", flush=True)

    n_ok = 0
    n_parsefail = 0
    t0 = time.time()
    for i, r in enumerate(ds, 1):
        q = r["instruction"]
        opts = [(l.upper(), r[f"option_{l}"]) for l in "abcde" if r.get(f"option_{l}")]

        if args.mode == "chat-mc":
            opts_str = "\n".join(f"{lab}) {v}" for lab, v in opts)
            body = f"{q}\n\n{opts_str}\n\nSvar med bogstavet på det korrekte svar."
            prompt = f"{USER}{body}{END}{ASST}"
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False,
                      return_token_type_ids=False).input_ids.cuda()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=eos_ids)
            gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            m = LETTER_RE.search(gen)
            if not m: n_parsefail += 1; pred = "?"
            else: pred = m.group(1)
        else:  # raw-logp: score each option as a continuation, pick highest
            base = f"{q.strip()}\nSvar: "
            lps = [(lab, score_cont(model, tok, base, v)) for lab, v in opts]
            pred = max(lps, key=lambda x: x[1])[0]
        if pred == r["answer"]: n_ok += 1
        if i % args.report_every == 0 or i == n:
            el = time.time() - t0
            eta = el * (n - i) / i
            print(f"  {i}/{n}  acc={n_ok/i:.3f}  parsefail={n_parsefail}  eta={eta:.0f}s",
                  flush=True)

    print(f"\n=== arc-da[{args.mode}]  n={n}  acc={100*n_ok/n:.2f}%  ({n_ok}/{n})  "
          f"parsefail={n_parsefail}  random~=25% ===", flush=True)


if __name__ == "__main__":
    main()
