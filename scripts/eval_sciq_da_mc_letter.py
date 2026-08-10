"""Generative Danish SciQ eval — present 4-choice question via chat template,
parse first A/B/C/D letter from model response, compare to gold letter.
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
LETTER_RE = re.compile(r"\b([ABCD])\b")


def build_prompt_and_gold(row, i):
    opts_list = [row["da_correct_answer"],
                 row["da_distractor1"],
                 row["da_distractor2"],
                 row["da_distractor3"]]
    rng = random.Random(42 + i)
    idxs = list(range(4))
    rng.shuffle(idxs)
    letters = "ABCD"
    opts_text = []
    gold_letter = None
    for slot, orig_idx in enumerate(idxs):
        opts_text.append(f"{letters[slot]}) {opts_list[orig_idx]}")
        if orig_idx == 0:
            gold_letter = letters[slot]
    user = (f"Spørgsmål: {row['da_question']}\n" + "\n".join(opts_text)
            + "\nSvar med bogstavet.")
    prompt = f"{USER}{user}{END}{ASST}"
    return prompt, gold_letter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dump", type=str, default=None,
                    help="jsonl output (per-row question/pred/gold/gen)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Generation needs left-padding so all sequences end at the same position.
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.float16).cuda().eval()
    end_id = tok.convert_tokens_to_ids(END)
    eos_ids = [tok.eos_token_id]
    if end_id != tok.unk_token_id:
        eos_ids.append(end_id)

    ds = load_dataset("jensjepsen/danish-sciq", "default", split=args.split)
    n = len(ds)
    rows = list(ds)

    dump_f = open(args.dump, "w") if args.dump else None
    n_ok = n_parse_fail = 0
    t0 = time.time()

    for bstart in range(0, n, args.batch_size):
        batch = rows[bstart:bstart + args.batch_size]
        B = len(batch)
        prompts = []
        golds = []
        for j, row in enumerate(batch):
            p, g = build_prompt_and_gold(row, bstart + j)
            prompts.append(p); golds.append(g)
        enc = tok(prompts, return_tensors="pt", add_special_tokens=False,
                  padding=True, return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_ids,
            )
        plen = enc["input_ids"].shape[1]
        for j in range(B):
            gen = tok.decode(out[j][plen:], skip_special_tokens=True).strip()
            m = LETTER_RE.search(gen)
            if not m:
                n_parse_fail += 1
                pred = "?"
            else:
                pred = m.group(1)
            if pred == golds[j]:
                n_ok += 1
            if dump_f:
                dump_f.write(json.dumps({
                    "idx": bstart + j, "q": batch[j]["da_question"],
                    "gold": golds[j], "pred": pred, "gen": gen,
                }, ensure_ascii=False) + "\n")

        done = bstart + B
        if done % (10 * args.batch_size) == 0 or done == n:
            el = time.time() - t0
            print(f"  {done}/{n} acc={n_ok/done:.3f} parsefail={n_parse_fail} "
                  f"eta={el*(n-done)/max(done,1):.0f}s", flush=True)

    if dump_f: dump_f.close()
    print(f"\nSciQ[da] gen: {n_ok/n:.4f} ({n_ok}/{n})  "
          f"parsefail={n_parse_fail} ({n_parse_fail/n:.2%})  "
          f"random baseline 0.25")


if __name__ == "__main__":
    main()
