"""Generative GSM8K eval on Danish — streams per-row JSONL so accuracy can be
tail'd as it runs.

Each output line: {idx, question, gold_num, pred_num, gen, correct}
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
NUM_RE = re.compile(r"####\s*(-?\d[\d,\.]*)")
LAST_NUM_RE = re.compile(r"(-?\d[\d,]*\.?\d*)")


def extract_num(text: str):
    m = NUM_RE.search(text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    # fallback: last number in the text
    nums = LAST_NUM_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "").rstrip(".")
    return None


def norm(s):
    if s is None: return None
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError, OverflowError):
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True, help="jsonl output path")
    ap.add_argument("--dataset", default="jensjepsen/danish-gsm8k")
    ap.add_argument("--config", default="sft")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Rows per generate() call. On 5090/32GB with a 400M "
                         "model + max_new=300, bs=64 fits comfortably.")
    args = ap.parse_args()

    print(f"loading {args.ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=dtype).cuda().eval()
    end_id = tok.convert_tokens_to_ids(END)
    eos_ids = [tok.eos_token_id]
    if end_id != tok.unk_token_id:
        eos_ids.append(end_id)

    ds = load_dataset(args.dataset, args.config, split=args.split)
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))
    n = len(ds)
    print(f"  {n} rows from {args.dataset}:{args.config}:{args.split}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    t0 = time.time()
    rows = list(ds)
    with open(args.out, "w") as f:
        for bstart in range(0, n, args.batch_size):
            batch = rows[bstart:bstart + args.batch_size]
            B = len(batch)
            prompts = [f"{USER} {r['messages'][0]['content']} {ASST}" for r in batch]
            golds = [extract_num(r['messages'][1]['content']) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=False, return_token_type_ids=False).to("cuda")
            with torch.no_grad():
                out = model.generate(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    max_new_tokens=args.max_new, do_sample=False, num_beams=1,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    repetition_penalty=1.1, eos_token_id=eos_ids,
                )
            plen = enc["input_ids"].shape[1]
            for row_ix in range(B):
                gen = tok.decode(out[row_ix, plen:], skip_special_tokens=True).strip()
                pred_num = extract_num(gen)
                ok = norm(pred_num) == norm(golds[row_ix])
                n_ok += ok
                i = bstart + row_ix + 1
                f.write(json.dumps({
                    "idx": i - 1, "q": batch[row_ix]['messages'][0]['content'],
                    "gold_num": golds[row_ix], "pred_num": pred_num,
                    "gen": gen, "correct": bool(ok),
                }, ensure_ascii=False) + "\n")
            f.flush()

            i = bstart + B
            if (i // 100) != ((i - B) // 100) or i == n:
                el = time.time() - t0
                print(f"  {i}/{n} acc={n_ok/i:.3f} ({n_ok}/{i})  "
                      f"eta={el*(n-i)/i:.0f}s", flush=True)

    print(f"\n=== gsm8k[da] {n_ok}/{n} = {100*n_ok/n:.2f}% ===")


if __name__ == "__main__":
    main()
