"""Eval LFM2.5-350M on the Esperanto ICL benchmark via on-the-fly translation.

Pipeline per question:
    EO question --v5b eo→en--> EN question --LFM--> EN answer --v5b en→eo--> EO answer
Then score against the EO eval's accepted answers using the project's matcher.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sp_tokenizer import SPMTokenizer

# Pull the project's lenient Esperanto matcher
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from eval_icl import normalize as eo_normalize


def matches(pred: str, gold: str) -> bool:
    p, g = eo_normalize(pred), eo_normalize(gold)
    if not p or not g:
        return False
    return p == g or g in p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfm-model", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--mt-checkpoint", default="/mnt/data/espllm/runs/mt/eneo_v5b/final")
    ap.add_argument("--mt-tokenizer", default="mt/data/tokenizer/spm_eneo_32k.model")
    ap.add_argument("--eval", default="data/causal_corpus/eval_handcrafted_v31.jsonl")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--lfm-batch-size", type=int, default=32)
    ap.add_argument("--mt-batch-size", type=int, default=32)
    ap.add_argument("--lfm-max-new-tokens", type=int, default=128)
    ap.add_argument("--mt-num-beams", type=int, default=4)
    ap.add_argument("--mt-max-length", type=int, default=192)
    ap.add_argument("--out", default="mt/runs/lfm25_eo_via_mt.jsonl")
    ap.add_argument("--hf-cache", default="/mnt/data/hf_cache")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_cache)
    from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel

    # --- load v5b translator ---
    print(f"Loading MT {args.mt_checkpoint} on cuda…", flush=True)
    mt_tok = SPMTokenizer(args.mt_tokenizer)
    mt_model = MarianMTModel.from_pretrained(args.mt_checkpoint).half().to("cuda").eval()
    mt_model.generation_config.no_repeat_ngram_size = 5

    # --- load LFM ---
    print(f"Loading LFM {args.lfm_model} on cuda fp16…", flush=True)
    lfm_tok = AutoTokenizer.from_pretrained(args.lfm_model)
    lfm_tok.padding_side = "left"
    if lfm_tok.pad_token_id is None:
        lfm_tok.pad_token_id = lfm_tok.eos_token_id
    lfm_model = AutoModelForCausalLM.from_pretrained(args.lfm_model, dtype=torch.float16).to("cuda").eval()
    print(f"  MT mem after loads = {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    def translate_batch(texts: list[str], tgt_lang: str) -> list[str]:
        """eo→en if tgt_lang=='en', else en→eo. Batched."""
        out_strs: list[str] = []
        for i in range(0, len(texts), args.mt_batch_size):
            chunk = texts[i : i + args.mt_batch_size]
            ids = [mt_tok.encode(t, lang=tgt_lang) for t in chunk]
            be = mt_tok.pad_batch(ids)
            inp = be.input_ids.to("cuda")
            attn = be.attention_mask.to("cuda")
            with torch.no_grad():
                out = mt_model.generate(
                    input_ids=inp, attention_mask=attn,
                    num_beams=args.mt_num_beams, max_length=args.mt_max_length,
                    early_stopping=True, no_repeat_ngram_size=5,
                )
            for seq in out:
                out_strs.append(mt_tok.decode(seq))
        return out_strs

    # --- read EO eval ---
    rows = [json.loads(l) for l in open(args.eval)]
    if args.n:
        rows = rows[: args.n]
    N = len(rows)
    print(f"Eval rows: {N}")

    # --- step 1: translate EO prompts to EN ---
    eo_prompts = [r["messages"][0]["content"] for r in rows]
    print("translating prompts eo→en…", flush=True)
    t0 = time.perf_counter()
    en_prompts = translate_batch(eo_prompts, tgt_lang="en")
    print(f"  {time.perf_counter()-t0:.0f}s")

    # --- step 2: run LFM on EN prompts (no system prompt, per our finding) ---
    chat_prompts = []
    for ep in en_prompts:
        msgs = [{"role": "user", "content": ep}]
        chat_prompts.append(lfm_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))

    print("LFM generating…", flush=True)
    t0 = time.perf_counter()
    en_answers: list[str] = []
    for bs in range(0, N, args.lfm_batch_size):
        batch = chat_prompts[bs : bs + args.lfm_batch_size]
        enc = lfm_tok(batch, return_tensors="pt", padding=True, truncation=False).to("cuda")
        with torch.no_grad():
            out = lfm_model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=args.lfm_max_new_tokens,
                do_sample=False,
                pad_token_id=lfm_tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        en_answers.extend(lfm_tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  LFM batch {bs//args.lfm_batch_size + 1}/{(N+args.lfm_batch_size-1)//args.lfm_batch_size} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    # --- step 3: translate EN answers back to EO ---
    # Keep first sentence only — verbose LFM rambles otherwise.
    en_first = []
    for a in en_answers:
        a = a.strip()
        # crude first-sentence split
        m = re.search(r"^[^.!?]+[.!?]", a)
        en_first.append(m.group(0).strip() if m else a[:200])
    print("translating answers en→eo…", flush=True)
    t0 = time.perf_counter()
    eo_answers = translate_batch(en_first, tgt_lang="eo")
    print(f"  {time.perf_counter()-t0:.0f}s")

    # --- step 4: score ---
    correct = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fout:
        for i, r in enumerate(rows):
            accepted = list(r.get("accepted_answers") or [])
            gold = r["messages"][1]["content"]
            if gold not in accepted:
                accepted = [gold] + accepted
            pred_eo = eo_answers[i]
            ok = any(matches(pred_eo, a) for a in accepted)
            correct += ok
            fout.write(json.dumps({
                "i": i,
                "eo_question": eo_prompts[i],
                "en_question": en_prompts[i],
                "en_answer": en_answers[i],
                "en_answer_first": en_first[i],
                "eo_answer": pred_eo,
                "gold": gold,
                "accepted": accepted,
                "ok": ok,
            }, ensure_ascii=False) + "\n")

    print(f"\n=== pass@1 = {correct}/{N} = {100*correct/N:.2f}% ===")


if __name__ == "__main__":
    main()
