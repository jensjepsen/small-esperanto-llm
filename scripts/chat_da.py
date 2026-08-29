"""Interactive multi-turn chat with a local Danish SFT checkpoint.

Streams response tokens as they generate and prints in/out token counts
per turn plus running totals.

Commands:
    /reset   — start a fresh conversation
    /quit    — exit
    /stats   — print total token counts
    (Ctrl+C — cancel current generation)

Usage:
    uv run python scripts/chat_da.py \\
        --ckpt runs/sft/da_v6_mix9wpreword/final \\
        [--max-new 400] [--temp 0.7]  # temp 0 = greedy
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def format_history(messages: list[dict]) -> str:
    """Trainer format: `<|user|> {u} <|assistant|> {a} <|end|>` per turn,
    then trailing `<|assistant|>` on the current turn to prompt generation."""
    parts = []
    for m in messages:
        if m["role"] == "user":
            parts.append(f"<|user|> {m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"<|assistant|> {m['content']} <|end|>")
    parts.append("<|assistant|>")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/jepsen/src/espllm/runs/sft/da_v6_mix9wpreword/final")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    args = ap.parse_args()

    print(f"loading tokenizer + model from {args.ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=dtype).cuda().eval()

    end_id = tok.convert_tokens_to_ids("<|end|>")
    user_id = tok.convert_tokens_to_ids("<|user|>")
    eos_ids = [tok.eos_token_id]
    if end_id != tok.unk_token_id and end_id is not None:
        eos_ids.append(end_id)
    if user_id != tok.unk_token_id and user_id is not None:
        eos_ids.append(user_id)  # stop if model tries to open a new user turn

    print(f"ready. temp={args.temp} max_new={args.max_new}. /reset /stats /quit")
    print("-" * 60)

    messages: list[dict] = []
    total_in = total_out = 0

    while True:
        try:
            user_msg = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user_msg:
            continue

        if user_msg == "/quit":
            break
        if user_msg == "/reset":
            messages = []
            print("[conversation reset]")
            continue
        if user_msg == "/stats":
            print(f"[in={total_in} out={total_out} total={total_in + total_out} tokens]")
            continue

        messages.append({"role": "user", "content": user_msg})
        prompt = format_history(messages)
        ids = tok(prompt, return_tensors="pt", return_token_type_ids=False,
                  add_special_tokens=False).to("cuda")
        in_toks = ids["input_ids"].shape[1]

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **ids,
            max_new_tokens=args.max_new,
            do_sample=(args.temp > 0),
            temperature=args.temp if args.temp > 0 else 1.0,
            top_p=args.top_p,
            pad_token_id=tok.eos_token_id,
            eos_token_id=eos_ids,
            streamer=streamer,
        )
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        print("bot > ", end="", flush=True)
        collected: list[str] = []
        t0 = time.time()
        try:
            for piece in streamer:
                cleaned = piece.replace("<|end|>", "").replace("<|user|>", "")
                if cleaned:
                    print(cleaned, end="", flush=True)
                    collected.append(cleaned)
        except KeyboardInterrupt:
            print("\n[cancelled]")
        thread.join()
        elapsed = time.time() - t0

        reply = "".join(collected).strip()
        messages.append({"role": "assistant", "content": reply})

        out_toks = len(tok(reply, add_special_tokens=False)["input_ids"])
        total_in += in_toks
        total_out += out_toks
        rate = out_toks / max(elapsed, 0.001)
        print(f"\n[in={in_toks} out={out_toks} ({rate:.1f} tok/s)  "
              f"total in={total_in} out={total_out}]")

    print(f"\n=== session end: in={total_in} out={total_out} total={total_in + total_out} tokens ===")


if __name__ == "__main__":
    main()
