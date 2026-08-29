"""Quick eyeball probe: load a Danish LM checkpoint and generate from a few prompts.

Purpose: at very early training (step 5000 = ~1% done), confirm the model is
producing Danish-shaped output rather than random garbage. Not a benchmark,
just a smoke test.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "Danmarks hovedstad er",
    "Hun gik hjem, fordi",
    "I dag er vejret",
    "Den kendte danske forfatter H.C. Andersen skrev",
    "For at bage brød skal man",
    "Solen er en stjerne, og planeterne",
    "Fodbold er en sport hvor",
    "Købenavns Universitet blev grundlagt i",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint dir")
    ap.add_argument("--tokenizer", default="jensjepsen/danish-tokenizer")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Loading model: {args.ckpt}")
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32
    ).to(args.device)
    model.eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {args.device}, dtype: {next(model.parameters()).dtype}")
    print()

    for i, prompt in enumerate(PROMPTS, 1):
        ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        print(f"[{i}] {text!r}")
        print()


if __name__ == "__main__":
    main()
