"""Generate text from a trained Esperanto LLaMA model."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from esperanto_lm.data import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Esperanto estas",
        help="Text prompt to continue from",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--tokenizer", type=str, default="tokenizer_morpheme",
                        help="Path to tokenizer directory")
    args = parser.parse_args()

    tokenizer = load_tokenizer(Path(args.tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    # Morpheme preprocess: decompose prompt into morphemes with <w> boundaries
    import re
    from esperanto_lm.morphology import decompose

    words = re.findall(r'[a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ]+|[^\s]', args.prompt)
    parts = []
    has_w_token = "<w>" in tokenizer.get_vocab()
    for word in words:
        if parts:
            if has_w_token:
                parts.append("<w>")
        if word[0].isalpha():
            parts.extend(decompose(word))
        else:
            parts.append(word)
    prompt = " ".join(parts)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    for i in range(args.num_samples):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
            )

        # Decode: convert tokens back to text
        gen_tokens = tokenizer.convert_ids_to_tokens(output[0])
        # Filter special tokens
        gen_tokens = [t for t in gen_tokens if t not in ("<s>", "</s>", "<pad>", "<unk>")]

        if has_w_token:
            # Join tokens, replace <w> with space
            text = "".join(t if t != "<w>" else " " for t in gen_tokens)
        else:
            # No word boundary token — use heuristic spacing
            # Insert space before morphemes that are known word-starters
            # (particles, articles, prepositions, etc.)
            from esperanto_lm.morphology import DO_NOT_DECOMPOSE, get_prefixes
            word_starters = DO_NOT_DECOMPOSE | get_prefixes()
            text_parts = []
            for t in gen_tokens:
                if t in (".", ",", ";", ":", "!", "?", ")", "]"):
                    text_parts.append(t)
                elif text_parts and t not in ("(", "["):
                    text_parts.append(" " + t)
                else:
                    text_parts.append(t)
            text = "".join(text_parts)

        if args.num_samples > 1:
            print(f"--- Sample {i + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
