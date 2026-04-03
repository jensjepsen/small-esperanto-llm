"""Generate text from a trained Esperanto LLaMA model."""

import argparse

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
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

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
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        if args.num_samples > 1:
            print(f"--- Sample {i + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
