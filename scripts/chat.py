"""Chat with an SFT-trained Esperanto model."""

import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from esperanto_lm.data import load_tokenizer, _morpheme_preprocess
from esperanto_lm.morphology import decompose

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"

SKIP_TOKENS = {"<s>", "</s>", "<pad>", "<unk>", USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN}


def decode_tokens(tokenizer, token_ids):
    """Decode token IDs to text, handling <w> word boundaries.
    Same logic as generate.py.
    """
    gen_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    gen_tokens = [t for t in gen_tokens if t not in SKIP_TOKENS]

    has_w = "<w>" in tokenizer.get_vocab()
    if has_w:
        text = "".join(t if t != "<w>" else " " for t in gen_tokens)
    else:
        text_parts = []
        for t in gen_tokens:
            if t in (".", ",", ";", ":", "!", "?", ")", "]"):
                text_parts.append(t)
            elif text_parts and t not in ("(", "["):
                text_parts.append(" " + t)
            else:
                text_parts.append(t)
        text = "".join(text_parts)

    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with an SFT model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="tokenizer_morpheme")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive mode)")
    args = parser.parse_args()

    tokenizer = load_tokenizer(Path(args.tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    def encode_chat(messages: list[dict]) -> str:
        """Format messages and morpheme-preprocess the content."""
        parts = []
        for msg in messages:
            content = _morpheme_preprocess(msg["content"])
            if msg["role"] == "user":
                parts.append(f"{USER_TOKEN} {content}")
            elif msg["role"] == "assistant":
                parts.append(f"{ASSISTANT_TOKEN} {content} {END_TOKEN}")
        # Add assistant token to prompt generation
        parts.append(ASSISTANT_TOKEN)
        return " ".join(parts)

    def generate_response(messages: list[dict]) -> str:
        """Generate a response given conversation history."""
        prompt = encode_chat(messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        end_id = tokenizer.convert_tokens_to_ids(END_TOKEN)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=end_id if end_id != tokenizer.unk_token_id else None,
            )

        # Get only new tokens
        new_ids = output[0][inputs["input_ids"].shape[1]:].tolist()

        # Trim at END_TOKEN
        if end_id in new_ids:
            new_ids = new_ids[:new_ids.index(end_id)]

        return decode_tokens(tokenizer, new_ids)

    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        response = generate_response(messages)
        print(response)
    else:
        print("Esperanto SFT Chat (tajpu 'quit' por eliri)")
        print()
        messages = []
        while True:
            try:
                user_input = input("Vi: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input or user_input.lower() in ("quit", "exit", "eliru"):
                break

            messages.append({"role": "user", "content": user_input})
            response = generate_response(messages)
            print(f"AI: {response}")
            print()
            messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
