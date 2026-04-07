"""Chat with an SFT-trained Esperanto model."""

import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from esperanto_lm.data import load_tokenizer, _morpheme_preprocess

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"


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

    special_tokens = [USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN]

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

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.convert_tokens_to_ids(END_TOKEN),
            )

        # Decode only the new tokens
        new_tokens = output[0][inputs["input_ids"].shape[1]:]

        # Stop at END_TOKEN
        end_id = tokenizer.convert_tokens_to_ids(END_TOKEN)
        trimmed = []
        for tok_id in new_tokens.tolist():
            if tok_id == end_id:
                break
            trimmed.append(tok_id)

        # Use convert_ids_to_tokens for <w> handling
        gen_tokens = tokenizer.convert_ids_to_tokens(trimmed)

        # Filter out chat tokens
        gen_tokens = [t for t in gen_tokens if t not in special_tokens]

        # Join with <w> handling
        has_w = "<w>" in tokenizer.get_vocab()
        if has_w:
            text = "".join(t if t != "<w>" else " " for t in gen_tokens)
        else:
            text = " ".join(gen_tokens)

        return text.strip()

    if args.prompt:
        # Single prompt mode
        messages = [{"role": "user", "content": args.prompt}]
        response = generate_response(messages)
        print(response)
    else:
        # Interactive mode
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
