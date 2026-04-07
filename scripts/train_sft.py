"""Fine-tune a pretrained Esperanto model on SFT conversation data."""

import argparse
import json
import re
from pathlib import Path

from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from esperanto_lm.data import load_tokenizer, _morpheme_preprocess

console = Console()

# Chat template tokens
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"


def format_conversation(messages: list[dict]) -> str:
    """Format a conversation into a training string with role tokens."""
    parts = []
    for msg in messages:
        if msg["role"] == "user":
            parts.append(f"{USER_TOKEN} {msg['content']}")
        elif msg["role"] == "assistant":
            parts.append(f"{ASSISTANT_TOKEN} {msg['content']} {END_TOKEN}")
    return " ".join(parts)


def load_sft_data(path: Path) -> list[str]:
    """Load SFT conversations and format them as training strings."""
    conversations = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            text = format_conversation(data["messages"])
            conversations.append(text)
    return conversations


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on SFT data")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--sft-data", type=str, default="jensjepsen/esperanto-sft-factoid",
                        help="Path to local SFT JSONL or HF Hub dataset name")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <checkpoint>-sft)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer_morpheme")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    output_dir = args.output_dir or f"{args.checkpoint.rstrip('/')}-sft"

    console.print(f"[bold green]Loading model from {args.checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)

    console.print(f"[bold green]Loading tokenizer from {args.tokenizer}...")
    tokenizer = load_tokenizer(Path(args.tokenizer))

    # Add chat template tokens
    special_tokens = [USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        console.print(f"[bold]Added {num_added} special tokens, resized embeddings to {len(tokenizer)}")

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print(f"[bold green]Loading SFT data from {args.sft_data}...")
    sft_path = Path(args.sft_data)
    if sft_path.exists():
        # Local JSONL file
        conversations = load_sft_data(sft_path)
    else:
        # HF Hub dataset
        from datasets import load_dataset as hf_load
        ds = hf_load(args.sft_data, split="train")
        conversations = [format_conversation(row["messages"]) for row in ds]
    console.print(f"[bold]Conversations loaded:[/] {len(conversations):,}")

    # Morpheme-preprocess and tokenize
    console.print("[bold green]Tokenizing conversations...")

    def preprocess_and_tokenize(text: str) -> dict:
        # Split on chat tokens, preprocess the content parts, rejoin
        parts = re.split(f'({re.escape(USER_TOKEN)}|{re.escape(ASSISTANT_TOKEN)}|{re.escape(END_TOKEN)})', text)
        processed = []
        for part in parts:
            if part in special_tokens:
                processed.append(part)
            elif part.strip():
                processed.append(_morpheme_preprocess(part.strip()))
            else:
                processed.append(part)
        return tokenizer(
            " ".join(processed),
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )

    tokenized = [preprocess_and_tokenize(conv) for conv in conversations]

    # Create dataset
    from datasets import Dataset

    dataset = Dataset.from_dict({
        "input_ids": [t["input_ids"] for t in tokenized],
        "attention_mask": [t["attention_mask"] for t in tokenized],
        "labels": [t["input_ids"].copy() for t in tokenized],
    })

    splits = dataset.train_test_split(test_size=0.05, seed=42)
    console.print(f"[bold]Train:[/] {len(splits['train']):,}  [bold]Eval:[/] {len(splits['test']):,}")

    # Training arguments
    import torch
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        fp16=not use_bf16 and torch.cuda.is_available(),
        bf16=use_bf16,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=2,
    )

    console.print("[bold green]Starting SFT training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
    )

    trainer.train()

    console.print("[bold green]Saving final model...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    console.print(f"[bold green]Done! Saved to {output_dir}/final")


if __name__ == "__main__":
    main()
