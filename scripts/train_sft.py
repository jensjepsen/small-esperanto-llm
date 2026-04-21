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


def _clean_gsm8k_markers(text: str) -> str:
    """Strip <<calculation>> markers from GSM8K answers, keep #### final answer."""
    return re.sub(r"<<[^>]*>>", "", text)


def format_conversation(messages: list[dict]) -> str:
    """Format a conversation into a training string with role tokens."""
    parts = []
    for msg in messages:
        content = _clean_gsm8k_markers(msg["content"])
        if msg["role"] == "user":
            parts.append(f"{USER_TOKEN} {content}")
        elif msg["role"] == "assistant":
            parts.append(f"{ASSISTANT_TOKEN} {content} {END_TOKEN}")
    return " ".join(parts)


def load_sft_data(path: Path, max_examples: int = 0) -> list[str]:
    """Load SFT conversations and format them as training strings."""
    conversations = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            data = json.loads(line)
            text = format_conversation(data["messages"])
            conversations.append(text)
    return conversations


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on SFT data")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--sft-data", type=str, nargs="+",
                        default=["jensjepsen/esperanto-sft-factoid", "jensjepsen/esperanto-sft-creative",
                                 "jensjepsen/esperanto-gsm8k", "jensjepsen/esperanto-arithmetic-cot",
                                 "jensjepsen/esperanto-sft-atomic-icl", "jensjepsen/esperanto-sft-atomic-qa",
                                 "jensjepsen/esperanto-sft-wikidata-icl", "jensjepsen/esperanto-sft-morphology-icl",
                                 "jensjepsen/esperanto-sft-quantity-reasoning",
                                 "jensjepsen/esperanto-sft-dolly"],
                        help="Paths to local SFT JSONL files or HF Hub dataset names")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <checkpoint>-sft)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer_morpheme")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Cap total examples (split evenly across sources). "
                             "0 = no cap. Useful for cold-start runs where you "
                             "want format-learning without memorization.")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--completion-only-loss", action="store_true",
                        help="Mask loss on the user-prompt tokens; only train "
                             "on the assistant response. Important when the "
                             "prompt template is highly repetitive (e.g. NLI).")
    parser.add_argument("--wandb-project", default="jepsen/espllm",
                        help="`entity/project` for Weights & Biases. "
                             "Pass empty string to disable wandb logging.")
    parser.add_argument("--wandb-run-name", default=None,
                        help="Optional run name (default: auto from output-dir).")
    parser.add_argument("--wandb-tags", nargs="*", default=None,
                        help="Optional tags for the wandb run.")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        base = f"{args.checkpoint.rstrip('/')}-sft"
        if not Path(base).exists():
            output_dir = base
        else:
            n = 2
            while Path(f"{base}-{n}").exists():
                n += 1
            output_dir = f"{base}-{n}"

    if args.wandb_project:
        import os
        import wandb
        if "/" in args.wandb_project:
            entity, project = args.wandb_project.split("/", 1)
        else:
            entity, project = None, args.wandb_project
        os.environ.setdefault("WANDB_PROJECT", project)
        if entity:
            os.environ.setdefault("WANDB_ENTITY", entity)
        wandb.init(
            entity=entity,
            project=project,
            name=args.wandb_run_name or Path(output_dir).name,
            tags=args.wandb_tags,
            config={
                "task": "sft",
                "checkpoint": args.checkpoint,
                "sft_data": args.sft_data,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
            },
        )

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

    per_source_cap = (args.max_examples // len(args.sft_data)
                      if args.max_examples else 0)
    conversations = []
    for source in args.sft_data:
        console.print(f"[bold green]Loading SFT data from {source}...")
        sft_path = Path(source)
        if sft_path.exists():
            conversations.extend(load_sft_data(sft_path, per_source_cap))
        else:
            from datasets import load_dataset as hf_load
            ds = hf_load(source, split="train")
            if per_source_cap:
                ds = ds.select(range(min(per_source_cap, len(ds))))
            conversations.extend([format_conversation(row["messages"]) for row in ds])
        console.print(f"[bold]  Loaded, total so far:[/] {len(conversations):,}")
    console.print(f"[bold]Total conversations:[/] {len(conversations):,}")

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
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        warmup_steps=100,
        weight_decay=0.01,
        fp16=not use_bf16 and torch.cuda.is_available(),
        bf16=use_bf16,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=50,
        report_to="wandb" if args.wandb_project else "none",
        dataloader_num_workers=2,
    )

    # Data collator that pads sequences to equal length.
    # With --completion-only-loss the loss is masked for everything up to
    # and including the <|assistant|> token, so gradient only flows through
    # the response. Necessary for NLI-style fine-tunes where the user-turn
    # template is so repetitive that whole-sequence loss drowns out the
    # label signal.
    from transformers import DataCollatorForLanguageModeling
    import torch as _torch

    base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if args.completion_only_loss:
        assistant_id = tokenizer.convert_tokens_to_ids(ASSISTANT_TOKEN)
        if assistant_id is None or assistant_id == tokenizer.unk_token_id:
            raise RuntimeError(
                f"--completion-only-loss requires {ASSISTANT_TOKEN!r} in the "
                f"tokenizer vocab; got id={assistant_id}"
            )
        console.print(f"[bold]Masking loss before/including token "
                      f"{ASSISTANT_TOKEN!r} (id={assistant_id})[/]")

        def data_collator(features):
            batch = base_collator(features)
            for i, ids in enumerate(batch["input_ids"]):
                hits = (ids == assistant_id).nonzero(as_tuple=True)[0]
                if len(hits) > 0:
                    cutoff = hits[0].item() + 1
                    batch["labels"][i, :cutoff] = -100
                else:
                    # No assistant token found — mask whole sequence to avoid
                    # training on malformed rows rather than default back to
                    # full-sequence loss.
                    batch["labels"][i, :] = -100
            return batch
    else:
        data_collator = base_collator

    console.print("[bold green]Starting SFT training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        data_collator=data_collator,
    )

    trainer.train()

    console.print("[bold green]Saving final model...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    console.print(f"[bold green]Done! Saved to {output_dir}/final")


if __name__ == "__main__":
    main()
