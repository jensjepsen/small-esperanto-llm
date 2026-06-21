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

# Tool-call template tokens. Tool calls are emitted by the assistant
# ({TOOL_CALL_OPEN}EXPR{TOOL_CALL_CLOSE}); tool results are environment
# input ({TOOL_RESULT_OPEN}VAL{TOOL_RESULT_CLOSE}) and masked from loss.
TOOL_CALL_OPEN = "<|tool_call|>"
TOOL_CALL_CLOSE = "<|/tool_call|>"
TOOL_RESULT_OPEN = "<|tool_result|>"
TOOL_RESULT_CLOSE = "<|/tool_result|>"

SPECIAL_TOKENS = [
    USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN,
    TOOL_CALL_OPEN, TOOL_CALL_CLOSE,
    TOOL_RESULT_OPEN, TOOL_RESULT_CLOSE,
]


def _clean_gsm8k_markers(text: str) -> str:
    """Strip <<calculation>> markers from GSM8K answers, keep #### final answer."""
    return re.sub(r"<<[^>]*>>", "", text)


def format_conversation(messages: list[dict]) -> str:
    """Format a conversation into a training string with role tokens.

    Roles handled:
      user      → "<|user|> CONTENT"
      assistant → "<|assistant|> CONTENT <|end|>"
      tool      → "CONTENT"  (content already carries
                  <|tool_result|>...<|/tool_result|> from the converter,
                  so no role wrapper is added — loss is masked over the
                  span by the collator).
    """
    parts = []
    for msg in messages:
        content = _clean_gsm8k_markers(msg["content"])
        role = msg["role"]
        if role == "user":
            parts.append(f"{USER_TOKEN} {content}")
        elif role == "assistant":
            parts.append(f"{ASSISTANT_TOKEN} {content} {END_TOKEN}")
        elif role == "tool":
            parts.append(content)
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
    parser.add_argument("--no-best", action="store_true",
                        help="Disable load_best_model_at_end. Default is to track "
                             "eval_loss across checkpoints, keep the best one, and "
                             "load it before saving --output-dir/final.")
    parser.add_argument(
        "--lr-scheduler", type=str, default="cosine_with_min_lr",
        choices=["cosine_with_min_lr", "constant",
                 "constant_with_warmup", "linear", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--no-completion-loss", action="store_true",
                        help="Disable completion-only loss. Default is to mask "
                             "loss on user-prompt tokens so gradient flows "
                             "only through the assistant response — matches "
                             "the inference objective and stops the model "
                             "wasting capacity learning to recite questions.")
    parser.add_argument("--wandb-project", default="jepsen/espllm",
                        help="`entity/project` for Weights & Biases. "
                             "Pass empty string to disable wandb logging.")
    parser.add_argument("--wandb-run-name", default=None,
                        help="Optional run name (default: auto from output-dir).")
    parser.add_argument("--wandb-tags", nargs="*", default=None,
                        help="Optional tags for the wandb run.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in "
                             "--output-dir. Useful when restarting "
                             "after an OOM or to switch batch size "
                             "without losing prior progress.")
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

    # Add chat + tool template tokens. add_special_tokens is a no-op for
    # tokens already in the vocab, so this is safe across resume / fresh
    # runs / continued fine-tunes from v4 (no tool tokens) → v6 (with).
    special_tokens = list(SPECIAL_TOKENS)
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
            for row in ds:
                # Accept either {messages: [...]} or Alpaca-style
                # {instruction, input, output}.
                if "messages" in row:
                    msgs = row["messages"]
                else:
                    instr = (row.get("instruction") or "").strip()
                    inp = (row.get("input") or "").strip()
                    out = row.get("output") or row.get("response") or ""
                    user = f"{instr}\n\n{inp}" if inp else instr
                    msgs = [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": out},
                    ]
                conversations.append(format_conversation(msgs))
        console.print(f"[bold]  Loaded, total so far:[/] {len(conversations):,}")
    console.print(f"[bold]Total conversations:[/] {len(conversations):,}")

    # Morpheme-preprocess and tokenize
    console.print("[bold green]Tokenizing conversations...")

    # Build a regex that protects any of the registered special tokens
    # from being morpheme-split. Sort longest-first so e.g. "<|/tool_call|>"
    # is matched before "<|tool_call|>".
    _special_pat = "(" + "|".join(
        re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)
    ) + ")"

    def preprocess_and_tokenize(text: str) -> dict:
        parts = re.split(_special_pat, text)
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
        lr_scheduler_type=args.lr_scheduler,
        lr_scheduler_kwargs=(
            {"min_lr_rate": 0.1}
            if args.lr_scheduler == "cosine_with_min_lr" else {}),
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        fp16=not use_bf16 and torch.cuda.is_available(),
        bf16=use_bf16,
        eval_strategy="steps",
        # eval_steps must match save_steps when load_best_model_at_end=True
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=not args.no_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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

    if not args.no_completion_loss:
        assistant_id = tokenizer.convert_tokens_to_ids(ASSISTANT_TOKEN)
        if assistant_id is None or assistant_id == tokenizer.unk_token_id:
            raise RuntimeError(
                f"--completion-only-loss requires {ASSISTANT_TOKEN!r} in the "
                f"tokenizer vocab; got id={assistant_id}"
            )
        # Tool result spans are environment input; model receives them
        # but does not emit them. Mask the closed interval [open, close].
        # IDs may be unk if the tokens were never added (pre-funcall
        # checkpoints) — in that case there are no spans to mask.
        unk_id = tokenizer.unk_token_id
        tr_open_id = tokenizer.convert_tokens_to_ids(TOOL_RESULT_OPEN)
        tr_close_id = tokenizer.convert_tokens_to_ids(TOOL_RESULT_CLOSE)
        mask_tool_results = (
            tr_open_id is not None and tr_open_id != unk_id
            and tr_close_id is not None and tr_close_id != unk_id
        )
        console.print(
            f"[bold]Masking loss before/including token "
            f"{ASSISTANT_TOKEN!r} (id={assistant_id}); "
            f"tool-result spans masked: {mask_tool_results}[/]"
        )

        def data_collator(features):
            batch = base_collator(features)
            for i, ids in enumerate(batch["input_ids"]):
                # 1) Mask initial prefix up to and including the first
                #    <|assistant|>.
                hits = (ids == assistant_id).nonzero(as_tuple=True)[0]
                if len(hits) == 0:
                    # No assistant token — mask whole sequence to avoid
                    # training on malformed rows rather than default back
                    # to full-sequence loss.
                    batch["labels"][i, :] = -100
                    continue
                cutoff = hits[0].item() + 1
                batch["labels"][i, :cutoff] = -100
                # 2) Mask each <|tool_result|>...<|/tool_result|> span.
                if mask_tool_results:
                    opens = (ids == tr_open_id).nonzero(as_tuple=True)[0].tolist()
                    closes = (ids == tr_close_id).nonzero(as_tuple=True)[0].tolist()
                    for o, c in zip(opens, closes):
                        if c >= o:
                            batch["labels"][i, o:c + 1] = -100
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

    trainer.train(resume_from_checkpoint=args.resume or None)

    console.print("[bold green]Saving final model...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    console.print(f"[bold green]Done! Saved to {output_dir}/final")


if __name__ == "__main__":
    main()
