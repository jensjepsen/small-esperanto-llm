"""Fine-tune a pretrained Esperanto model on SFT data with sequence packing.

Differences vs `train_sft.py`:
- Labels are computed at tokenize-time (prompt + tool-result spans masked
  to -100), not at collate-time. This lets us pack short conversations
  into max_length-token sequences without losing the masking.
- Greedy packing with `<eos>` separator between conversations. Each
  packed example is exactly max_length tokens (padded at the end).
  Causal-attention cross-leak between packed conversations is left in
  place (option 1 of the design discussion) — the eos marker is the
  learned boundary signal.
- Uses plain `transformers.Trainer` with no custom collator (labels
  are already correct).

Throughput motivation: our SFT mix is dominated by short rows
(arith chains ~30 tok, easy algebra ~50 tok). Without packing, every
batch pads short rows up to the longest in the batch, wasting compute
on pad tokens. Packing typically gives 3-5× throughput on
short-row-dominated mixes.
"""

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

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"
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
    return re.sub(r"<<[^>]*>>", "", text)


def format_conversation(messages: list[dict]) -> str:
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
    conversations = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            data = json.loads(line)
            text = format_conversation(data["messages"])
            conversations.append(text)
    return conversations


def _build_preprocess_and_tokenize(tokenizer, special_tokens, max_length):
    """Return a tokenize fn that yields {input_ids, attention_mask}.
    Morpheme-preprocesses non-special-token text spans, leaves special
    tokens intact for the tokenizer to map to single ids.
    """
    pat = "(" + "|".join(
        re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)
    ) + ")"
    pat = re.compile(pat)

    def fn(text: str) -> dict:
        parts = pat.split(text)
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
            max_length=max_length,
            truncation=True,
            padding=False,
        )
    return fn


def _build_label_masker(assistant_id, tool_open_id, tool_close_id, unk_id):
    """Return a fn that takes input_ids and yields a labels list with
    -100 for prompt tokens (everything up to and including the first
    <|assistant|>) and tool-result span tokens. Returns None if the row
    has no <|assistant|> (malformed → drop)."""
    mask_tool = (tool_open_id is not None and tool_open_id != unk_id
                 and tool_close_id is not None and tool_close_id != unk_id)

    def mask(input_ids: list[int]) -> list[int] | None:
        try:
            first = input_ids.index(assistant_id)
        except ValueError:
            return None
        labels = list(input_ids)
        for i in range(first + 1):
            labels[i] = -100
        if mask_tool:
            opens = [i for i, t in enumerate(input_ids) if t == tool_open_id]
            closes = [i for i, t in enumerate(input_ids) if t == tool_close_id]
            for o, c in zip(opens, closes):
                if c >= o:
                    for i in range(o, c + 1):
                        labels[i] = -100
        return labels
    return mask


def _pack_chunk(args):
    """Pack a single chunk of rows sequentially. Called in a worker process
    via multiprocessing. Returns a list of packed rows (each a dict).

    Each chunk's final buffer is emitted as a padded pack — this creates a
    small number of extra sub-full packs at chunk boundaries (1 per worker)
    but that's negligible when rows/chunk ≫ 1.
    """
    rows, max_length, eos_id, pad_id = args
    packed = []
    cur_ids: list[int] = []
    cur_labels: list[int] = []

    def _emit():
        if not cur_ids:
            return
        n = len(cur_ids)
        # Pre-size once; extend with pads if short. Faster than list(cur_ids).
        pad = max_length - n
        if pad > 0:
            ids = cur_ids + [pad_id] * pad
            labels = cur_labels + [-100] * pad
            attn = [1] * n + [0] * pad
        else:
            ids = cur_ids
            labels = cur_labels
            attn = [1] * n
        packed.append({"input_ids": ids, "attention_mask": attn, "labels": labels})

    for row in rows:
        ids = row["input_ids"]
        labels = row["labels"]
        needed = len(ids) + (1 if cur_ids else 0)
        if len(cur_ids) + needed > max_length:
            _emit()
            cur_ids, cur_labels = [], []
        if cur_ids:
            cur_ids.append(eos_id)
            cur_labels.append(-100)
        cur_ids.extend(ids)
        cur_labels.extend(labels)

    _emit()
    return packed


def _pack_rows(rows, max_length: int, eos_id: int, pad_id: int, num_proc: int = 1):
    """Greedy-pack tokenized rows into max_length sequences. Parallelizable
    by chunking the input across `num_proc` workers; each chunk packs
    independently, and results are concatenated.

    num_proc=1 falls back to a single sequential pack.
    """
    if num_proc <= 1 or len(rows) < 10_000:
        return _pack_chunk((rows, max_length, eos_id, pad_id))

    import multiprocessing as mp

    # Slice with HF Dataset.select() instead of materializing the whole
    # thing to a Python list — arrow-backed views are cheap to construct
    # and each worker iterates its slice in-process. Materializing 1M rows
    # into a Python list on the main process was the bottleneck.
    n = len(rows)
    chunk_size = (n + num_proc - 1) // num_proc
    if hasattr(rows, "select"):
        chunks = [rows.select(range(i, min(i + chunk_size, n)))
                  for i in range(0, n, chunk_size)]
    else:
        chunks = [rows[i:i + chunk_size] for i in range(0, n, chunk_size)]
    tasks = [(c, max_length, eos_id, pad_id) for c in chunks]

    # Use forkserver so we don't inherit main's large tokenized-data heap
    # AND don't need to re-pickle spawn-style. Falls back to fork if unavail.
    # Fork poisons CUDA init later when main touches torch.cuda; forkserver
    # spawns children from a small stub interpreter, avoiding that.
    try:
        ctx = mp.get_context("forkserver")
    except ValueError:
        ctx = mp.get_context("fork")
    with ctx.Pool(num_proc) as pool:
        results = pool.map(_pack_chunk, tasks)
    return [p for chunk_result in results for p in chunk_result]


def main():
    parser = argparse.ArgumentParser(description="Pack-aware SFT")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sft-data", type=str, nargs="+", required=True,
                        help="Local JSONL files or HF Hub dataset names.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="tokenizer_morpheme")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Per-row truncation AND packed-sequence length.")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Cap total examples (split evenly across sources). "
                             "0 = no cap.")
    parser.add_argument("--save-steps", type=int, default=None,
                        help="Explicit save/eval interval. If set, overrides "
                             "--save-fraction-of-epoch. If neither is set, "
                             "defaults to every half epoch.")
    parser.add_argument("--save-fraction-of-epoch", type=float, default=0.5,
                        help="Save every N fraction of an epoch (computed "
                             "from train dataset size / effective batch). "
                             "Ignored when --save-steps is given.")
    parser.add_argument("--eval-fraction-of-epoch", type=float, default=None,
                        help="Eval every N fraction of an epoch. If unset, "
                             "matches --save-fraction-of-epoch (legacy "
                             "coupled behavior). Set a larger value than "
                             "save-fraction to save frequently but eval "
                             "less often — useful when eval is unstable "
                             "and you want frequent checkpoints for crash "
                             "recovery.")
    parser.add_argument("--no-pin-memory", action="store_true",
                        help="Disable dataloader_pin_memory. Workaround for "
                             "sporadic 'CUDA error: unknown error' in "
                             "pin_memory thread during eval on some GPUs "
                             "(e.g. Blackwell 5090 under load).")
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--no-best", action="store_true")
    parser.add_argument(
        "--lr-scheduler", type=str, default="cosine_with_min_lr",
        choices=["cosine_with_min_lr", "constant",
                 "constant_with_warmup", "linear", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--wandb-project", default="jepsen/espllm")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--attn-impl", default="auto",
                        choices=["auto", "flash_attention_2", "sdpa", "eager"])
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        base = f"{args.checkpoint.rstrip('/')}-sft-packed"
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
            entity=entity, project=project,
            name=args.wandb_run_name or Path(output_dir).name,
            tags=args.wandb_tags,
            config={
                "task": "sft-packed", "checkpoint": args.checkpoint,
                "sft_data": args.sft_data, "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "packing": True,
            },
        )

    console.print(f"[bold green]Loading model from {args.checkpoint}...")
    attn_impl = args.attn_impl
    if attn_impl == "auto":
        try:
            import flash_attn  # noqa: F401
            import torch as _t
            if _t.cuda.is_available() and _t.cuda.is_bf16_supported():
                attn_impl = "flash_attention_2"
            else:
                attn_impl = "sdpa"
        except ImportError:
            attn_impl = "sdpa"
    console.print(f"[bold]Attention impl:[/] {attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, attn_implementation=attn_impl)

    console.print(f"[bold green]Loading tokenizer from {args.tokenizer}...")
    tokenizer = load_tokenizer(Path(args.tokenizer))

    special_tokens = list(SPECIAL_TOKENS)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens})
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        console.print(f"[bold]Added {num_added} special tokens, "
                      f"resized embeddings to {len(tokenizer)}")

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
            source_cap = per_source_cap
            if ":" in source:
                parts = source.split(":")
                repo_id, config = parts[0], parts[1]
                split = parts[2] if len(parts) > 2 else "train"
                if len(parts) > 3 and parts[3]:
                    source_cap = int(parts[3])
                ds = hf_load(repo_id, config, split=split)
            else:
                ds = hf_load(source, split="train")
            if source_cap:
                ds = ds.shuffle(seed=42).select(
                    range(min(source_cap, len(ds))))
            for row in ds:
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

    # Tokenize + compute labels per conversation. Drop rows without
    # an <|assistant|> token (malformed) — they'd contribute zero loss
    # anyway and would just waste packed bytes.
    assistant_id = tokenizer.convert_tokens_to_ids(ASSISTANT_TOKEN)
    if assistant_id is None or assistant_id == tokenizer.unk_token_id:
        raise RuntimeError(
            f"Completion-only loss requires {ASSISTANT_TOKEN!r} in vocab; "
            f"got id={assistant_id}")
    unk_id = tokenizer.unk_token_id
    tr_open_id = tokenizer.convert_tokens_to_ids(TOOL_RESULT_OPEN)
    tr_close_id = tokenizer.convert_tokens_to_ids(TOOL_RESULT_CLOSE)
    mask_tool = (tr_open_id != unk_id and tr_close_id != unk_id)
    console.print(
        f"[bold]Masking prompt before+including {ASSISTANT_TOKEN!r} "
        f"(id={assistant_id}); tool-result spans masked: {mask_tool}")

    tokenize_fn = _build_preprocess_and_tokenize(
        tokenizer, special_tokens, args.max_length)
    label_fn = _build_label_masker(
        assistant_id, tr_open_id, tr_close_id, unk_id)

    console.print("[bold green]Tokenizing + computing labels...")
    from datasets import Dataset
    from esperanto_lm.data import num_proc

    raw_ds = Dataset.from_dict({"text": conversations})

    def _tok_with_labels(row):
        enc = tokenize_fn(row["text"])
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", [1] * len(input_ids))
        labels = label_fn(input_ids)
        if labels is None:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        # Return exactly these 3 keys regardless of what the tokenizer
        # added — HF datasets' batched .map() across workers crashes
        # with KeyError if the per-row return dict has inconsistent keys
        # (some tokenizer configs slip in `token_type_ids`).
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels}

    # Cache the tokenized+filtered+split dataset on disk so a crashed
    # training run can resume without redoing 10+ minutes of tokenize.
    import hashlib as _h
    fingerprint = _h.md5(
        (str(sorted(args.sft_data)) + str(args.max_length)
         + str(args.max_examples) + str(len(raw_ds))
         ).encode()
    ).hexdigest()[:12]
    cache_root = Path(args.output_dir or ".") / "prep_cache"
    cache_dir = cache_root / fingerprint
    if cache_dir.exists() and (cache_dir / "dataset_dict.json").exists():
        from datasets import load_from_disk
        console.print(f"[bold cyan]Loading cached splits from {cache_dir}")
        splits = load_from_disk(str(cache_dir))
    else:
        tokenized = raw_ds.map(
            _tok_with_labels,
            num_proc=num_proc(),
            remove_columns=["text"],
            desc="Tokenizing",
        )
        # Filter out malformed rows (empty after dropping no-<|assistant|>)
        n_pre = len(tokenized)
        tokenized = tokenized.filter(lambda r: len(r["input_ids"]) > 0,
                                      num_proc=num_proc())
        n_dropped = n_pre - len(tokenized)
        if n_dropped:
            console.print(f"[bold yellow]Dropped {n_dropped:,} malformed rows "
                          f"(no {ASSISTANT_TOKEN})")

        splits = tokenized.train_test_split(test_size=0.05, seed=42)
        cache_dir.mkdir(parents=True, exist_ok=True)
        splits.save_to_disk(str(cache_dir))
        console.print(f"[bold cyan]Saved splits to {cache_dir}")

    console.print(f"[bold]Pre-pack:[/] train={len(splits['train']):,} "
                  f"eval={len(splits['test']):,}")

    # Greedy-pack each split. Shuffle first so packed examples mix sources.
    eos_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("</s>")
    pad_id = tokenizer.pad_token_id

    import os
    # Touch CUDA in main BEFORE any child processes are spawned. Otherwise
    # child procs get a poisoned CUDA context state that propagates back
    # to main via shared handles/env, and the trainer's later CUDA init
    # fails with 'CUDA initialization: unknown error' — model falls back
    # to CPU silently.
    if torch.cuda.is_available():
        torch.cuda.init()
        _ = torch.cuda.device_count()
    pack_workers = min(num_proc(), max(1, os.cpu_count() - 2))
    console.print(f"[bold green]Packing into max_length sequences "
                  f"(parallel, workers={pack_workers})...")
    train_shuffled = splits["train"].shuffle(seed=0)
    test_shuffled = splits["test"].shuffle(seed=0)
    train_packed = _pack_rows(
        train_shuffled, args.max_length, eos_id, pad_id, num_proc=pack_workers)
    test_packed = _pack_rows(
        test_shuffled, args.max_length, eos_id, pad_id, num_proc=pack_workers)

    train_dataset = Dataset.from_list(train_packed)
    eval_dataset = Dataset.from_list(test_packed)

    # Packing efficiency: token count vs theoretical max
    total_train_tokens = sum(
        (1 if l != -100 else 0)
        for ex in train_packed for l in ex["labels"]
    )
    total_train_slots = len(train_packed) * args.max_length
    console.print(
        f"[bold]Post-pack:[/] train={len(train_packed):,} packed sequences "
        f"({args.max_length} tok each); eval={len(test_packed):,}")
    console.print(
        f"[bold]Train completion-token utilization:[/] "
        f"{total_train_tokens:,} / {total_train_slots:,} = "
        f"{100 * total_train_tokens / total_train_slots:.1f}%")

    import torch
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Compute effective save/eval interval. Explicit --save-steps wins;
    # otherwise derive from --save-fraction-of-epoch × steps_per_epoch.
    effective_batch = args.batch_size * args.gradient_accumulation
    steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    if args.save_steps is not None:
        save_steps = args.save_steps
    else:
        save_steps = max(1, int(steps_per_epoch * args.save_fraction_of_epoch))
    # Eval interval — decoupled from save when --eval-fraction-of-epoch given.
    if args.eval_fraction_of_epoch is not None:
        raw_eval_steps = max(1, int(steps_per_epoch * args.eval_fraction_of_epoch))
        # Round to a multiple of save_steps so every eval step is also a save
        # step — required so the best-eval checkpoint actually gets saved and
        # tracked for load_best_at_end.
        eval_steps = max(save_steps, (raw_eval_steps // save_steps) * save_steps)
    else:
        eval_steps = save_steps
    console.print(f"[cyan]save every {save_steps} steps "
                  f"({save_steps / steps_per_epoch:.2%} of epoch), "
                  f"eval every {eval_steps} steps "
                  f"({eval_steps / steps_per_epoch:.2%} of epoch), "
                  f"~{steps_per_epoch} steps/epoch")

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
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        dataloader_pin_memory=not args.no_pin_memory,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=not args.no_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="wandb" if args.wandb_project else "none",
        dataloader_num_workers=2,
        optim=args.optim,
    )

    # Default collator. Packed rows already have correct input_ids,
    # attention_mask, and labels with -100 in the right places —
    # nothing to do at collate time but stack into a batch tensor.
    from transformers import default_data_collator

    console.print("[bold green]Starting packed SFT training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume or None)

    final = Path(output_dir) / "final"
    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    console.print(f"[bold green]Saved final model to {final}")


if __name__ == "__main__":
    main()
