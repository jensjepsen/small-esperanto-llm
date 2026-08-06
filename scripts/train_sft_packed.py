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

# Liger kernel — monkey-patches transformers' Llama classes with fused
# CUDA kernels (RoPE, RMSNorm, SwiGLU, Fused-Linear Cross-Entropy).
# Must run BEFORE `from transformers import AutoModelForCausalLM` to
# reach the classes at import time. `fused_linear_cross_entropy=True`
# swaps the LM head → optimizer state from a non-Liger run won't
# resume cleanly, so we only apply it on fresh SFT-from-base runs.
try:
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    apply_liger_kernel_to_llama(
        rope=True, rms_norm=True, swiglu=True,
        fused_linear_cross_entropy=True, cross_entropy=False,
    )
    _LIGER_ON = True
except ImportError:
    _LIGER_ON = False

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


_WORLD_ROLES = {"user", "tool_result", "tool"}
_MODEL_ROLES = {"assistant", "tool_call"}


def format_conversation(messages: list[dict]) -> str:
    """Render a chat-format message list into the flat token stream the
    trainer consumes.

    Roles:
      user         → `<|user|> {content}`
      assistant    → `<|assistant|> {content}` (+ `<|end|>` when next turn is world-provided)
      tool_call    → `<|tool_call|>{content}<|/tool_call|>` (+ `<|end|>` when next turn is world-provided)
      tool_result  → `<|tool_result|>{content}<|/tool_result|>`  (no `<|end|>`, world-provided)
      tool         → same as tool_result (legacy alias)

    `<|end|>` is emitted after a MODEL-generated turn iff the next turn is
    WORLD-provided (or the conversation ends). This keeps reasoning +
    tool_call as a single autoregressive burst (`<|assistant|> reasoning
    <|tool_call|>{...}<|/tool_call|> <|end|>`) so the model learns to
    stream from reasoning straight into the call without an artificial stop.

    Unknown roles raise — silent drops used to be a footgun.
    """
    parts = []
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = _clean_gsm8k_markers(msg["content"])
        next_role = messages[i + 1]["role"] if i + 1 < len(messages) else None
        needs_end = role in _MODEL_ROLES and (
            next_role is None or next_role in _WORLD_ROLES)

        if role == "user":
            parts.append(f"{USER_TOKEN} {content}")
        elif role == "assistant":
            block = f"{ASSISTANT_TOKEN} {content}"
            if needs_end:
                block += f" {END_TOKEN}"
            parts.append(block)
        elif role == "tool_call":
            block = f"{TOOL_CALL_OPEN}{content}{TOOL_CALL_CLOSE}"
            if needs_end:
                block += f" {END_TOKEN}"
            parts.append(block)
        elif role in ("tool_result", "tool"):
            parts.append(f"{TOOL_RESULT_OPEN}{content}{TOOL_RESULT_CLOSE}")
        else:
            raise ValueError(f"unknown message role: {role!r}")
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


def _build_preprocess_and_tokenize(tokenizer, special_tokens, max_length,
                                    morpheme_preprocess: bool = True):
    """Return a tokenize fn that yields {input_ids, attention_mask}.
    When morpheme_preprocess=True (EO default), decomposes non-special text
    spans into space-separated morphemes with <w> word boundaries. When False
    (Danish and other non-EO), passes text through unchanged so it tokenizes
    the same way the pretrain did.
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
            elif part.strip() and morpheme_preprocess:
                processed.append(_morpheme_preprocess(part.strip()))
            else:
                processed.append(part)
        joiner = " " if morpheme_preprocess else ""
        return tokenizer(
            joiner.join(processed),
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
    parser.add_argument("--tokenized-cache", type=str, default=None,
                        help="HF Hub dataset id or local path holding an "
                             "already-tokenized+split DatasetDict "
                             "(from a prior run's save_to_disk / "
                             "push_to_hub). When set, --sft-data becomes "
                             "optional and tokenize/filter/split are skipped.")
    parser.add_argument("--sft-data", type=str, nargs="+", required=False,
                        help="Local JSONL files or HF Hub dataset names.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="tokenizer_morpheme")
    parser.add_argument("--epochs", type=float, default=3.0,
                        help="Can be fractional (e.g. 0.2 for a short anneal).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=None,
                        help="Per-row truncation AND packed-sequence length. "
                             "Default: auto-detected from the loaded model's "
                             "config.max_position_embeddings (so a 512-context "
                             "base uses 512, a 2048-context base uses 2048). "
                             "Explicit values are capped at the model's "
                             "max_position_embeddings — the trainer errors out "
                             "if you request more, since RoPE at unseen positions "
                             "produces noise.")
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
    parser.add_argument("--best-eval-ckpt", action="store_true",
                        help="If set, /final = best-eval-loss checkpoint via "
                             "HF Trainer's load_best_model_at_end. Default "
                             "OFF — /final = last checkpoint. Rationale: DA "
                             "v12 A/B showed best-eval-loss ckpt was weaker "
                             "on 4/5 downstream metrics than the last ckpt "
                             "(project_v12_best_ckpt_selection).")
    parser.add_argument(
        "--lr-scheduler", type=str, default="cosine_with_min_lr",
        choices=["cosine_with_min_lr", "constant",
                 "constant_with_warmup", "linear", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--downstream-evals", nargs="*",
                        default=["gsm8k", "sciq", "citgen", "citmc"],
                        choices=["gsm8k", "sciq", "citgen", "citmc"],
                        help="Run these downstream evals on every eval step. "
                             "Empty list disables. See "
                             "esperanto_lm.downstream_eval_callback.")
    parser.add_argument("--downstream-n", type=int, default=0,
                        help="Rows per downstream eval. 0 = full test set "
                             "(default, no sampling bias). Non-zero: random "
                             "subsample, seed rotated per eval step so bias "
                             "averages out. NEVER use fixed shuffle(seed)+"
                             "first-N — v15 investigation showed a specific "
                             "seed pinned the first 200 GSM rows to an ~8pp-"
                             "easier subset than the full set. "
                             "(project_v15_callback_subsample_bias)")
    parser.add_argument("--downstream-batch-size", type=int, default=32,
                        help="Batch size for downstream generation.")
    parser.add_argument("--top-k-downstream", type=int, default=0,
                        help="Preserve the top-K checkpoints ranked by mean "
                             "downstream accuracy (moved into best/ so HF's "
                             "save_total_limit rotation can't delete them). "
                             "0 disables. Requires --downstream-evals.")
    parser.add_argument("--wandb-project", default="jepsen/espllm")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-run-id", default=None,
                        help="Resume the given wandb run id (sets "
                             "WANDB_RESUME=allow). Use with --wandb-step-"
                             "offset to make continuation-run steps line up "
                             "past the parent run's endpoint.")
    parser.add_argument("--wandb-step-offset", type=int, default=0,
                        help="Add this to every wandb-logged step (both "
                             "HF Trainer's built-in logs and the downstream "
                             "callback). Set to sum of prior training-phase "
                             "step counts so a continuation chart is "
                             "gapless past those phases.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-optim-from", type=str, default=None,
                        help="Path to a checkpoint dir. Loads optimizer.pt "
                             "AFTER Trainer init, but does NOT touch the LR "
                             "scheduler, global step, or RNG state. Intended "
                             "for anneal-from-constant-LR runs: reuse the "
                             "well-conditioned Adam moments from the base, "
                             "start a fresh linear-decay schedule from step 0. "
                             "Mutually exclusive with --resume.")
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--attn-impl", default="auto",
                        choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--no-morpheme-preprocess", action="store_true",
                        help="Skip morpheme preprocessing (Danish and other "
                             "non-Esperanto languages). Default (off) applies "
                             "morpheme decomposition matching EO pretrain.")
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
        # Resume the parent run when asked so continuation charts are one line
        if args.wandb_run_id:
            os.environ["WANDB_RESUME"] = "allow"
            os.environ["WANDB_RUN_ID"] = args.wandb_run_id
        # Bump every wandb.log(step=...) by --wandb-step-offset so a new
        # phase (fresh HF Trainer starting at global_step=0) lands past the
        # parent run's endpoint instead of overwriting its early steps.
        if args.wandb_step_offset:
            _wandb_orig_log = wandb.log
            _step_offset = args.wandb_step_offset
            def _offset_log(data, step=None, **kw):
                if step is not None:
                    step = step + _step_offset
                return _wandb_orig_log(data, step=step, **kw)
            wandb.log = _offset_log
        wandb.init(
            entity=entity, project=project,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
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
    console.print(f"[bold]Attention impl:[/] {attn_impl}   [bold]Liger:[/] {_LIGER_ON}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, attn_implementation=attn_impl)

    # Resolve --max-length against the model's own position-embedding ceiling.
    # Autodetect when unset; error out when a user request exceeds what RoPE
    # was trained for (positions past model.config.max_position_embeddings
    # produce noise on this arch).
    model_max_pos = int(model.config.max_position_embeddings)
    if args.max_length is None:
        args.max_length = model_max_pos
        console.print(f"[bold]max_length auto-set to model's max_position_embeddings = {args.max_length}")
    elif args.max_length > model_max_pos:
        raise SystemExit(
            f"--max-length={args.max_length} exceeds the model's "
            f"max_position_embeddings={model_max_pos}. RoPE past that position "
            f"produces noise (see project_rope_extension memory). "
            f"Either drop --max-length to ≤{model_max_pos} or use a checkpoint "
            f"whose config was already RoPE-extended.")
    else:
        console.print(f"[bold]max_length={args.max_length}  (model supports up to {model_max_pos})")

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

    # Fast path: if --tokenized-cache points at an already-tokenized
    # DatasetDict (HF Hub or local), skip loading + tokenizing + filtering
    # + splitting entirely and jump straight to packing.
    splits = None
    if args.tokenized_cache:
        from datasets import load_dataset as _hf_load
        from datasets import load_from_disk as _load_from_disk
        cache_src = Path(args.tokenized_cache)
        if cache_src.exists():
            console.print(f"[bold cyan]Loading tokenized cache from disk: "
                          f"{args.tokenized_cache}")
            splits = _load_from_disk(str(cache_src))
        else:
            console.print(f"[bold cyan]Loading tokenized cache from HF: "
                          f"{args.tokenized_cache}")
            splits = _hf_load(args.tokenized_cache)
    elif not args.sft_data:
        raise SystemExit("--sft-data is required unless --tokenized-cache is set")

    if splits is None:
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
            tokenizer, special_tokens, args.max_length,
            morpheme_preprocess=not args.no_morpheme_preprocess)
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
            return {"input_ids": input_ids, "attention_mask": attention_mask,
                    "labels": labels}

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
            n_pre = len(tokenized)
            tokenized = tokenized.filter(lambda r: len(r["input_ids"]) > 0,
                                          num_proc=num_proc())
            n_dropped = n_pre - len(tokenized)
            if n_dropped:
                console.print(f"[bold yellow]Dropped {n_dropped:,} malformed "
                              f"rows (no {ASSISTANT_TOKEN})")

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
    import torch
    from datasets import Dataset
    from esperanto_lm.data import num_proc as _num_proc
    # Touch CUDA in main BEFORE any child processes are spawned. Otherwise
    # child procs get a poisoned CUDA context state that propagates back
    # to main via shared handles/env, and the trainer's later CUDA init
    # fails with 'CUDA initialization: unknown error' — model falls back
    # to CPU silently.
    if torch.cuda.is_available():
        torch.cuda.init()
        _ = torch.cuda.device_count()
    pack_workers = min(_num_proc(), max(1, os.cpu_count() - 2))
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
        load_best_model_at_end=args.best_eval_ckpt,
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
    callbacks = []
    if args.downstream_evals:
        from esperanto_lm.downstream_eval_callback import DownstreamEvalCallback
        callbacks.append(DownstreamEvalCallback(
            tokenizer=tokenizer,
            evals=args.downstream_evals,
            n_per_eval=args.downstream_n,
            batch_size=args.downstream_batch_size,
            top_k=args.top_k_downstream,
            output_dir=output_dir if args.top_k_downstream > 0 else None,
        ))
        n_label = "full" if not args.downstream_n else str(args.downstream_n)
        console.print(f"[green]Downstream evals every eval step: "
                      f"{args.downstream_evals} (n={n_label})"
                      + (f" — preserving top-{args.top_k_downstream} in best/"
                         if args.top_k_downstream > 0 else ""))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    if args.resume and args.resume_optim_from:
        raise ValueError("--resume and --resume-optim-from are mutually "
                         "exclusive; pick one.")

    if args.resume_optim_from:
        # HF's optimizer is created lazily on first train step. Force create
        # here so we can load state into it before .train() begins.
        trainer.create_optimizer()
        optim_path = Path(args.resume_optim_from) / "optimizer.pt"
        if not optim_path.is_file():
            raise FileNotFoundError(
                f"--resume-optim-from: no optimizer.pt at {optim_path}")
        console.print(f"[bold cyan]Loading optimizer state from {optim_path}"
                      f" (scheduler/step/RNG stay fresh)")
        # weights_only=False needed for adamw_torch_fused's pickled state
        state = torch.load(str(optim_path), map_location="cpu",
                           weights_only=False)
        trainer.optimizer.load_state_dict(state)

    trainer.train(resume_from_checkpoint=args.resume or None)

    final = Path(output_dir) / "final"
    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    console.print(f"[bold green]Saved final model to {final}")


if __name__ == "__main__":
    main()
