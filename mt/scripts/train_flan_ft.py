"""Fine-tune google/flan-t5-small on our EN<->EO parallel corpus.

Uses flan-t5-small's own SentencePiece tokenizer (32,128 vocab) — DO NOT
substitute our spm_eneo_48k_v3, since the pretrained embeddings/output
head are tied to the T5 tokenizer's ids.

Data format: task-prefix + source, target is plain text.
  EN->EO:  "translate English to Esperanto: <en>" -> "<eo>"
  EO->EN:  "translate Esperanto to English: <eo>" -> "<en>"

Bidirectional training: each parallel pair is used twice per epoch (once
in each direction), giving both directions equal budget. This differs
from the Marian bidir path where each pair fires ONE direction per epoch
(50/50 random).

Val sets and eval flow reuse the ones from train.py: FLORES devtest,
opus100_validation, MMLU stem/full, SciQ, math_wp.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import sacrebleu
import torch
import yaml
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]


EN2EO_TAG = "translate English to Esperanto: "
EO2EN_TAG = "translate Esperanto to English: "


def load_parallel_ds(spec: str) -> Dataset:
    """Return a Dataset of `{en, eo}` rows from local JSONL or hf://repo[::config][/split].

    Arrow-native: never materializes into a Python list.
    """
    if spec.startswith("hf://"):
        tail = spec[len("hf://"):]
        config = None
        if "::" in tail:
            head, rest = tail.split("::", 1)
            parts = rest.split("/")
            config = parts[0]
            split = parts[1] if len(parts) > 1 else "train"
            repo = head
        else:
            parts = tail.split("/")
            if len(parts) == 2:
                repo, split = "/".join(parts), "train"
            elif len(parts) == 3:
                repo, split = "/".join(parts[:2]), parts[2]
            else:
                raise ValueError(f"bad hf:// path: {spec}")
        ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    else:
        ds = load_dataset("json", data_files=str(spec), split="train")
    # keep only en/eo columns; drop empties
    keep = [c for c in ("en", "eo") if c in ds.column_names]
    if len(keep) != 2:
        raise ValueError(f"{spec}: missing en/eo columns; got {ds.column_names}")
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


def to_bidir(batch, tag_en2eo=EN2EO_TAG, tag_eo2en=EO2EN_TAG):
    """Batched map function that expands each row into 2 directed rows."""
    ens = batch["en"]
    eos = batch["eo"]
    src, tgt = [], []
    for e, o in zip(ens, eos):
        src.append(tag_en2eo + e); tgt.append(o)
        src.append(tag_eo2en + o); tgt.append(e)
    return {"source": src, "target": tgt}


def to_direction(batch, direction: str):
    tag = EN2EO_TAG if direction == "en2eo" else EO2EN_TAG
    if direction == "en2eo":
        src = [tag + e for e in batch["en"]]
        tgt = list(batch["eo"])
    else:
        src = [tag + o for o in batch["eo"]]
        tgt = list(batch["en"])
    return {"source": src, "target": tgt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/flan-t5-small",
                    help="HF model id to fine-tune from")
    ap.add_argument("--train-files", nargs="+", default=[
        "hf://jensjepsen/esperanto-mt-parallel-v13",
        "hf://jensjepsen/esperanto-mt-math-parallel-v2::sentences",
        "hf://jensjepsen/esperanto-mt-math-parallel-v2::rows",
        "hf://jensjepsen/esperanto-mt-yago-parallel-v2::labels",
        "hf://jensjepsen/esperanto-mt-yago-parallel-v2::comments",
        "hf://jensjepsen/esperanto-orca-math-gemini-parallel",
    ])
    ap.add_argument("--val-files", nargs="+", type=Path, default=[
        Path("mt/data/parallel/flores_devtest.jsonl"),
        Path("mt/data/parallel/opus100_validation.jsonl"),
        Path("mt/data/parallel/eval_mmlu_stem.jsonl"),
        Path("mt/data/parallel/eval_mmlu_full.jsonl"),
        Path("mt/data/parallel/eval_sciq.jsonl"),
        Path("mt/data/parallel/eval_math_wp.jsonl"),
    ])
    ap.add_argument("--val-names", nargs="+", default=[
        "flores_devtest", "opus100_validation",
        "eval_mmlu_stem", "eval_mmlu_full", "eval_sciq", "eval_math_wp",
    ])
    ap.add_argument("--val-direction", default="en2eo", choices=["en2eo", "eo2en"])
    ap.add_argument("--val-cap-per-set", type=int, default=1500)
    ap.add_argument("--direction", default="bidir", choices=["bidir", "en2eo", "eo2en"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-source-length", type=int, default=192)
    ap.add_argument("--max-target-length", type=int, default=192)
    ap.add_argument("--per-device-train-batch-size", type=int, default=64)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--eval-fraction", type=float, default=0.25,
                    help="Eval + save every N epochs")
    ap.add_argument("--logging-steps", type=int, default=100)
    ap.add_argument("--num-beams", type=int, default=1,
                    help="Beam width for eval-time generation (1 = greedy)")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--wandb-project", default="espllm-mt")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb-tags", nargs="*", default=[])
    args = ap.parse_args()

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    print(f"[model] loading {args.model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = T5ForConditionalGeneration.from_pretrained(args.model_id)
    print(f"[model] vocab={tok.vocab_size}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M",
          flush=True)

    print("[data] loading train sources (arrow-native)...", flush=True)
    train_shards = []
    for spec in args.train_files:
        ds = load_parallel_ds(spec)
        print(f"  {spec}: {len(ds):,}", flush=True)
        train_shards.append(ds)
    train_pairs_ds = concatenate_datasets(train_shards)
    print(f"[data] merged pair pool: {len(train_pairs_ds):,}", flush=True)

    if args.direction == "bidir":
        train_ds = train_pairs_ds.map(
            to_bidir, batched=True, batch_size=1000,
            remove_columns=["en", "eo"],
            num_proc=8, desc="expand bidir",
        )
    else:
        d = args.direction
        train_ds = train_pairs_ds.map(
            lambda batch: to_direction(batch, d), batched=True, batch_size=1000,
            remove_columns=["en", "eo"],
            num_proc=8, desc=f"format {d}",
        )
    print(f"[data] train examples: {len(train_ds):,}  (direction={args.direction})", flush=True)

    val_sets = {}
    val_dir = args.val_direction
    for name, path in zip(args.val_names, args.val_files):
        ds = load_parallel_ds(str(path))
        if args.val_cap_per_set and len(ds) > args.val_cap_per_set:
            ds = ds.select(range(args.val_cap_per_set))
        ds = ds.map(
            lambda batch: to_direction(batch, val_dir), batched=True, batch_size=500,
            remove_columns=["en", "eo"],
            desc=f"format val[{name}]",
        )
        val_sets[name] = ds
        print(f"  val[{name}]: {len(ds):,}  direction={args.val_direction}", flush=True)

    def tokenize(batch):
        model_inputs = tok(
            batch["source"],
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tok(
            text_target=batch["target"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(
        tokenize, batched=True, remove_columns=["source", "target"],
        desc="tok train", num_proc=8,
    )
    val_sets = {
        name: ds.map(tokenize, batched=True, remove_columns=["source", "target"],
                     desc=f"tok val[{name}]", num_proc=4)
        for name, ds in val_sets.items()
    }

    collator = DataCollatorForSeq2Seq(
        tok, model=model, padding="longest",
        label_pad_token_id=-100,
    )

    steps_per_epoch = max(1, len(train_ds) // (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    eval_steps = max(1, int(steps_per_epoch * args.eval_fraction))
    total_steps = steps_per_epoch * args.epochs
    print(f"[schedule] steps/epoch={steps_per_epoch:,}  eval every {eval_steps:,}  total={total_steps:,}",
          flush=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="flores_devtest_loss",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        report_to=["wandb"] if os.environ.get("WANDB_PROJECT") else [],
        run_name=args.run_name,
        bf16=args.bf16,
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        generation_max_length=args.max_target_length,
        remove_unused_columns=True,
        seed=1337,
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        # replace -100 with pad for decoding
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels]).score
        chrfpp = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2).score
        return {"bleu": bleu, "chrf": chrf, "chrfpp": chrfpp}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_sets,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("[train] starting...", flush=True)
    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
