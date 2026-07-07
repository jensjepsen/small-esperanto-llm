"""Train a small MarianMT encoder-decoder for en↔eo translation.

Defaults match the 60M scaffold (6+6 layers, d=512, ff=2048). Uses HF
Seq2SeqTrainer for generation-aware eval and sacrebleu metric.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent))
from dataset import ParallelDataset, Seq2SeqCollator
from model import build_model, param_summary
from sp_tokenizer import SPMTokenizer

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_size_config(name: str) -> dict:
    """Load mt/configs/{name}.yaml. Used to override --d-model etc. defaults."""
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"unknown --config {name}; expected one of "
            f"{sorted(p.stem for p in CONFIGS_DIR.glob('*.yaml'))}")
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    # Stage 1: peek at --config so config-derived defaults are in effect.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="tiny")
    pre_args, _ = pre.parse_known_args()
    cfg = load_size_config(pre_args.config)
    m, t = cfg["model"], cfg["training"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="tiny",
                    help="Size preset from mt/configs/{name}.yaml (tiny|small|"
                         "medium|large). Sets model arch + training defaults. "
                         "CLI flags override config values.")
    ap.add_argument("--tokenizer", type=str, default="mt/data/tokenizer/spm_eneo_32k.model")
    ap.add_argument("--train-files", nargs="+", type=str, default=[
        "hf://jensjepsen/esperanto-mt-parallel",
        # Math-domain parallel (closes v6's "left → maldekstre" gap on math CoT).
        # Sentences slice = deduped step-level chain lines + short question sentences.
        # Rows slice = full multi-sentence Q+A pairs (turns off if disk-thin).
        "hf://jensjepsen/esperanto-mt-math-parallel::sentences",
        "hf://jensjepsen/esperanto-mt-math-parallel::rows",
        # YAGO-derived named-entity label pairs + short descriptions
        # (closes v6's "Horace → Horacio" proper-noun bug).
        "hf://jensjepsen/esperanto-mt-yago-parallel::labels",
        "hf://jensjepsen/esperanto-mt-yago-parallel::comments",
        # OPUS bundle: QED (educational subtitles), GlobalVoices (news),
        # KDE4/GNOME/Ubuntu (software localization). ~159k pairs across 5 registers.
        "hf://jensjepsen/esperanto-mt-opus-parallel",
    ], help="Local JSONL paths or hf://repo[::config][/split] "
            "(default: merged v6 corpus + math + YAGO + OPUS supplements)")
    ap.add_argument("--val-files", nargs="+", type=Path, default=[
        Path("mt/data/parallel/flores_devtest.jsonl"),
        Path("mt/data/parallel/opus100_validation.jsonl"),
        Path("mt/data/parallel/eval_mmlu_stem.jsonl"),
        Path("mt/data/parallel/eval_sciq.jsonl"),
        # Held-out math CoT eval (wp_compose seed=99999, ~1k rows).
        # Signals whether math-domain BLEU is moving during v8-mt training.
        Path("mt/data/parallel/eval_math_wp.jsonl"),
    ])
    ap.add_argument("--val-names", nargs="*", type=str, default=None,
                    help="Names for each val file (defaults to file stem). When >1 val file is "
                         "given, eval metrics are prefixed eval_<name>_bleu/chrf.")
    ap.add_argument("--metric-for-best-model", type=str,
                    default="flores_devtest_bleu",
                    help="Which metric to track for best-model. Default is "
                         "flores_devtest_bleu (industry-standard OOD bench). "
                         "Other options: opus100_validation_bleu, "
                         "eval_mmlu_stem_bleu, eval_sciq_bleu.")
    ap.add_argument("--direction", default="en2eo", choices=["en2eo", "eo2en", "bidir"],
                    help="Training direction. 'bidir' randomizes per-pair so both directions "
                         "share encoder/decoder learning. Eval direction stays fixed via --val-direction.")
    ap.add_argument("--val-direction", default="en2eo", choices=["en2eo", "eo2en"],
                    help="Direction used for all val sets (kept fixed for stable metric tracking).")
    ap.add_argument("--output-dir", type=str, default="/mnt/data/espllm/runs/mt/eneo_v1")
    ap.add_argument("--init-from", type=str, default=None,
                    help="Load weights from this checkpoint (fresh optimizer). For phase-2 fine-tunes.")
    ap.add_argument("--resume-from-checkpoint", type=str, default=None,
                    help="Resume training from a checkpoint dir (full optimizer/scheduler/rng state).")

    ap.add_argument("--d-model", type=int, default=m["d_model"])
    ap.add_argument("--encoder-layers", type=int, default=m["encoder_layers"])
    ap.add_argument("--decoder-layers", type=int, default=m["decoder_layers"])
    ap.add_argument("--heads", type=int, default=m["heads"])
    ap.add_argument("--ffn-dim", type=int, default=m["ffn_dim"])
    ap.add_argument("--max-position-embeddings", type=int,
                    default=m["max_position_embeddings"])

    ap.add_argument("--max-src-len", type=int, default=128)
    ap.add_argument("--max-tgt-len", type=int, default=128)

    ap.add_argument("--epochs", type=int, default=t["epochs"])
    ap.add_argument("--batch-size", type=int, default=t["batch_size"])
    ap.add_argument("--gradient-accumulation", type=int,
                    default=t["gradient_accumulation"])
    ap.add_argument("--learning-rate", type=float, default=t["learning_rate"])
    ap.add_argument("--warmup-steps", type=int, default=t["warmup_steps"])
    ap.add_argument("--weight-decay", type=float, default=t["weight_decay"])
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="Gradient clipping norm. Default 1.0 (HF Trainer "
                         "default). Lower (0.5) for tighter clipping when "
                         "training a larger/deeper model in bf16.")
    ap.add_argument("--skip-nan-inf", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Zero out gradients on any step where loss or grad is "
                         "NaN/Inf, so a single bad batch doesn't corrupt Adam "
                         "moments. Default on.")
    ap.add_argument("--label-smoothing", type=float,
                    default=t["label_smoothing"])
    ap.add_argument("--dropout", type=float, default=t["dropout"])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--eval-steps", type=int, default=None,
                    help="Explicit eval interval in steps. When unset, "
                         "derived from --eval-fraction-of-epoch.")
    ap.add_argument("--save-steps", type=int, default=None,
                    help="Explicit save interval in steps. When unset, "
                         "derived from --save-fraction-of-epoch.")
    ap.add_argument("--eval-fraction-of-epoch", type=float, default=0.25,
                    help="Fraction of an epoch between evals (default 0.25 → "
                         "4 evals per epoch). Ignored if --eval-steps is set.")
    ap.add_argument("--save-fraction-of-epoch", type=float, default=0.25,
                    help="Fraction of an epoch between checkpoints "
                         "(default 0.25). Ignored if --save-steps is set.")
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--eval-max-samples", type=int, default=500,
                    help="Cap eval set size for fast in-loop sacrebleu")
    ap.add_argument("--predict-with-generate", action="store_true", default=True)

    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False,
                    help="Use FP16 mixed precision (default off; prefer bf16 on Ampere+)")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                    help="Use BF16 mixed precision (default on; requires Ampere or newer)")
    ap.add_argument("--wandb-tags", nargs="*", default=[])
    ap.add_argument("--wandb-project", default="espllm-mt",
                    help="W&B project name (also sets WANDB_PROJECT env)")
    ap.add_argument("--run-name", default=None,
                    help="W&B run_name; defaults to basename of --output-dir")
    return ap.parse_args()


def make_compute_metrics(tokenizer: SPMTokenizer):
    import sacrebleu

    def _decode(seqs):
        out = []
        for s in seqs:
            s = [int(t) for t in s if int(t) != -100]
            out.append(tokenizer.decode(s))
        return out

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        # replace -100 in labels for decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_id)
        decoded_preds = _decode(preds)
        decoded_labels = _decode(labels)
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])
        chrfpp = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2)
        return {"bleu": bleu.score, "chrf": chrf.score, "chrfpp": chrfpp.score}

    return compute_metrics


def main():
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/mnt/data/hf_cache")
    # Wire wandb env BEFORE TrainingArguments constructs — HF reads these once.
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_tags:
        os.environ["WANDB_TAGS"] = ",".join(args.wandb_tags)
    torch.manual_seed(args.seed)

    tok = SPMTokenizer(args.tokenizer)
    print(f"Tokenizer: vocab={tok.vocab_size}  pad={tok.pad_id} eos={tok.eos_id}")

    if args.init_from:
        from transformers import MarianMTModel
        print(f"Init from {args.init_from} (fresh optimizer)")
        model = MarianMTModel.from_pretrained(args.init_from)
    else:
        model = build_model(
            vocab_size=tok.vocab_size,
            d_model=args.d_model,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers,
            encoder_attention_heads=args.heads,
            decoder_attention_heads=args.heads,
            encoder_ffn_dim=args.ffn_dim,
            decoder_ffn_dim=args.ffn_dim,
            max_position_embeddings=args.max_position_embeddings,
            dropout=args.dropout,
            pad_token_id=tok.pad_id,
            bos_token_id=tok.bos_id,
            eos_token_id=tok.eos_id,
            decoder_start_token_id=tok.pad_id,
        )
    # Decode-time hardening — forbid repeated 5-grams to stop degeneration loops
    # while still allowing legitimate structural repetition (enumerations like
    # "the first class has X. the second class has Y."). Was 3; bumped to 5
    # after a roundtrip test showed 3 truncated valid repeating-structure output.
    model.generation_config.no_repeat_ngram_size = 5
    param_summary(model)

    train_ds = ParallelDataset(args.train_files, direction=args.direction)

    val_names = args.val_names or [p.stem for p in args.val_files]
    assert len(val_names) == len(args.val_files), \
        f"--val-names count ({len(val_names)}) must match --val-files count ({len(args.val_files)})"
    val_sets = {}
    for name, p in zip(val_names, args.val_files):
        ds = ParallelDataset([p], direction=args.val_direction)
        if args.eval_max_samples and len(ds) > args.eval_max_samples:
            ds.pairs = ds.pairs[: args.eval_max_samples]
        val_sets[name] = ds
        print(f"Val[{name}]: {len(ds)}  ({p})  direction={args.val_direction}")
    # If only one val set, pass it directly (HF prefixes metrics with eval_); if many, pass dict.
    eval_arg = val_sets[val_names[0]] if len(val_sets) == 1 else val_sets
    print(f"Train: {len(train_ds)}  direction={args.direction}")

    collator = Seq2SeqCollator(
        tok,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    compute_metrics = make_compute_metrics(tok)

    # Derive eval/save intervals from fraction-of-epoch when not explicitly set.
    # steps_per_epoch = train_len / (batch_size * grad_accum). HF trainer counts
    # optimizer steps, so this matches TrainingArguments.
    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.gradient_accumulation))
    eval_steps = args.eval_steps or max(1, int(steps_per_epoch * args.eval_fraction_of_epoch))
    save_steps = args.save_steps or max(1, int(steps_per_epoch * args.save_fraction_of_epoch))
    # HF requires save_steps % eval_steps == 0 when load_best_model_at_end=True.
    # Round save_steps to nearest multiple of eval_steps (>= eval_steps).
    if save_steps % eval_steps != 0:
        save_steps = max(eval_steps, ((save_steps + eval_steps - 1) // eval_steps) * eval_steps)
    print(f"Schedule: steps_per_epoch={steps_per_epoch:,}  "
          f"eval every {eval_steps} steps ({eval_steps/steps_per_epoch:.2f} ep)  "
          f"save every {save_steps} steps ({save_steps/steps_per_epoch:.2f} ep)", flush=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(8, args.batch_size // 2),
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        label_smoothing_factor=args.label_smoothing,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.max_tgt_len,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"],
        run_name=args.run_name or Path(args.output_dir).name,
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    if args.skip_nan_inf:
        class NaNSkippingSeq2SeqTrainer(Seq2SeqTrainer):
            """Zero out gradients on any step with NaN/Inf loss, so a single
            pathological batch doesn't corrupt Adam moments. Returns zero loss
            so the running average stays clean."""
            _nan_count = 0

            def training_step(self, model, inputs, num_items_in_batch=None):
                loss = super().training_step(model, inputs, num_items_in_batch)
                if not torch.isfinite(loss).all():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
                    self.__class__._nan_count += 1
                    print(f"[NaN-skip] step {self.state.global_step}: zeroing "
                          f"grads on non-finite loss (total skipped: "
                          f"{self.__class__._nan_count})", flush=True)
                    return torch.zeros_like(loss)
                return loss

        TrainerCls = NaNSkippingSeq2SeqTrainer
    else:
        TrainerCls = Seq2SeqTrainer

    trainer = TrainerCls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_arg,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(f"{args.output_dir}/final")
    print(f"Saved final model to {args.output_dir}/final")


if __name__ == "__main__":
    main()
