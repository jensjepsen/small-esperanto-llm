"""GRPO on Danish IF or GSM8K using verifier rewards (TRL GRPOTrainer).

Two supported modes:
  --task gsm8k     reward = numeric-answer match on danish-gsm8k
  --task ifeval    reward = mean constraint-pass on danish-instruction-following-v4

Usage:
  uv run python scripts/train_grpo_verifier.py \\
      --task gsm8k --checkpoint jensjepsen/danish-lm-400m-sft-v32-avg-top3 \\
      --output-dir /workspace/runs/grpo/gsm8k \\
      --epochs 1 --batch-size 4 --grad-accum 8 \\
      --num-generations 4 --max-prompt-length 512 --max-completion-length 256 \\
      --learning-rate 1e-6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from esperanto_lm.rl_rewards import reward_gsm8k, reward_ifeval


# transformers 4.55+ calls _get_train_sampler(dataset); TRL <=1.10 still
# overrides without the arg. Shim: accept & ignore, use the internal path.
_orig_sampler = GRPOTrainer._get_train_sampler


def _patched_sampler(self, *args, **kwargs):
    return _orig_sampler(self)


GRPOTrainer._get_train_sampler = _patched_sampler

USER = "<|user|>"
ASST = "<|assistant|>"
END = "<|end|>"


def build_gsm8k_dataset(split="train", max_rows: int = 0):
    ds = load_dataset("jensjepsen/danish-gsm8k", "sft", split=split)
    if max_rows and len(ds) > max_rows:
        ds = ds.select(range(max_rows))
    rows = []
    for r in ds:
        u = r["messages"][0]["content"]
        a = r["messages"][1]["content"]
        rows.append({"prompt": f"{USER} {u} {ASST}", "gold": a})
    return Dataset.from_list(rows)


def build_ifeval_dataset(split="train", max_rows: int = 0):
    ds = load_dataset("jensjepsen/danish-instruction-following-v4",
                      "default", split=split)
    rows = []
    for i, r in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        u = r["messages"][0]["content"]
        rows.append({
            "prompt": f"{USER}{u}{END}{ASST}",
            "constraints": r["constraints"],
            "params": r["params"],
        })
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gsm8k", "ifeval"], required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="HF repo or local path — SFT starting model")
    ap.add_argument("--tokenizer", default=None,
                    help="Defaults to --checkpoint")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="per-device batch of PROMPTS "
                         "(each generates --num-generations completions)")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--num-generations", type=int, default=4,
                    help="rollouts per prompt for group advantage")
    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.04,
                    help="KL coefficient vs reference policy")
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=0,
                    help="If >0, eval on test split every N steps.")
    ap.add_argument("--eval-max-rows", type=int, default=200,
                    help="Cap test rows for periodic eval.")
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--wandb-project", default="danish-lm-grpo")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="Cap training rows (0=all). Handy for smoke tests.")
    args = ap.parse_args()

    tok_path = args.tokenizer or args.checkpoint
    print(f"loading tokenizer {tok_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(tok_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"building dataset for task={args.task}...", flush=True)
    eval_ds = None
    if args.task == "gsm8k":
        ds = build_gsm8k_dataset("train", max_rows=args.max_rows or 0)
        reward_fn = reward_gsm8k
        if args.eval_steps > 0:
            eval_ds = build_gsm8k_dataset("test", max_rows=args.eval_max_rows)
    else:
        ds = build_ifeval_dataset("train", max_rows=args.max_rows or 0)
        reward_fn = reward_ifeval
        if args.eval_steps > 0:
            eval_ds = build_ifeval_dataset("eval", max_rows=args.eval_max_rows)
    if args.max_rows and len(ds) > args.max_rows:
        ds = ds.select(range(args.max_rows))
    print(f"  train {len(ds)} rows"
          + (f", eval {len(eval_ds)} rows" if eval_ds is not None else ""),
          flush=True)

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        lr_scheduler_type="constant_with_warmup",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps or None,
        per_device_eval_batch_size=args.batch_size,
        save_total_limit=2,
        report_to=["wandb"],
        run_name=args.wandb_run_name or f"grpo_{args.task}",
        bf16=True,
        optim="adamw_bnb_8bit",
        remove_unused_columns=False,
    )

    print(f"loading model {args.checkpoint}...", flush=True)
    trainer = GRPOTrainer(
        model=args.checkpoint,
        processing_class=tok,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=ds,
        eval_dataset=eval_ds,
    )
    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")


if __name__ == "__main__":
    import os
    os.environ.setdefault("WANDB_PROJECT", "danish-lm-grpo")
    main()
