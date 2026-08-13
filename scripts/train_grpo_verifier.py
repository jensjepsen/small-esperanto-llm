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

import re
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from esperanto_lm.rl_rewards import (
    reward_gsm8k, reward_ifeval, _extract_num, _norm_num,
)

_NUM_RE_INT = re.compile(r"-?\d[\d,]*\.?\d*")


# transformers 4.55+ calls _get_train_sampler(dataset); TRL <=1.10 still
# overrides without the arg. Shim: accept & ignore, use the internal path.
_orig_sampler = GRPOTrainer._get_train_sampler


def _patched_sampler(self, *args, **kwargs):
    return _orig_sampler(self)


GRPOTrainer._get_train_sampler = _patched_sampler


class GreedyEvalCallback(TrainerCallback):
    """Every N steps, run greedy pass@1 on a fixed test-set subset and log.
    Reports `eval_greedy_pass@1` — apples-to-apples with SFT downstream eval,
    unlike the TRL built-in sampled eval_reward."""

    def __init__(self, tokenizer, items, task,
                 every_n_steps: int, max_new_tokens: int = 256,
                 batch_size: int = 16):
        """items schema:
             gsm8k: list of (prompt, gold_answer_string)
             ifeval: list of (prompt, constraints_list, params_json_string)"""
        self.tok = tokenizer
        self.items = items
        self.task = task
        self.every = every_n_steps
        self.max_new = max_new_tokens
        self.bs = batch_size
        self._pending = None  # metric to inject on next on_log

    def on_log(self, args, state, control, logs=None, **kw):
        if self._pending is not None and logs is not None:
            logs.update(self._pending)
            self._pending = None

    def on_step_end(self, args, state, control, model=None, **kw):
        if self.every <= 0 or state.global_step % self.every != 0 or state.global_step == 0:
            return
        if model is None:
            return
        model.eval()
        tok = self.tok
        prev_side = tok.padding_side
        prev_pad = tok.pad_token
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        eos_ids = [tok.eos_token_id]
        end_id = tok.convert_tokens_to_ids("<|end|>")
        if end_id is not None and end_id != tok.unk_token_id:
            eos_ids.append(end_id)
        outs = []
        try:
            for i in range(0, len(self.items), self.bs):
                batch = [row[0] for row in self.items[i:i + self.bs]]
                enc = tok(batch, return_tensors="pt", padding=True,
                          add_special_tokens=False,
                          return_token_type_ids=False).to(model.device)
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        max_new_tokens=self.max_new,
                        do_sample=False, num_beams=1,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=eos_ids, repetition_penalty=1.1)
                plen = enc["input_ids"].shape[1]
                for row in gen:
                    outs.append(tok.decode(row[plen:], skip_special_tokens=True).strip())
                done = min(i + self.bs, len(self.items))
                # running metric
                if self.task == "gsm8k":
                    n_ok = sum(1 for o, (_, g) in zip(outs, self.items[:done])
                               if _norm_num(_extract_num(o)) is not None
                               and _norm_num(_extract_num(o)) == _norm_num(g if g and _NUM_RE_INT.fullmatch(g.strip()) else _extract_num(g)))
                    print(f"  [greedy-eval] {done}/{len(self.items)} acc={n_ok/done:.4f}", flush=True)
                else:  # ifeval
                    scores_so_far = reward_ifeval(
                        outs, [row[1] for row in self.items[:done]],
                        [row[2] for row in self.items[:done]])
                    print(f"  [greedy-eval] {done}/{len(self.items)} mean_pass={sum(scores_so_far)/len(scores_so_far):.4f}", flush=True)
        finally:
            tok.padding_side = prev_side
            tok.pad_token = prev_pad
        # Final metric
        if self.task == "gsm8k":
            n_ok = 0
            for o, (_, gold) in zip(outs, self.items):
                pred = _norm_num(_extract_num(o))
                target = _norm_num(gold if gold and _NUM_RE_INT.fullmatch((gold or "").strip())
                                   else _extract_num(gold))
                if pred is not None and pred == target:
                    n_ok += 1
            acc = n_ok / max(1, len(self.items))
        else:  # ifeval
            scores = reward_ifeval(outs,
                                   [row[1] for row in self.items],
                                   [row[2] for row in self.items])
            acc = sum(scores) / max(1, len(scores))
        print(f"  [greedy-eval] step={state.global_step} {self.task}={100*acc:.2f}%",
              flush=True)
        key = ("eval_greedy_gsm8k_pass@1" if self.task == "gsm8k"
               else "eval_greedy_ifeval_mean_pass")
        self._pending = {key: acc}
        model.train()
        control.should_log = True  # trigger on_log so _pending gets flushed

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
    ap.add_argument("--greedy-eval-steps", type=int, default=0,
                    help="If >0, run greedy pass@1 (deterministic, single-"
                         "sample) on test rows every N steps. Logged as "
                         "eval_greedy_pass@1. Apples-to-apples with SFT "
                         "downstream eval, unlike TRL's sampled eval_reward.")
    ap.add_argument("--greedy-eval-max-rows", type=int, default=200)
    ap.add_argument("--skip-zero-adv", action="store_true",
                    help="Zero out completion_mask for groups where all "
                         "rollouts scored the same reward (std==0 → "
                         "advantage==0). Excludes them cleanly from the "
                         "loss + KL + optimizer noise. Doesn't save fwd/bwd "
                         "compute (TRL still generates them) but avoids the "
                         "noise-only Adam step. Ported from train_grpo.py.")
    ap.add_argument("--use-vllm-server", action="store_true",
                    help="Use a separate vLLM server for rollouts (huge "
                         "speedup — ~10-20× rollout throughput). Requires "
                         "a running `trl vllm-serve` on a second GPU. See "
                         "scripts/launch_grpo_vllm.sh for the 2-GPU launcher.")
    ap.add_argument("--vllm-host", default="localhost")
    ap.add_argument("--vllm-port", type=int, default=8000)
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.55,
                    help="GPU mem fraction for vLLM server (only used if "
                         "trainer + server share a GPU; irrelevant with 2 GPUs).")
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
        use_vllm=args.use_vllm_server,
        vllm_server_host=args.vllm_host,
        vllm_server_port=args.vllm_port,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )

    print(f"loading model {args.checkpoint}...", flush=True)
    if args.skip_zero_adv:
        # Zero-out completion_mask for groups where reward.std()==0 (all
        # rollouts scored the same → advantage=0 → no gradient signal).
        # Excludes them from loss + KL + Adam. Doesn't save fwd/bwd compute
        # (TRL still generates them) but avoids the noise-only optimizer step.
        # Ported from scripts/train_grpo.py (Esperanto verifier trainer).
        import torch as _t
        _orig_prepare = GRPOTrainer._prepare_inputs

        def _prepare_with_skip(self, inputs):
            result = _orig_prepare(self, inputs)
            adv = result.get("advantages")
            cm = result.get("completion_mask")
            mode = "eval" if self.control.should_evaluate else "train"
            self._metrics.setdefault(mode, {}).setdefault("skip_frac", [])
            if adv is None or cm is None:
                self._metrics[mode]["skip_frac"].append(0.0)
                return result
            n_gen = self.num_generations
            n_local = adv.shape[0]
            if n_local < n_gen:
                self._metrics[mode]["skip_frac"].append(0.0)
                return result
            n_groups = n_local // n_gen
            adv_g = adv[:n_groups * n_gen].view(n_groups, n_gen)
            active = (adv_g.abs() > 1e-6).any(dim=1)
            frac_skipped = 1.0 - active.float().mean().item()
            self._metrics[mode]["skip_frac"].append(frac_skipped)
            if bool(active.all()) or not bool(active.any()):
                return result
            sample_mask = active.repeat_interleave(n_gen).to(adv.device)
            if sample_mask.numel() < n_local:
                pad = _t.zeros(n_local - sample_mask.numel(),
                               dtype=_t.bool, device=adv.device)
                sample_mask = _t.cat([sample_mask, pad])
            result["completion_mask"] = cm * sample_mask.to(cm.dtype).unsqueeze(1)
            return result

        GRPOTrainer._prepare_inputs = _prepare_with_skip
        print("skip-zero-adv: enabled (masking completion_mask on zero-std groups)", flush=True)

    trainer = GRPOTrainer(
        model=args.checkpoint,
        processing_class=tok,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=ds,
        eval_dataset=eval_ds,
    )
    if args.greedy_eval_steps > 0:
        if args.task == "gsm8k":
            gds = build_gsm8k_dataset("test", max_rows=args.greedy_eval_max_rows)
            g_items = [(r["prompt"], r["gold"]) for r in gds]
        else:  # ifeval
            gds = build_ifeval_dataset("eval", max_rows=args.greedy_eval_max_rows)
            g_items = [(r["prompt"], r["constraints"], r["params"]) for r in gds]
        trainer.add_callback(GreedyEvalCallback(
            tokenizer=tok, items=g_items, task=args.task,
            every_n_steps=args.greedy_eval_steps, max_new_tokens=256,
            batch_size=args.batch_size,
        ))
    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")


if __name__ == "__main__":
    import os
    os.environ.setdefault("WANDB_PROJECT", "danish-lm-grpo")
    main()
