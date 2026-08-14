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
    reward_gsm8k, reward_ifeval, reward_ifeval_combined, reward_mixed,
    _extract_num, _norm_num,
)

_NUM_RE_INT = re.compile(r"-?\d[\d,]*\.?\d*")


# transformers 4.55+ calls _get_train_sampler(dataset); TRL <=1.10 still
# overrides without the arg. Shim: accept & ignore, use the internal path.
_orig_sampler = GRPOTrainer._get_train_sampler


def _patched_sampler(self, *args, **kwargs):
    return _orig_sampler(self)


GRPOTrainer._get_train_sampler = _patched_sampler


class IFEvalDACallback(TrainerCallback):
    """Step-triggered greedy eval on the full ifeval-da benchmark (541 rows).
    Reports the standard 4-way IFEval metric split: prompt-strict, prompt-loose,
    inst-strict, inst-loose. Fires every `every_n_steps`."""

    def __init__(self, tokenizer, every_n_steps: int = 125,
                 max_new_tokens: int = 512, batch_size: int = 16):
        import time as _time
        self.tok = tokenizer
        self.every = int(every_n_steps)
        self.max_new = max_new_tokens
        self.bs = batch_size
        self._time = _time
        self._pending = None
        self._trainer = None  # set by main() after trainer is built
        # Precompute prompts + instruction objects once (541 rows).
        from eval_ifeval_da import build_instructions as _bi  # noqa: E402
        ds = load_dataset("danish-foundation-models/ifeval-da", split="train")
        self._prompts = [f"{USER}{r['prompt']}{END}{ASST}" for r in ds]
        self._insts = [_bi(r) for r in ds]

    def on_log(self, args, state, control, logs=None, **kw):
        if self._pending is not None and logs is not None:
            logs.update(self._pending)
            self._pending = None

    def on_step_end(self, args, state, control, model=None, **kw):
        if model is None or self.every <= 0:
            return
        if state.global_step == 0 or state.global_step % self.every != 0:
            return
        from eval_ifeval_da import score_row as _score  # noqa: E402
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
        t0 = self._time.time()
        responses: list[str] = []
        try:
            for i in range(0, len(self._prompts), self.bs):
                batch = self._prompts[i:i + self.bs]
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
                    responses.append(tok.decode(row[plen:], skip_special_tokens=True).strip())
                done = len(responses)
                if done % 128 == 0 or done == len(self._prompts):
                    print(f"  [ifeval-da] step={state.global_step} gen {done}/{len(self._prompts)}",
                          flush=True)
        finally:
            tok.padding_side = prev_side
            tok.pad_token = prev_pad
        # Score
        p_strict = p_loose = 0
        i_strict = i_loose = 0
        i_tot = 0
        rows_scored = 0
        for resp, insts in zip(responses, self._insts):
            if not insts:
                continue
            s, l = _score(resp, insts)
            i_tot += len(insts)
            i_strict += sum(s); i_loose += sum(l)
            if all(s): p_strict += 1
            if all(l): p_loose += 1
            rows_scored += 1
        if rows_scored == 0 or i_tot == 0:
            model.train()
            return
        m = {
            "eval_ifeval_da_prompt_strict": p_strict / rows_scored,
            "eval_ifeval_da_prompt_loose": p_loose / rows_scored,
            "eval_ifeval_da_inst_strict": i_strict / i_tot,
            "eval_ifeval_da_inst_loose": i_loose / i_tot,
        }
        dt = self._time.time() - t0
        print(f"  [ifeval-da] step={state.global_step}  "
              f"prompt-strict={100*m['eval_ifeval_da_prompt_strict']:.2f}%  "
              f"prompt-loose={100*m['eval_ifeval_da_prompt_loose']:.2f}%  "
              f"inst-strict={100*m['eval_ifeval_da_inst_strict']:.2f}%  "
              f"inst-loose={100*m['eval_ifeval_da_inst_loose']:.2f}%  "
              f"({dt:.0f}s)", flush=True)
        # Route metrics through the trainer's own logging path so wandb
        # gets them without the two-process attach conflict the sidecar
        # ran into. trainer.log() writes directly through HF's registered
        # wandb reporter.
        if self._trainer is not None:
            try:
                self._trainer.log(m)
            except Exception as e:
                print(f"  [ifeval-da] trainer.log() failed: {e}", flush=True)
        self._pending = m  # kept as fallback for on_log injection path
        model.train()
        control.should_log = True


class BestCkptSaverCallback(TrainerCallback):
    """Rolling top-K snapshotter — writes best model weights (no optim) to
    <output_dir>/_best_ckpts/ when the composite score improves.

    Composite = sum of whichever of the tracked metric keys land in `logs`
    for the current step. Tolerant of partial results (e.g. only ifeval-da
    landing before gsm8k for the mixed task — snapshot updates as more
    metrics accumulate for the same step).

    Skips optimizer/scheduler/rng in snapshots (best-of is for EVAL not
    resume) — saves ~1GB per snapshot on tight /workspace quotas."""

    _METRIC_KEYS = (
        "eval_ifeval_da_prompt_strict",
        "eval_ifeval_da_inst_strict",
        "eval_greedy_gsm8k_pass@1",
        "eval_greedy_ifeval_mean_pass",
        "eval_greedy_ifeval_combined_mean_pass",
    )

    def __init__(self, output_dir, tokenizer, top_k: int = 3,
                 subdir: str = "_best_ckpts"):
        from pathlib import Path as _P
        import shutil as _sh
        self._P = _P
        self._sh = _sh
        self.best_dir = _P(output_dir) / subdir
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = int(top_k)
        self.tok = tokenizer
        # (score, step, path). Sorted best-first.
        self.best: list[tuple[float, int, "_P"]] = []
        self._buf: dict[int, dict[str, float]] = {}
        self._snapped: dict[int, float] = {}  # step -> last snapshotted composite

    def on_log(self, args, state, control, logs=None, model=None, **kw):
        if logs is None or model is None:
            return
        step = state.global_step
        hits = {k: v for k, v in logs.items() if k in self._METRIC_KEYS}
        if not hits:
            return
        buf = self._buf.setdefault(step, {})
        buf.update(hits)
        score = sum(buf.values())
        # Same or lower score already snapshotted for this step — skip
        if self._snapped.get(step, -1.0) >= score:
            return
        # New candidate: must beat the current threshold to enter top-K
        threshold = min((s for s, _, _ in self.best), default=-1.0)
        if len(self.best) >= self.top_k and score <= threshold:
            return

        snap_dir = self.best_dir / f"step{step:06d}_score{score:07.3f}"
        # Remove any prior snapshot for this step (score got upgraded)
        prev = next((b for b in self.best if b[1] == step), None)
        if prev is not None:
            self._sh.rmtree(prev[2], ignore_errors=True)
            self.best = [b for b in self.best if b[1] != step]
        try:
            model.save_pretrained(snap_dir)
            if self.tok is not None:
                self.tok.save_pretrained(snap_dir)
            self._snapped[step] = score
        except Exception as e:
            print(f"  [best-ckpt] save err step={step}: {e}", flush=True)
            return
        self.best.append((score, step, snap_dir))
        self.best.sort(key=lambda x: -x[0])
        # Drop out-of-top-K snapshots
        for dropped in self.best[self.top_k:]:
            self._sh.rmtree(dropped[2], ignore_errors=True)
        self.best = self.best[:self.top_k]
        print(f"  [best-ckpt] step={step} score={score:.3f} snapped ({len(self.best)} tracked)",
              flush=True)


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
        self._pending = None  # kept as fallback
        self._trainer = None  # set by main() after trainer is built

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
                elif self.task == "combined":
                    scores_so_far = reward_ifeval_combined(
                        outs, [row[1] for row in self.items[:done]],
                        [row[2] for row in self.items[:done]])
                    print(f"  [greedy-eval] {done}/{len(self.items)} mean_pass={sum(scores_so_far)/len(scores_so_far):.4f}", flush=True)
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
        elif self.task == "combined":
            scores = reward_ifeval_combined(outs,
                                            [row[1] for row in self.items],
                                            [row[2] for row in self.items])
            acc = sum(scores) / max(1, len(scores))
        else:  # ifeval
            scores = reward_ifeval(outs,
                                   [row[1] for row in self.items],
                                   [row[2] for row in self.items])
            acc = sum(scores) / max(1, len(scores))
        print(f"  [greedy-eval] step={state.global_step} {self.task}={100*acc:.2f}%",
              flush=True)
        key = ("eval_greedy_gsm8k_pass@1" if self.task == "gsm8k"
               else "eval_greedy_ifeval_combined_mean_pass" if self.task == "combined"
               else "eval_greedy_ifeval_mean_pass")
        m = {key: acc}
        # Direct trainer.log() bypasses TRL 0.16's mid-step log-decision
        # (control.should_log=True doesn't force a log at the current step).
        if self._trainer is not None:
            try:
                self._trainer.log(m)
            except Exception as e:
                print(f"  [greedy-eval] trainer.log() failed: {e}", flush=True)
        self._pending = m  # fallback for on_log path
        model.train()
        control.should_log = True

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


def build_combined_dataset(source: str, max_rows: int = 0):
    """Loader for grpo_if_rewrite_v1 (or successor). `source` can be an HF
    repo id (jensjepsen/...) or a local save_to_disk directory path.

    Expected schema per row:
        prompt         str    — full user-facing prompt (task + rules woven in)
        constraints    list[str]     — mixed our-46 + google:...
        params         list[dict]    — one dict per constraint, parallel
    """
    from pathlib import Path as _P
    p = _P(source)
    if p.exists() and (p / "state.json").exists():
        # HF load_from_disk
        from datasets import load_from_disk as _lfd
        ds = _lfd(str(p))
    elif p.exists() and (p / "hf" / "state.json").exists():
        from datasets import load_from_disk as _lfd
        ds = _lfd(str(p / "hf"))
    else:
        ds = load_dataset(source, split="train")
    rows = []
    for i, r in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        u = r["prompt"]
        rows.append({
            "prompt": f"{USER}{u}{END}{ASST}",
            "constraints": r["constraints"],
            "params": r["params"],
        })
    return Dataset.from_list(rows)


def build_mixed_dataset(combined_source: str, max_rows: int = 0,
                        gsm_frac: float = 0.5, seed: int = 42):
    """Interleave gsm8k train + combined-IF into a single dataset with a
    `task` marker per row. `gsm_frac` sets the ratio of gsm8k rows
    relative to IF rows in the final mix (0.5 = ~equal). Unused columns
    are filled with defaults so all rows share the same schema:
       prompt (str), task ("gsm8k"|"ifeval"), gold (str),
       constraints (list[str]), params (list[dict])
    """
    import random as _r
    rng = _r.Random(seed)
    # IF side
    ifds = build_combined_dataset(combined_source, max_rows=0)
    if_rows = [{"prompt": r["prompt"], "task": "ifeval",
                "gold": "",
                "constraints": r["constraints"], "params": r["params"]}
               for r in ifds]
    # gsm8k side
    gds = build_gsm8k_dataset("train", max_rows=0)
    gsm_rows = [{"prompt": f"{USER} {r['prompt'][len(USER):].strip()}" if r["prompt"].startswith(USER) else r["prompt"],
                 "task": "gsm8k",
                 "gold": r["gold"],
                 "constraints": [], "params": []}
                for r in gds]
    # Cap gsm to `gsm_frac` share of total (bias if_rows to fully use)
    n_if = len(if_rows)
    n_gsm_target = int(round(n_if * (gsm_frac / max(1e-6, 1 - gsm_frac))))
    if len(gsm_rows) > n_gsm_target:
        rng.shuffle(gsm_rows)
        gsm_rows = gsm_rows[:n_gsm_target]
    rows = if_rows + gsm_rows
    rng.shuffle(rows)
    if max_rows and len(rows) > max_rows:
        rows = rows[:max_rows]
    return Dataset.from_list(rows)


def build_ifeval_da_dataset(max_rows: int = 0):
    """Load the danish-foundation-models/ifeval-da benchmark (541 rows).
    Used as a greedy-eval target for `combined` task so we see actual
    benchmark movement per checkpoint instead of a proxy metric."""
    ds = load_dataset("danish-foundation-models/ifeval-da", split="train")
    rows = []
    for i, r in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        # ifeval-da uses google's schema; prefix constraint ids so our
        # reward dispatches correctly. Build params-list aligned with
        # constraint list.
        cons = [f"google:{c}" for c in r.get("instruction_id_list", [])]
        params_raw = r.get("kwargs", [])
        # kwargs is a list[dict] parallel to instruction_id_list already
        rows.append({
            "prompt": f"{USER}{r['prompt']}{END}{ASST}",
            "constraints": cons,
            "params": params_raw,
        })
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gsm8k", "ifeval", "combined", "mixed"], required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="HF repo or local path — SFT starting model")
    ap.add_argument("--combined-source", default=None,
                    help="Required when --task=combined or mixed. HF repo id or "
                         "local save_to_disk dir for the mixed our-46 + google "
                         "training set (e.g. data/grpo_if_rewrite_v1 or "
                         "jensjepsen/danish-if-grpo-combined-v1).")
    ap.add_argument("--gsm-frac", type=float, default=0.5,
                    help="For --task=mixed: fraction of gsm8k rows in the "
                         "interleaved mix (0.5 = ~equal count vs IF rows).")
    ap.add_argument("--greedy-eval-task", choices=["auto", "ifeval-da", "same", "both"],
                    default="auto",
                    help="Which dataset the greedy-eval callback uses. "
                         "'same' = same as --task's train dataset (default for "
                         "gsm8k/ifeval). 'ifeval-da' = the 541-row benchmark "
                         "(default for combined). 'both' = attach both the "
                         "ifeval-da benchmark AND the gsm8k test-set callback "
                         "(default for --task=mixed). 'auto' picks per --task.")
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
    ap.add_argument("--save-align-eval", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Force save_steps = greedy_eval_steps so every "
                         "saved ckpt has a matching wandb-logged eval "
                         "point (and vice-versa) for apples-to-apples "
                         "offline reruns. ON by default; pass "
                         "--no-save-align-eval to disable.")
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
    ap.add_argument("--best-k", type=int, default=3,
                    help="Rolling top-K best-model snapshots (model weights + "
                         "tokenizer, no optim/scheduler) under "
                         "<output_dir>/_best_ckpts/. Composite score is sum of "
                         "whichever eval_* metrics land in the log for that step. "
                         "0 = disable.")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="Cap training rows (0=all). Handy for smoke tests.")
    ap.add_argument("--resume", default=None, nargs="?", const="latest",
                    help="Resume from a saved ckpt. Value can be a local dir "
                         "(preloaded via `huggingface-cli download`) or "
                         "'latest' to autodetect the newest ckpt in "
                         "--output-dir. Passes through to HF Trainer's "
                         "resume_from_checkpoint. Pair with "
                         "WANDB_RUN_ID=... WANDB_RESUME=allow to keep the "
                         "wandb chart continuous.")
    args = ap.parse_args()

    # Post-parse: apply save/eval alignment if requested
    if args.save_align_eval and args.greedy_eval_steps > 0:
        if args.save_steps != args.greedy_eval_steps:
            print(f"[save-align-eval] overriding save_steps "
                  f"{args.save_steps} -> {args.greedy_eval_steps}",
                  flush=True)
            args.save_steps = args.greedy_eval_steps

    tok_path = args.tokenizer or args.checkpoint
    print(f"loading tokenizer {tok_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(tok_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # TRL builds rollout GenerationConfig from tokenizer.eos_token_id (single
    # int). SFT models never emit their default eos (e.g. </s>); the correct
    # stop is <|end|>. Without this swap every rollout runs to
    # max_completion_length, torching gen time and reward signal.
    # Also collect <|user|> as an extra chat stop (catches the model spawning
    # a fake follow-up turn mid-completion for reward farming).
    chat_stops = []
    end_id = tok.convert_tokens_to_ids("<|end|>")
    if end_id is not None and end_id != tok.unk_token_id:
        original_eos = tok.eos_token
        tok.eos_token = "<|end|>"
        chat_stops.append(end_id)
        print(f"tok.eos_token: {original_eos!r} -> '<|end|>' (id={end_id})",
              flush=True)
    user_id = tok.convert_tokens_to_ids("<|user|>")
    if user_id is not None and user_id != tok.unk_token_id:
        chat_stops.append(user_id)

    print(f"building dataset for task={args.task}...", flush=True)
    eval_ds = None
    if args.task == "gsm8k":
        ds = build_gsm8k_dataset("train", max_rows=args.max_rows or 0)
        reward_fn = reward_gsm8k
        if args.eval_steps > 0:
            eval_ds = build_gsm8k_dataset("test", max_rows=args.eval_max_rows)
    elif args.task == "ifeval":
        ds = build_ifeval_dataset("train", max_rows=args.max_rows or 0)
        reward_fn = reward_ifeval
        if args.eval_steps > 0:
            eval_ds = build_ifeval_dataset("eval", max_rows=args.eval_max_rows)
    elif args.task == "combined":
        assert args.combined_source, "--combined-source required for --task=combined"
        ds = build_combined_dataset(args.combined_source, max_rows=args.max_rows or 0)
        reward_fn = reward_ifeval_combined
        # No separate eval split; greedy-eval callback handles benchmarking.
    else:  # mixed
        assert args.combined_source, "--combined-source required for --task=mixed"
        ds = build_mixed_dataset(args.combined_source,
                                 max_rows=args.max_rows or 0,
                                 gsm_frac=args.gsm_frac)
        reward_fn = reward_mixed
        # Greedy-eval callback attaches BOTH ifeval-da and gsm8k below.
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
    # Rebuild trainer.generation_config with the full chat_stops list so
    # rollouts stop on any of <|end|> / <|user|>. TRL only puts a single
    # eos_token_id in the GenerationConfig by default.
    if len(chat_stops) > 1:
        from transformers import GenerationConfig
        gc = trainer.generation_config
        cfg_gen = gc.to_dict()
        cfg_gen["eos_token_id"] = chat_stops
        trainer.generation_config = GenerationConfig(**cfg_gen)
    print(f"trainer.generation_config.eos_token_id = "
          f"{trainer.generation_config.eos_token_id}", flush=True)
    if args.greedy_eval_steps > 0:
        # Resolve which callbacks to attach
        eval_task = args.greedy_eval_task
        if eval_task == "auto":
            if args.task == "combined":
                eval_task = "ifeval-da"
            elif args.task == "mixed":
                eval_task = "both"
            else:
                eval_task = "same"

        def _attach_ifeval_da():
            cb = IFEvalDACallback(
                tokenizer=tok,
                every_n_steps=args.greedy_eval_steps,
                max_new_tokens=args.max_completion_length,
                batch_size=args.batch_size,
            )
            cb._trainer = trainer  # so callback can call trainer.log()
            trainer.add_callback(cb)

        def _attach_gsm8k():
            gds = build_gsm8k_dataset("test", max_rows=args.greedy_eval_max_rows)
            items = [(r["prompt"], r["gold"]) for r in gds]
            cb = GreedyEvalCallback(
                tokenizer=tok, items=items, task="gsm8k",
                every_n_steps=args.greedy_eval_steps,
                max_new_tokens=args.max_completion_length,
                batch_size=args.batch_size,
            )
            cb._trainer = trainer  # direct trainer.log() bypasses on_log
            trainer.add_callback(cb)

        if eval_task == "ifeval-da":
            _attach_ifeval_da()
        elif eval_task == "both":
            _attach_ifeval_da()
            _attach_gsm8k()
        else:  # 'same' — task-specific single greedy callback
            if args.task == "gsm8k":
                _attach_gsm8k()
            elif args.task == "combined":
                gds = build_combined_dataset(args.combined_source,
                                             max_rows=args.greedy_eval_max_rows)
                g_items = [(r["prompt"], r["constraints"], r["params"]) for r in gds]
                cb = GreedyEvalCallback(
                    tokenizer=tok, items=g_items, task="combined",
                    every_n_steps=args.greedy_eval_steps,
                    max_new_tokens=args.max_completion_length,
                    batch_size=args.batch_size,
                )
                cb._trainer = trainer
                trainer.add_callback(cb)
            else:  # ifeval task
                gds = build_ifeval_dataset("eval", max_rows=args.greedy_eval_max_rows)
                g_items = [(r["prompt"], r["constraints"], r["params"]) for r in gds]
                cb = GreedyEvalCallback(
                    tokenizer=tok, items=g_items, task="ifeval",
                    every_n_steps=args.greedy_eval_steps,
                    max_new_tokens=args.max_completion_length,
                    batch_size=args.batch_size,
                )
                cb._trainer = trainer
                trainer.add_callback(cb)
    # Rolling top-K snapshot of the best-scoring model weights (no optim).
    # Reads whatever metric keys land in on_log; composite = sum.
    if args.best_k > 0:
        trainer.add_callback(BestCkptSaverCallback(
            output_dir=args.output_dir,
            tokenizer=tok,
            top_k=args.best_k,
        ))

    resume = args.resume
    if resume == "latest":
        resume = True  # HF Trainer autodetects newest ckpt in output_dir
    if resume:
        print(f"resuming from {resume!r}", flush=True)
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()
    trainer.save_model(f"{args.output_dir}/final")


if __name__ == "__main__":
    import os
    os.environ.setdefault("WANDB_PROJECT", "danish-lm-grpo")
    main()
