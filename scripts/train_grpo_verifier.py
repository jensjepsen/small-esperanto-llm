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
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Apply Liger kernels (RoPE, RMSNorm, SwiGLU) before any Llama model is
# created. Skips fused_linear_cross_entropy since GRPO computes per-token
# log-probs manually (no standard CE loss path); enabling it just risks
# subtle interactions with TRL's log-prob math. Liger is required — fail
# loudly if missing so we notice a broken env rather than silently
# regressing to vanilla-slow training.
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama(
    rope=True, rms_norm=True, swiglu=True,
    fused_linear_cross_entropy=False, cross_entropy=False,
)
print("[liger] RoPE + RMSNorm + SwiGLU kernels applied", flush=True)

import os as _os
import re
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# GRPO_VLLM_DTYPE: force vLLM colocate to load with a specific dtype.
# Accepts "float32", "float16", "bfloat16". Overrides vLLM's default
# "auto" (which downcasts fp32 model configs to bf16 on Blackwell/H100).
# Use "float32" for cleanest match with trainer's fp32 masters (loss:
# ~2× VRAM, ~2× slower rollouts). Use "float16" for Wu et al. 2025's
# FP16-everywhere training-inference-mismatch mitigation (must combo with
# GRPO_FP16_EVERYWHERE=1 to also flip trainer to fp16 autocast).
# GRPO_FP16_EVERYWHERE: kept as legacy alias — implies GRPO_VLLM_DTYPE=float16
# AND trainer fp16 autocast (fp16=True) in the GRPOConfig block below.
_vllm_dtype = _os.environ.get("GRPO_VLLM_DTYPE")
if not _vllm_dtype and _os.environ.get("GRPO_FP16_EVERYWHERE") == "1":
    _vllm_dtype = "float16"  # legacy alias
if _vllm_dtype:
    assert _vllm_dtype in ("float32", "float16", "bfloat16"), (
        f"GRPO_VLLM_DTYPE must be one of float32/float16/bfloat16, got {_vllm_dtype!r}")
    from vllm import LLM as _LLM
    _orig_llm_init = _LLM.__init__

    def _forced_dtype_llm_init(self, *args, **kwargs):
        if "dtype" not in kwargs:
            kwargs["dtype"] = _vllm_dtype
        return _orig_llm_init(self, *args, **kwargs)

    _LLM.__init__ = _forced_dtype_llm_init
    print(f"[vllm-dtype] vLLM.LLM patched to force dtype={_vllm_dtype}", flush=True)


# GRPO_VLLM_ATTENTION_BACKEND: force vLLM's attention backend, e.g.
# "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION". vLLM >=0.19 dropped the
# VLLM_ATTENTION_BACKEND env var; the backend is now an engine arg
# (AttentionConfig.backend, parsed from an upper-case string), so the only way
# to set it through TRL — which doesn't forward arbitrary vLLM kwargs — is to
# patch LLM.__init__ the same way GRPO_VLLM_DTYPE does.
#
# Why you may need this: vLLM's bundled FlashAttention-2 kernels
# (torch.ops._vllm_fa2_C) ship PTX built with a newer CUDA toolkit than some
# drivers can JIT. On a 12.8 driver that raises
#   AcceleratorError: the provided PTX was compiled with an unsupported toolchain
# on the first attention call. FLASHINFER avoids those kernels entirely.
_vllm_attn = _os.environ.get("GRPO_VLLM_ATTENTION_BACKEND")
if _vllm_attn:
    from vllm import LLM as _LLM_A
    _orig_llm_init_attn = _LLM_A.__init__

    def _forced_attn_llm_init(self, *args, **kwargs):
        if "attention_backend" not in kwargs:
            kwargs["attention_backend"] = _vllm_attn.upper()
        return _orig_llm_init_attn(self, *args, **kwargs)

    _LLM_A.__init__ = _forced_attn_llm_init
    print(f"[vllm-attn] vLLM.LLM patched to force "
          f"attention_backend={_vllm_attn.upper()}", flush=True)


# NB: TRL >=1.0 sets logprobs_mode="processed_logprobs" natively in
# trl/generation/vllm_generation.py::_init_vllm (fixes TRL #4159).
# Our old GRPO_VLLM_PROCESSED_LOGPROBS monkey-patch is no longer needed.
# Divergence itself is logged natively too as
# `sampling/sampling_logp_difference/{mean,max}` when
# vllm_importance_sampling_correction=True (default), which also enables
# Truncated Importance Sampling to correct for the mismatch.


# GRPO_VLLM_STOP_TOKEN_IDS: comma-separated token ids to force vLLM to stop on.
# TRL 0.18.2 does NOT set stop_token_ids in SamplingParams, so vLLM only
# stops on tokenizer.eos_token_id. Our SFT tokenizer's eos is `<|end|>` (16002)
# but the model was also trained to emit `<|user|>` (16000) at end of assistant
# turn. Without this, ~5-15% of rollouts run to max_new_tokens because they
# emit `<|user|>` and vLLM doesn't stop. Monkey-patch LLM.generate to inject
# stops if the caller didn't already set them.
# GRPO_DAPO_RESAMPLE=N: DAPO dynamic sampling — retry the whole generation
# batch up to N times if any prompt-group has zero reward std, keep the
# attempt with the most active groups. Costs up to (N+1)× generation but
# recovers gradient signal from otherwise-dead groups. Composes with
# --skip-zero-adv: DAPO runs first (retries to reduce zero-std count),
# then skip-adv masks any remaining zero-std groups.
_DAPO_RETRIES = _os.environ.get("GRPO_DAPO_RESAMPLE")
_DAPO_FRESH_PROMPTS = _os.environ.get("GRPO_DAPO_FRESH_PROMPTS") == "1"
_DAPO_FRESH_MATCH_TASK = _os.environ.get("GRPO_DAPO_FRESH_MATCH_TASK") == "1"


from esperanto_lm.dapo_slot_swap import active_mask as _dapo_active_mask, slot_swap as _dapo_slot_swap


def _dapo_active_count(result, num_gens):
    """Back-compat wrapper — returns (n_active_int, n_groups)."""
    m, g = _dapo_active_mask(result, num_gens)
    if m is None:
        return 0, g
    return int(m.sum().item()), g


if _DAPO_RETRIES:
    _dapo_n = max(1, int(_DAPO_RETRIES))
    _orig_dapo_gen = GRPOTrainer._generate_and_score_completions

    def _dapo_gen(self, inputs):
        best = _orig_dapo_gen(self, inputs)
        best_mask, n_groups = _dapo_active_mask(best, self.num_generations)
        best_active = 0 if best_mask is None else int(best_mask.sum().item())
        first_active = best_active
        attempts_used = 1
        rescued_by_fresh = 0
        if _DAPO_FRESH_PROMPTS:
            if not hasattr(self, "_dapo_spare_iter"):
                self._dapo_spare_iter = iter(self.get_train_dataloader())
        def _pull_fresh_batch(want_task=None, max_reject=50):
            """Fetch next batch from spare iterator, rejecting task
            mismatches when want_task is set (rejection-sample matched
            fresh prompt). Falls back to last drawn if budget exhausted."""
            last = None
            for _ in range(max_reject):
                try:
                    b = next(self._dapo_spare_iter)
                except StopIteration:
                    self._dapo_spare_iter = iter(self.get_train_dataloader())
                    b = next(self._dapo_spare_iter)
                last = b
                if want_task is None:
                    return b
                got = _extract_task(b)
                if got == want_task:
                    return b
            return last

        def _extract_task(x):
            """inputs may be a dict of columns or a list of per-microbatch
            dicts (TRL 0.18 hands us a generation_batch that can be either).
            Reach into the first entry's 'task' column and grab the first
            row's task string."""
            d = None
            if isinstance(x, dict):
                d = x
            elif isinstance(x, (list, tuple)) and x:
                for item in x:
                    if isinstance(item, dict):
                        d = item
                        break
            if d is None:
                return None
            t = d.get("task")
            if isinstance(t, (list, tuple)) and t:
                return t[0]
            return t

        want_task = None
        if _DAPO_FRESH_PROMPTS and _DAPO_FRESH_MATCH_TASK:
            want_task = _extract_task(inputs)

        for _attempt in range(_dapo_n):
            if best_active >= n_groups:
                break
            attempts_used += 1
            if _DAPO_FRESH_PROMPTS:
                fresh = _pull_fresh_batch(want_task=want_task)
                attempt_result = _orig_dapo_gen(self, fresh)
            else:
                attempt_result = _orig_dapo_gen(self, inputs)
            att_mask, att_n = _dapo_active_mask(attempt_result, self.num_generations)
            # Per-slot swap: for each slot dead in best but active in attempt,
            # overwrite best's slot rows with attempt's slot rows (pad-
            # reconciled). Works for n_groups >= 1. When attempt has a
            # different n_groups (drop_last edge), slot_swap returns 0 and we
            # fall back to whole-batch replacement only if strictly better.
            n_swapped = 0
            if best_mask is not None and att_mask is not None and att_n == n_groups:
                n_swapped = _dapo_slot_swap(best, attempt_result,
                                            self.num_generations,
                                            best_mask=best_mask,
                                            src_mask=att_mask)
                if n_swapped > 0:
                    best_active = int(best_mask.sum().item())
                    if _DAPO_FRESH_PROMPTS:
                        rescued_by_fresh += n_swapped
            elif att_mask is not None:
                a = int(att_mask.sum().item())
                if a > best_active:
                    best = attempt_result
                    best_mask = att_mask
                    n_groups = att_n
                    best_active = a
                    if _DAPO_FRESH_PROMPTS:
                        rescued_by_fresh += a
        mode_train = self._metrics.setdefault("train", {})
        mode_train.setdefault("dapo_active", []).append(
            best_active / max(1, n_groups)
        )
        mode_train.setdefault("dapo_first_active", []).append(
            first_active / max(1, n_groups)
        )
        mode_train.setdefault("dapo_attempts_used", []).append(float(attempts_used))
        if _DAPO_FRESH_PROMPTS:
            mode_train.setdefault("dapo_rescued_by_fresh", []).append(float(rescued_by_fresh))
        return best

    GRPOTrainer._generate_and_score_completions = _dapo_gen
    if _DAPO_FRESH_PROMPTS:
        _mode = "FRESH-prompts (per-slot swap, supports n_groups>=1)"
        if _DAPO_FRESH_MATCH_TASK:
            _mode += " +TASK-MATCH (rejection-sample on inputs['task'][0])"
    else:
        _mode = "same-prompts re-rollout (per-slot swap)"
    print(f"[dapo] dynamic sampling enabled: up to {_dapo_n} extra generation "
          f"attempts, mode={_mode}", flush=True)


# NB: log_rho was our custom |log(π_train/π_vllm)| metric, computed via
# monkey-patches on GRPOTrainer._get_per_token_logps + compute_loss. TRL
# 1.0+ removed _get_per_token_logps AND logs the same statistic natively
# as `sampling/sampling_logp_difference/{mean,max}` whenever
# vllm_importance_sampling_correction=True (default). Related metrics:
# `sampling/importance_sampling_ratio/{min,mean,max}`. Full block deleted.


_STOPS = _os.environ.get("GRPO_VLLM_STOP_TOKEN_IDS")
if _STOPS:
    from vllm import LLM as _LLM_stop
    _stop_ids = [int(x) for x in _STOPS.split(",") if x.strip()]
    _orig_llm_generate = _LLM_stop.generate

    def _llm_generate_with_stops(self, prompts=None, sampling_params=None, **kwargs):
        if sampling_params is not None:
            _sp_list = (sampling_params if isinstance(sampling_params, list)
                        else [sampling_params])
            for _sp in _sp_list:
                if getattr(_sp, "stop_token_ids", None) is None:
                    _sp.stop_token_ids = list(_stop_ids)
        return _orig_llm_generate(self, prompts=prompts,
                                  sampling_params=sampling_params, **kwargs)

    _LLM_stop.generate = _llm_generate_with_stops
    print(f"[vllm-stops] vLLM.LLM.generate patched, stop_token_ids={_stop_ids}",
          flush=True)

from esperanto_lm.rl_rewards import (
    reward_gsm8k, reward_ifeval, reward_ifeval_combined, reward_mixed,
    reward_json_schema, reward_ner, ner_prompt,
    _extract_num, _norm_num,
)

_NUM_RE_INT = re.compile(r"-?\d[\d,]*\.?\d*")


# _get_train_sampler shim REMOVED: TRL 0.18.2's signature natively accepts
# `dataset=None`, matching transformers 4.55+ expectations.


def _greedy_via_vllm_colocate(trainer, prompts, max_new, stop_token_ids=None):
    """Route greedy eval generation through the trainer's colocate vLLM engine.
    Returns list[str] or None (caller falls back to HF generate).

    Colocate mode only — server mode is skipped intentionally: server-mode
    eval would add a full HTTP round-trip per eval, and none of the current
    launchers use server mode on the target hardware.

    Weight sync: TRL keeps the colocate engine in step with the trainer
    weights automatically at each optimizer step, so no manual sync is
    needed here. deterministic sampling (temperature=0). repetition_penalty
    matched to the HF path (1.1).

    Engine attribute location changed between TRL versions:
      TRL 0.18.x: trainer.llm
      TRL 1.10.x: trainer.vllm_generation.llm
    Check both, prefer whichever exists."""
    llm = getattr(trainer, "llm", None)
    if llm is None:
        vg = getattr(trainer, "vllm_generation", None)
        llm = getattr(vg, "llm", None) if vg is not None else None
    if llm is None:
        return None
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(max_new),
        n=1,
        repetition_penalty=1.1,
        stop_token_ids=list(stop_token_ids) if stop_token_ids else None,
    )
    # vLLM preserves input order in the returned outputs, so we can
    # zip 1:1 against `prompts` downstream. use_tqdm=False keeps the
    # per-fire eval quiet since callback already prints a header.
    outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
    return [o.outputs[0].text for o in outs]


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
            vllm_out = _greedy_via_vllm_colocate(
                self._trainer, self._prompts, self.max_new, stop_token_ids=eos_ids)
            if vllm_out is not None:
                responses = vllm_out
                print(f"  [ifeval-da] step={state.global_step} vllm gen "
                      f"{len(responses)}/{len(self._prompts)}", flush=True)
            else:
                # HF fallback (non-vLLM runs)
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
        "eval_greedy_json_mean_reward",
        "eval_greedy_ner_f1",
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
                 batch_size: int = 16, metric_name: str | None = None):
        """items schema:
             gsm8k:   list of (prompt, gold_answer_string)
             ifeval:  list of (prompt, constraints_list, params_json_string)
             json:    list of (prompt, fields_list, types_list, strict_bool, passage_str, gold_dict_or_None)
             struct:  list of (prompt, gold, fields, fmt, passage_or_None)"""
        # Several callbacks can share a task (two `struct` evals: NER and
        # ICL). Without an explicit name they collide on one log key and
        # silently overwrite each other --- and `struct` fell through to the
        # ifeval key, corrupting that curve too.
        self.metric_name = metric_name
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
            all_prompts = [row[0] for row in self.items]
            vllm_out = _greedy_via_vllm_colocate(
                self._trainer, all_prompts, self.max_new, stop_token_ids=eos_ids)
            if vllm_out is not None:
                outs = vllm_out
                print(f"  [greedy-eval] vllm gen {len(outs)}/{len(self.items)}",
                      flush=True)
            else:
                # HF fallback (non-vLLM runs)
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
                    elif self.task == "json":
                        scores_so_far = [
                            reward_json_schema(o, r[1], r[3], passage=(r[4] or None),
                                               types=r[2],
                                               gold_values=(r[5] if len(r) > 5 else None))
                            for o, r in zip(outs, self.items[:done])
                        ]
                        print(f"  [greedy-eval] {done}/{len(self.items)} mean_reward={sum(scores_so_far)/len(scores_so_far):.4f}", flush=True)
                    elif self.task == "struct":
                        # items are (prompt, gold, fields, fmt, passage)
                        from esperanto_lm.rl_rewards import reward_structured
                        scores_so_far = [reward_structured(o, r[1], r[2], r[3], r[4])
                                         for o, r in zip(outs, self.items[:done])]
                        print(f"  [greedy-eval] {done}/{len(self.items)} mean_f1={sum(scores_so_far)/len(scores_so_far):.4f}", flush=True)
                    elif self.task == "ner":
                        # items are (prompt, gold_values_json)
                        scores_so_far = [reward_ner(o, r[1])
                                         for o, r in zip(outs, self.items[:done])]
                        print(f"  [greedy-eval] {done}/{len(self.items)} mean_f1={sum(scores_so_far)/len(scores_so_far):.4f}", flush=True)
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
        elif self.task == "json":
            scores = [
                reward_json_schema(o, r[1], r[3], passage=(r[4] or None), types=r[2],
                                   gold_values=(r[5] if len(r) > 5 else None))
                for o, r in zip(outs, self.items)
            ]
            acc = sum(scores) / max(1, len(scores))
        elif self.task == "struct":
            from esperanto_lm.rl_rewards import reward_structured
            scores = [reward_structured(o, r[1], r[2], r[3], r[4])
                      for o, r in zip(outs, self.items)]
            acc = sum(scores) / max(1, len(scores))
        elif self.task == "ner":
            scores = [reward_ner(o, r[1]) for o, r in zip(outs, self.items)]
            acc = sum(scores) / max(1, len(scores))
        else:  # ifeval
            scores = reward_ifeval(outs,
                                   [row[1] for row in self.items],
                                   [row[2] for row in self.items])
            acc = sum(scores) / max(1, len(scores))
        print(f"  [greedy-eval] step={state.global_step} "
              f"{self.metric_name or self.task}={100*acc:.2f}%",
              flush=True)
        key = (self.metric_name
               or ("eval_greedy_gsm8k_pass@1" if self.task == "gsm8k"
                   else "eval_greedy_ifeval_combined_mean_pass" if self.task == "combined"
                   else "eval_greedy_json_mean_reward" if self.task == "json"
                   else "eval_greedy_ner_f1" if self.task == "ner"
                   else "eval_greedy_ifeval_mean_pass"))
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


def build_json_dataset(source: str, split: str = "train", max_rows: int = 0):
    """Loader for danish-json-grpo-v1 (or successor). `source` = HF repo id
    or local save_to_disk path. Emits mixed-compatible rows with task='json'
    and JSON-specific columns for reward_json_schema.

    Fixup: v1 dataset has ~19% rows (mostly fill_template @ 87%) where the
    Gemini rewriter dropped the source passage from the prompt text even
    though it stored one in the `passage` field. Without the passage inline,
    the model has nothing to extract from and the reward is forced-fail
    noise. When passage exists but isn't inline, we prepend it explicitly.
    """
    from pathlib import Path as _P
    p = _P(source)
    if p.exists() and (p / "state.json").exists():
        from datasets import load_from_disk as _lfd
        ds = _lfd(str(p))
    else:
        ds = load_dataset(source, split=split)
    rows = []
    n_fixed = 0
    for i, r in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        u = r["prompt"]
        passage = r.get("passage") or ""
        if passage and len(passage) > 30:
            # Not inline if first 40 chars of passage don't appear in prompt.
            if passage.strip()[:40] not in u:
                u = f"{u}\n\nKildetekst:\n{passage.strip()}"
                n_fixed += 1
        rows.append({
            "prompt": f"{USER}{u}{END}{ASST}",
            "task": "json",
            "gold": "",
            "constraints": [],
            "params": [],
            "fields": list(r["fields"]),
            "types": list(r["types"]),
            "strict": bool(r["strict"]),
            "passage": passage,
            "gold_values": r.get("gold_values") or "",  # JSON string; reward decodes
        })
    if n_fixed:
        print(f"  [build_json_dataset] appended passage inline for {n_fixed}/{len(rows)} "
              f"rows (Gemini-drop fixup)", flush=True)
    return Dataset.from_list(rows)


_NER_TYPE_MAP = {"PERSON": "person", "PER": "person",
                 "ORGANIZATION": "org", "ORG": "org",
                 "GPE": "sted", "LOCATION": "sted", "LOC": "sted",
                 "FACILITY": "sted", "DATE": "dato"}

# Prompt is built per-example by ner_prompt() in rl_rewards, because each row
# requests a SUBSET of the entity types — see build_ner_dataset.


def build_ner_dataset(source: str = "KennethEnevoldsen/dane_plus",
                      split: str = "train", max_rows: int = 0,
                      empty_frac: float = 0.28, seed: int = 42):
    """dane_plus → mixed-compatible rows with task='ner'.

    Gold entities ride in `gold_values` (serialized [[surface, type], ...]) so
    no new column is needed — Arrow requires one schema across all interleaved
    parts, and adding a column would mean touching every other builder.

    `empty_frac` caps the share of entity-FREE sentences. The natural split is
    ~53% empty, and empty-on-empty scores 1.0, so at the natural rate a policy
    that never emits anything earns mean reward ~0.53 — a high plateau that the
    v31 base already sits near (82-93% empty completions). Downsampling the
    abstention half keeps that from being the easiest way to a good score.
    Set empty_frac<0 to keep the natural distribution.
    """
    import random as _rnd
    ds = load_dataset(source, split=split)
    with_e, no_e = [], []
    for r in ds:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        ents = set()
        for e in r.get("ents") or []:
            raw = str(e.get("label", "")).upper()
            lab = _NER_TYPE_MAP.get(raw) or _NER_TYPE_MAP.get(raw.replace(" ", "_"))
            if not lab:
                continue
            surf = text[e["start"]:e["end"]].strip()
            if surf:
                ents.add((surf.lower(), lab))
        row = {"text": text, "ents": sorted(ents)}
        (with_e if ents else no_e).append(row)

    rng = _rnd.Random(seed)
    if empty_frac is not None and empty_frac >= 0 and with_e:
        # n_empty such that n_empty / (n_empty + n_with) == empty_frac
        want = int(round(len(with_e) * empty_frac / max(1e-6, 1.0 - empty_frac)))
        no_e = rng.sample(no_e, min(want, len(no_e)))
    rows_src = with_e + no_e
    rng.shuffle(rows_src)
    if max_rows and len(rows_src) > max_rows:
        rows_src = rows_src[:max_rows]

    # Each row requests a SUBSET of the four types. Asking for all four every
    # time lets the policy emit one memorised object and never read the schema
    # — which is what happened twice (SFT textman keys; then `org` emitted in
    # 565/565 rows and populated in none). Weighted toward larger subsets so
    # the full-schema case stays common, and any row whose gold would become
    # empty under its subset is re-drawn once to avoid manufacturing extra
    # abstention on top of the --ner-empty-frac balance.
    ALL_B = ("person", "org", "sted", "dato")
    SUBSET_SIZES = [1, 2, 3, 4]
    SUBSET_W = [0.10, 0.20, 0.30, 0.40]
    rows = []
    for r in rows_src:
        have = {t for _, t in r["ents"]}
        for _attempt in range(2):
            k = rng.choices(SUBSET_SIZES, weights=SUBSET_W)[0]
            buckets = sorted(rng.sample(ALL_B, k), key=ALL_B.index)
            if not have or (have & set(buckets)):
                break
        ents = [[sfc, t] for sfc, t in r["ents"] if t in buckets]
        rows.append({
            "prompt": f"{USER}{ner_prompt(buckets).format(t=r['text'])}{END}{ASST}",
            "task": "ner",
            "gold": "",
            "constraints": [], "params": [],
            "fields": [], "types": [], "strict": False, "passage": r["text"],
            "gold_values": json.dumps({"ents": ents, "buckets": buckets},
                                      ensure_ascii=False)})
    from collections import Counter as _C
    _sizes = _C(len(json.loads(r["gold_values"])["buckets"]) for r in rows)
    n_empty = sum(1 for r in rows if not json.loads(r["gold_values"])["ents"])
    print(f"  [build_ner] {source}:{split} → {len(rows)} rows "
          f"({len(rows)-n_empty} with entities, {n_empty} entity-free = "
          f"{100*n_empty/max(1,len(rows)):.0f}%)  "
          f"key-subset sizes: {dict(sorted(_sizes.items()))}", flush=True)
    return Dataset.from_list(rows)


def _drop_overlong(rows, tokenizer, max_prompt_tokens, label):
    """Drop rows whose prompt exceeds the vLLM window.

    TRL 1.x dropped max_prompt_length as a TRUNCATION control -- it now only
    feeds vllm_max_model_length = max_prompt + max_completion. Nothing clips
    the prompt, so a single over-long row aborts the whole run mid-training
    ("maximum context length is 2432 tokens ... your prompt contains 2699").
    That killed mixed5_v2 at step 9,795 after four hours.

    Filtering here rather than truncating: a truncated ICL prompt loses its
    demonstrations and becomes unanswerable, which would feed the reward
    channel pure noise instead of crashing loudly.
    """
    if not tokenizer or not max_prompt_tokens:
        return rows
    keep, dropped, longest = [], 0, 0
    for r in rows:
        n = len(tokenizer(r["prompt"], add_special_tokens=False)["input_ids"])
        longest = max(longest, n)
        if n <= max_prompt_tokens:
            keep.append(r)
        else:
            dropped += 1
    print(f"  [{label}] dropped {dropped}/{len(rows)} rows over "
          f"{max_prompt_tokens} prompt tokens (longest seen {longest})",
          flush=True)
    return keep


def build_ner_sft_dataset(source: str = "jensjepsen/danish-ner-sft-v1",
                          split: str = "train", max_rows: int = 0,
                          seed: int = 42, tokenizer=None,
                          max_prompt_tokens: int = 0):
    """NER GRPO rows from the SFT set rather than raw dane_plus.

    dane_plus goes through a synthesised prompt and a JSON-only verifier, which
    reaches neither of the two things v33 is actually weak at: instruction mode
    (48.4 exact vs 60.4 for demonstrations) and span-wrap on an unseen
    delimiter (7.5). This set carries all three prompt modes and all fourteen
    formats with gold answers verified to strip back to their own passage, so
    GRPO pressure lands where the headroom is.

    passage is recovered from the prompt because span-wrap faithfulness is
    scored against it; the rendering is fixed ("Tekst:\n...\nSvar:") and is the
    same slice eval_ner_sft.py takes.
    """
    import random as _rnd
    from datasets import load_dataset as _ld
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_icl_schema_format import SYMBOLS as _SYM

    ds = _ld(source, "default", split=split)
    rows = []
    for r in ds:
        fmt = r.get("format")
        if not fmt:
            continue
        if r.get("symbols", "none") == "none":
            keys = sorted(set(r["types"].split("|")))
        else:
            keys = sorted(set(_SYM[r["symbols"]][:r["n_types"]]))
        if not keys:
            continue
        prompt = r["messages"][0]["content"]
        if "Tekst:\n" not in prompt or "\nSvar:" not in prompt:
            continue
        passage = prompt.rsplit("Tekst:\n", 1)[1].split("\nSvar:")[0]
        rows.append({
            "prompt": prompt,
            "task": "ner",
            "gold": r["messages"][1]["content"],
            "constraints": [], "params": [],
            "fields": keys, "types": [fmt], "strict": False,
            "passage": passage, "gold_values": "",
        })
    rows = _drop_overlong(rows, tokenizer, max_prompt_tokens, "build_ner_sft")
    _rnd.Random(seed).shuffle(rows)
    if max_rows:
        rows = rows[:max_rows]
    print(f"  [build_ner_sft] {len(rows)} rows from {source}:{split}", flush=True)
    return Dataset.from_list(rows)


def build_icl_dataset(source: str, split: str = "train", max_rows: int = 0,
                      seed: int = 42, tokenizer=None,
                      max_prompt_tokens: int = 0):
    """ICL schema/format induction rows for GRPO.

    The prompt already carries the demonstrations, so the policy has to induce
    both the key set and the output format from them. The verifier needs three
    things: the gold answer, the key set, and which format to parse with ---
    carried in `gold`, `fields` and `types[0]` respectively, reusing the union
    columns rather than widening the schema for every other builder.

    Keys come from metadata (schema / symbol scheme), never by regexing the
    rendered answer --- a per-format pattern table is the maintenance trap that
    broke twice when bracket_pair/brace_pair were added.
    """
    import random as _rnd
    from datasets import load_dataset as _ld
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_icl_schema_format import SYMBOLS as _SYM

    ds = _ld(source, "default", split=split)
    rows = []
    for r in ds:
        if r.get("symbols", "none") == "none":
            keys = sorted(set(r["schema"].split("|")))
        else:
            keys = sorted(set(_SYM[r["symbols"]][:r["n_fields"]]))
        if not keys or not r.get("format"):
            continue
        rows.append({
            "prompt": r["messages"][0]["content"],
            "task": "icl",
            "gold": r["messages"][1]["content"],
            "constraints": [], "params": [],
            "fields": keys, "types": [r["format"]], "strict": False,
            "passage": "", "gold_values": "",
        })
    rows = _drop_overlong(rows, tokenizer, max_prompt_tokens, "build_icl")
    _rnd.Random(seed).shuffle(rows)
    if max_rows:
        rows = rows[:max_rows]
    print(f"  [build_icl] {len(rows)} rows from {source}:{split}", flush=True)
    return Dataset.from_list(rows)


def build_mixed_dataset(combined_source: str, max_rows: int = 0,
                        gsm_frac: float = 0.5, seed: int = 42,
                        json_source: str | None = None,
                        json_frac: float = 0.0,
                        ner_source: str | None = None,
                        ner_frac: float = 0.0,
                        ner_empty_frac: float = 0.28,
                        icl_source: str | None = None,
                        icl_frac: float = 0.0,
                        tokenizer=None, max_prompt_tokens: int = 0,
                        interleave_strategy: str = "all_exhausted"):
    """Mix gsm8k train + combined-IF (+ optional json) via
    datasets.interleave_datasets with per-task probabilities [if_share,
    gsm_frac, json_frac].

    interleave_strategy:
      - "first_exhausted": stop when the smallest source runs out (throws
        away rows from the larger sources — matches prior subsample behavior)
      - "all_exhausted":   stop when the LARGEST source is fully consumed
        (small sources are cycled with-replacement to keep target ratios);
        no rows discarded

    Schema uniformity: all rows carry the union of columns
      prompt, task, gold, constraints, params,
      fields, types, strict, passage, gold_values
    with empty defaults where a task doesn't use a given column. This is
    required by Arrow: mixed-shape rows can't be concatenated.
    """
    from datasets import interleave_datasets

    _EMPTY_JSON = {"fields": [], "types": [], "strict": False,
                   "passage": "", "gold_values": ""}

    # Sources are loaded ONLY when their fraction is non-zero — otherwise a
    # json+ner run still downloads and builds the IF and gsm8k splits it will
    # never sample from.
    _if_share = 1.0 - gsm_frac - json_frac - ner_frac - icl_frac
    if_ds = None
    if _if_share > 1e-9:
        ifds = build_combined_dataset(combined_source, max_rows=0)
        if_ds = Dataset.from_list(
        [{"prompt": r["prompt"], "task": "ifeval",
          "gold": "",
          "constraints": r["constraints"], "params": r["params"],
          **_EMPTY_JSON}
             for r in ifds])

    # gsm8k side
    gsm_ds = None
    if gsm_frac > 1e-9:
        gds = build_gsm8k_dataset("train", max_rows=0)
        gsm_ds = Dataset.from_list(
        [{"prompt": (f"{USER} {r['prompt'][len(USER):].strip()}"
                     if r["prompt"].startswith(USER) else r["prompt"]),
          "task": "gsm8k",
          "gold": r["gold"],
          "constraints": [], "params": [],
          **_EMPTY_JSON}
         for r in gds])

    # JSON side (optional)
    json_ds = None
    if json_source and json_frac > 0:
        jds = build_json_dataset(json_source, split="train", max_rows=0)
        json_ds = Dataset.from_list(
            [{"prompt": r["prompt"], "task": "json",
              "gold": "", "constraints": [], "params": [],
              "fields": list(r["fields"]),
              "types": list(r["types"]),
              "strict": bool(r["strict"]),
              "passage": r["passage"],
              "gold_values": r["gold_values"]}
             for r in jds])

    # NER side (optional)
    ner_ds = None
    if ner_frac > 0:
        src = ner_source or "KennethEnevoldsen/dane_plus"
        if "ner-sft" in src:
            ner_ds = build_ner_sft_dataset(src, split="train", max_rows=0,
                                           seed=seed, tokenizer=tokenizer,
                                           max_prompt_tokens=max_prompt_tokens)
        else:
            ner_ds = build_ner_dataset(src, split="train", max_rows=0,
                                       empty_frac=ner_empty_frac, seed=seed)

    # Drop zero-weight sources rather than passing probability 0. Keeping them
    # forces if_share to a 1e-6 floor so the probabilities no longer sum to 1,
    # and it leaves an unused source in the interleave. Dropping them makes
    # any subset a valid mix (e.g. json+ner only).
    # ICL side (optional)
    icl_ds = None
    if icl_frac > 0:
        icl_ds = build_icl_dataset(
            icl_source or "jensjepsen/danish-icl-schema-format-v3",
            split="train", max_rows=0, seed=seed, tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens)

    if_share = 1.0 - gsm_frac - json_frac - ner_frac - icl_frac
    cand = [(if_ds, if_share, "if"), (gsm_ds, gsm_frac, "gsm"),
            (json_ds, json_frac, "json"), (ner_ds, ner_frac, "ner"),
            (icl_ds, icl_frac, "icl")]
    parts, probs, names = [], [], []
    for d, w, nm in cand:
        if d is not None and w > 1e-9:
            parts.append(d); probs.append(w); names.append(nm)
    assert parts, "all task fractions are zero — nothing to train on"
    tot = sum(probs)
    probs = [p / tot for p in probs]        # renormalise after dropping
    print(f"  [build_mixed] active sources: "
          f"{', '.join(f'{n}={w:.3f}' for n, w in zip(names, probs))}", flush=True)

    mixed = interleave_datasets(parts, probabilities=probs,
                                stopping_strategy=interleave_strategy,
                                seed=seed)
    if max_rows and len(mixed) > max_rows:
        mixed = mixed.select(range(max_rows))

    _sz = lambda d: len(d) if d is not None else 0   # sources may be skipped
    print(f"  [build_mixed:interleave={interleave_strategy}] "
          f"if_ds={_sz(if_ds)} gsm_ds={_sz(gsm_ds)} "
          f"json_ds={_sz(json_ds)} ner_ds={_sz(ner_ds)} "
          f"→ mixed={len(mixed)} (probs={[round(p,3) for p in probs]})",
          flush=True)
    return mixed


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
    ap.add_argument("--task", choices=["gsm8k", "ifeval", "combined", "mixed", "json", "ner", "icl"],
                    required=True)
    ap.add_argument("--icl-source",
                    default="jensjepsen/danish-icl-schema-format-v3",
                    help="Source for --icl-frac / --task=icl.")
    ap.add_argument("--icl-frac", type=float, default=0.0,
                    help="Share of the mixed dataset drawn from the ICL "
                         "schema/format set. Verified by reward_icl.")
    ap.add_argument("--json-source", default="jensjepsen/danish-json-grpo-v1",
                    help="Source for --task=json (HF repo or local path).")
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
    ap.add_argument("--json-frac", type=float, default=0.0,
                    help="For --task=mixed: fraction of json rows in the "
                         "interleaved mix (0 = no json). --json-source is "
                         "used as the row source.")
    ap.add_argument("--ner-frac", type=float, default=0.0,
                    help="For --task=mixed: fraction of NER rows in the "
                         "interleaved mix (0 = no NER).")
    ap.add_argument("--ner-source", default="jensjepsen/danish-ner-sft-v1",
                    help="Source for NER rows. A repo whose name contains "
                         "'ner-sft' is read as the SFT set (3 prompt modes, "
                         "14 formats, gold verified to strip back to its own "
                         "passage) and scored by reward_structured; anything "
                         "else is read as the dane_plus schema (text + ents "
                         "with char offsets) and keeps the legacy JSON-only "
                         "verifier.")
    ap.add_argument("--ner-empty-frac", type=float, default=0.28,
                    help="Share of entity-FREE sentences in the NER split. "
                         "Natural rate is ~53%%, but empty-on-empty scores 1.0, "
                         "so at the natural rate an always-abstain policy earns "
                         "~0.53 mean reward. Negative = keep natural rate.")
    ap.add_argument("--interleave-strategy",
                    choices=["first_exhausted", "all_exhausted"],
                    default="all_exhausted",
                    help="For --task=mixed: HF datasets.interleave_datasets "
                         "stopping strategy. 'all_exhausted' preserves ALL "
                         "rows by cycling small sources; 'first_exhausted' "
                         "stops when the smallest source runs out (throws "
                         "away the excess from larger sources).")
    ap.add_argument("--greedy-eval-task",
                    choices=["auto", "ifeval-da", "same", "both", "json", "all3",
                             "ner", "all4"],
                    default="auto",
                    help="Which dataset the greedy-eval callback uses. "
                         "'same' = same as --task's train dataset (default for "
                         "gsm8k/ifeval). 'ifeval-da' = the 541-row benchmark "
                         "(default for combined). 'both' = ifeval-da + gsm8k "
                         "(default for --task=mixed without json). 'all3' = "
                         "ifeval-da + gsm8k + json (recommended for mixed with "
                         "--json-frac > 0). 'json' = 200-row json eval split "
                         "(default for --task=json). 'auto' picks per --task.")
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
    ap.add_argument("--beta", type=float, default=0.004,
                    help="KL coefficient vs reference policy. Empirically "
                         "beta=0.004 dominates the TRL default 0.04 by "
                         "+3-5pp per eval on Danish IF+GSM8K mixed3 runs "
                         "(memory: grpo-low-beta-and-fresh-optim). Bump "
                         "back to 0.04 if you want tighter policy anchor.")
    ap.add_argument("--adam-beta1", type=float, default=0.9,
                    help="AdamW β1 (first-moment EMA). Default 0.9. Lower "
                         "for more responsive momentum to current gradients.")
    ap.add_argument("--adam-beta2", type=float, default=0.999,
                    help="AdamW β2 (second-moment EMA). Default 0.999. "
                         "Lower (e.g. 0.99, 0.95) for shorter RMS window "
                         "→ more local per-param adaptation. Expect "
                         "temporary effective-LR bump for ~100 steps on "
                         "mid-run change.")
    ap.add_argument("--adam-epsilon", type=float, default=1e-8,
                    help="AdamW ε (denominator floor). Default 1e-8. "
                         "Larger (e.g. 1e-6) softens per-param LR "
                         "amplification for small-gradient params.")
    ap.add_argument("--loss-type", choices=["grpo", "bnpo", "dr_grpo"],
                    default="dr_grpo",
                    help="GRPO loss reduction. 'dr_grpo' (default): sum-"
                         "tokens/(bs*max_len) — length-weighted AND ga-"
                         "invariant, matches gold's bnpo bs=32/ga=1 gradient "
                         "direction. 'grpo': per-sample-mean then batch-mean "
                         "(sample-weighted, ga-invariant, ignores length). "
                         "'bnpo': sum-tokens/sum-mask per microbatch (length-"
                         "weighted BUT drifts under ga>1 vs ga=1). Empirical "
                         "on IF-only from-v31: dr_grpo pulled ~2-4pp ahead of "
                         "grpo on IF after ~2000 steps.")
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
    ap.add_argument("--greedy-eval-batch-size", type=int, default=128,
                    help="Batch size for greedy-eval callbacks. Decoupled "
                         "from --batch-size (train rollout) since eval has "
                         "no gradients + no rollout expansion → much less "
                         "VRAM per example. Default 128 = ~4x train batch.")
    ap.add_argument("--skip-zero-adv", action="store_true",
                    help="Zero out completion_mask for groups where all "
                         "rollouts scored the same reward (std==0 → "
                         "advantage==0). Excludes them cleanly from the "
                         "loss + KL + optimizer noise. Doesn't save fwd/bwd "
                         "compute (TRL still generates them) but avoids the "
                         "noise-only Adam step. Ported from train_grpo.py.")
    ap.add_argument("--use-vllm-server", action="store_true",
                    help="Use vLLM for rollouts (huge speedup, ~10-20× rollout "
                         "throughput). Requires TRL 0.18+. See --vllm-mode.")
    ap.add_argument("--vllm-mode", choices=["server", "colocate"],
                    default="server",
                    help="'server' — connect to a separately-running "
                         "`trl vllm-serve` (see launch_grpo_vllm.sh, 2-GPU). "
                         "'colocate' — run vLLM inside the trainer process, "
                         "same GPU (see launch_grpo_vllm_h100.sh, single H100).")
    ap.add_argument("--vllm-host", default="localhost")
    ap.add_argument("--vllm-port", type=int, default=8000)
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.4,
                    help="GPU mem fraction for vLLM (colocate mode: fraction "
                         "of the shared GPU that vLLM pre-allocates; leave the "
                         "rest for training).")
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
    ap.add_argument("--lr-scheduler-type", default="constant_with_warmup",
                    help="HF scheduler name — 'constant_with_warmup' (default), "
                         "'cosine', 'cosine_with_restarts', 'linear', 'polynomial', etc.")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Cap training at N optimizer steps (overrides --epochs). "
                         "Required for 'cosine'/'linear' schedulers so they know "
                         "the horizon to anneal to 0 over.")
    ap.add_argument("--reset-scheduler", action="store_true",
                    help="With --resume: keep optimizer.pt (Adam moments) but "
                         "wipe scheduler.pt AND reset trainer_state.global_step "
                         "to 0 so LR warmup fires fresh. Isolates warmup as the "
                         "restart-mechanism to test. NOTE: resets epoch counter "
                         "and step-linked callbacks — treat as a fresh run "
                         "for wandb purposes.")
    ap.add_argument("--ref-anchor-checkpoint", default=None,
                    help="Override the KL reference model. Accepts an HF repo id "
                         "(e.g. jensjepsen/danish-lm-400m-sft-v31-avg-top3) or a "
                         "local ckpt dir. TRL normally clones ref_model from the "
                         "policy at init; this swaps it AFTER trainer construction "
                         "with the specified weights. Isolates reference-anchor "
                         "from resume mechanics (fresh optim vs loaded optim).")
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
    elif args.task == "json":
        ds = build_json_dataset(args.json_source, split="train",
                                max_rows=args.max_rows or 0)
        from esperanto_lm.rl_rewards import reward_mixed as _reward_mixed
        reward_fn = _reward_mixed
        # Held-out eval handled by JSON eval callback (or greedy-eval `--greedy-eval-task json`).
    elif args.task == "ner":
        # build_ner_dataset already emits task="ner" plus the full union
        # schema, so reward_mixed dispatches to reward_ner per row — same
        # pattern as the json single-task branch above.
        ds = build_ner_dataset(args.ner_source, split="train",
                               max_rows=args.max_rows or 0,
                               empty_frac=args.ner_empty_frac)
        from esperanto_lm.rl_rewards import reward_mixed as _reward_mixed
        reward_fn = _reward_mixed
        # Held-out eval is the dane_plus dev split via --greedy-eval-task ner.
    elif args.task == "icl":
        ds = build_icl_dataset(args.icl_source, split="train",
                               max_rows=args.max_rows or 0)
        from esperanto_lm.rl_rewards import reward_mixed as _reward_mixed
        reward_fn = _reward_mixed
    else:  # mixed
        assert args.combined_source, "--combined-source required for --task=mixed"
        ds = build_mixed_dataset(args.combined_source,
                                 max_rows=args.max_rows or 0,
                                 gsm_frac=args.gsm_frac,
                                 json_source=(args.json_source
                                              if args.json_frac > 0 else None),
                                 json_frac=args.json_frac,
                                 ner_source=(args.ner_source
                                             if args.ner_frac > 0 else None),
                                 ner_frac=args.ner_frac,
                                 ner_empty_frac=args.ner_empty_frac,
                                 icl_source=(args.icl_source
                                             if args.icl_frac > 0 else None),
                                 icl_frac=args.icl_frac,
                                 tokenizer=tok,
                                 max_prompt_tokens=args.max_prompt_length,
                                 interleave_strategy=args.interleave_strategy)
        reward_fn = reward_mixed
        # Greedy-eval callback attaches ifeval-da, gsm8k, and optionally json below.
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
        # TRL 1.x dropped max_prompt_length; use vllm_max_model_length to
        # cap prompt+completion combined (only relevant when use_vllm=True).
        max_completion_length=args.max_completion_length,
        vllm_max_model_length=(
            args.max_prompt_length + args.max_completion_length
            if args.use_vllm_server else None
        ),
        # TRL 1.10 forwards `generation_kwargs` to vLLM's SamplingParams.
        # Mirror HF's stop semantics: HF's GenerationConfig(eos_token_id=...)
        # takes a single int, and we swap tokenizer.eos_token_id to <|end|>
        # (16002) above. Pass the same single stop to vLLM so both paths
        # terminate on the same criterion — otherwise A/B comparisons are
        # skewed by asymmetric stopping. (Prior monkey-patch injected both
        # <|end|> AND <|user|> into vLLM only, unfair.)
        generation_kwargs=(
            {"stop_token_ids": [tok.eos_token_id]}
            if (args.use_vllm_server and tok.eos_token_id is not None)
            else None
        ),
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        lr_scheduler_type=args.lr_scheduler_type,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps or None,
        per_device_eval_batch_size=args.batch_size,
        save_total_limit=2,
        report_to=["wandb"],
        run_name=args.wandb_run_name or f"grpo_{args.task}",
        # BF16 by default; GRPO_FP16_EVERYWHERE=1 flips trainer to fp16
        # autocast + forces vLLM to fp16 (via the monkey-patch at the top
        # of this file) to minimize training-inference mismatch (Wu et al.
        # 2025). Both paths: load model in fp32, use --bf16 or --fp16
        # autocast for compute, keep fp32 master weights (avoids bf16-
        # weight rounding on tiny GRPO updates). Matches train_sft.py.
        bf16=(_os.environ.get("GRPO_FP16_EVERYWHERE") != "1"),
        fp16=(_os.environ.get("GRPO_FP16_EVERYWHERE") == "1"),
        optim="adamw_bnb_8bit",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        # TRL 1.x bump on torch 2.10 → no flash-attn prebuilt wheel (would
        # source-compile ~2h). Fall back to SDPA (still fast enough for
        # 400M models). Revisit if we can pin torch back to 2.8 with a
        # vllm range that supports it.
        model_init_kwargs={"attn_implementation": "sdpa"},
        # loss_type via --loss-type CLI. Default 'dr_grpo' (length-weighted,
        # ga-invariant, matches gold's bnpo bs=32/ga=1 gradient direction).
        # 'grpo' is sample-weighted; 'bnpo' drifts under ga>1. See --help.
        loss_type=args.loss_type,
        remove_unused_columns=False,
        # Prefetch next batch on worker threads so the rollout+reward step
        # isn't gated on main-thread data prep (tokenize + collate).
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        use_vllm=args.use_vllm_server,
        vllm_mode=args.vllm_mode,
        vllm_server_host=args.vllm_host,
        vllm_server_port=args.vllm_port,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    print(f"[optim] adamw_bnb_8bit  β1={cfg.adam_beta1}  β2={cfg.adam_beta2}  "
          f"ε={cfg.adam_epsilon}  weight_decay={cfg.weight_decay}  "
          f"max_grad_norm={cfg.max_grad_norm}", flush=True)

    print(f"loading model {args.checkpoint}...", flush=True)
    if args.skip_zero_adv:
        # Zero-out completion_mask for groups where reward.std()==0 (all
        # rollouts scored the same → advantage=0 → no gradient signal).
        # Excludes them from loss + KL + Adam. Doesn't save fwd/bwd compute
        # (TRL still generates them) but avoids the noise-only optimizer step.
        #
        # TRL 0.18.2 note: MUST patch _generate_and_score_completions (not
        # _prepare_inputs) because the latter is called on per-slice batches
        # AFTER `steps_per_generation` has split the full generation batch —
        # by then group-level advantages are no longer visible per-group in
        # each slice. _generate_and_score_completions is the point where the
        # full generation-batch's advantages are computed, matching TRL 0.16's
        # patch semantics.
        import torch as _t
        _orig_gen_score = GRPOTrainer._generate_and_score_completions

        def _log_per_task(self, mode, adv_g, n_gen):
            """Bucket the current batch's groups by task and push per-task
            reward / advantage / zero-std stats. Uses reward_mixed's stash of
            per-example (task, reward). Groups are pure-task (num_gen
            completions per prompt), so a group's task is unambiguous."""
            from esperanto_lm import rl_rewards as _rr
            tasks = _rr.LAST_MIXED_TASKS
            rewards = _rr.LAST_MIXED_REWARDS
            if not tasks or not rewards or len(tasks) != len(rewards):
                return
            n_local = adv_g.shape[0] * adv_g.shape[1]
            if len(tasks) < n_local:
                return
            import torch as _t2
            tasks = tasks[:n_local]
            rewards = rewards[:n_local]
            # Groups are contiguous num_gen slabs per prompt. Task of a group
            # is the task of any of its samples (all identical).
            n_groups = adv_g.shape[0]
            group_tasks = [tasks[g * n_gen] for g in range(n_groups)]
            r_t = _t2.tensor(rewards, device=adv_g.device, dtype=_t2.float32)
            r_g = r_t[:n_groups * n_gen].view(n_groups, n_gen)
            for tname in set(group_tasks):
                idx = [g for g, t in enumerate(group_tasks) if t == tname]
                if not idx:
                    continue
                r_sel = r_g[idx]
                a_sel = adv_g[idx]
                rsd = r_sel.std(dim=1)
                fzs = (rsd < 1e-6).float().mean().item()
                self._metrics[mode].setdefault(f"rewards/{tname}/mean", []).append(r_sel.mean().item())
                self._metrics[mode].setdefault(f"rewards/{tname}/std", []).append(rsd.mean().item())
                self._metrics[mode].setdefault(f"rewards/{tname}/fzs", []).append(fzs)
                self._metrics[mode].setdefault(f"advantages/{tname}/absmean", []).append(a_sel.abs().mean().item())
                self._metrics[mode].setdefault(f"advantages/{tname}/std", []).append(a_sel.std().item())
                self._metrics[mode].setdefault(f"count/{tname}", []).append(float(len(idx)))

        def _gen_score_with_skip(self, inputs):
            result = _orig_gen_score(self, inputs)
            adv = result.get("advantages")
            cm = result.get("completion_mask")
            mode = "train" if self.model.training else "eval"
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
            _log_per_task(self, mode, adv_g, n_gen)
            if bool(active.all()) or not bool(active.any()):
                return result
            sample_mask = active.repeat_interleave(n_gen).to(adv.device)
            if sample_mask.numel() < n_local:
                pad = _t.zeros(n_local - sample_mask.numel(),
                               dtype=_t.bool, device=adv.device)
                sample_mask = _t.cat([sample_mask, pad])
            result["completion_mask"] = cm * sample_mask.to(cm.dtype).unsqueeze(1)
            return result

        GRPOTrainer._generate_and_score_completions = _gen_score_with_skip
        print("skip-zero-adv: enabled (masking completion_mask on zero-std "
              "groups at _generate_and_score_completions; TRL 0.18.2 compat)",
              flush=True)

    trainer = GRPOTrainer(
        model=args.checkpoint,
        processing_class=tok,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=ds,
        eval_dataset=eval_ds,
    )

    # Post-load sanity: confirm attn_implementation actually engaged. Under
    # fp32 model load + FA2 request, HF may silently fall back to SDPA with
    # only a warning ("Flash Attention 2 only supports fp16/bf16 dtypes").
    _pm = trainer.model
    _attn_cfg = getattr(_pm.config, "_attn_implementation", None) or \
                getattr(_pm.config, "attn_implementation", None)
    try:
        _first_attn = _pm.model.layers[0].self_attn
        _attn_class = type(_first_attn).__name__
    except AttributeError:
        _attn_class = "unknown"
    print(f"[attn] config._attn_implementation={_attn_cfg!r}  "
          f"first_layer={_attn_class}  model_param_dtype={next(_pm.parameters()).dtype}",
          flush=True)

    # Swap ref_model (for KL penalty) with a different anchor. Runs AFTER
    # trainer __init__ (which does the default create_reference_model(model)
    # + accelerator.prepare_model), so we tear down and rebuild in place.
    if args.ref_anchor_checkpoint:
        if args.beta <= 0:
            print("[ref-anchor] --beta is 0; ref_model unused — skipping swap.",
                  flush=True)
        else:
            print(f"[ref-anchor] loading {args.ref_anchor_checkpoint} as KL ref",
                  flush=True)
            _ref = AutoModelForCausalLM.from_pretrained(
                args.ref_anchor_checkpoint,
                torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            )
            _ref.eval()
            for _p in _ref.parameters():
                _p.requires_grad_(False)
            _ref = trainer.accelerator.prepare_model(_ref, evaluation_mode=True)
            # Free the old ref_model first (TRL default clone of policy weights).
            _old = trainer.ref_model
            trainer.ref_model = _ref
            del _old
            torch.cuda.empty_cache()
            print(f"[ref-anchor] swapped: trainer.ref_model = "
                  f"{args.ref_anchor_checkpoint}", flush=True)

    # Rebuild trainer.generation_config with the full chat_stops list so
    # rollouts stop on any of <|end|> / <|user|>. TRL only puts a single
    # eos_token_id in the GenerationConfig by default.
    # Only applies when vLLM is OFF — with vLLM, generation goes through
    # the vLLM engine (which uses its own SamplingParams), and TRL 0.18.2
    # skips the HF GenerationConfig setup entirely.
    if not args.use_vllm_server and len(chat_stops) > 1:
        from transformers import GenerationConfig
        gc = trainer.generation_config
        cfg_gen = gc.to_dict()
        cfg_gen["eos_token_id"] = chat_stops
        trainer.generation_config = GenerationConfig(**cfg_gen)
    if not args.use_vllm_server:
        print(f"trainer.generation_config.eos_token_id = "
              f"{trainer.generation_config.eos_token_id}", flush=True)
    else:
        # vLLM: stop tokens go into SamplingParams via the vLLM engine.
        # TRL 0.18.2 uses tokenizer.eos_token_id for stop when constructing
        # its internal SamplingParams. Extra stops (<|end|>) can be added
        # via GRPOConfig.vllm_guided_decoding_regex or a custom subclass;
        # for now the tokenizer's eos handles the common case.
        print(f"vLLM colocate: stop-token = tokenizer.eos_token_id "
              f"(chat_stops={chat_stops} for reference)", flush=True)
    if args.greedy_eval_steps > 0:
        # Resolve which callbacks to attach
        eval_task = args.greedy_eval_task
        if eval_task == "auto":
            if args.task == "combined":
                eval_task = "ifeval-da"
            elif args.task == "mixed":
                eval_task = ("all4" if args.ner_frac > 0 and args.json_frac > 0
                             else "all3" if args.json_frac > 0
                             else "both")
            elif args.task == "json":
                eval_task = "json"
            elif args.task == "ner":
                eval_task = "ner"
            else:
                eval_task = "same"

        def _attach_ifeval_da():
            cb = IFEvalDACallback(
                tokenizer=tok,
                every_n_steps=args.greedy_eval_steps,
                max_new_tokens=args.max_completion_length,
                batch_size=args.greedy_eval_batch_size,
            )
            cb._trainer = trainer  # so callback can call trainer.log()
            trainer.add_callback(cb)

        def _attach_gsm8k():
            # Always full test set (1317 rows). The first-N cap left over from
            # `--greedy-eval-max-rows` is biased — the first ~200 rows of
            # danish-gsm8k:test are systematically easier (translation preserves
            # the original GSM8K order), so a 200-row subset overreports by ~5pp.
            gds = build_gsm8k_dataset("test", max_rows=0)
            items = [(r["prompt"], r["gold"]) for r in gds]
            cb = GreedyEvalCallback(
                tokenizer=tok, items=items, task="gsm8k",
                every_n_steps=args.greedy_eval_steps,
                max_new_tokens=args.max_completion_length,
                batch_size=args.greedy_eval_batch_size,
            )
            cb._trainer = trainer  # direct trainer.log() bypasses on_log
            trainer.add_callback(cb)

        def _attach_json():
            jds = build_json_dataset(args.json_source, split="eval",
                                     max_rows=args.greedy_eval_max_rows)
            def _decode_gold(g):
                if not g:
                    return None
                if isinstance(g, dict):
                    return g
                try:
                    return json.loads(g)
                except (TypeError, ValueError):
                    return None
            j_items = [(r["prompt"], r["fields"], r["types"],
                        bool(r["strict"]), r.get("passage") or "",
                        _decode_gold(r.get("gold_values")))
                       for r in jds]
            cb = GreedyEvalCallback(
                tokenizer=tok, items=j_items, task="json",
                every_n_steps=args.greedy_eval_steps,
                max_new_tokens=args.max_completion_length,
                batch_size=args.greedy_eval_batch_size,
            )
            cb._trainer = trainer
            trainer.add_callback(cb)

        def _attach_ner():
            # The SFT set has no `dev` split (train/val/eval/eval_format) and
            # its rows are format-tagged, so it needs both the other builder
            # and the other scorer. Missing this dispatch crashed the first
            # real launch after the training path had already been switched.
            if "ner-sft" in (args.ner_source or ""):
                nds = build_ner_sft_dataset(args.ner_source, split="val",
                                            max_rows=args.greedy_eval_max_rows,
                                            tokenizer=tok,
                                            max_prompt_tokens=args.max_prompt_length)
                n_items = [(r["prompt"], r["gold"], r["fields"],
                            r["types"][0], r["passage"]) for r in nds]
                n_task = "struct"
            else:
                # dev split — train is used for rollouts, test stays held out
                nds = build_ner_dataset(args.ner_source, split="dev",
                                        max_rows=args.greedy_eval_max_rows,
                                        empty_frac=-1.0)
                n_items = [(r["prompt"], r["gold_values"]) for r in nds]
                n_task = "ner"
            cb = GreedyEvalCallback(
                tokenizer=tok, items=n_items, task=n_task,
                metric_name=("eval_greedy_ner_reward_mean"
                             if n_task == "struct" else None),
                every_n_steps=args.greedy_eval_steps,
                max_new_tokens=args.max_completion_length,
                batch_size=args.greedy_eval_batch_size,
            )
            cb._trainer = trainer
            trainer.add_callback(cb)

        def _attach_icl():
            # eval_schema: unseen schemas, seen formats. Never trained on --- the
            # GRPO rows come from the train split.
            ids_ = build_icl_dataset(args.icl_source, split="eval_schema",
                                     max_rows=args.greedy_eval_max_rows,
                                     tokenizer=tok,
                                     max_prompt_tokens=args.max_prompt_length)
            i_items = [(r["prompt"], r["gold"], r["fields"], r["types"][0], None)
                       for r in ids_]
            cb = GreedyEvalCallback(
                tokenizer=tok, items=i_items, task="struct",
                metric_name="eval_greedy_icl_reward_mean",
                every_n_steps=args.greedy_eval_steps,
                max_new_tokens=args.max_completion_length,
                batch_size=args.greedy_eval_batch_size,
            )
            cb._trainer = trainer
            trainer.add_callback(cb)

        if eval_task == "ifeval-da":
            _attach_ifeval_da()
        elif eval_task == "both":
            _attach_ifeval_da()
            _attach_gsm8k()
        elif eval_task == "all3":
            _attach_ifeval_da()
            _attach_gsm8k()
            _attach_json()
        elif eval_task == "all4":
            _attach_ifeval_da()
            _attach_gsm8k()
            _attach_json()
            _attach_ner()
            if args.icl_frac > 0:
                _attach_icl()
        elif eval_task == "json":
            _attach_json()
        elif eval_task == "ner":
            _attach_ner()
        else:  # 'same' — task-specific single greedy callback
            if args.task == "gsm8k":
                _attach_gsm8k()
            elif args.task == "ner":
                _attach_ner()
            elif args.task == "combined":
                gds = build_combined_dataset(args.combined_source,
                                             max_rows=args.greedy_eval_max_rows)
                g_items = [(r["prompt"], r["constraints"], r["params"]) for r in gds]
                cb = GreedyEvalCallback(
                    tokenizer=tok, items=g_items, task="combined",
                    every_n_steps=args.greedy_eval_steps,
                    max_new_tokens=args.max_completion_length,
                    batch_size=args.greedy_eval_batch_size,
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
                    batch_size=args.greedy_eval_batch_size,
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

    # --reset-scheduler: pre-mutate the ckpt dir before Trainer picks it up.
    # Delete scheduler.pt (forces re-init from training_args) and rewrite
    # trainer_state.json to set global_step=0/epoch=0 so warmup counts from
    # scratch under constant_with_warmup. Optimizer.pt is preserved.
    if resume and args.reset_scheduler:
        import glob as _glob
        # Find the ckpt dir HF Trainer would resume from
        if resume is True:
            _cks = sorted(_glob.glob(f"{args.output_dir}/checkpoint-*"),
                          key=lambda p: int(p.rsplit("-", 1)[-1]))
            ckdir = _cks[-1] if _cks else None
        else:
            ckdir = str(resume)
        if not ckdir or not Path(ckdir).exists():
            raise SystemExit(f"--reset-scheduler: could not find ckpt dir "
                             f"(resume={resume!r})")
        sched_p = Path(ckdir) / "scheduler.pt"
        if sched_p.exists():
            sched_p.unlink()
            print(f"[reset-scheduler] removed {sched_p}", flush=True)
        state_p = Path(ckdir) / "trainer_state.json"
        if state_p.exists():
            st = json.loads(state_p.read_text())
            st["global_step"] = 0
            st["epoch"] = 0.0
            # Wipe the log history too so wandb doesn't try to backfill.
            st["log_history"] = []
            state_p.write_text(json.dumps(st, indent=2))
            print(f"[reset-scheduler] zeroed global_step + epoch in {state_p}",
                  flush=True)

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
