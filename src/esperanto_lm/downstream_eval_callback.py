"""TrainerCallback that runs downstream generation evals during training.

Motivation: HF Trainer's eval_loss on the held-out mix has been shown to
mislead about downstream capability for our DA SFT setup (see memory
`project-v12-best-ckpt-selection`). Running actual downstream evals
during training gives an honest capability trajectory.

Runs on each `on_evaluate` step:
  - GSM8K greedy on danish-gsm8k:test (substring match on numeric answer)
  - SciQ open-Q on danish-sciq:test (substring match on gold answer)
  - Cit-gen on danish-citizen-tests (substring match on gold option)

Adds metrics to the eval-loss dict:
  eval_downstream_gsm8k   — greedy GSM accuracy
  eval_downstream_sciq    — SciQ open-Q accuracy
  eval_downstream_citgen  — citizen-tests generative accuracy

These show up in wandb and stdout alongside eval_loss. Overhead: ~1-2 min
per eval step with n=100 rows and bs=32 on a 5090.
"""
from __future__ import annotations

import re
import time
import unicodedata

import torch
from datasets import load_dataset
from transformers import TrainerCallback

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"

# ── metric helpers ──────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"####\s*(-?\d[\d,\.]*)")
_LAST_NUM_RE = re.compile(r"(-?\d[\d,]*\.?\d*)")

_DA_STOP = {"en", "et", "den", "det", "de", "at", "og", "i", "på", "af",
            "til", "for", "med", "er", "som", "der", "har"}


def _extract_num(text: str) -> str | None:
    m = _NUM_RE.search(text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = _LAST_NUM_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "").rstrip(".")
    return None


def _norm_num(s):
    if s is None:
        return None
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError, OverflowError):
        return s


def _norm_text(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(w for w in s.split() if w not in _DA_STOP)


def _matches_text(pred: str, gold: str) -> bool:
    np, ng = _norm_text(pred), _norm_text(gold)
    return bool(ng) and ng in np


# ── the callback ────────────────────────────────────────────────────────────

class DownstreamEvalCallback(TrainerCallback):
    """Runs downstream generation evals on every `on_evaluate` step and
    injects the accuracy metrics into the eval-metrics dict so HF Trainer
    logs them (and wandb picks them up)."""

    def __init__(self, tokenizer, evals=("gsm8k", "sciq", "citgen"),
                 n_per_eval=100, batch_size=32, max_new_gsm=300,
                 max_new_short=48, seed=42):
        self.tokenizer = tokenizer
        self.evals = tuple(evals)
        self.n = n_per_eval
        self.bs = batch_size
        self.max_new_gsm = max_new_gsm
        self.max_new_short = max_new_short
        self.seed = seed
        self._cache = {}  # eval_name → list of (prompt, gold) tuples
        self._end_id = tokenizer.convert_tokens_to_ids(END)

    # ── dataset loaders (called lazily on first eval) ──────────────────────

    def _load_gsm8k(self):
        ds = load_dataset("jensjepsen/danish-gsm8k", "sft", split="test")
        ds = ds.shuffle(seed=self.seed).select(range(min(self.n, len(ds))))
        items = []
        for r in ds:
            q = r["messages"][0]["content"]
            gold = _extract_num(r["messages"][1]["content"])
            items.append((q, gold))
        return items

    def _load_sciq(self):
        ds = load_dataset("jensjepsen/danish-sciq", "default", split="test")
        ds = ds.shuffle(seed=self.seed).select(range(min(self.n, len(ds))))
        return [(r["da_question"], r["da_correct_answer"]) for r in ds]

    def _load_citgen(self):
        ds = load_dataset("alexandrainst/danish-citizen-tests", split="train")
        ds = ds.shuffle(seed=self.seed).select(range(min(self.n, len(ds))))
        items = []
        for r in ds:
            gold_letter = r["answer"]
            gold_text = r.get(f"option_{gold_letter.lower()}")
            if not gold_text:
                continue
            items.append((r["question"], gold_text))
        return items

    def _get(self, name: str):
        if name not in self._cache:
            loader = getattr(self, f"_load_{name}")
            self._cache[name] = loader()
        return self._cache[name]

    # ── batched greedy generation ──────────────────────────────────────────

    def _generate(self, model, prompts: list[str], max_new: int) -> list[str]:
        tok = self.tokenizer
        # Save + set left padding for generation, restore after
        prev_side = tok.padding_side
        prev_pad = tok.pad_token
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        outs = []
        try:
            eos_ids = [tok.eos_token_id]
            if self._end_id is not None and self._end_id != tok.unk_token_id:
                eos_ids.append(self._end_id)
            for i in range(0, len(prompts), self.bs):
                batch = prompts[i:i + self.bs]
                enc = tok(batch, return_tensors="pt", padding=True,
                          add_special_tokens=False,
                          return_token_type_ids=False).to(model.device)
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        max_new_tokens=max_new, do_sample=False, num_beams=1,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id,
                        eos_token_id=eos_ids,
                        repetition_penalty=1.1,
                    )
                plen = enc["input_ids"].shape[1]
                for row in gen:
                    outs.append(tok.decode(row[plen:], skip_special_tokens=True).strip())
        finally:
            tok.padding_side = prev_side
            tok.pad_token = prev_pad
        return outs

    # ── per-eval scorers ───────────────────────────────────────────────────

    def _score_gsm8k(self, model) -> float:
        items = self._get("gsm8k")
        prompts = [f"{USER} {q} {ASST}" for q, _ in items]
        outs = self._generate(model, prompts, self.max_new_gsm)
        n_ok = sum(1 for out, (_, gold) in zip(outs, items)
                   if _norm_num(_extract_num(out)) == _norm_num(gold))
        return n_ok / len(items)

    def _score_sciq(self, model) -> float:
        items = self._get("sciq")
        prompts = [f"{USER}{q}{END}{ASST}" for q, _ in items]
        outs = self._generate(model, prompts, self.max_new_short)
        n_ok = sum(1 for out, (_, gold) in zip(outs, items)
                   if _matches_text(out, gold))
        return n_ok / len(items)

    def _score_citgen(self, model) -> float:
        items = self._get("citgen")
        prompts = [f"{USER}{q}{END}{ASST}" for q, _ in items]
        outs = self._generate(model, prompts, self.max_new_short)
        n_ok = sum(1 for out, (_, gold) in zip(outs, items)
                   if _matches_text(out, gold))
        return n_ok / len(items)

    # ── HF Trainer hook ────────────────────────────────────────────────────

    def on_evaluate(self, args, state, control, model=None, metrics=None,
                    **kwargs):
        if model is None:
            return control
        model.eval()
        t0 = time.time()
        downstream_metrics = {}
        for name in self.evals:
            score = getattr(self, f"_score_{name}")(model)
            key = f"eval_downstream_{name}"
            downstream_metrics[key] = score
            if metrics is not None:
                metrics[key] = score  # for HF logging on same-step
            print(f"  [downstream] {name}: {100*score:.1f}%", flush=True)
        elapsed = time.time() - t0
        print(f"  [downstream] {len(self.evals)} evals in {elapsed:.0f}s "
              f"(n={self.n} each, bs={self.bs})", flush=True)
        # Explicit wandb.log() — Trainer's built-in log already fired for
        # the eval metrics dict BEFORE on_evaluate ran, so mutating `metrics`
        # doesn't reach wandb. Push our downstream metrics directly at the
        # current global_step.
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(downstream_metrics, step=state.global_step)
        except ImportError:
            pass
        return control
