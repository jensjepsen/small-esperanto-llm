"""TrainerCallback that reports per-position-half NLL on every eval step.

Purpose: during a RoPE-extension continued pretrain (or any long-context
adaptation), we need to see BOTH numbers evolve live:

  - eval/short_nll : mean NLL over positions [0, short_len)
                     — should stay flat (~= baseline) or improve slightly.
                       Growing = catastrophic forgetting on trained range.
  - eval/long_nll  : mean NLL over positions [short_len, max_len)
                     — should DROP over training as the extended positions
                       learn to attend properly. A flat/rising curve means
                       the extension isn't taking.
  - eval/long_short_ratio : long_nll / short_nll. Target ≤ 1.15 by end of run.

Held-out data is a small fixed subset of Danish text (default: streamed
from `jensjepsen/danish-pretrain`), tokenized once to `max_len`, cached to
disk so restarts don't re-fetch. Only long-enough docs are kept.

Cost: N docs × max_len tokens per eval. At 32 × 2048 on a 400M model, one
eval pass is a few seconds — cheap enough to run on every eval step.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import TrainerCallback


class LongShortPerplexityCallback(TrainerCallback):
    def __init__(self, tokenizer, cache_dir: str,
                 n_docs: int = 128,
                 max_len: int = 2048,
                 short_len: int = 512,
                 batch_size: int = 4,
                 dataset_name: str = "jensjepsen/danish-pretrain",
                 dataset_split: str = "train",
                 text_field: str = "text",
                 min_char_len: int = 8000,
                 stream_skip: int = 500_000,
                 seed: int = 42):
        """
        cache_dir: where to persist the tokenized held-out set (survives restarts).
        n_docs: number of eval docs; each is exactly max_len tokens.
        stream_skip: how many rows to skip in the source stream before starting
            collection — for danish-pretrain, ensures we don't accidentally hit
            the same distributional slice as early training shards.
        """
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir)
        self.n_docs = n_docs
        self.max_len = max_len
        self.short_len = short_len
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.text_field = text_field
        self.min_char_len = min_char_len
        self.stream_skip = stream_skip
        self.seed = seed
        self._eval_ids: torch.Tensor | None = None  # [N, max_len] long tensor on CPU

    # ── data preparation ─────────────────────────────────────────────────

    def _cache_path(self) -> Path:
        # Include tokenizer name + params in the cache key so different runs
        # don't collide.
        tok_name = getattr(self.tokenizer, "name_or_path", "unknown").replace("/", "_")
        key = f"{tok_name}_n{self.n_docs}_L{self.max_len}_s{self.stream_skip}_seed{self.seed}"
        return self.cache_dir / f"long_short_eval_{key}.pt"

    def _prepare_eval_ids(self):
        if self._eval_ids is not None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = self._cache_path()
        if cache.exists():
            self._eval_ids = torch.load(cache, weights_only=True)
            print(f"[long_short_eval] loaded {self._eval_ids.shape[0]} cached "
                  f"eval docs from {cache}", flush=True)
            return

        # Fetch via streaming so we don't download all 111GB.
        from datasets import load_dataset
        print(f"[long_short_eval] streaming {self.dataset_name} to collect "
              f"{self.n_docs} docs of ≥{self.max_len} tokens…", flush=True)
        stream = load_dataset(self.dataset_name, split=self.dataset_split,
                              streaming=True)
        # Skip early rows to reduce overlap with pretrained shards.
        if self.stream_skip:
            stream = stream.skip(self.stream_skip)

        collected: list[list[int]] = []
        for row in stream:
            txt = row.get(self.text_field, "") or ""
            if len(txt) < self.min_char_len:
                continue
            ids = self.tokenizer(txt, add_special_tokens=False)["input_ids"]
            if len(ids) < self.max_len:
                continue
            collected.append(ids[:self.max_len])
            if len(collected) >= self.n_docs:
                break

        if len(collected) < self.n_docs:
            print(f"[long_short_eval] WARN: only found {len(collected)} "
                  f"long-enough docs (wanted {self.n_docs})", flush=True)

        self._eval_ids = torch.tensor(collected, dtype=torch.long)
        torch.save(self._eval_ids, cache)
        print(f"[long_short_eval] cached {self._eval_ids.shape[0]} docs to {cache}",
              flush=True)

    # ── the eval itself ──────────────────────────────────────────────────

    @torch.no_grad()
    def _measure(self, model) -> dict[str, float]:
        self._prepare_eval_ids()
        assert self._eval_ids is not None and self._eval_ids.numel() > 0

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        short_losses: list[torch.Tensor] = []
        long_losses: list[torch.Tensor] = []

        for i in range(0, self._eval_ids.shape[0], self.batch_size):
            batch = self._eval_ids[i:i + self.batch_size].to(device)
            logits = model(input_ids=batch).logits  # [B, L, V]
            shift_logits = logits[:, :-1, :]        # predict positions [1:]
            shift_labels = batch[:, 1:]
            per_tok = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
            ).view(batch.size(0), -1)               # [B, L-1]

            # Halves in the LABEL space (positions 1..max_len-1 are label
            # positions). "short" = predicting labels at positions 1..short_len-1;
            # "long" = predicting labels at positions short_len..max_len-1.
            short_losses.append(per_tok[:, : self.short_len - 1].reshape(-1))
            long_losses.append(per_tok[:, self.short_len - 1:].reshape(-1))

        if was_training:
            model.train()

        short_nll = torch.cat(short_losses).mean().item()
        long_nll = torch.cat(long_losses).mean().item()
        return {
            "eval/short_nll": short_nll,
            "eval/long_nll": long_nll,
            "eval/long_short_ratio": long_nll / max(short_nll, 1e-9),
            "eval/short_ppl": float(torch.exp(torch.tensor(short_nll))),
            "eval/long_ppl": float(torch.exp(torch.tensor(long_nll))),
        }

    # ── callback hook ────────────────────────────────────────────────────

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kw):
        if model is None:
            return
        try:
            new_metrics = self._measure(model)
        except Exception as e:
            print(f"[long_short_eval] error: {type(e).__name__}: {e}", flush=True)
            return
        # HF's Trainer already logged `metrics` before calling this callback,
        # so injecting here is too late for the current step's log line. Log
        # explicitly so wandb picks it up under this step.
        try:
            from transformers.integrations import WandbCallback  # noqa
            import wandb
            if wandb.run is not None:
                wandb.log(new_metrics, step=state.global_step)
        except Exception:
            pass
        # Also mirror into the metrics dict for downstream consumers.
        if metrics is not None:
            metrics.update(new_metrics)
        s = new_metrics["eval/short_nll"]; l = new_metrics["eval/long_nll"]
        r = new_metrics["eval/long_short_ratio"]
        print(f"[long_short_eval] step={state.global_step}  "
              f"short_nll={s:.4f} (ppl={float(torch.exp(torch.tensor(s))):.2f})  "
              f"long_nll={l:.4f} (ppl={float(torch.exp(torch.tensor(l))):.2f})  "
              f"ratio={r:.3f}", flush=True)
