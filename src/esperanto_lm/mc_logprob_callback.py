"""TrainerCallback: MC-logprob scoring on SciQ-DA and Citizen-tests MC.

For each item, score each option as a length-normalized continuation of
"{question}\nSvar: {option}", pick argmax, compute accuracy. Reports:
  - eval/sciq_mc_logprob
  - eval/citmc_logprob

Items cached in memory on first call. Cheap on H100 (~5-15s per pass).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from transformers import TrainerCallback


def _score_option_batch(model, tok, prompt: str, options: list[str], batch_size: int = 8) -> list[float]:
    """Length-normalized log P(option | prompt) for each option."""
    device = model.device
    scores = []
    for i in range(0, len(options), batch_size):
        batch_opts = options[i:i + batch_size]
        prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=True,
                         return_token_type_ids=False).input_ids.to(device)
        for opt in batch_opts:
            full = prompt + opt
            full_ids = tok(full, return_tensors="pt", add_special_tokens=True,
                           return_token_type_ids=False).input_ids.to(device)
            p_len = prompt_ids.shape[1]
            n_cont = full_ids.shape[1] - p_len
            if n_cont <= 0:
                scores.append(float("-inf")); continue
            with torch.no_grad():
                logits = model(full_ids).logits
            cont_logits = logits[0, p_len - 1: -1, :].float()
            cont_targets = full_ids[0, p_len:]
            lp = F.log_softmax(cont_logits, dim=-1).gather(1, cont_targets.unsqueeze(1)).squeeze(1)
            scores.append(lp.sum().item() / n_cont)
    return scores


class MCLogprobCallback(TrainerCallback):
    def __init__(self, tokenizer, n_sciq: int = 200, n_citmc: int = 300):
        self.tok = tokenizer
        self.n_sciq = n_sciq
        self.n_citmc = n_citmc
        self._sciq = None
        self._citmc = None

    def _load_sciq(self):
        if self._sciq is not None: return
        from datasets import load_dataset
        ds = load_dataset("jensjepsen/danish-sciq", "default", split="test")
        ds = ds.select(range(min(self.n_sciq, len(ds))))
        items = []
        for r in ds:
            correct = r["da_correct_answer"]
            options = [correct, r["da_distractor1"], r["da_distractor2"], r["da_distractor3"]]
            items.append({
                "prompt": f"{r['da_question'].strip()}\nSvar: ",
                "options": options,
                "gold_idx": 0,  # correct is always index 0
            })
        self._sciq = items

    def _load_citmc(self):
        if self._citmc is not None: return
        from datasets import load_dataset
        ds = load_dataset("alexandrainst/danish-citizen-tests", split="train")
        items = []
        for r in ds:
            gold = r.get("answer")
            if not gold: continue
            gold = gold.upper()
            opts = {}
            for ll in "abcd":
                v = r.get(f"option_{ll}")
                if v: opts[ll.upper()] = v
            if len(opts) < 2 or gold not in opts: continue
            options = list(opts.values())
            gold_text = opts[gold]
            items.append({
                "prompt": f"{r['question'].strip()}\nSvar: ",
                "options": options,
                "gold_idx": options.index(gold_text),
            })
            if len(items) >= self.n_citmc: break
        self._citmc = items

    def _score_items(self, model, items) -> float:
        n_ok = 0
        for it in items:
            scores = _score_option_batch(model, self.tok, it["prompt"], it["options"])
            pred = max(range(len(scores)), key=lambda i: scores[i])
            if pred == it["gold_idx"]:
                n_ok += 1
        return n_ok / max(1, len(items))

    def on_evaluate(self, args, state, control, model=None, metrics=None,
                    **kwargs):
        if model is None:
            print("[mc-logprob] SKIP: model is None", flush=True); return
        was_training = model.training
        model.eval()
        try:
            self._load_sciq()
            self._load_citmc()
            sciq_acc = round(self._score_items(model, self._sciq), 4)
            cit_acc  = round(self._score_items(model, self._citmc), 4)
        except Exception as e:
            import traceback
            print(f"[mc-logprob] ERROR: {e}", flush=True)
            traceback.print_exc()
            return
        finally:
            if was_training: model.train()
        # Also mutate metrics dict (for callers that read it) but print explicitly
        # since HF Trainer already logged before on_evaluate runs.
        if metrics is not None:
            metrics["eval/sciq_mc_logprob"] = sciq_acc
            metrics["eval/citmc_logprob"] = cit_acc
        print(f"[mc-logprob] step={state.global_step}  "
              f"sciq_mc_logprob={sciq_acc:.4f}  citmc_logprob={cit_acc:.4f}",
              flush=True)
        # Push to wandb if active
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"eval/sciq_mc_logprob": sciq_acc,
                           "eval/citmc_logprob": cit_acc,
                           "train/global_step": state.global_step},
                          step=state.global_step)
        except Exception:
            pass
