"""Batched generation + continuation scoring for the per-task eval scripts.

Written once here rather than three times in eval_arc_da / eval_gpqa_da /
eval_piqa_da. Those scored one item at a time: arc_easy alone is 2,376 rows,
and in raw-logp mode each row scores 4 options, so a single eval was ~9,500
sequential forward passes with the GPU almost idle. That made the per-task
scripts too slow to use in a full suite, which is why the suite reached for the
training callback instead -- and then the numbers no longer matched the harness
the model cards were produced with.

Batching must be numerically equivalent to the per-item version or the scores
move, so the padding side differs by task:

  generate_batch  pads LEFT. A right-padded prompt puts PAD immediately before
                  the first generated token, so the model conditions on padding
                  and the continuation changes.
  score_cont_batch pads RIGHT and masks. Each row's continuation log-probs are
                  read at that row's own prompt length and pad positions are
                  excluded from both the sum and the length normalisation.

Both restore the tokenizer's padding_side/pad_token, since callers share one
tokenizer across modes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _with_pad(tok):
    prev = (tok.padding_side, tok.pad_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return prev


def generate_batch(model, tok, prompts, max_new, eos_ids, bs=64,
                   progress=None, **gen_kw):
    """Greedy-generate each prompt; returns decoded continuations in order."""
    prev = _with_pad(tok)
    tok.padding_side = "left"
    outs: list[str] = []
    try:
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False,
                      return_token_type_ids=False).to(model.device)
            with torch.no_grad():
                g = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=max_new, do_sample=False, num_beams=1,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    eos_token_id=eos_ids, **gen_kw)
            plen = enc["input_ids"].shape[1]
            outs += [tok.decode(r[plen:], skip_special_tokens=True).strip()
                     for r in g]
            if progress:
                progress(len(outs), len(prompts))
    finally:
        tok.padding_side, tok.pad_token = prev
    return outs


def score_cont_batch(model, tok, pairs, bs=32, progress=None):
    """Length-normalized log P(cont | prompt) for each (prompt, cont) pair.

    Equivalent to scoring one at a time: position j of the shifted log-probs
    predicts token j+1, so a continuation starting at prompt length L is read
    from j >= L-1, and only positions inside the attention mask count.
    """
    prev = _with_pad(tok)
    tok.padding_side = "right"
    out: list[float] = []
    try:
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            p_lens = [len(tok(p, add_special_tokens=False)["input_ids"])
                      for p, _ in chunk]
            enc = tok([p + c for p, c in chunk], return_tensors="pt",
                      padding=True, add_special_tokens=False,
                      return_token_type_ids=False).to(model.device)
            ids, mask = enc["input_ids"], enc["attention_mask"]
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=mask).logits
            lp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
            tgt = ids[:, 1:]
            tok_lp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            L = torch.tensor(p_lens, device=ids.device).unsqueeze(1)
            pos = torch.arange(tgt.shape[1], device=ids.device).unsqueeze(0)
            keep = mask[:, 1:].bool() & (pos >= L - 1)
            n = keep.sum(1)
            s = (tok_lp * keep).sum(1)
            out += [(s[k] / n[k]).item() if n[k] > 0 else -float("inf")
                    for k in range(len(chunk))]
            if progress:
                progress(len(out), len(pairs))
    finally:
        tok.padding_side, tok.pad_token = prev
    return out


def ratchet(label):
    """Progress callback that prints every batch -- these evals run for
    minutes and a silent script is indistinguishable from a hung one."""
    import time
    t0 = time.time()

    def _p(done, total):
        el = time.time() - t0
        eta = el * (total - done) / done if done else 0
        print(f"  [{label}] {done}/{total}  eta={eta:.0f}s", flush=True)
    return _p
