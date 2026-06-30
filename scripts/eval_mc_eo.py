"""Multiple-choice eval for SciQ-EO and COPA-EO via log-likelihood scoring.

For each item, build a prompt and score each candidate completion by
sum log-prob of its tokens (length-normalized by default). Highest = predicted.

Usage:
  eval_mc_eo.py <ckpt> [sciq|copa] [n] [--no-length-norm] [--no-support]
"""
import re
import sys
import argparse
import torch
import torch.nn.functional as F
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, "src")
from esperanto_lm.data import _morpheme_preprocess

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
SPECIAL = (USER, ASST, END)
SKIP = {"<s>", "</s>", "<pad>", "<unk>", USER, ASST, END}

# Set by main() based on CUDA availability.
_DEVICE = "cpu"


def pp(s):
    pat = "(" + "|".join(re.escape(t) for t in SPECIAL) + ")"
    return " ".join(p if p in SPECIAL else _morpheme_preprocess(p)
                    for p in re.split(pat, s))


def score_completion(model, tok, prompt: str, completion: str, length_norm=True):
    """Sum log P(completion tokens | prompt).  Optionally length-normalize."""
    full = pp(prompt + " " + completion)
    prefix = pp(prompt)
    full_ids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids.to(_DEVICE)
    prefix_ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(_DEVICE)
    n_pref = prefix_ids.shape[1]
    # the completion span in full_ids starts at n_pref
    with torch.no_grad():
        logits = model(full_ids).logits  # (1, T, V)
    # at position i we predict token i+1, so for tokens [n_pref..T-1]
    # we use logits at [n_pref-1..T-2]
    targets = full_ids[0, n_pref:]
    logits_for = logits[0, n_pref - 1: -1]
    logp = F.log_softmax(logits_for.float(), dim=-1)
    tok_logps = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    total = tok_logps.sum().item()
    n = tok_logps.shape[0]
    return total / n if length_norm and n > 0 else total


def eval_sciq(model, tok, n, use_support, length_norm, prompt_format="chat"):
    ds = load_dataset("jensjepsen/esperanto-sciq", split="test").select(range(n))
    rng = random.Random(42)
    n_ok = 0
    for i, row in enumerate(ds, 1):
        q = row["question"]
        correct = row["correct_answer"]
        opts = [correct] + list(row["distractors"])
        # shuffle deterministically
        order = list(range(4))
        rng.shuffle(order)
        opts = [opts[j] for j in order]
        gold_idx = order.index(0)

        if prompt_format == "demando":
            # Matches the "Demando: ... \nRespondo: ..." pretrain Q/A text
            # format (load_benchmark_qa_dataset). Use for base-LM eval.
            body = f"{row['support']}\n\n{q}" if use_support else q
            prompt_q = f"Demando: {body}\nRespondo:"
        else:
            # SFT chat-token wrapped prompt
            prompt_q = f"{USER} {q}"
            if use_support:
                prompt_q = f"{USER} {row['support']}\n\n{q}"
            prompt_q = f"{prompt_q} {ASST}"

        scores = [score_completion(model, tok, prompt_q, o, length_norm) for o in opts]
        pred = max(range(4), key=lambda k: scores[k])
        ok = pred == gold_idx
        n_ok += ok
        flag = "✓" if ok else "✗"
        print(f"[{i}/{n}] {flag} gold={gold_idx} pred={pred}  scores={[round(s,3) for s in scores]}", flush=True)
        if i <= 5:
            print(f"   Q: {q[:120]}")
            for k, o in enumerate(opts):
                mark = " ← gold" if k == gold_idx else (" ← pred" if k == pred else "")
                print(f"     [{k}] {o[:80]}{mark}")
    print(f"\n=== sciq {n_ok}/{n} = {100*n_ok/n:.1f}% ===")


def eval_copa(model, tok, n, length_norm, prompt_format="chat"):
    ds = load_dataset("jensjepsen/esperanto-balanced-copa", split="test").select(range(n))
    n_ok = 0
    for i, row in enumerate(ds, 1):
        premise = row["premise"]
        q = row["question_eo"]
        opts = [row["choice1"], row["choice2"]]
        gold_idx = int(row["label"])

        connector = "Ĉar" if q == "kaŭzo" else "Tial"
        if prompt_format == "demando":
            prompt = f"{premise} {connector}"
        else:
            prompt = f"{USER} {premise} {connector} {ASST}"

        scores = [score_completion(model, tok, prompt, o, length_norm) for o in opts]
        pred = max(range(2), key=lambda k: scores[k])
        ok = pred == gold_idx
        n_ok += ok
        flag = "✓" if ok else "✗"
        print(f"[{i}/{n}] {flag} gold={gold_idx} pred={pred}  scores={[round(s,3) for s in scores]}", flush=True)
        if i <= 5:
            print(f"   {premise}  ({q})")
            print(f"     [0] {opts[0]}{' ← gold' if gold_idx==0 else ''}{' ← pred' if pred==0 else ''}")
            print(f"     [1] {opts[1]}{' ← gold' if gold_idx==1 else ''}{' ← pred' if pred==1 else ''}")
    print(f"\n=== copa {n_ok}/{n} = {100*n_ok/n:.1f}% ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("task", choices=["sciq", "copa"])
    ap.add_argument("n", type=int, default=100, nargs="?")
    ap.add_argument("--no-length-norm", action="store_true")
    ap.add_argument("--no-support", action="store_true",
                    help="SciQ only: hide support paragraph (zero-shot recall)")
    ap.add_argument("--prompt-format", choices=["chat", "demando"], default="chat",
                    help="chat = <|user|>…<|assistant|> (SFT default). "
                         "demando = 'Demando: …\\nRespondo: …' (matches the "
                         "pretrain benchmark Q/A format — use for base-LM eval).")
    args = ap.parse_args()

    print(f"loading {args.ckpt}  task={args.task}  n={args.n}  "
          f"length_norm={not args.no_length_norm}", flush=True)
    tok = PreTrainedTokenizerFast.from_pretrained("tokenizer_morpheme")
    tok.add_special_tokens({"additional_special_tokens": list(SPECIAL)})
    # Auto-pick device: GPU if available (fp16), else CPU (fp32).
    # CPU fallback lets you run alongside an in-flight GPU training job.
    global _DEVICE
    if torch.cuda.is_available():
        _DEVICE = "cuda"
        dtype = torch.float16
        threads_msg = ""
    else:
        _DEVICE = "cpu"
        dtype = torch.float32
        torch.set_num_threads(8)
        threads_msg = " (CPU, 8 threads)"
    print(f"device: {_DEVICE}{threads_msg}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=dtype).to(_DEVICE).eval()
    model.resize_token_embeddings(len(tok))

    if args.task == "sciq":
        eval_sciq(model, tok, args.n, not args.no_support,
                  not args.no_length_norm, args.prompt_format)
    else:
        eval_copa(model, tok, args.n, not args.no_length_norm, args.prompt_format)


if __name__ == "__main__":
    main()
