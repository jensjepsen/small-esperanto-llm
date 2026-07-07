"""Canonical GSM8K probe for SFT-trained EO chat models.

Runs greedy (default) or sampled pass@K generation on
`jensjepsen/esperanto-gsm8k` test split, using the checkpoint's own
tokenizer and the exact training-time chat format from
`train_sft_packed.format_conversation`.

Writes per-row JSONL for post-hoc analysis and prints hit/total every
--log-every rows so long runs give live progress.

Usage:

    # Greedy on full 1279-row test split (~1h on 1080Ti, ~10min on 5090):
    uv run python scripts/probe_gsm8k.py \\
        /mnt/data2/v16_sft/checkpoint-22498

    # Sampled pass@6 on first 200 rows:
    uv run python scripts/probe_gsm8k.py CKPT --n 200 --k 6 --temperature 0.7

    # Explicit output path (defaults to /mnt/data2/gsm8k_probe_<basename>.jsonl):
    uv run python scripts/probe_gsm8k.py CKPT --out /tmp/foo.jsonl
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from esperanto_lm.data import _morpheme_preprocess
from train_sft_packed import format_conversation, USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN

SPECIAL_TOKENS = (
    USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN,
    "<|tool_call|>", "<|/tool_call|>",
    "<|tool_result|>", "<|/tool_result|>",
)
_SPECIAL_SPLIT = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")
_SKIP_TOKS = {"<s>", "</s>", "<pad>", "<unk>", USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN}

# Match "#### N" — allow spaced #### and comma-formatted numbers ("19 , 500").
_HASH_RE = re.compile(r"(?:#\s*){2,}\s*(-?[\d,\s]+(?:\.\d+)?)")


def train_prep(text: str) -> str:
    """Apply the same morpheme preprocessing used at training time to
    non-special-token spans of `text`."""
    out = []
    for p in _SPECIAL_SPLIT.split(text):
        if p in SPECIAL_TOKENS:
            out.append(p)
        elif p.strip():
            out.append(_morpheme_preprocess(p.strip()))
        else:
            out.append(p)
    return " ".join(out)


def decode(tok, ids) -> str:
    toks = [t for t in tok.convert_ids_to_tokens(ids) if t not in _SKIP_TOKS]
    return "".join(t if t != "<w>" else " " for t in toks).strip()


def extract_answer(text: str) -> str | None:
    """Pull the numeric answer after '####'. Handles '19 , 500' → '19500'."""
    m = _HASH_RE.search(text)
    if not m:
        # Fallback: last bare number
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "").replace(" ", ""))
        return nums[-1] if nums else None
    raw = m.group(1).replace(",", "").replace(" ", "")
    # Strip trailing junk
    m2 = re.match(r"-?\d+(?:\.\d+)?", raw)
    return m2.group(0) if m2 else None


def matches(pred: str | None, gold: str) -> bool:
    if not pred:
        return False
    try:
        return float(pred) == float(gold)
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt", help="Path to SFT checkpoint directory")
    ap.add_argument("--n", type=int, default=None,
                    help="Number of rows to eval (default: full test = 1279)")
    ap.add_argument("--k", type=int, default=1,
                    help="Samples per row for pass@K (default: 1 = greedy)")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature when --k > 1 (default 0.7)")
    ap.add_argument("--top-p", type=float, default=0.9,
                    help="Nucleus sampling threshold (default 0.9, "
                         "matches earlier v15 pass@k protocol)")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--out", type=str, default=None,
                    help="Per-row JSONL output path (default: "
                         "/mnt/data2/gsm8k_probe_<basename>.jsonl)")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--rows-per-batch", type=int, default=1,
                    help="Pack B rows into a single forward pass "
                         "(B*K sequences per batch). Left-pads shorter prompts. "
                         "Default 1 = one row at a time. 5090 comfortably handles B=8, K=12.")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    assert ckpt.exists(), f"ckpt not found: {ckpt}"
    if args.out is None:
        args.out = f"/mnt/data2/gsm8k_probe_{ckpt.parent.name}_{ckpt.name}.jsonl"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    print(f"loading {ckpt}  N={args.n or 'all'}  pass@{args.k}"
          f"{f' T={args.temperature}' if args.k > 1 else ''}"
          f"  dtype={args.dtype}  rows/batch={args.rows_per_batch}"
          f"  out={out_path}", flush=True)
    tok = PreTrainedTokenizerFast.from_pretrained(str(ckpt))
    # Left-padding for decoder-only generation (padding on LEFT so all
    # sequences resume from the SAME position on the right).
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(str(ckpt), torch_dtype=dtype).cuda().eval()
    end_id = tok.convert_tokens_to_ids(END_TOKEN)

    def _fmt_prompt(question: str) -> str:
        raw = (format_conversation([{"role": "user", "content": question}])
               + " " + ASSISTANT_TOKEN + " ")
        return train_prep(raw)

    def ask_batch(questions: list[str], k: int) -> list[list[str]]:
        """Return B lists of k completions each. Packs B rows into ONE forward
        pass with left-padding + attention mask. Total batch = B*k sequences."""
        prompts = [_fmt_prompt(q) for q in questions]
        enc = tok(prompts, return_tensors="pt", padding=True,
                    add_special_tokens=False)
        ids = enc.input_ids.cuda()
        mask = enc.attention_mask.cuda()
        with torch.no_grad():
            out = model.generate(
                ids,
                attention_mask=mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=(k > 1),
                temperature=args.temperature if k > 1 else None,
                top_p=args.top_p if k > 1 else None,
                num_beams=1,
                num_return_sequences=k,
                pad_token_id=tok.pad_token_id,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=end_id,
            )
        prompt_len = ids.shape[1]
        # out shape: (B*k, prompt_len + max_new_tokens). Split into B groups of k.
        results: list[list[str]] = []
        for b in range(len(questions)):
            group = [decode(tok, out[b * k + i][prompt_len:].tolist())
                       for i in range(k)]
            results.append(group)
        return results

    ds = load_dataset("jensjepsen/esperanto-gsm8k", split="test")
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))
    print(f"test rows: {len(ds)}", flush=True)

    # Pre-collect valid (idx, question, gold) tuples; skip rows without gold.
    todo = []
    for i, row in enumerate(ds):
        msgs = row["messages"]
        gm = _HASH_RE.search(msgs[1]["content"])
        if not gm:
            continue
        gold = gm.group(1).replace(",", "").replace(" ", "").rstrip(".")
        todo.append((i, msgs[0]["content"], gold))

    t0 = time.time()
    hit_p1 = hit_pk = total = 0
    B = max(1, args.rows_per_batch)
    with out_path.open("w") as f:
        for chunk_start in range(0, len(todo), B):
            chunk = todo[chunk_start:chunk_start + B]
            questions = [q for _, q, _ in chunk]
            batch_resps = ask_batch(questions, args.k)
            for (i, _, gold), resps in zip(chunk, batch_resps):
                preds = [extract_answer(r) for r in resps]
                p1_ok = matches(preds[0], gold)
                pk_ok = any(matches(p, gold) for p in preds)
                hit_p1 += int(p1_ok)
                hit_pk += int(pk_ok)
                total += 1
                f.write(json.dumps({
                    "idx": i, "gold": gold, "preds": preds,
                    "p1_correct": p1_ok, "pk_correct": pk_ok,
                    "resps": resps if args.k > 1 else resps[0],
                }, ensure_ascii=False) + "\n")
            f.flush()
            if total // args.log_every > (total - len(chunk)) // args.log_every:
                if args.k == 1:
                    print(f"[{total}/{len(todo)}] hit={hit_p1} "
                          f"({100*hit_p1/total:.1f}%) e={time.time()-t0:.0f}s",
                          flush=True)
                else:
                    print(f"[{total}/{len(todo)}] pass@1={hit_p1} "
                          f"({100*hit_p1/total:.1f}%)  pass@{args.k}={hit_pk} "
                          f"({100*hit_pk/total:.1f}%) e={time.time()-t0:.0f}s",
                          flush=True)

    if args.k == 1:
        print(f"\n=== {ckpt.name} greedy GSM8K: {hit_p1}/{total} = "
              f"{100*hit_p1/total:.2f}% ({time.time()-t0:.0f}s) ===")
    else:
        print(f"\n=== {ckpt.name} sampled GSM8K T={args.temperature}: "
              f"pass@1={hit_p1}/{total}={100*hit_p1/total:.2f}%  "
              f"pass@{args.k}={hit_pk}/{total}={100*hit_pk/total:.2f}% "
              f"({time.time()-t0:.0f}s) ===")


if __name__ == "__main__":
    main()
