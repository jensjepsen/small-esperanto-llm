"""Probe SFT on Orca-math: greedy, grade via #### N.

Orca-math has no test split; we pick records from the END of train (the
training set was shuffled, so this is closer to a 'memorization-tinged'
quality check than true held-out — informative for format + multi-step
correctness, not generalization.
"""
import re
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, "src")
from esperanto_lm.data import _morpheme_preprocess

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
SPECIAL = (USER, ASST, END)
SKIP = {"<s>", "</s>", "<pad>", "<unk>", USER, ASST, END}


def pp(s):
    pat = "(" + "|".join(re.escape(t) for t in SPECIAL) + ")"
    return " ".join(p if p in SPECIAL else _morpheme_preprocess(p)
                    for p in re.split(pat, s))


def decode(tok, ids):
    toks = tok.convert_ids_to_tokens(ids)
    toks = [t for t in toks if t not in SKIP]
    return "".join(t if t != "<w>" else " " for t in toks).strip()


GOLD_RE = re.compile(r"####\s*([\-0-9,.]+)")
# Loose 'last number' extractor for predictions that drop ####
PRED_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_gold(s: str) -> str | None:
    """orca-math has no #### marker — gold is embedded prose. Use the
    LAST standalone number in the gold answer, mirroring extract_pred."""
    m = GOLD_RE.search(s)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = PRED_NUM_RE.findall(s.replace(",", "").replace(" ", ""))
    return nums[-1] if nums else None


def extract_pred(s: str) -> str | None:
    """Prefer #### marker; fall back to last standalone number."""
    m = GOLD_RE.search(s)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = PRED_NUM_RE.findall(s.replace(",", "").replace(" ", ""))
    return nums[-1] if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("n", type=int, default=50, nargs="?")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--from-end", action="store_true",
                    help="sample from end of train (default: from start)")
    args = ap.parse_args()

    print(f"loading {args.ckpt}  n={args.n}", flush=True)
    tok = PreTrainedTokenizerFast.from_pretrained("tokenizer_morpheme")
    tok.add_special_tokens({"additional_special_tokens": list(SPECIAL)})
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.float16).cuda().eval()
    model.resize_token_embeddings(len(tok))

    ds = load_dataset("jensjepsen/esperanto-orca-math", split="train")
    total = len(ds)
    if args.from_end:
        ds = ds.select(range(total - args.n, total))
    else:
        ds = ds.select(range(args.n))

    n_ok = 0
    n_marker_ok = 0  # had #### marker AND was correct
    n_marker = 0     # had #### marker at all
    for i, row in enumerate(ds, 1):
        msgs = row["messages"]
        q = msgs[0]["content"]
        a_full = msgs[1]["content"]
        gold = extract_gold(a_full)

        prompt = pp(f"{USER} {q} {ASST} ")
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=args.max_new, do_sample=False, num_beams=1,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                repetition_penalty=1.1,
                eos_token_id=tok.convert_tokens_to_ids(END),
            )
        pred = decode(tok, out[0][ids.shape[1]:].tolist())
        pred_num = extract_pred(pred)
        has_marker = GOLD_RE.search(pred) is not None
        if has_marker:
            n_marker += 1
        ok = pred_num is not None and gold is not None and float(pred_num) == float(gold)
        if ok:
            n_ok += 1
            if has_marker:
                n_marker_ok += 1
        flag = "✓" if ok else "✗"
        marker = "@" if has_marker else "·"
        print(f"[{i}/{args.n}] {flag}{marker} gold={gold} pred={pred_num}", flush=True)
        if i <= 5 or not ok:
            print(f"   Q: {q[:200]}")
            print(f"   pred: {pred[:300]}")

    print(f"\n=== orca-math {n_ok}/{args.n} = {100*n_ok/args.n:.1f}%  "
          f"(#### marker emitted in {n_marker}/{args.n}, of which {n_marker_ok} correct) ===")


if __name__ == "__main__":
    main()
