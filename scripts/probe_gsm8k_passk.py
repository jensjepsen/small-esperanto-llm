"""Probe GSM8K-EO test with pass@K (sampled, OR-merge)."""
import re
import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, "src")
from esperanto_lm.data import _morpheme_preprocess

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/sft/sft_v7/checkpoint-16000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
TEMP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
MAX_NEW = int(sys.argv[5]) if len(sys.argv) > 5 else 400

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


def extract_gold(s):
    m = re.search(r"####\s*([\-0-9,.]+)", s)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    return None


def extract_pred(s):
    nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", "").replace(" ", ""))
    return nums[-1] if nums else None


print(f"loading {CKPT}  N={N}  pass@{K}  T={TEMP}", flush=True)
tok = PreTrainedTokenizerFast.from_pretrained("tokenizer_morpheme")
tok.add_special_tokens({"additional_special_tokens": list(SPECIAL)})
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float16).cuda().eval()
model.resize_token_embeddings(len(tok))

ds = load_dataset("jensjepsen/esperanto-gsm8k", split="test").select(range(N))


def ask_k(text, k):
    p = pp(f"{USER} {text} {ASST} ")
    ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=True, num_beams=1,
            num_return_sequences=k, temperature=TEMP, top_p=0.95,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            repetition_penalty=1.1,
            eos_token_id=tok.convert_tokens_to_ids(END),
        )
    return [decode(tok, out[i][ids.shape[1]:].tolist()) for i in range(k)]


n_ok = 0
n_any_close = 0  # at least one of K has the gold token anywhere
for i, row in enumerate(ds, 1):
    msgs = row["messages"]
    q = msgs[0]["content"]
    a_full = msgs[1]["content"]
    gold = extract_gold(a_full)
    preds = ask_k(q, K)
    nums = [extract_pred(p) for p in preds]
    matches = [
        n is not None and gold is not None and float(n) == float(gold)
        for n in nums
    ]
    ok = any(matches)
    n_ok += ok
    flag = "✓" if ok else "✗"
    print(f"\n[{i}/{N}] {flag} gold={gold} pred_lasts={nums}")
    print(f"  Q: {q}")
    for j, (p, m) in enumerate(zip(preds, matches), 1):
        h = "✓" if m else "✗"
        print(f"  [{j}/{K}] {h} {p[:250]}")

print(f"\n=== pass@{K} {n_ok}/{N} = {100*n_ok/N:.0f}% ===")
