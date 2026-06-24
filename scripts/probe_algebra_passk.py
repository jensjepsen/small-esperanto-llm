"""Probe algebra with pass@K (sampled, K attempts, OR-merge correctness)."""
import re
import sys
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, "src")
from esperanto_lm.data import _morpheme_preprocess
from scripts.probe_algebra import PROBES, has_answer  # reuse

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/sft/v10_sftv4/checkpoint-16000"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7

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


print(f"loading {CKPT} (pass@{K}, T={TEMP})", flush=True)
tok = PreTrainedTokenizerFast.from_pretrained("tokenizer_morpheme")
tok.add_special_tokens({"additional_special_tokens": list(SPECIAL)})
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float16).cuda().eval()
model.resize_token_embeddings(len(tok))


def ask_sampled(text, k):
    p = pp(f"{USER} {text} {ASST} ")
    ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    preds = []
    for _ in range(k):
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=300, do_sample=True, num_beams=1,
                temperature=TEMP, top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                repetition_penalty=1.1,
                eos_token_id=tok.convert_tokens_to_ids(END),
            )
        preds.append(decode(tok, out[0][ids.shape[1]:].tolist()))
    return preds


from collections import Counter
hit_by = Counter()
tot_by = Counter()
print()
for level, q, gold in PROBES:
    preds = ask_sampled(q, K)
    ok = any(has_answer(p, gold) for p in preds)
    tot_by[level] += 1
    if ok:
        hit_by[level] += 1
    flag = "✓" if ok else "✗"
    print(f"[{level:11s}] {flag} gold={gold:>5}  Q: {q}")
    for i, p in enumerate(preds):
        h = "✓" if has_answer(p, gold) else "✗"
        print(f"              [{i+1}/{K}] {h} {p[:200]}")
    print()

print(f"\n{'='*55}")
for level in ["trivial","one-step","two-step","distrib","both-sides","fraction","word","quad"]:
    if tot_by[level]:
        print(f"  {level:11s} {hit_by[level]}/{tot_by[level]}")
total = sum(hit_by.values())
totsum = sum(tot_by.values())
print(f"  {'TOTAL':11s} {total}/{totsum} = {100*total/totsum:.0f}%  (pass@{K})")
