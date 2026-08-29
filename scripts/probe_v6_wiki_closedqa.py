"""Probe local v6 checkpoint on a sample of wiki_closedqa Q/A.

Picks a mix of easy/hard questions across topics, generates greedy
answers, prints (Q, gold, v6 answer) side-by-side.
"""
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT = "/home/jepsen/src/espllm/runs/sft/da_v6_mix9wpreword/final"

# Sample from the dataset on disk (before it's pushed to HF, but the local
# jsonl is fine too).
JSONL = Path("/mnt/data2/wiki_closedqa_v4/rows.jsonl")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15

rows = [json.loads(l) for l in JSONL.open()]
# Sample across many topics — dedup by title so we get varied subjects
by_title = {}
for r in rows:
    by_title.setdefault(r["orig_title"], []).append(r)

random.seed(42)
titles = random.sample(list(by_title.keys()), N)
# One question per title
sample = [random.choice(by_title[t]) for t in titles]

print(f"loading v6 …", flush=True)
tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float16).cuda().eval()
end_id = tok.convert_tokens_to_ids("<|end|>")
stop = [tok.eos_token_id, end_id]

for i, r in enumerate(sample, 1):
    prompt = f"<|user|> {r['q']} <|assistant|>"
    ids = tok(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda")
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=120, do_sample=False,
                             pad_token_id=tok.eos_token_id, eos_token_id=stop)
    gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    gen = gen.replace("<|end|>", "").strip()
    print("=" * 72)
    print(f"[{i}] {r['orig_title']}  ({r['tier']})")
    print(f"  Q: {r['q']}")
    print(f"  GOLD: {r['a']}")
    print(f"  V6  : {gen[:250]}")
