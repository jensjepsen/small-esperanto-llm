"""Show what v5 model emits on title_wrapped constraint rows."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = "/root/runs/sft/da_v5_mix9if2/final"
tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16).cuda().eval()
end_id = tok.convert_tokens_to_ids("<|end|>")
stop = [tok.eos_token_id, end_id]

ds = load_dataset("jensjepsen/danish-instruction-following-v2", "default", split="eval")
ds = ds.shuffle(seed=0).select(range(200))
title_rows = [r for r in ds if "title_wrapped" in r["constraints"]][:8]
print(f"got {len(title_rows)} title_wrapped rows")

for i, r in enumerate(title_rows):
    prompt = f"<|user|> {r['messages'][0]['content']} <|assistant|>"
    ids = tok(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda")
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=384, do_sample=False,
                             pad_token_id=tok.eos_token_id, eos_token_id=stop)
    gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).replace("<|end|>", "").strip()
    print("=" * 72)
    print(f"[{i}] constraints:", r["constraints"])
    print("PROMPT:", r["messages"][0]["content"][:250])
    print("GOLD-ish (from Gemini):", r["messages"][1]["content"][:250])
    print("V5 GEN:", gen[:300])
