"""Same article, same chunk size, same beam — only dtype differs.
Verifies whether bf16-on-Pascal causes tail collapses vs fp16."""
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import MarianMTModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mt" / "scripts"))
from sp_tokenizer import SPMTokenizer

CKPT = "/mnt/data2/checkpoints/mt/eneo_v6/final"
SPM = "mt/data/tokenizer/spm_eneo_32k.model"

TARGETS = ["Manos Hatzidakis", "Ludwig Prandtl", "Multituberculata",
           "Ashura", "Western grey kangaroo"]

_BREAK = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')

def sents(t):
    return [p.strip() for p in _BREAK.split(t.strip()) if p.strip()]

def chunks(sents, n=2):
    return [" ".join(sents[i:i+n]) for i in range(0, len(sents), n)]


print("loading tokenizer...", flush=True)
tok = SPMTokenizer(SPM)
device = "cuda"
print(f"GPU compute cap: {torch.cuda.get_device_capability()}", flush=True)

# load all target rows once
all_rows = {}
for line in open("/mnt/data2/wiki_gaps/eo_vital_level5_translated.jsonl"):
    r = json.loads(line)
    if r["title"] in TARGETS:
        all_rows[r["title"]] = r


def translate(model, chunk_list):
    outs = []
    BATCH = 12
    for i in range(0, len(chunk_list), BATCH):
        batch = chunk_list[i:i+BATCH]
        ids_lists = [tok.encode(c, lang="eo", add_eos=True)[:256] for c in batch]
        ml = max(len(x) for x in ids_lists)
        in_ids = torch.full((len(batch), ml), tok.pad_id,
                            dtype=torch.long, device=device)
        attn = torch.zeros_like(in_ids)
        for j, ids in enumerate(ids_lists):
            in_ids[j, :len(ids)] = torch.tensor(ids, device=device)
            attn[j, :len(ids)] = 1
        with torch.no_grad():
            out = model.generate(input_ids=in_ids, attention_mask=attn,
                                 num_beams=4, max_length=256, early_stopping=True,
                                 no_repeat_ngram_size=5, repetition_penalty=1.2,
                                 encoder_repetition_penalty=1.1)
        for seq in out:
            outs.append(tok.decode(seq))
    return " ".join(outs)


results = {}
for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
    print(f"\n{'='*70}", flush=True)
    print(f"### Loading model in {dtype_name}", flush=True)
    model = MarianMTModel.from_pretrained(CKPT, torch_dtype=dtype).to(device).eval()
    for title in TARGETS:
        row = all_rows[title]
        ss = sents(row["en_text"])
        t0 = time.time()
        eo = translate(model, chunks(ss, 2))
        results.setdefault(title, {})[dtype_name] = eo
        print(f"  {title:30s} {dtype_name} {len(eo)} chars in {time.time()-t0:.1f}s",
              flush=True)
    del model
    torch.cuda.empty_cache()

print(f"\n{'='*70}\n### TAILS\n")
for title in TARGETS:
    print(f"\n--- {title} ---")
    print(f"  fp16 TAIL: ...{results[title]['fp16'][-200:]}")
    print(f"  bf16 TAIL: ...{results[title]['bf16'][-200:]}")
    print(f"  STORED:    ...{all_rows[title]['eo_text'][-200:]}")
