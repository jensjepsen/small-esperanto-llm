"""Re-translate several known-degenerate articles with v6 in fp16
across chunk sizes; report tail per article × mode."""
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

TARGETS = sys.argv[1:] or [
    "Multituberculata", "Ashura", "Computer-aided design",
    "Western grey kangaroo", "Petty kingdoms of Norway", "Résumé",
]

_BREAK = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')

def sents(t):
    return [p.strip() for p in _BREAK.split(t.strip()) if p.strip()]

def chunks(sents, n):
    return [" ".join(sents[i:i+n]) for i in range(0, len(sents), n)]


print("loading model...", flush=True)
tok = SPMTokenizer(SPM)
device = "cuda"
dtype = torch.float16 if torch.cuda.get_device_capability()[0] < 8 else torch.bfloat16
print(f"dtype={dtype}", flush=True)
model = MarianMTModel.from_pretrained(CKPT, torch_dtype=dtype).to(device).eval()


def translate(chunk_list):
    outs = []
    BATCH = 12
    for i in range(0, len(chunk_list), BATCH):
        batch = chunk_list[i:i+BATCH]
        ids_lists = [tok.encode(c, lang="eo", add_eos=True)[:256] for c in batch]
        ml = max(len(x) for x in ids_lists)
        in_ids = torch.full((len(batch), ml), tok.pad_id, dtype=torch.long, device=device)
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


# preload rows once
all_rows = {}
for line in open("/mnt/data2/wiki_gaps/eo_vital_level5_translated.jsonl"):
    r = json.loads(line)
    if r["title"] in TARGETS:
        all_rows[r["title"]] = r

for title in TARGETS:
    if title not in all_rows:
        print(f"\nSKIP {title}: not found"); continue
    row = all_rows[title]
    ss = sents(row["en_text"])
    print(f"\n{'='*70}")
    print(f"### {title}  ({len(row['en_text'])} chars en, {len(ss)} sents)")
    for n, label in [(1, "SINGLE"), (2, "PAIR"), (4, "QUAD")]:
        t0 = time.time()
        eo = translate(chunks(ss, n))
        print(f"\n  [{label} n={n}] {len(eo)} chars in {time.time()-t0:.1f}s")
        print(f"    TAIL: ...{eo[-200:]}", flush=True)
    print(f"\n  [STORED bf16] TAIL: ...{row['eo_text'][-200:]}", flush=True)
