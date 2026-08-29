"""Re-translate one article with v6 in 3 modes: pair (current), single sentence,
half-doc chunks. Compare tail quality."""
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
TARGET = sys.argv[1] if len(sys.argv) > 1 else "Manos Hatzidakis"

_BREAK = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')

def sents(t):
    return [p.strip() for p in _BREAK.split(t.strip()) if p.strip()]

def chunks(sents, n):
    return [" ".join(sents[i:i+n]) for i in range(0, len(sents), n)]

print(f"loading model...", flush=True)
tok = SPMTokenizer(SPM)
device = "cuda"
dtype = torch.float16 if torch.cuda.get_device_capability()[0] < 8 else torch.bfloat16
model = MarianMTModel.from_pretrained(CKPT, torch_dtype=dtype).to(device).eval()

print(f"locating {TARGET}...", flush=True)
row = None
for line in open("/mnt/data2/wiki_gaps/eo_vital_level5_translated.jsonl"):
    r = json.loads(line)
    if r["title"] == TARGET:
        row = r
        break
assert row, f"not found: {TARGET}"

en = row["en_text"]
ss = sents(en)
print(f"  {len(en)} chars en, {len(ss)} sentences", flush=True)


def translate_chunks(chunk_list, label):
    print(f"\n=== {label}: {len(chunk_list)} chunks ===", flush=True)
    t0 = time.time()
    outs = []
    BATCH = 16
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
    eo = " ".join(outs)
    print(f"  {len(eo)} chars eo in {time.time()-t0:.1f}s", flush=True)
    print(f"  TAIL: ...{eo[-200:]}", flush=True)
    return eo


single = translate_chunks(ss, "SINGLE sentences")
pair = translate_chunks(chunks(ss, 2), "PAIR sentences (current)")
quad = translate_chunks(chunks(ss, 4), "QUAD sentences")

print("\n=== ORIGINAL tail in stored translation ===")
print(f"  TAIL: ...{row['eo_text'][-200:]}")
