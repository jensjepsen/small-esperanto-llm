"""Push eneo_v7_mini final/ to HF Hub."""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO = "jensjepsen/eo-mt-v7-mini"
SRC = Path("/mnt/data2/runs/mt/eneo_v7_mini/final")
# tokenizer + SP model live in mt/data/tokenizer/
TOK_SPM = Path("mt/data/tokenizer/spm_eneo_32k.model")

token = os.getenv("HF_HUB_TOKEN") or (Path.home() / ".cache/huggingface/token").read_text().strip()
api = HfApi(token=token)
create_repo(REPO, repo_type="model", private=False, exist_ok=True, token=token)

# model + configs
for fname in ["model.safetensors", "config.json", "generation_config.json", "training_args.bin"]:
    src = SRC / fname
    if not src.exists():
        print(f"skip {fname} (missing)")
        continue
    print(f"uploading {fname}", flush=True)
    api.upload_file(path_or_fileobj=str(src), path_in_repo=fname, repo_id=REPO, token=token)

# tokenizer (SentencePiece model)
if TOK_SPM.exists():
    print(f"uploading tokenizer: {TOK_SPM.name}", flush=True)
    api.upload_file(path_or_fileobj=str(TOK_SPM), path_in_repo="spm_eneo_32k.model",
                    repo_id=REPO, token=token)

# model card
card = """---
language: [eo, en]
license: apache-2.0
tags: [translation, marian, esperanto]
---
# eo-mt-v7-mini

30M-parameter Marian MT model for English ↔ Esperanto, trained on
`jensjepsen/esperanto-mt-parallel` (5.1M sentence pairs).

**Offline FLORES devtest, en→eo, sacrebleu lowercase=True (full 1012 pairs):**

| Metric | Score |
|---|---|
| BLEU | **30.55** |
| chrF | **61.08** |
| chrF++ | **58.20** |

Roughly matches [eo-mt-v6](https://huggingface.co/jensjepsen/eo-mt-v6) (BLEU 31.00,
60M params) at half the parameter count. Above NLLB-200-distilled-600M (BLEU 29.28).

Tokenizer is SentencePiece, **case-folding**: outputs are always lowercase. Use
`--lowercase` for apples-to-apples sacrebleu.

Training: fp16 on a 1080 Ti, 3 epochs, 24h wall time, batch 64 × grad-accum 2.
Architecture: d_model 384, 4+4 enc/dec layers, 6 heads, ffn 1536.
"""
print("uploading README.md", flush=True)
api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                repo_id=REPO, token=token)

print(f"\ndone → https://huggingface.co/{REPO}")
