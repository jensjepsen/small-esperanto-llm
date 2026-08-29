"""Push SFT-v4 model to HF as a private repo for fast remote pull."""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO = "jensjepsen/eo-sft-v4-25686"
SRC = Path("runs/sft/v10_sftv4/checkpoint-25686")
TOK = Path("tokenizer_morpheme")

token = os.getenv("HF_HUB_TOKEN") or (Path.home() / ".cache/huggingface/token").read_text().strip()

api = HfApi(token=token)
create_repo(REPO, repo_type="model", private=True, exist_ok=True, token=token)

# Push model files (skip optimizer.pt — not needed for a fresh fine-tune)
files = ["model.safetensors", "config.json", "generation_config.json"]
for fname in files:
    src = SRC / fname
    print(f"uploading {src} → {REPO}/{fname}", flush=True)
    api.upload_file(path_or_fileobj=str(src), path_in_repo=fname,
                    repo_id=REPO, token=token)

# Also push tokenizer files alongside (so the remote box can load both from one repo)
for f in TOK.glob("*"):
    if f.is_file():
        print(f"uploading {f} → {REPO}/{f.name}", flush=True)
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=REPO, token=token)

print(f"\ndone → https://huggingface.co/{REPO}")
