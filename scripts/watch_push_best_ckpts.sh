#!/usr/bin/env bash
# Push new best-checkpoint snapshots to the Hub as they appear.
#
# Exists because a rented pod vanished mid-run and took six weights-only GRPO
# snapshots with it -- including the best one -- along with four hours of L40S
# time. `--best-k` writes to local disk only, and local-only checkpoints on a
# rented box are one connection-refused away from gone.
#
# Polls <output-dir>/_best_ckpts and uploads any directory it has not already
# uploaded, each as a subfolder of one HF repo. Cheap: a 400M model is ~1.6 GB
# and only new snapshots are sent.
#
# Usage:
#   HF_TOKEN=$(cat /root/hf_token) \
#   bash scripts/watch_push_best_ckpts.sh <output-dir> <hf-repo> [interval_s]
#
# For a packed-SFT run set BEST_SUBDIR=best (the callback's layout):
#   BEST_SUBDIR=best bash scripts/watch_push_best_ckpts.sh /root/runs/X repo 300
set -u
OUT="${1:?output dir required}"
REPO="${2:?hf repo required}"
EVERY="${3:-300}"
# Which subdirectory holds the preserved snapshots. The GRPO trainer writes
# `_best_ckpts/`; the SFT downstream-eval callback writes `best/step-N-agg-X`.
# Pointing this at the wrong one is silent -- the loop just never finds a
# directory and reports nothing -- which is worse than not running it, so the
# name is explicit rather than guessed.
SUB="${BEST_SUBDIR:-_best_ckpts}"
STATE="${STATE:-/root/.pushed_ckpts}"
touch "$STATE"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/tmp/hf-cache}"
PY="${PY:-uv run --no-sync python}"

echo "[watch] $OUT/$SUB -> $REPO every ${EVERY}s"
while true; do
  if [ -d "$OUT/$SUB" ]; then
    for d in "$OUT"/"$SUB"/*/; do
      [ -d "$d" ] || continue
      name=$(basename "$d")
      grep -qxF "$name" "$STATE" && continue
      # only upload a finished snapshot -- a dir being written has no weights yet
      [ -f "$d/model.safetensors" ] || continue
      echo "[watch] pushing $name"
      if $PY - "$d" "$REPO" "$name" <<'PYEOF'
import sys
from huggingface_hub import HfApi
src, repo, name = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=src, repo_id=repo, path_in_repo=name,
                  commit_message=f"{name}")
print(f"pushed {name}")
PYEOF
      then
        echo "$name" >> "$STATE"
      else
        echo "[watch] push FAILED for $name (will retry next tick)"
      fi
    done
  fi
  sleep "$EVERY"
done
