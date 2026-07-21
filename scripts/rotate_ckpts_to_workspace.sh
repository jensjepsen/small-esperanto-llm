#!/usr/bin/env bash
# Keep the latest N training checkpoints copied to /workspace so they're
# reachable via the pod's S3-compatible endpoint from outside.
#
# Preserves the FULL resume-capable checkpoint (weights + optimizer +
# scheduler + per-rank RNG + trainer_state + training_args). At ~2.6 GB
# per ckpt × KEEP=4 = ~10 GB, well under the 30 GB /workspace quota.
# Skipping to inference-only would save ~1 GB/ckpt but forfeits smooth
# resume-from-checkpoint if the pod dies.
#
# Runs as a background daemon on the pod. Skips already-synced files
# so it's cheap after the first pass. Removes any workspace checkpoints
# that fall out of the "latest N" window.
#
# Usage on pod: nohup bash scripts/rotate_ckpts_to_workspace.sh </dev/null >/tmp/ckpt_rotate.log 2>&1 &
set -uo pipefail

SRC="${SRC:-/tmp/runs/v1_danish_400m}"
WORK="${WORK:-/workspace}"
KEEP="${KEEP:-4}"
INTERVAL="${INTERVAL:-180}"   # seconds between passes
# Fixed-name files (rng_state_*.pth handled separately as it's per-rank)
FILES=(config.json generation_config.json special_tokens_map.json
       tokenizer.json tokenizer_config.json trainer_state.json
       training_args.bin scheduler.pt optimizer.pt
       model.safetensors)

ts() { date -Iseconds; }

log() { echo "$(ts) $*"; }

while true; do
  # Latest N checkpoints by numeric step
  latest=$(ls -1 "$SRC" 2>/dev/null \
           | grep -oE '^checkpoint-[0-9]+$' \
           | sort -t- -k2 -n | tail -n "$KEEP")

  # Copy any missing
  for ck in $latest; do
    step=${ck#checkpoint-}
    dst="$WORK/da_ckpt_$step"
    src="$SRC/$ck"
    if [ ! -f "$src/model.safetensors" ]; then
      continue  # checkpoint still being written
    fi
    mkdir -p "$dst"
    newly_synced=0
    for f in "${FILES[@]}"; do
      if [ -f "$src/$f" ] && [ ! -f "$dst/$f" ]; then
        cp "$src/$f" "$dst/$f" && newly_synced=1
      fi
    done
    # rng_state_<rank>.pth — one per DDP rank, count varies with world_size
    for rng in "$src"/rng_state_*.pth; do
      [ -f "$rng" ] || continue
      base=$(basename "$rng")
      if [ ! -f "$dst/$base" ]; then
        cp "$rng" "$dst/$base" && newly_synced=1
      fi
    done
    if [ "$newly_synced" = 1 ]; then
      sz=$(du -sh "$dst" 2>/dev/null | cut -f1)
      log "synced $ck → $dst ($sz)"
    fi
  done

  # Prune stale workspace dirs not in the latest-N list
  for dir in "$WORK"/da_ckpt_*; do
    [ -d "$dir" ] || continue
    step=$(basename "$dir" | sed 's/^da_ckpt_//')
    ck="checkpoint-$step"
    keep=0
    for L in $latest; do
      if [ "$L" = "$ck" ]; then keep=1; break; fi
    done
    if [ "$keep" = 0 ]; then
      rm -rf "$dir"
      log "pruned $dir"
    fi
  done

  sleep "$INTERVAL"
done
