"""Average N checkpoint safetensors into one — simple SWA-style model soup.

Usage:
    python scripts/avg_ckpts.py --output /root/v21_avg_1600_2000 \\
        /path/to/ckpt_a /path/to/ckpt_b [more...]

Loads each ckpt's model.safetensors, verifies shapes match, averages the
tensors (equal weight), writes model.safetensors + copies config.json /
tokenizer* / generation_config.json from the FIRST ckpt so
AutoModelForCausalLM.from_pretrained(output_dir) works out of the box.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Output dir for averaged ckpt")
    ap.add_argument("ckpts", nargs="+", help="Two or more checkpoint dirs")
    args = ap.parse_args()

    if len(args.ckpts) < 2:
        raise SystemExit("Need at least 2 ckpts to average")

    ckpts = [Path(p) for p in args.ckpts]
    for p in ckpts:
        if not (p / "model.safetensors").is_file():
            raise SystemExit(f"missing model.safetensors in {p}")

    print(f"Averaging {len(ckpts)} ckpts → {args.output}", flush=True)
    for p in ckpts:
        print(f"  {p}", flush=True)

    print("Loading first ckpt as template...", flush=True)
    avg = load_file(str(ckpts[0] / "model.safetensors"))

    for p in ckpts[1:]:
        print(f"Adding {p}...", flush=True)
        other = load_file(str(p / "model.safetensors"))
        if set(avg) != set(other):
            missing = set(avg) - set(other)
            extra = set(other) - set(avg)
            raise SystemExit(f"key mismatch — missing:{missing} extra:{extra}")
        for k in avg:
            if avg[k].shape != other[k].shape:
                raise SystemExit(f"shape mismatch for {k}: "
                                 f"{avg[k].shape} vs {other[k].shape}")
            avg[k] = avg[k] + other[k]

    n = float(len(ckpts))
    print(f"Dividing by n={int(n)}...", flush=True)
    for k in avg:
        # Convert to float32 for the divide if the dtype is integer-ish;
        # weight tensors are float so this is a no-op divide.
        avg[k] = (avg[k] / n).to(avg[k].dtype)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving averaged model.safetensors → {out}...", flush=True)
    save_file(avg, str(out / "model.safetensors"))

    print("Copying config + tokenizer from first ckpt...", flush=True)
    src = ckpts[0]
    for name in ["config.json", "generation_config.json",
                 "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json"]:
        p = src / name
        if p.is_file():
            shutil.copy2(str(p), str(out / name))
            print(f"  {name}", flush=True)

    print(f"\nDone. Load with: AutoModelForCausalLM.from_pretrained('{out}')",
          flush=True)


if __name__ == "__main__":
    main()
