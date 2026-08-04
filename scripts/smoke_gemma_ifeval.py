"""Smoke: how well does gemma-3-12b (OpenRouter) follow ifeval-da constraints?

Samples N ifeval-da rows, sends each prompt verbatim to gemma via OR,
runs the google IFEval verifiers on the response, reports strict/loose
pass rates + per-family breakdown. Same eval logic as eval_ifeval_da.py
but points at OR instead of a local model.

Purpose: decide whether gemma is a viable data generator for a compliance-
verified IF training set that fills the format/combination gaps we saw
on our v22-avg baseline (detectable_format 9.5%, combination 1.6%).

Usage:
    python scripts/smoke_gemma_ifeval.py --key ~/or --n 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ifeval_google import instructions_registry as reg  # noqa: E402

MODEL = "google/gemma-3-12b-it"
API = "https://openrouter.ai/api/v1/chat/completions"


def build_instructions(row):
    out = []
    for i, name in enumerate(row["instruction_id_list"]):
        cls = reg.INSTRUCTION_DICT.get(name)
        if cls is None:
            continue
        inst = cls(name)
        raw_kwargs = row["kwargs"][i] if i < len(row["kwargs"]) else {}
        kwargs = {k: v for k, v in raw_kwargs.items() if v is not None}
        try:
            inst.build_description(**kwargs)
        except (TypeError, KeyError):
            try:
                inst.build_description()
            except Exception:
                continue
        out.append((name, inst))
    return out


def score_row(response, insts):
    strict, loose = [], []
    for _, inst in insts:
        try:
            strict.append(inst.check_following(response))
        except Exception:
            strict.append(False)
        variants = [response, response.strip(),
                    "\n".join(response.split("\n")[1:]).strip() if "\n" in response else response,
                    response.replace("*", "")]
        ok = False
        for v in variants:
            try:
                if inst.check_following(v):
                    ok = True
                    break
            except Exception:
                pass
        loose.append(ok)
    return strict, loose


def call_gemma(prompt, key, retries=3):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://claude-code-ifeval-smoke",
        "X-Title": "ifeval-smoke",
    }
    for attempt in range(retries):
        try:
            r = requests.post(API, json=body, headers=headers, timeout=90)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, type=Path)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    key = args.key.expanduser().read_text().strip()
    ds = load_dataset("danish-foundation-models/ifeval-da", split="train")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))
    print(f"gemma-3-12b on {len(ds)} ifeval-da rows", flush=True)

    prompt_strict = 0
    prompt_loose = 0
    inst_strict_ok = 0
    inst_loose_ok = 0
    inst_strict_tot = 0
    per_fam = defaultdict(lambda: [0, 0])
    scored = 0

    for i, row in enumerate(ds):
        insts = build_instructions(row)
        if not insts:
            continue
        try:
            resp = call_gemma(row["prompt"], key)
        except Exception as e:
            print(f"  {i+1}: API fail: {e}", flush=True)
            continue
        s, l = score_row(resp, insts)
        prompt_strict += all(s)
        prompt_loose += all(l)
        inst_strict_ok += sum(s)
        inst_loose_ok += sum(l)
        inst_strict_tot += len(s)
        scored += 1
        for (name, _), ok_s in zip(insts, s):
            fam = name.split(":")[0]
            per_fam[fam][0] += ok_s
            per_fam[fam][1] += 1
        print(f"  {i+1}/{len(ds)}  strict:{'/'.join('T' if x else 'F' for x in s)}  "
              f"running prompt-strict={100*prompt_strict/scored:.1f}%  "
              f"inst-strict={100*inst_strict_ok/inst_strict_tot:.1f}%",
              flush=True)

    print(f"\n=== gemma-3-12b on ifeval-da (n={scored}) ===")
    print(f"  prompt-strict: {prompt_strict}/{scored} = {100*prompt_strict/scored:.1f}%")
    print(f"  prompt-loose:  {prompt_loose}/{scored} = {100*prompt_loose/scored:.1f}%")
    print(f"  inst-strict:   {inst_strict_ok}/{inst_strict_tot} = {100*inst_strict_ok/inst_strict_tot:.1f}%")
    print(f"  inst-loose:    {inst_loose_ok}/{inst_strict_tot} = {100*inst_loose_ok/inst_strict_tot:.1f}%")
    print("\nPer-family (strict):")
    for fam in sorted(per_fam, key=lambda f: -per_fam[f][1]):
        p, t = per_fam[fam]
        print(f"  {100*p/t:5.1f}%  {p:3d}/{t:3d}  {fam}")


if __name__ == "__main__":
    main()
