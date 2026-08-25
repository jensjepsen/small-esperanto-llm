"""Generative json-schema eval on Danish — mirrors the GreedyEvalCallback
(task='json') exactly. Loads the `eval` split of a HF json-grpo dataset,
generates greedily, scores via reward_json_schema, prints mean_reward.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esperanto_lm.rl_rewards import reward_json_schema  # noqa: E402

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"


def _load_json_eval(source, split="eval"):
    p = Path(source)
    if p.exists() and (p / "state.json").exists():
        ds = load_from_disk(str(p))
    else:
        ds = load_dataset(source, split=split)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--source", default="jensjepsen/danish-json-grpo-v1")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None, help="optional jsonl output")
    args = ap.parse_args()

    dt = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    print(f"loading {args.ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    end_id = tok.convert_tokens_to_ids(END)
    eos_ids = [tok.eos_token_id] + ([end_id] if end_id != tok.unk_token_id else [])
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=dt).cuda().eval()

    print(f"loading {args.source} split={args.split}", flush=True)
    ds = _load_json_eval(args.source, args.split)
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))
    print(f"  n={len(ds)}", flush=True)

    # Build items with the passage-fixup mirroring build_json_dataset
    items = []
    n_fixed = 0
    for r in ds:
        u = r["prompt"]
        passage = r.get("passage") or ""
        if passage and len(passage) > 30 and passage.strip()[:40] not in u:
            u = f"{u}\n\nKildetekst:\n{passage.strip()}"
            n_fixed += 1
        items.append({
            "prompt": f"{USER}{u}{END}{ASST}",
            "fields": list(r["fields"]),
            "types": list(r["types"]),
            "strict": bool(r["strict"]),
            "passage": passage,
            "gold_values": r.get("gold_values") or "",
        })
    if n_fixed:
        print(f"  [fixup] appended passage inline for {n_fixed}/{len(items)} rows", flush=True)

    def _decode_gold(g):
        if not g:
            return None
        if isinstance(g, dict):
            return g
        try:
            return json.loads(g)
        except (TypeError, ValueError):
            return None

    n = len(items)
    bs = args.batch_size
    t0 = time.time()
    scores = []
    fout = open(args.out, "w", encoding="utf-8") if args.out else None
    for i in range(0, n, bs):
        batch = items[i:i + bs]
        enc = tok([b["prompt"] for b in batch], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=args.max_new, do_sample=False, num_beams=1,
                pad_token_id=tok.pad_token_id, eos_token_id=eos_ids,
                repetition_penalty=1.1,
            )
        plen = enc["input_ids"].shape[1]
        for j, b in enumerate(batch):
            resp = tok.decode(gen[j][plen:], skip_special_tokens=True).strip()
            s = reward_json_schema(
                resp, b["fields"], b["strict"],
                passage=(b["passage"] or None), types=b["types"],
                gold_values=_decode_gold(b["gold_values"]),
            )
            scores.append(s)
            if fout:
                fout.write(json.dumps({
                    "idx": i + j, "reward": s, "gen": resp[:2000],
                }, ensure_ascii=False) + "\n")
        if fout: fout.flush()
        done = i + len(batch)
        el = time.time() - t0
        mean_r = sum(scores) / max(1, len(scores))
        print(f"  {done}/{n}  mean_reward={mean_r:.4f}  eta={el*(n-done)/done:.0f}s", flush=True)
    if fout: fout.close()

    mean_r = sum(scores) / max(1, len(scores))
    print(f"\n=== json[da] mean_reward = {mean_r:.4f} on n={n} ===")


if __name__ == "__main__":
    main()
