"""RL-data farming: sample K chains per problem, grade against procedural gold,
output (problem, sampled chain, reward) triples filtered to the RL-positive
bucket (1 ≤ wins < K).

Works on any HF dataset config that exposes both `question_eo` and `answer`
fields. Designed for jensjepsen/esperanto-word-problems default config.

For each problem:
  1. Build chat prompt
  2. Generate K samples via num_return_sequences=K (one forward pass)
  3. Grade each sample via probe_algebra.has_answer against gold
  4. Classify problem by wins/K
  5. Emit one row per sample with reward {0.0, 1.0} + a `bucket` field

Usage:
  uv run python scripts/rl_data_farm.py \\
    --checkpoint runs/sft/sft_v6/checkpoint-24000 \\
    --dataset jensjepsen/esperanto-word-problems --config default \\
    --n 200 --k 8 --temperature 0.7 \\
    --out runs/rl/wp_sft_v6_ck24k_k8.jsonl

Output JSONL fields:
  problem_idx, type, question_eo, gold, k, wins_in_k, bucket,
  sample_idx, chain_eo, reward
where bucket ∈ {"never" (0/K), "rl-positive" (1..K-1), "always" (K/K)}.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from esperanto_lm.data import _morpheme_preprocess
from probe_algebra import has_answer

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"
SPECIAL = (USER, ASST, END)
SKIP = {"<s>", "</s>", "<pad>", "<unk>", USER, ASST, END}


def pp(s):
    pat = "(" + "|".join(re.escape(t) for t in SPECIAL) + ")"
    return " ".join(p if p in SPECIAL else _morpheme_preprocess(p)
                    for p in re.split(pat, s))


def decode(tok, ids):
    toks = tok.convert_ids_to_tokens(ids)
    toks = [t for t in toks if t not in SKIP]
    return "".join(t if t != "<w>" else " " for t in toks).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="jensjepsen/esperanto-word-problems",
                    help="HF repo id, or local JSONL path with `question_eo`+`answer` fields")
    ap.add_argument("--config", default="default")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=200, help="problems to sample")
    ap.add_argument("--k", type=int, default=8, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--tokenizer", default="tokenizer_morpheme")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-every", type=int, default=20,
                    help="print progress every N problems")
    args = ap.parse_args()

    print(f"loading checkpoint {args.checkpoint}", flush=True)
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    tok.add_special_tokens({"additional_special_tokens": list(SPECIAL)})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.float16
    ).cuda().eval()
    model.resize_token_embeddings(len(tok))
    end_id = tok.convert_tokens_to_ids(END)

    if Path(args.dataset).exists():
        print(f"loading local JSONL {args.dataset}", flush=True)
        rows = [json.loads(l) for l in Path(args.dataset).open()]
        from datasets import Dataset as HFDataset
        ds = HFDataset.from_list(rows)
    else:
        print(f"loading dataset {args.dataset}:{args.config}[{args.split}]", flush=True)
        ds = load_dataset(args.dataset, args.config, split=args.split)
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))
    print(f"  selected {len(ds)} problems  (k={args.k}, T={args.temperature})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.out.open("w")

    buckets = Counter()
    bucket_for = lambda w, K: "never" if w == 0 else ("always" if w == K else "rl-positive")
    sample_writes = 0
    rl_positive_problems = 0
    t0 = time.time()

    for prob_idx, row in enumerate(ds):
        q = row["question_eo"]
        gold = str(row["answer"])
        problem_type = row.get("type", "?")
        prompt = pp(f"{USER} {q} {ASST} ")
        in_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()

        with torch.no_grad():
            out = model.generate(
                in_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.k,
                pad_token_id=tok.pad_token_id,
                repetition_penalty=1.1,
                eos_token_id=end_id,
            )
        # out shape: [K, total_len]; strip prompt from each
        chains = [decode(tok, seq[in_ids.shape[1]:].tolist()) for seq in out]

        sample_results = [has_answer(c, gold) for c in chains]
        wins = sum(sample_results)
        bucket = bucket_for(wins, args.k)
        buckets[bucket] += 1
        if bucket == "rl-positive":
            rl_positive_problems += 1

        for s_idx, (c, ok) in enumerate(zip(chains, sample_results)):
            out_f.write(json.dumps({
                "problem_idx": prob_idx,
                "type": problem_type,
                "question_eo": q,
                "gold": gold,
                "k": args.k,
                "wins_in_k": wins,
                "bucket": bucket,
                "sample_idx": s_idx,
                "chain_eo": c,
                "reward": 1.0 if ok else 0.0,
            }, ensure_ascii=False) + "\n")
            sample_writes += 1
        out_f.flush()

        if (prob_idx + 1) % args.report_every == 0:
            elapsed = time.time() - t0
            rate = (prob_idx + 1) / elapsed * 60
            eta = (len(ds) - prob_idx - 1) / max(1, rate) * 60
            print(f"  [{prob_idx+1}/{len(ds)}]  rate={rate:.1f}/min  "
                  f"eta={eta:.0f}s  buckets={dict(buckets)}  "
                  f"rl-positive={rl_positive_problems}",
                  flush=True)

    out_f.close()
    print(f"\ndone: {sample_writes} samples → {args.out}")
    print(f"  bucket distribution over {len(ds)} problems: {dict(buckets)}")
    print(f"  RL-positive problems: {rl_positive_problems}/{len(ds)} "
          f"({100*rl_positive_problems/len(ds):.1f}%)")
    print(f"  wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
