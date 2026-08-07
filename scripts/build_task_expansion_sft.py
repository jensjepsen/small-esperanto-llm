"""Flatten raw task-expansion JSONL into SFT rows.

Reads rc.jsonl / reason.jsonl / textman.jsonl (one row per wiki article with
multiple subtype outputs each) + prompt_templates.json, emits one flat SFT row
per (article, subtype[, question]) with the standard chat schema.

Output row schema:
    {"messages": [{"role": "user",      "content": <prompt>},
                  {"role": "assistant", "content": <answer>}],
     "subtype":     "rc_numeric" | "reason_causal_chain" | ...,
     "orig_idx":    12345,
     "passage_len": 2400}

Prompt construction is randomized per row:
    task_str = task_template.format(**subtype_fields)
    prompt   = wrapper.format(title=title, text=text, TASK=task_str)

Sample counts:
    rc      = ~21k articles × ~4 Qs   ≈ 84k rows
    reason  = ~21k articles × 6 types = 126k rows
    textman = ~21k articles × 6 types = 126k rows
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# Subtype → task-variant pool key
POOL_FOR = {
    # RC (5 sub-types all share the same QA-style prompt pool)
    "rc_multi_fact":        "rc_qa",
    "rc_numeric":           "rc_qa",
    "rc_attribution":       "rc_qa",
    "rc_ordering":          "rc_qa",
    "rc_causal_inference":  "rc_qa",
    # Reason (5 QA-style + 1 fact-check)
    "reason_causal_chain":  "reason_qa",
    "reason_argumentation": "reason_qa",
    "reason_multi_step":    "reason_qa",
    "reason_ranking":       "reason_qa",
    "reason_analogy":       "reason_qa",
    "reason_fact_check":    "reason_fact_check",
    # Textman (6 distinct)
    "textman_summary":         "textman_summary",
    "textman_rewrite":         "textman_rewrite",
    "textman_style_transfer":  "textman_style_transfer",
    "textman_extraction":      "textman_extraction",
    "textman_elaborate":       "textman_elaborate",
    "textman_genre_transform": "textman_genre_transform",
}


def build_prompt(wrappers, task_pool, task_kwargs, title, text, rng):
    """Sample wrapper + task template, compose the user prompt."""
    task_tpl = rng.choice(task_pool)
    task_str = task_tpl.format(**task_kwargs) if task_kwargs else task_tpl
    wrapper  = rng.choice(wrappers)
    return wrapper.format(title=title, text=text, TASK=task_str)


def rc_rows(raw, wrappers, pools, rng):
    """One SFT row per (article, question)."""
    for q in raw["qs"]:
        subtype = f"rc_{q['type']}"
        pool = pools[POOL_FOR[subtype]]
        prompt = build_prompt(wrappers, pool, {"q": q["q"]},
                              raw["title"], raw["text"], rng)
        yield {
            "messages": [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": q["a"]},
            ],
            "subtype": subtype,
            "orig_idx": raw["orig_idx"],
            "passage_len": raw["text_len"],
        }


def reason_rows(raw, wrappers, pools, rng):
    it = raw["items"]
    # 5 QA-style
    for key in ("causal_chain", "argumentation", "multi_step", "ranking", "analogy"):
        item = it.get(key)
        if not isinstance(item, dict) or not item.get("q") or not item.get("a"):
            continue
        subtype = f"reason_{key}"
        pool = pools[POOL_FOR[subtype]]
        prompt = build_prompt(wrappers, pool, {"q": item["q"]},
                              raw["title"], raw["text"], rng)
        yield {
            "messages": [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": item["a"]},
            ],
            "subtype": subtype,
            "orig_idx": raw["orig_idx"],
            "passage_len": raw["text_len"],
        }
    # fact_check — answer = "VERDICT. REASONING"
    fc = it.get("fact_check")
    if isinstance(fc, dict) and fc.get("claim") and fc.get("verdict") and fc.get("reasoning"):
        subtype = "reason_fact_check"
        pool = pools[POOL_FOR[subtype]]
        prompt = build_prompt(wrappers, pool, {"claim": fc["claim"]},
                              raw["title"], raw["text"], rng)
        yield {
            "messages": [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": f"{fc['verdict']}. {fc['reasoning']}"},
            ],
            "subtype": subtype,
            "orig_idx": raw["orig_idx"],
            "passage_len": raw["text_len"],
        }


def textman_rows(raw, wrappers, pools, rng):
    it = raw["items"]
    for subtype, key, answer_fn, task_kwargs in [
        ("textman_summary",         "summary",         lambda i: i["summary"],           {}),
        ("textman_rewrite",         "rewrite",         lambda i: i["rewrite"],           {}),
        ("textman_style_transfer",  "style_transfer",  lambda i: i["style_transfer"]["text"],
                                                                                          {"style": raw["style_target"]}),
        ("textman_extraction",      "extraction",      lambda i: json.dumps(i["extraction"], ensure_ascii=False, separators=(',', ':')),
                                                                                          {}),
        ("textman_elaborate",       "elaborate",       lambda i: f"KILDEPASSAGE: {i['elaborate'].get('source_passage','')}\n\nUDVIDET:\n{i['elaborate']['expanded']}",
                                                                                          {}),
        ("textman_genre_transform", "genre_transform", lambda i: i["genre_transform"]["text"],
                                                                                          {"genre": raw["genre_target"]}),
    ]:
        if key not in it: continue
        try:
            answer = answer_fn(it)
        except (KeyError, TypeError):
            continue
        if not answer: continue
        pool = pools[POOL_FOR[subtype]]
        prompt = build_prompt(wrappers, pool, task_kwargs,
                              raw["title"], raw["text"], rng)
        yield {
            "messages": [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            "subtype": subtype,
            "orig_idx": raw["orig_idx"],
            "passage_len": raw["text_len"],
        }


BUILDERS = {"rc": rc_rows, "reason": reason_rows, "textman": textman_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/task_expansion_v1"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/task_expansion_v1/sft"))
    ap.add_argument("--templates", type=Path,
                    default=Path("data/task_expansion_v1/prompt_templates.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="Drop rows whose (prompt + answer) exceeds this. "
                         "Set 0 to disable filtering.")
    ap.add_argument("--tokenizer", default="jensjepsen/danish-tokenizer")
    args = ap.parse_args()

    templates = json.loads(args.templates.read_text())
    wrappers = templates["wrapper"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}

    tok = None
    if args.max_tokens > 0:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer {args.tokenizer} for length filter...", flush=True)
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    for name, build in BUILDERS.items():
        src = args.data_dir / f"{name}.jsonl"
        dst = args.out_dir / f"{name}.jsonl"
        if not src.exists():
            print(f"  {name}: SKIP (no {src})")
            continue
        subtype_counts = {}
        dropped = {}
        n = 0
        with dst.open("w") as out:
            for line in src.open():
                raw = json.loads(line)
                if raw.get("reject"): continue
                rng = random.Random(f"{args.seed}:{name}:{raw['orig_idx']}")
                for row in build(raw, wrappers, templates, rng):
                    if tok is not None:
                        n_tok = (len(tok(row["messages"][0]["content"]).input_ids)
                                 + len(tok(row["messages"][1]["content"]).input_ids))
                        if n_tok > args.max_tokens:
                            dropped[row["subtype"]] = dropped.get(row["subtype"], 0) + 1
                            continue
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    subtype_counts[row["subtype"]] = subtype_counts.get(row["subtype"], 0) + 1
                    n += 1
        counts[name] = (n, subtype_counts, dropped)
        print(f"  {name}: {n:,} SFT rows kept, {sum(dropped.values()):,} dropped (>{args.max_tokens} tok)")
        for st, c in sorted(subtype_counts.items()):
            d = dropped.get(st, 0)
            drop_pct = 100*d/(c+d) if c+d else 0
            print(f"    {st}: {c:,}  (dropped {d}, {drop_pct:.1f}%)")

    total = sum(n for n, _, _ in counts.values())
    print(f"\nTotal: {total:,} rows across {len(counts)} datasets → {args.out_dir}")


if __name__ == "__main__":
    main()
