"""Generate English user questions from a topic list using LFM2.5.

Topic-grounded self-instruct (Test 3 winner from the feasibility probe):
- Loads topics from a built-in seed list, a JSONL file, or a HF Hub dataset
- For each topic, asks LFM to write K diverse user questions
- Dedupes by lowercased+normalized text (Jaccard threshold optional)
- Writes JSONL of {"topic": "...", "question": "...", "i": N, "gen_idx": K}

The output JSONL is ready to feed `distill_lfm_to_eo.py` as a new source
(set `--source self_instruct --self-instruct-path <path>` once wired up).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


# Hand-curated seeds across diverse domains. Acts as the bootstrap pool
# and as a fallback when no --topics-file/--topics-hf is provided.
DEFAULT_SEEDS = [
    # science
    "photosynthesis", "DNA replication", "the immune system", "black holes",
    "plate tectonics", "the water cycle", "the Periodic Table",
    "Newton's laws of motion", "the structure of an atom", "evolution by natural selection",
    "vaccines", "antibiotics", "the human nervous system", "climate change",
    # math
    "the Pythagorean theorem", "prime numbers", "the concept of zero",
    "logarithms", "set theory basics", "imaginary numbers",
    # history
    "the French Revolution", "the fall of the Roman Empire", "World War I",
    "the Renaissance", "the Silk Road", "ancient Egypt", "the printing press",
    "the Industrial Revolution", "the moon landing",
    # geography
    "the Amazon rainforest", "Mount Everest", "the Sahara desert",
    "the Mediterranean climate", "the Great Barrier Reef",
    # cooking
    "how to bake bread", "how to make a basic stock", "fermentation in cooking",
    "knife skills basics", "how to cook rice properly",
    # everyday
    "raising a puppy", "growing tomatoes", "saving for retirement",
    "buying a used car", "improving sleep quality", "managing stress",
    "learning a new language", "remote work productivity",
    # arts & culture
    "the structure of a haiku", "impressionist painting", "jazz music origins",
    "classical mythology", "modern architecture",
    # tech
    "what HTTP is", "how Wi-Fi works", "the basics of public-key cryptography",
    "the difference between RAM and storage", "what an operating system does",
    # philosophy
    "the trolley problem", "Stoicism", "the Socratic method",
    "Kant's categorical imperative",
    # creative
    "write a short poem about autumn", "write a haiku about the ocean",
    "tell a 3-sentence ghost story", "write a love letter from a robot",
    # listicle / opinion / how-to
    "five benefits of regular exercise", "three tips for studying for an exam",
    "pros and cons of remote work", "how to prepare for a job interview",
    "ideas for a weekend trip near a city",
]


# Per-topic question variants to encourage diversity within a single topic.
# Sampled cyclically when K > len(VARIANTS).
QUESTION_VARIANTS = [
    "explain it to a beginner",
    "ask a comparison question",
    "ask for a step-by-step explanation",
    "ask a 'why' question",
    "ask for an opinion or recommendation",
    "ask a follow-up that requires reasoning",
    "ask a short factual question",
    "ask for a list",
]


def load_topics(path: str | None, hf_spec: str | None) -> list[str]:
    if hf_spec:
        from datasets import load_dataset
        # hf_spec = "namespace/dataset:split:field"
        parts = hf_spec.split(":")
        name = parts[0]
        split = parts[1] if len(parts) > 1 else "train"
        field = parts[2] if len(parts) > 2 else "title"
        ds = load_dataset(name, split=split)
        return [str(r[field]) for r in ds if r.get(field)]
    if path:
        topics = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # support {"topic": "..."} or {"title": "..."} or {"name": "..."}
                for k in ("topic", "title", "name", "label"):
                    if k in obj:
                        topics.append(str(obj[k]))
                        break
                else:
                    topics.append(json.dumps(obj))
            except json.JSONDecodeError:
                topics.append(line)
        return topics
    return list(DEFAULT_SEEDS)


_NORM_RE = re.compile(r"[^a-z0-9 ]+")


def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = _NORM_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfm-model", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--topics-file", type=str, default=None,
                    help="JSONL of topics (uses 'topic'/'title'/'name'/'label' field, "
                         "or one topic per line)")
    ap.add_argument("--topics-hf", type=str, default=None,
                    help="HF dataset spec: 'namespace/dataset[:split[:field]]'")
    ap.add_argument("--n-topics", type=int, default=0,
                    help="Cap topics processed (0 = all)")
    ap.add_argument("--k", type=int, default=2,
                    help="Questions per topic")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dedupe-jaccard", type=float, default=0.7,
                    help="Drop questions whose normalized-word Jaccard with any "
                         "kept question exceeds this. 1.0 = no dedupe.")
    ap.add_argument("--out", default="mt/runs/self_instruct_topics.jsonl")
    ap.add_argument("--hf-cache", default="/tmp/hf-cache")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_cache)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    topics = load_topics(args.topics_file, args.topics_hf)
    if args.n_topics:
        topics = topics[: args.n_topics]
    print(f"Loaded {len(topics)} topics. K={args.k} → up to {len(topics)*args.k} questions.")

    print(f"Loading {args.lfm_model}…")
    tok = AutoTokenizer.from_pretrained(args.lfm_model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.lfm_model, dtype=torch.float16
    ).to("cuda").eval()

    SYSTEM = (
        "You write user questions for a Q/A training dataset. "
        "Output ONLY the question, nothing else. The question should be a "
        "natural, self-contained user prompt that someone might ask a chatbot, "
        "answerable in plain English text."
    )

    def build_prompt(topic: str, variant: str) -> str:
        user = f"Topic: {topic}\n\nWrite a user question about this topic. Style hint: {variant}.\n\nQuestion:"
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False,
        )

    # Build (topic_idx, gen_idx, prompt_text) triples, batch generate.
    prompts = []
    meta = []  # (topic_idx, gen_idx, topic)
    for i, t in enumerate(topics):
        for g in range(args.k):
            v = QUESTION_VARIANTS[g % len(QUESTION_VARIANTS)]
            prompts.append(build_prompt(t, v))
            meta.append((i, g, t))

    # Resume: if out file exists, read which (i, gen_idx) keys are already done.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys: set[tuple[int, int]] = set()
    kept_norms: set[frozenset] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add((r["i"], r["gen_idx"]))
                    kept_norms.add(frozenset(norm_text(r["question"]).split()))
                except Exception:
                    pass
        print(f"Resume: {len(done_keys)} rows already in {out_path}")

    todo_idx = [j for j, (i, g, _) in enumerate(meta) if (i, g) not in done_keys]
    if not todo_idx:
        print("Nothing to do.")
        return
    print(f"  {len(todo_idx)} prompts to generate")

    BS = args.batch_size
    written = dropped_dup = 0
    with out_path.open("a") as fout:
        from tqdm import tqdm
        for s in tqdm(range(0, len(todo_idx), BS), desc="gen"):
            chunk = todo_idx[s : s + BS]
            batch = [prompts[j] for j in chunk]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=False).to("cuda")
            with torch.no_grad():
                out = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tok.pad_token_id,
                )
            in_len = enc["input_ids"].shape[1]
            decoded = tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
            for j, txt in zip(chunk, decoded):
                i, g, topic = meta[j]
                q = txt.strip().split("\n")[0].strip().strip('"').strip()
                # quality filters
                if len(q) < 8 or len(q) > 280:
                    continue
                if "?" not in q and not q.lower().startswith(("write", "give", "list", "explain", "describe", "compare", "suggest")):
                    continue
                qn = frozenset(norm_text(q).split())
                # jaccard dedupe vs kept set
                if args.dedupe_jaccard < 1.0:
                    is_dup = False
                    for prev in kept_norms:
                        if jaccard(qn, prev) > args.dedupe_jaccard:
                            is_dup = True
                            break
                    if is_dup:
                        dropped_dup += 1
                        continue
                kept_norms.add(qn)
                fout.write(json.dumps({
                    "i": i, "gen_idx": g, "topic": topic, "question": q,
                }, ensure_ascii=False) + "\n")
                written += 1
            fout.flush()

    print(f"\nWrote {written} questions. Dropped {dropped_dup} as near-duplicates.")


if __name__ == "__main__":
    main()
