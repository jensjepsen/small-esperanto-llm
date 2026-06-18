"""Distillation pipeline: synthesize EO chat SFT data from English instructions.

For each English instruction Q_EN:
    1. Translate Q_EN -> Q_EO via v5b (en->eo)
    2. Run LFM on the original Q_EN -> A_EN  (clean English, no translation noise)
    3. Translate A_EN -> A_EO via v5b (en->eo)
    4. Write JSONL row with all four fields.

Skipped/filtered rows are still written with a `skipped` flag and reason so the
pipeline is auditable end-to-end.

Resumable: appends to the output file and skips any indices already written.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sp_tokenizer import SPMTokenizer


PROSE_SYSTEM_PROMPT = (
    "Answer in plain prose. Do not use markdown formatting, bullet points, "
    "numbered lists, headers, or asterisks. Write a single flowing answer."
)


def strip_markdown(text: str) -> str:
    """Remove markdown formatting so v5b doesn't choke on it.

    LFM is markdown-happy; v5b's SP tokenizer doesn't know **, ##, list bullets
    and emits the unk token (⁇), and isolated words inside list items lose
    context for translation.
    """
    # Fenced code blocks
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Inline code
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Bold / italic — keep the inner text
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Headers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Numbered list markers ("1. ", "12) ")
    text = re.sub(r"^\s*\d+[.\)]\s+", "", text, flags=re.MULTILINE)
    # Bulleted list markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    # Blockquote markers
    text = re.sub(r"^\s*>\s*", "", text, flags=re.MULTILINE)
    # Collapse 2+ newlines → ". " so prose flows for the translator
    text = re.sub(r"\n{2,}", ". ", text)
    # Collapse single newlines → space
    text = re.sub(r"\n+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Multiple periods → single
    text = re.sub(r"\.{2,}", ".", text)
    return text


# Hedge / refusal patterns common in LFM2.5 outputs that are useless as SFT.
HEDGE_PATTERNS = re.compile(
    r"\b("
    r"i (?:cannot|can't|am unable|am not able)|"
    r"as an ai|as a language model|"
    r"i (?:don'?t|do not) have (?:access|the ability|information)|"
    r"i'?m sorry,? but|"
    r"the answer depends on (?:context|the context)|"
    r"it depends on (?:the )?context|"
    r"i'?m (?:not |un)sure"
    r")\b",
    re.IGNORECASE,
)

# Prompts that don't translate well (translation tasks, code, etc.) — drop upstream.
SKIP_INSTRUCTION_PATTERNS = re.compile(
    r"\b("
    r"translate\b.*\b(?:to|into)\b|"
    r"write (?:the )?(?:following|this) in|"
    r"convert .* to (?:json|xml|html|csv|yaml)|"
    r"in (?:french|spanish|german|chinese|japanese|korean|arabic|russian|italian|portuguese|dutch|polish|hindi)"
    r")\b",
    re.IGNORECASE,
)


def make_q_text(instruction: str, input_text: str) -> str:
    """Combine Alpaca's (instruction, input) into a single user-style question."""
    if input_text and input_text.strip():
        return instruction.strip() + "\n\n" + input_text.strip()
    return instruction.strip()


def load_alpaca(n: int, skip: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    out = []
    i = skip
    while len(out) < n and i < len(ds):
        row = ds[i]
        q = make_q_text(row["instruction"], row.get("input", ""))
        i += 1
        if SKIP_INSTRUCTION_PATTERNS.search(q):
            continue
        out.append(q)
    return out


def load_gsm8k(n: int, skip: int, split: str = "train") -> tuple[list[str], list[str]]:
    """Return (questions, gold_answers) for English GSM8K."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    questions, golds = [], []
    end = min(skip + n, len(ds))
    for i in range(skip, end):
        questions.append(ds[i]["question"].strip())
        golds.append(extract_final_number(ds[i]["answer"]))
    return questions, golds


_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def extract_final_number(text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K-style solution."""
    if not text:
        return None
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    m = re.search(r"\\boxed\{([-+]?\d[\d,]*\.?\d*)\}", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    m = re.search(r"(?:final answer|answer)\s*[:=]\s*\$?([-+]?\d[\d,]*\.?\d*)",
                  text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfm-model", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--mt-checkpoint", default="/mnt/data/espllm/runs/mt/eneo_v5b/final")
    ap.add_argument("--mt-tokenizer", default="mt/data/tokenizer/spm_eneo_32k.model")
    ap.add_argument("--source", default="alpaca", choices=["alpaca", "gsm8k"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--skip", type=int, default=0, help="Skip first N prompts in source dataset")
    ap.add_argument("--gsm8k-filter", action="store_true",
                    help="GSM8K-only: keep only LFM gens whose extracted final answer "
                         "matches the gold. Turns noisy teacher into clean SFT signal.")
    ap.add_argument("--lfm-batch-size", type=int, default=256)
    ap.add_argument("--mt-batch-size", type=int, default=256)
    ap.add_argument("--lfm-max-new-tokens", type=int, default=256)
    ap.add_argument("--mt-num-beams", type=int, default=2)
    ap.add_argument("--mt-max-length", type=int, default=256)
    ap.add_argument("--max-q-chars", type=int, default=800,
                    help="Drop prompts longer than this — v5b can't translate them well")
    ap.add_argument("--max-a-chars", type=int, default=1500,
                    help="Truncate LFM answers to this length before en->eo translation")
    ap.add_argument("--max-student-tokens", type=int, default=512,
                    help="Drop rows whose Q_EO+A_EO exceed this in the student tokenizer "
                         "(the base 44k model can't see past max_position_embeddings).")
    ap.add_argument("--student-tokenizer", default="tokenizer_morpheme",
                    help="Path or name of the student's tokenizer for budget filtering")
    ap.add_argument("--chunk-size", type=int, default=1024,
                    help="Length-sort and write progress every N prompts")
    ap.add_argument("--n-gens", type=int, default=1,
                    help="Number of distinct LFM answers per prompt. >1 implies --sample.")
    ap.add_argument("--sample", action="store_true",
                    help="Use sampling (do_sample=True) for LFM. Required for n-gens > 1.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--out", default="mt/runs/distill_alpaca_lfm.jsonl")
    ap.add_argument("--hf-cache", default="/mnt/data/hf_cache")
    args = ap.parse_args()

    if args.n_gens > 1 and not args.sample:
        print(f"--n-gens={args.n_gens} forces --sample")
        args.sample = True

    os.environ.setdefault("HF_HOME", args.hf_cache)
    from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel

    # --- resume: scan existing output. With multi-gen, a prompt is "done" when
    # all K gens have been written; otherwise we re-do the prompt fully. ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys: set[tuple[int, int]] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add((r["i"], r.get("gen_idx", 0)))
                except Exception:
                    pass
        print(f"Resume: {len(done_keys)} rows already in {out_path}")

    # --- load source ---
    print(f"Loading {args.source} prompts (n={args.n}, skip={args.skip})…")
    en_gold: list[str | None] = []
    if args.source == "alpaca":
        en_questions = load_alpaca(args.n, args.skip)
        en_gold = [None] * len(en_questions)
    elif args.source == "gsm8k":
        en_questions, en_gold = load_gsm8k(args.n, args.skip)
    print(f"  {len(en_questions)} prompts after upstream filter")
    if args.gsm8k_filter and args.source != "gsm8k":
        raise SystemExit("--gsm8k-filter requires --source gsm8k")
    todo_indices = [
        i for i in range(len(en_questions))
        if sum(1 for g in range(args.n_gens) if (i, g) in done_keys) < args.n_gens
    ]
    if not todo_indices:
        print("Nothing to do.")
        return
    print(f"  {len(todo_indices)} prompts to process ({args.n_gens} gen{'s' if args.n_gens>1 else ''} each)")

    # --- load models ---
    print(f"Loading MT {args.mt_checkpoint} on cuda fp16…")
    mt_tok = SPMTokenizer(args.mt_tokenizer)
    mt_model = MarianMTModel.from_pretrained(args.mt_checkpoint).half().to("cuda").eval()
    mt_model.generation_config.no_repeat_ngram_size = 5

    print(f"Loading LFM {args.lfm_model} on cuda fp16…")
    lfm_tok = AutoTokenizer.from_pretrained(args.lfm_model)
    lfm_tok.padding_side = "left"
    if lfm_tok.pad_token_id is None:
        lfm_tok.pad_token_id = lfm_tok.eos_token_id
    lfm_model = AutoModelForCausalLM.from_pretrained(args.lfm_model, dtype=torch.float16).to("cuda").eval()
    print(f"  GPU mem after loads = {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # --- student tokenizer (for budget filtering only — CPU is fine) ---
    print(f"Loading student tokenizer {args.student_tokenizer}…")
    student_tok = AutoTokenizer.from_pretrained(args.student_tokenizer)

    def student_token_count(text: str) -> int:
        return len(student_tok(text, add_special_tokens=False).input_ids)

    def _sorted_batches(texts: list[str], bs: int) -> list[tuple[list[int], list[str]]]:
        """Return [(original_indices, sorted_texts), ...] grouped into length-uniform
        batches. Caller is responsible for un-sorting results back to original order.
        """
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        batches = []
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            batches.append((idx, [texts[j] for j in idx]))
        return batches

    def mt_translate(texts: list[str], tgt_lang: str, desc: str) -> list[str]:
        out: list[str | None] = [None] * len(texts)
        batches = _sorted_batches(texts, args.mt_batch_size)
        for idx, chunk in tqdm(batches, desc=desc, leave=False):
            ids = [mt_tok.encode(t, lang=tgt_lang) for t in chunk]
            be = mt_tok.pad_batch(ids)
            inp = be.input_ids.to("cuda")
            attn = be.attention_mask.to("cuda")
            with torch.no_grad():
                gen = mt_model.generate(
                    input_ids=inp, attention_mask=attn,
                    num_beams=args.mt_num_beams, max_length=args.mt_max_length,
                    early_stopping=True, no_repeat_ngram_size=5,
                )
            for orig_i, seq in zip(idx, gen):
                out[orig_i] = mt_tok.decode(seq)
        return [t if t is not None else "" for t in out]

    def lfm_generate(prompts: list[str]) -> list[list[str]]:
        """Return per-prompt list of n_gens answers (length == args.n_gens)."""
        K = args.n_gens
        out: list[list[str] | None] = [None] * len(prompts)
        chat_prompts = [
            lfm_tok.apply_chat_template(
                [{"role": "system", "content": PROSE_SYSTEM_PROMPT},
                 {"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=False,
            )
            for p in prompts
        ]
        # When sampling with batch_size B, num_return_sequences=K produces a
        # tensor of shape (B*K, T). Smaller LFM batch to keep memory bounded.
        eff_bs = max(1, args.lfm_batch_size // max(1, K))
        batches = _sorted_batches(chat_prompts, eff_bs)
        for idx, batch in tqdm(batches, desc="LFM gen", leave=False):
            enc = lfm_tok(batch, return_tensors="pt", padding=True, truncation=False).to("cuda")
            with torch.no_grad():
                gen = lfm_model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=args.lfm_max_new_tokens,
                    do_sample=args.sample,
                    temperature=args.temperature if args.sample else 1.0,
                    top_p=args.top_p if args.sample else 1.0,
                    num_return_sequences=K,
                    pad_token_id=lfm_tok.pad_token_id,
                )
            in_len = enc["input_ids"].shape[1]
            decoded = lfm_tok.batch_decode(gen[:, in_len:], skip_special_tokens=True)
            # decoded has shape len(batch)*K; gens of batch[i] live at i*K..i*K+K
            for i_in_batch, orig_i in enumerate(idx):
                out[orig_i] = decoded[i_in_batch * K : (i_in_batch + 1) * K]
        return [t if t is not None else [""] * K for t in out]

    # Process in chunks. Each chunk: length-sort across all 3 stages (so batches
    # are uniform-length), then write the whole chunk before moving on. Lets us
    # see progress + resume mid-run, at small efficiency cost vs full-set sort.
    CHUNK = args.chunk_size
    t_start = time.perf_counter()
    total_written = total_kept = total_skipped_long = total_skipped_hedge = total_skipped_budget = 0
    t_q_total = t_lfm_total = t_a_total = 0.0
    BUDGET_Q_ONLY = int(args.max_student_tokens * 0.7)   # leave at least 30% for answer

    with out_path.open("a") as fout:
        for chunk_start in tqdm(range(0, len(todo_indices), CHUNK), desc="chunks"):
            chunk_idx = todo_indices[chunk_start : chunk_start + CHUNK]
            chunk_q_en = [en_questions[i] for i in chunk_idx]

            # Pre-filter: prompts too long in characters (proxy for v5b translation cost)
            long_mask = [len(q) > args.max_q_chars for q in chunk_q_en]

            keep_q_idx = [j for j, m in enumerate(long_mask) if not m]
            keep_q_en = [chunk_q_en[j] for j in keep_q_idx]

            # Pass 1: Q_EN -> Q_EO
            t0 = time.perf_counter()
            q_eo_keep = mt_translate(keep_q_en, tgt_lang="eo", desc="Q en→eo") if keep_q_en else []
            t_q_total += time.perf_counter() - t0

            # Budget filter on Q_EO — drop rows whose Q_EO alone already eats most of the budget
            q_budget_mask = [student_token_count(eo) > BUDGET_Q_ONLY for eo in q_eo_keep]
            # Keep going only on rows that pass the budget filter
            q_pass = [j for j, m in enumerate(q_budget_mask) if not m]
            lfm_input = [keep_q_en[j] for j in q_pass]

            # Pass 2: LFM on original EN (only rows that passed budget filter).
            # Returns per-prompt list of K answers; flatten to length P*K.
            K = args.n_gens
            t0 = time.perf_counter()
            a_en_nested = lfm_generate(lfm_input) if lfm_input else []
            a_en_kept = [a for gens in a_en_nested for a in gens]
            t_lfm_total += time.perf_counter() - t0

            # Strip markdown + hedge filter (all K*P answers, flat)
            a_en_stripped = [strip_markdown(a)[: args.max_a_chars].strip() for a in a_en_kept]
            hedge_mask = [not s or bool(HEDGE_PATTERNS.search(s)) for s in a_en_stripped]

            # GSM8K answer-correctness filter: drop gens whose extracted final number
            # doesn't match the gold. Uses gold from chunk_idx → q_pass → keep_q_idx mapping.
            wrong_mask = [False] * len(a_en_stripped)
            if args.gsm8k_filter:
                for flat in range(len(a_en_stripped)):
                    if hedge_mask[flat]:
                        continue
                    p_idx = flat // K
                    j = keep_q_idx[q_pass[p_idx]]  # position back in chunk_idx
                    gold = en_gold[chunk_idx[j]]
                    pred = extract_final_number(a_en_stripped[flat])
                    if pred is None or gold is None or pred != gold:
                        wrong_mask[flat] = True
            a_to_translate = [(k, s) for k, s in enumerate(a_en_stripped)
                              if not hedge_mask[k] and not wrong_mask[k]]

            # Pass 3: A_EN -> A_EO
            t0 = time.perf_counter()
            a_eo_translated = []
            if a_to_translate:
                a_eo_translated = mt_translate([s for _, s in a_to_translate], tgt_lang="eo", desc="A en→eo")
            t_a_total += time.perf_counter() - t0
            a_eo_out = [""] * len(a_en_kept)
            for (k, _), eo in zip(a_to_translate, a_eo_translated):
                a_eo_out[k] = eo

            # Final budget filter: total Q_EO + A_EO must fit in student.
            # Index k into flat arrays maps to (p_idx = k // K, gen_idx = k % K).
            budget_mask = []
            for k in range(len(a_eo_out)):
                if hedge_mask[k]:
                    budget_mask.append(False)
                    continue
                p_idx = k // K
                eo_idx = q_pass[p_idx]
                total = student_token_count(q_eo_keep[eo_idx]) + student_token_count(a_eo_out[k])
                budget_mask.append(total > args.max_student_tokens)

            # Reassemble all rows for this chunk and write in order.
            # For long/q-budget skips we write 1 row (gen_idx=0). For prompts
            # that reached LFM we write K rows (one per gen).
            rows_to_write: list[dict] = []
            for j, idx_in_chunk in enumerate(chunk_idx):
                if long_mask[j]:
                    rows_to_write.append({"i": idx_in_chunk, "gen_idx": 0,
                                          "skipped": True, "reason": "q_too_long",
                                          "q_en": chunk_q_en[j]})
                    continue
                k_in_keep = keep_q_idx.index(j)
                q_eo = q_eo_keep[k_in_keep]
                if q_budget_mask[k_in_keep]:
                    rows_to_write.append({"i": idx_in_chunk, "gen_idx": 0,
                                          "skipped": True, "reason": "q_eo_too_long",
                                          "q_en": chunk_q_en[j], "q_eo": q_eo})
                    continue
                p_idx = q_pass.index(k_in_keep)
                for g in range(K):
                    flat = p_idx * K + g
                    a_en = a_en_kept[flat]
                    a_en_clean = a_en_stripped[flat]
                    if hedge_mask[flat]:
                        rows_to_write.append({"i": idx_in_chunk, "gen_idx": g,
                                              "skipped": True, "reason": "hedge_or_empty",
                                              "q_en": chunk_q_en[j], "q_eo": q_eo,
                                              "a_en": a_en, "a_en_clean": a_en_clean})
                        continue
                    if args.gsm8k_filter and wrong_mask[flat]:
                        rows_to_write.append({"i": idx_in_chunk, "gen_idx": g,
                                              "skipped": True, "reason": "wrong_answer",
                                              "q_en": chunk_q_en[j], "q_eo": q_eo,
                                              "a_en": a_en, "a_en_clean": a_en_clean,
                                              "gold": en_gold[idx_in_chunk]})
                        continue
                    a_eo = a_eo_out[flat]
                    if budget_mask[flat]:
                        rows_to_write.append({"i": idx_in_chunk, "gen_idx": g,
                                              "skipped": True, "reason": "total_too_long",
                                              "q_en": chunk_q_en[j], "q_eo": q_eo,
                                              "a_en": a_en, "a_en_clean": a_en_clean,
                                              "a_eo": a_eo})
                        continue
                    rows_to_write.append({"i": idx_in_chunk, "gen_idx": g,
                                          "skipped": False,
                                          "q_en": chunk_q_en[j], "q_eo": q_eo,
                                          "a_en": a_en, "a_en_clean": a_en_clean,
                                          "a_eo": a_eo})

            for r in rows_to_write:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                total_written += 1
                if r.get("skipped"):
                    reason = r.get("reason", "")
                    if reason == "q_too_long" or reason == "q_eo_too_long" or reason == "total_too_long":
                        total_skipped_long += 1
                    elif reason == "hedge_or_empty":
                        total_skipped_hedge += 1
                else:
                    total_kept += 1
            fout.flush()

    dt = time.perf_counter() - t_start
    print(f"\nWrote {total_written} rows in {dt:.0f}s ({total_written/max(1,dt):.2f} rows/s)")
    print(f"  kept={total_kept}  skipped_too_long={total_skipped_long}  skipped_hedge={total_skipped_hedge}")
    print(f"  per-stage wall time:")
    print(f"    Q  en→eo : {t_q_total:6.1f}s  ({t_q_total/dt*100:.0f}%)")
    print(f"    LFM gen  : {t_lfm_total:6.1f}s  ({t_lfm_total/dt*100:.0f}%)")
    print(f"    A  en→eo : {t_a_total:6.1f}s  ({t_a_total/dt*100:.0f}%)")


if __name__ == "__main__":
    main()
