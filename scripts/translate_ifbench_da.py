"""Translate the 300 English prompts in `allenai/IFBench_test` to Danish
via google/gemini-2.5-flash-lite (OpenRouter, async).

Constraint semantics MUST survive the translation intact:
  - keep every literal keyword the row's kwargs pin (letters, phrases,
    forbidden words, start/end tokens) EXACTLY as in the English source;
  - keep every integer (N, num_sentences, ...) EXACTLY as in kwargs;
  - keep literal format markers ([address], Section 1, ***, etc.);
  - preserve the constraint sentence order and phrasing style; only
    render the base task in fluent Danish. Constraint tails often carry
    English tokens by design (e.g. "the word 'meridian' three times") —
    those tokens must not be translated.

Output: a HF-native dataset at data/ifbench_da_v1 (parquet), with the
same schema as the source (`key`, `prompt` [Danish], `instruction_id_list`,
`kwargs`). Then pushed to hub as `jensjepsen/ifbench-da-v1`.

Usage:
  uv run python scripts/translate_ifbench_da.py \\
    --out data/ifbench_da_v1 \\
    --concurrency 32 --seed 42
  uv run python scripts/translate_ifbench_da.py --push
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from datasets import Dataset, load_dataset

MODEL = "google/gemini-2.5-flash-lite"
SRC_DS = "allenai/IFBench_test"
HUB_REPO = "jensjepsen/ifbench-da-v1"


def _read_key(names):
    for name in names:
        for p in [Path.home() / name, Path.home() / f".{name}"]:
            if p.exists():
                return p.read_text().strip()
    return None


_SESSION = None


async def _get_session():
    global _SESSION
    if _SESSION is None:
        import aiohttp
        key = (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_API_KEY")
               or _read_key(["or", "openrouter"]))
        if not key:
            raise SystemExit("No OPENROUTER_API_KEY set and no ~/or key file.")
        _SESSION = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://claude-code-if",
                     "X-Title": "translate-ifbench-da"},
            timeout=aiohttp.ClientTimeout(total=90))
    return _SESSION


_SCHEMA = {
    "type": "object",
    "properties": {"da_prompt": {"type": "string"}},
    "required": ["da_prompt"],
    "additionalProperties": False,
}


_TMPL = """Translate the following English instruction-following prompt to natural, fluent Danish.

CRITICAL RULES — the translation is scored by an automated constraint checker:
1. Keep EVERY integer in the prompt EXACTLY as-is (word counts, sentence counts, "3 times", "at least 5", etc.).
2. Keep EVERY quoted keyword EXACTLY as-is — never translate a word that appears in quotes.
3. Keep every LITERAL format token EXACTLY as-is: bracketed placeholders like [address], section markers like "Section 1", separator strings like "***" or "******", markdown patterns like "*highlighted*", postscript tags like "P.S." or "P.P.S", angle-bracket titles like <<title>>.
4. Keep English letters/tokens inside the CONSTRAINT clauses when the constraint checker cares about them (e.g. "the letter 'e'", "words starting with A", "start with the word Actually"). Translating those breaks the verifier.
5. Translate the base task (the request the user is actually asking about) into fluent Danish. Translate only the natural-language framing of the constraints (imperatives like "Please respond in ..."), never the anchor tokens they name.
6. Keep the overall structure and ordering identical: same number of sentences, same placement of constraint clauses.
7. If the prompt asks the answer to be in a specific language (English, Chinese, Bengali, ...), keep that language name EXACTLY — do not change it to "danish".

Output JSON only: {{"da_prompt": "..."}}.

--- KWARGS PINNED BY THE VERIFIER (do NOT translate any string that appears here) ---
{kwargs_hint}

--- ENGLISH PROMPT ---
{prompt}

Produce the JSON now."""


def _pin_hint(row) -> str:
    """Collect every string kwarg that must survive verbatim."""
    pins = []
    for kw in row["kwargs"]:
        if not kw:
            continue
        for k, v in kw.items():
            if v is None:
                continue
            if isinstance(v, str):
                pins.append(f"{k}={v!r}")
            elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                pins.append(f"{k}={v!r}")
            elif isinstance(v, (int, float)):
                pins.append(f"{k}={v!r}")
    return "; ".join(pins) if pins else "(none)"


async def call_gemini(prompt: str, max_retries: int = 5) -> dict | None:
    session = await _get_session()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "translate", "strict": True, "schema": _SCHEMA},
        },
    }
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions", json=body
            ) as resp:
                if resp.status == 429:
                    await asyncio.sleep(backoff); backoff *= 2; continue
                if resp.status != 200:
                    text = await resp.text()
                    print(f"  http {resp.status}: {text[:200]}", file=sys.stderr)
                    await asyncio.sleep(backoff); backoff *= 2; continue
                data = await resp.json()
                raw = data["choices"][0]["message"]["content"]
                return json.loads(raw)
        except Exception as e:
            print(f"  err {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            await asyncio.sleep(backoff); backoff *= 2
    return None


async def translate_row(row: dict) -> dict | None:
    prompt = _TMPL.format(prompt=row["prompt"], kwargs_hint=_pin_hint(row))
    result = await call_gemini(prompt)
    if not result:
        return None
    da = (result.get("da_prompt") or "").strip()
    if not da or len(da) < 20:
        return None
    return {
        "key": row["key"],
        "prompt": da,
        "instruction_id_list": row["instruction_id_list"],
        "kwargs": row["kwargs"],
        "prompt_en": row["prompt"],
    }


async def bounded(sem, coro):
    async with sem:
        return await coro


async def main_translate(args):
    print(f"Loading {SRC_DS}...", flush=True)
    ds = load_dataset(SRC_DS, split="train")
    print(f"  {len(ds)} rows", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "rows.jsonl"

    # allow --resume: don't retranslate already-done keys
    done_keys = set()
    if jsonl_path.exists() and not args.overwrite:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add(r["key"])
                except Exception:
                    pass
        print(f"  found {len(done_keys)} already-translated keys → skipping", flush=True)

    todo = [r for r in ds if r["key"] not in done_keys]
    print(f"  {len(todo)} to translate", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    n_ok = 0; n_err = 0

    with open(jsonl_path, "a", encoding="utf-8") as fout:
        # process in chunks so we can flush + print
        CHUNK = 60
        for i in range(0, len(todo), CHUNK):
            batch = todo[i:i + CHUNK]
            tasks = [bounded(sem, translate_row(r)) for r in batch]
            results = await asyncio.gather(*tasks)
            for r in results:
                if r is None:
                    n_err += 1
                else:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_ok += 1
            fout.flush()
            done = i + len(batch)
            dt = time.time() - t0
            rate = n_ok / max(1e-6, dt)
            print(f"  {done}/{len(todo)}  ok={n_ok}  err={n_err}  "
                  f"rate={rate:.2f}/s  elapsed={dt:.0f}s", flush=True)

    print(f"\nTotal ok={n_ok} err={n_err} in {time.time()-t0:.0f}s", flush=True)

    # Re-read jsonl → HF Dataset (parquet on disk)
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["key"])
    ds_out = Dataset.from_list(rows)
    save_dir = out_dir / "hf"
    ds_out.save_to_disk(str(save_dir))
    parquet_path = out_dir / "ifbench_da.parquet"
    ds_out.to_parquet(str(parquet_path))
    print(f"Saved HF dataset: {save_dir}", flush=True)
    print(f"Saved parquet:    {parquet_path}", flush=True)
    print(f"n={len(ds_out)}", flush=True)

    if _SESSION is not None:
        await _SESSION.close()


def push(args):
    from huggingface_hub import HfApi
    from datasets import load_from_disk
    src = Path(args.out) / "hf"
    if not src.exists():
        raise SystemExit(f"{src} not found; run translation first.")
    ds = load_from_disk(str(src))
    print(f"Pushing {len(ds)} rows to {HUB_REPO}...", flush=True)
    ds.push_to_hub(HUB_REPO, private=False)
    print("Done.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ifbench_da_v1",
                    help="output dir for jsonl + hf + parquet")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true",
                    help="Ignore existing jsonl (default: resume/skip already-done keys)")
    ap.add_argument("--push", action="store_true",
                    help="Skip translation; only push existing --out/hf to HF hub.")
    args = ap.parse_args()

    if args.push:
        push(args)
        return
    asyncio.run(main_translate(args))


if __name__ == "__main__":
    main()
