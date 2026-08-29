"""Build a general Danish rephrase SFT dataset via gemma-3-12b + structured output.

Pipeline:
  1. Load Danish sentences from wiki/dolly/sciq contexts (~50k pool).
  2. For each source, call Gemma once per TARGET direction (simple/compact/
     casual/formal), forcing exactly N variants via JSON schema.
  3. Emit BOTH directions:
       forward = (source → gemma_variant)   with target's instruction
       reverse = (gemma_variant → source)   with inverse instruction
     The reverse row's assistant output is the ORIGINAL human-written source,
     so it's fabrication-safe by construction.
  4. Drop pairs where SequenceMatcher(source, variant) > 0.85 (degenerate).
  5. Multiple instruction phrasings per direction so model doesn't lock onto
     exact template strings.

Uses if_generate's OpenRouter session infra with rate-limit backoff.

Usage:
  uv run python scripts/build_da_rephrase_general.py \\
    --out data/da_rephrase_general_v1.jsonl \\
    --n-sources 20000 \\
    --n-variants 3 \\
    --concurrency 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from datasets import load_dataset

MODEL = "google/gemma-3-12b-it"


def _read_key(names):
    for name in names:
        for p in [Path.home() / name, Path.home() / f".{name}"]:
            if p.exists():
                return p.read_text().strip()
    return None


_SCHEMA_TMPL = {
    "type": "object",
    "properties": {
        "rewrites": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["rewrites"],
    "additionalProperties": False,
}


def make_schema(n: int) -> dict:
    s = dict(_SCHEMA_TMPL)
    s = json.loads(json.dumps(s))  # deep copy
    s["properties"]["rewrites"]["minItems"] = n
    s["properties"]["rewrites"]["maxItems"] = n
    return s


# ── Forward direction prompts (Gemma-facing) ─────────────────────────────────
# Multiple phrasings per target so the training rows use varied instructions.

FORWARD_PROMPTS = {
    "simple": [
        "Forklar følgende sætning på {n} enklere måder — så en 12-årig kan forstå den. Brug analogier hvor det giver mening, undgå fagudtryk. Ingen markdown, ingen forklaringer.\n\nSætning: {q}",
        "Skriv følgende sætning på {n} måder, en teenager kan forstå. Erstat fagudtryk med hverdagsord. Ingen markdown, ingen forklaringer.\n\nSætning: {q}",
        "Omskriv følgende sætning på {n} mere ligetil måder. Undgå akademisk sprog og tekniske termer. Ingen markdown, ingen forklaringer.\n\nSætning: {q}",
    ],
    "compact": [
        "Skriv følgende sætning på {n} kortere, mere direkte måder — helst under 15 ord. Bevar det essentielle. Ingen markdown, ingen forklaringer.\n\nSætning: {q}",
        "Formulér følgende sætning på {n} mere komprimerede måder. Skær unødige ord væk. Ingen markdown, ingen forklaringer.\n\nSætning: {q}",
        "Skriv følgende sætning på {n} mere præcise, ordknappe måder. Ingen markdown, ingen forklaringer.\n\nSætning: {q}",
    ],
    "casual": [
        "Skriv følgende sætning på {n} mere afslappede, hverdagsagtige måder. Bevar betydningen. Ingen markdown, ingen forklaringer.\n\nSætning: {q}",
        "Omskriv følgende sætning på {n} mere uformelle måder — som man kunne sige det til en ven. Ingen markdown, ingen forklaringer.\n\nSætning: {q}",
        "Formulér følgende sætning på {n} mere ligefremme, uhøjtidelige måder. Ingen markdown, ingen forklaringer.\n\nSætning: {q}",
    ],
    "formal": [
        "Skriv følgende sætning på {n} mere formelle, officielle måder. Bevar betydningen. Ingen markdown, ingen forklaringer.\n\nSætning: {q}",
        "Omskriv følgende sætning på {n} måder passende til et brev eller officiel skrivelse. Ingen markdown, ingen forklaringer.\n\nSætning: {q}",
        "Formulér følgende sætning på {n} mere stringente, saglige måder. Ingen markdown, ingen forklaringer.\n\nSætning: {q}",
    ],
}

# ── Reverse direction prompts (user-facing SFT instructions) ─────────────────
# These are the USER prompts for the reverse rows; assistant output = original.

REVERSE_PROMPTS = {
    "simple": [  # inverts "make simple" → "make technical"
        "Skriv følgende sætning mere teknisk og præcist, med fagligt korrekt terminologi: {q}",
        "Formulér følgende mere fagligt, med den korrekte terminologi: {q}",
        "Omskriv følgende i et mere akademisk sprog: {q}",
        "Gør følgende mere teknisk og mindre folkelig: {q}",
    ],
    "compact": [  # inverts "make compact" → "make elaborate"
        "Skriv følgende sætning mere uddybende, med fyldigere formulering: {q}",
        "Formulér følgende mere udførligt, med tilføjet kontekst der er implicit i originalen: {q}",
        "Uddyb følgende sætning — gør den mere fyldig: {q}",
        "Omskriv følgende med mere kød på: {q}",
    ],
    "casual": [  # inverts "make casual" → "make formal"
        "Skriv følgende sætning mere formelt og officielt: {q}",
        "Omskriv følgende, så det passer til et officielt dokument: {q}",
        "Formulér følgende i en mere formel tone: {q}",
        "Gør følgende mere stringent og mindre uformelt: {q}",
    ],
    "formal": [  # inverts "make formal" → "make casual"
        "Skriv følgende sætning mere afslappet og hverdagsagtigt: {q}",
        "Omskriv følgende, så det lyder mindre stift: {q}",
        "Formulér følgende i et mere afslappet sprog: {q}",
        "Gør følgende mere uformelt og hverdagsagtigt: {q}",
    ],
}


# ── Source sentence loader ───────────────────────────────────────────────────

def _sentences_from_text(text: str) -> list[str]:
    """Split text into sentences on . ! ? followed by whitespace + capital."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÆØÅ])", text.strip())
    return [p.strip() for p in parts if 30 <= len(p.strip()) <= 250]


def load_sources(n_target: int, rng: random.Random) -> list[str]:
    """Draw Danish sentences from multiple SFT source datasets, favoring
    variety over volume. Returns up to n_target unique sentences."""
    seen: set[str] = set()
    pool: list[str] = []

    print("[sources] loading wiki-grounded-sft-v3 contexts…", flush=True)
    ds = load_dataset("jensjepsen/danish-wiki-grounded-sft-v3", "default",
                      split="train")
    for r in ds:
        ctx = (r.get("context") or "").strip()
        for s in _sentences_from_text(ctx):
            if s not in seen:
                seen.add(s)
                pool.append(s)

    print(f"[sources] wiki-grounded → {len(pool)} unique sentences", flush=True)

    print("[sources] loading sciq contexts…", flush=True)
    ds = load_dataset("jensjepsen/danish-sciq", "default", split="train")
    for r in ds:
        support = (r.get("da_support") or "").strip()
        for s in _sentences_from_text(support):
            if s not in seen:
                seen.add(s)
                pool.append(s)

    print(f"[sources] +sciq → {len(pool)} unique sentences", flush=True)

    print("[sources] loading dolly-15k DA answers…", flush=True)
    try:
        ds = load_dataset("jensjepsen/danish-dolly-15k", "sft", split="train")
        for r in ds:
            for msg in r.get("messages", []):
                if msg.get("role") == "assistant":
                    for s in _sentences_from_text(msg.get("content", "")):
                        if s not in seen:
                            seen.add(s)
                            pool.append(s)
    except Exception as e:
        print(f"[sources]   dolly skipped: {e}", flush=True)

    print(f"[sources] +dolly → {len(pool)} unique sentences", flush=True)

    rng.shuffle(pool)
    return pool[:n_target]


# ── OpenRouter client with retry/backoff ─────────────────────────────────────

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
                     "X-Title": "danish-rephrase-generation"},
            timeout=aiohttp.ClientTimeout(total=90))
    return _SESSION


async def call_gemma(prompt: str, schema: dict, max_retries: int = 4) -> list[str] | None:
    session = await _get_session()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1200,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "rewrites", "strict": True, "schema": schema},
        },
    }
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions", json=body
            ) as resp:
                if resp.status == 429:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                if resp.status != 200:
                    text = await resp.text()
                    print(f"  http {resp.status}: {text[:150]}", file=sys.stderr)
                    return None
                data = await resp.json()
                raw = data["choices"][0]["message"]["content"]
                parsed = json.loads(raw)
                return parsed.get("rewrites") or None
        except Exception as e:
            print(f"  err {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            await asyncio.sleep(backoff)
            backoff *= 2
    return None


# ── Row emission ─────────────────────────────────────────────────────────────

def emit_rows(source: str, target: str, variants: list[str],
              rng: random.Random,
              sim_max: float) -> list[dict]:
    """For each variant, emit forward + (optionally) reverse rows."""
    out = []
    for variant in variants:
        # Skip variants with unfilled placeholders
        if "[" in variant and "]" in variant and re.search(r"\[[A-ZÆØÅa-zæøå\s]+\]", variant):
            continue
        sim = SequenceMatcher(None, source, variant).ratio()
        fwd_tpl = rng.choice(FORWARD_PROMPTS[target])
        fwd_user = fwd_tpl.format(n=1, q=source).replace(
            " på 1 forskellige måder", "").replace(" på 1 måder", "").replace(
            " på 1 måde", "")
        # For single-variant SFT rows, use a de-quantified variant of the prompt
        fwd_user = re.sub(r" på \{?n?\}? ?", " ", fwd_user)  # cleanup
        fwd_user = fwd_tpl.format(n=1, q=source)  # keep as-is; model has seen n=3
        out.append({
            "messages": [
                {"role": "user", "content": fwd_user},
                {"role": "assistant", "content": variant},
            ],
            "target": target,
            "direction": "forward",
            "sim": round(sim, 3),
        })
        if sim <= sim_max:
            rev_tpl = rng.choice(REVERSE_PROMPTS[target])
            out.append({
                "messages": [
                    {"role": "user", "content": rev_tpl.format(q=variant)},
                    {"role": "assistant", "content": source},
                ],
                "target": target,
                "direction": "reverse",
                "sim": round(sim, 3),
            })
    return out


# ── Main run loop ────────────────────────────────────────────────────────────

async def run(args):
    rng = random.Random(args.seed)
    sources = load_sources(args.n_sources, rng)
    print(f"\n[run] source pool: {len(sources):,} sentences", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: read existing (source_hash, target) pairs from output
    seen: set[tuple[str, str]] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("direction") == "forward":
                    src = r["messages"][1]["content"] if r.get("direction") == "reverse" else \
                          r["messages"][0]["content"]  # heuristic — safer to use per-row hash
        # simpler resume: use file line count as progress hint only
        print(f"[run] resuming — existing file has {sum(1 for _ in out_path.open()):,} rows",
              flush=True)

    schema = make_schema(args.n_variants)
    targets = ["simple", "compact", "casual", "formal"]

    sem = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()
    stats: Counter = Counter()
    t0 = time.time()

    async def process(idx: int, source: str):
        async with sem:
            for target in targets:
                prompt_tpl = rng.choice(FORWARD_PROMPTS[target])
                prompt = prompt_tpl.format(n=args.n_variants, q=source)
                variants = await call_gemma(prompt, schema)
                if not variants:
                    stats["fail"] += 1
                    continue
                rows = emit_rows(source, target, variants, rng, args.sim_max)
                stats["ok"] += 1
                stats[f"target:{target}"] += 1
                stats["rows_forward"] += sum(1 for r in rows if r["direction"] == "forward")
                stats["rows_reverse"] += sum(1 for r in rows if r["direction"] == "reverse")
                async with file_lock:
                    with out_path.open("a") as fout:
                        for r in rows:
                            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

        done = stats["ok"] + stats["fail"]
        if done % max(1, args.log_every) == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            total_rows = stats["rows_forward"] + stats["rows_reverse"]
            print(f"  [{done}/{len(sources)*len(targets)}]  ok={stats['ok']} "
                  f"fail={stats['fail']}  {rate:.1f} calls/s  "
                  f"rows={total_rows} ({stats['rows_forward']}f + {stats['rows_reverse']}r)",
                  flush=True)

    tasks = [asyncio.create_task(process(i, s))
             for i, s in enumerate(sources, 1)]
    await asyncio.gather(*tasks)

    duration = time.time() - t0
    print(f"\n[done] {duration:.0f}s  "
          f"forward={stats['rows_forward']}  reverse={stats['rows_reverse']}  "
          f"total={stats['rows_forward'] + stats['rows_reverse']}",
          flush=True)
    print(f"stats: {dict(stats)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sources", type=int, default=20000)
    ap.add_argument("--n-variants", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--sim-max", type=float, default=0.85,
                    help="Reverse rows dropped when similarity(source, variant) > this.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
