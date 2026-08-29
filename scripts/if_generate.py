"""Build Danish instruction-following SFT data.

Loops:
    for seed in content_seeds:
        combo = sample_combo(rng)
        prompt = f"{seed_task}\\n\\nRegler ...\\n{rendered_rules}"
        for attempt in range(RETRY_K):
            answer = call_llm(prompt)
            ok, failures = verify_all(answer, combo)
            if ok:
                emit(...); break
            else: track_failures(failures)

Emits JSONL messages format ready for train_sft_packed.py:
    {"messages": [{"role":"user","content":...}, {"role":"assistant","content":...}],
     "source": "...", "constraints": [name, ...], "params": {...}, "attempts": N}

Also emits stats.json with per-constraint pass rate + retry distribution.

To swap the LLM backend, replace `call_llm` — it takes prompt string, returns
answer string (or None to trigger a retry).
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
from pathlib import Path

from datasets import load_dataset

# Same-directory import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from if_constraints import (  # noqa: E402
    ALL as ALL_CONSTRAINTS,
    sample_combo,
    render_rules,
    verify_all,
)


# ────────────────────────────────────────────────────────────────────────────
# LLM backend  (swap for your preferred provider)
# ────────────────────────────────────────────────────────────────────────────

MODEL_ID = os.environ.get("IF_MODEL_ID", "gemini-3.1-flash-lite")
_GEMINI_CLIENT = None
_OR_SESSION = None


def _read_key_file(names: list[str]) -> str | None:
    for name in names:
        for p in [Path.home() / name, Path.home() / f".{name}"]:
            if p.exists():
                return p.read_text().strip()
    return None


def _is_openrouter_model(model_id: str) -> bool:
    """Model IDs with a "/" go to OpenRouter (e.g. google/gemma-3-12b-it)."""
    return "/" in model_id


_GEMINI_CONFIG = None


async def _call_gemini(prompt: str) -> str | None:
    global _GEMINI_CLIENT, _GEMINI_CONFIG
    if _GEMINI_CLIENT is None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise SystemExit("`pip install google-genai` for the Gemini backend.")
        key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
               or _read_key_file(["gem", "gemini_key"]))
        if not key:
            raise SystemExit("No GOOGLE_API_KEY set and no ~/gem key file.")
        _GEMINI_CLIENT = genai.Client(api_key=key)
        # Explicitly disable thinking (2.5+ / 3.x defaults have drifted;
        # thinking_budget=0 forces zero reasoning tokens on any variant).
        _GEMINI_CONFIG = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
    try:
        resp = await _GEMINI_CLIENT.aio.models.generate_content(
            model=MODEL_ID, contents=prompt, config=_GEMINI_CONFIG)
        return (resp.text or "").strip() or None
    except Exception as e:
        print(f"  gemini error: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        return None


async def _call_openrouter(prompt: str) -> str | None:
    """OpenRouter chat-completions call. Uses aiohttp for concurrency."""
    global _OR_SESSION
    import aiohttp
    if _OR_SESSION is None:
        key = (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_API_KEY")
               or _read_key_file(["or", "openrouter"]))
        if not key:
            raise SystemExit("No OPENROUTER_API_KEY set and no ~/or key file.")
        _OR_SESSION = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://claude-code-if",
                     "X-Title": "danish-if-generation"},
            # Default connector caps at 100 sockets; bump so --concurrency > 100
            # actually gets used. limit_per_host=0 means "no per-host cap".
            connector=aiohttp.TCPConnector(limit=1000, limit_per_host=0),
            timeout=aiohttp.ClientTimeout(total=60))
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1500,
    }
    try:
        async with _OR_SESSION.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=body) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"  openrouter {resp.status}: {text[:200]}", file=sys.stderr)
                return None
            data = await resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip() or None
    except Exception as e:
        print(f"  or error: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        return None


async def call_llm_async(prompt: str) -> str | None:
    """Route to Gemini direct or OpenRouter based on MODEL_ID.

    Model IDs with "/" (e.g. google/gemma-3-12b-it) go to OpenRouter;
    plain names (e.g. gemini-3.1-flash-lite) go to the Google GenAI SDK.
    """
    if _is_openrouter_model(MODEL_ID):
        return await _call_openrouter(prompt)
    return await _call_gemini(prompt)


# ────────────────────────────────────────────────────────────────────────────
# Paraphrase cache
#
# Cache is keyed by (constraint_name, params_json). Value = list of LLM-
# generated Danish paraphrases of that constraint's canonical rule text.
#
# The SAME paraphrase is used for BOTH the Gemini responder prompt AND the
# stored SFT prompt — so if Gemini's answer passes the checker, we know the
# paraphrase preserved the rule's semantics (self-validating).
# ────────────────────────────────────────────────────────────────────────────

_PARAPHRASE_CACHE: dict[tuple[str, str], list[str]] = {}


def _params_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)


def load_paraphrase_cache(path: str | None) -> int:
    """Populate _PARAPHRASE_CACHE from a warmup JSONL. Returns entry count."""
    _PARAPHRASE_CACHE.clear()
    if not path:
        return 0
    p = Path(path)
    if not p.exists():
        return 0
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            _PARAPHRASE_CACHE[(r["name"], r["params_key"])] = r["variants"]
    return len(_PARAPHRASE_CACHE)


def maybe_swap_paraphrase(combo: list[dict], rng: random.Random, prob: float) -> None:
    """In-place: with probability `prob`, replace each rule's `render` with a
    cached paraphrase for (constraint, params)."""
    if not _PARAPHRASE_CACHE or prob <= 0:
        return
    for r in combo:
        if rng.random() >= prob:
            continue
        variants = _PARAPHRASE_CACHE.get((r["name"], _params_key(r["params"])))
        if variants:
            r["render"] = rng.choice(variants)


_PARAPHRASE_PROMPT = """Omskriv følgende danske instruktion på {n} forskellige måder.

Krav:
- Bevar den præcise betydning — alle tal, konkrete ordvalg-krav og strukturkrav skal være uændrede.
- Ret KUN formulering, tone og syntaks. Tilføj eller fjern IKKE regler eller begrænsninger.
- Rør IKKE ved tekst i anførselstegn — bevar dem ordret.
- Varier tonen: formel, uformel, kort, uddybende, imperativ, spørgsmål.
- Skriv én variant per linje, uden nummerering, bulletpoints eller anførselstegn omkring hele linjen.

Original: {canonical}

Varianter:"""


_LINE_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•–])\s*")


async def _paraphrase_one(canonical: str, n: int) -> list[str] | None:
    prompt = _PARAPHRASE_PROMPT.format(n=n, canonical=canonical)
    resp = await call_llm_async(prompt)
    if not resp:
        return None
    lines = [_LINE_PREFIX_RE.sub("", l).strip() for l in resp.split("\n")]
    lines = [l.strip(' "\'') for l in lines if len(l.strip()) > 3]
    # dedupe while preserving order, drop lines that exactly match the canonical
    seen: set[str] = set()
    uniq: list[str] = []
    for l in lines:
        if l == canonical or l in seen:
            continue
        seen.add(l)
        uniq.append(l)
    if len(uniq) < max(3, n // 2):
        return None
    return uniq[:n]


async def _warmup(args) -> None:
    """Enumerate (constraint, params) pairs, generate paraphrases up to a
    target count per pair, atomic-rewrite the cache JSONL.

    Semantics: `--target-variants N` = ensure EVERY pair has at least N
    paraphrases. If an entry already has M < N, generate (N-M) MORE and
    merge. Never regenerates existing paraphrases."""
    rng = random.Random(args.seed)
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    target = args.target_variants

    existing: dict[tuple[str, str], dict] = {}
    if cache_path.exists():
        with cache_path.open() as f:
            for line in f:
                r = json.loads(line)
                existing[(r["name"], r["params_key"])] = r
        counts = [len(r["variants"]) for r in existing.values()]
        below = sum(1 for c in counts if c < target)
        print(f"[warmup] cache has {len(existing)} entries "
              f"(min={min(counts) if counts else 0} "
              f"max={max(counts) if counts else 0}); "
              f"{below} below target of {target}", flush=True)

    todo: list[tuple[str, str, str, int]] = []  # (name, key, canonical, need)
    total_enumerated = 0
    for c in ALL_CONSTRAINTS:
        seen_params: set[str] = set()
        for _ in range(args.enumerate_k):
            try:
                params = c.sample(rng, None)
            except Exception:
                continue
            key = _params_key(params)
            if key in seen_params:
                continue
            seen_params.add(key)
            total_enumerated += 1
            have = len(existing.get((c.name, key), {"variants": []})["variants"])
            need = target - have
            if need <= 0:
                continue
            canonical = c.render_variants[0](params)
            todo.append((c.name, key, canonical, need))
    print(f"[warmup] enumerated {total_enumerated} unique (constraint, params) "
          f"pairs; need to top up {len(todo)}", flush=True)
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)
    state_lock = asyncio.Lock()
    ok = fail = 0
    t0 = time.time()

    async def one(name: str, key: str, canonical: str, need: int):
        nonlocal ok, fail
        async with sem:
            variants = await _paraphrase_one(canonical, need)
        if variants:
            async with state_lock:
                entry = existing.setdefault((name, key), {
                    "name": name, "params_key": key,
                    "canonical": canonical, "variants": [],
                })
                # Merge; dedupe while preserving order + drop canonical dups
                seen_v = set(entry["variants"])
                for v in variants:
                    if v == canonical or v in seen_v:
                        continue
                    seen_v.add(v)
                    entry["variants"].append(v)
                ok += 1
        else:
            fail += 1
        done = ok + fail
        if done % max(1, args.log_every) == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            print(f"  [{done}/{len(todo)}] ok={ok} fail={fail}  "
                  f"{rate:.1f} req/s", flush=True)

    await asyncio.gather(*[one(*t) for t in todo])

    # Atomic rewrite so partial state can't corrupt the cache
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("w") as fout:
        for entry in existing.values():
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    tmp.replace(cache_path)
    counts = [len(r["variants"]) for r in existing.values()]
    print(f"[warmup] done — ok={ok} fail={fail}  wrote {len(existing)} entries "
          f"(min={min(counts)} max={max(counts)}) to {cache_path}", flush=True)


def _smoke(args) -> None:
    """Print variation stats + sample paraphrases per constraint from cache."""
    n = load_paraphrase_cache(args.cache)
    print(f"[smoke] loaded {n} cache entries from {args.cache}\n")

    # Group by constraint
    by_name: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for (name, key), variants in _PARAPHRASE_CACHE.items():
        by_name[name].append((key, variants))

    print(f"{'constraint':38s}  {'#params':>8}  {'variants_min':>12}  "
          f"{'variants_max':>12}  {'variants_avg':>12}")
    print("-" * 90)
    for name in sorted(by_name):
        entries = by_name[name]
        counts = [len(v) for _, v in entries]
        print(f"{name:38s}  {len(entries):>8}  {min(counts):>12}  "
              f"{max(counts):>12}  {sum(counts)/len(counts):>12.1f}")

    # Detailed sample for a few constraints
    print("\n" + "=" * 90)
    print(f"Sample paraphrases (up to {args.n_samples} per shown constraint):")
    print("=" * 90)
    rng = random.Random(args.seed)
    focus = args.only.split(",") if args.only else sorted(by_name)[:args.max_shown]
    for name in focus:
        if name not in by_name:
            print(f"\n[{name}]  NOT IN CACHE")
            continue
        entries = by_name[name]
        for key, variants in entries[:args.n_samples]:
            # find the canonical string from the cache file for display
            print(f"\n[{name}]  params={key}")
            print(f"  canonical: {_lookup_canonical(args.cache, name, key)!r}")
            for v in variants[:args.n_variants]:
                print(f"    - {v}")


def _lookup_canonical(cache_path: str, name: str, key: str) -> str | None:
    """One-shot re-read of the cache file to fetch canonical (not kept in-memory)."""
    with open(cache_path) as f:
        for line in f:
            r = json.loads(line)
            if r["name"] == name and r["params_key"] == key:
                return r.get("canonical")
    return None


# ────────────────────────────────────────────────────────────────────────────
# Content seeds
# ────────────────────────────────────────────────────────────────────────────

def load_seeds(sources: list[str], max_per_source: int, rng: random.Random) -> list[dict]:
    """Return list of {task, source} dicts. `task` = base instruction to which
    IF rules will be appended."""
    out = []

    if "wiki" in sources:
        # Use ALL wiki-grounded categories — every non-empty instruction is a
        # valid self-contained IF task once we prepend context (when present).
        # Categories: open_qa, closed_qa, information_extraction, summarization,
        #             classification, brainstorming, creative_writing, general_qa
        ds = load_dataset("jensjepsen/danish-wiki-grounded-sft-v3", "default", split="train")
        pool = [r for r in ds if (r.get("instruction") or "").strip()]
        pool = rng.sample(pool, min(max_per_source, len(pool)))
        for r in pool:
            ctx = (r.get("context") or "").strip()
            instr = r["instruction"].strip()
            if ctx:
                task = f"Tekst:\n{ctx}\n\n{instr}"
            else:
                task = instr
            out.append({"task": task,
                        "source": f"wiki:{r.get('category', 'unknown')}",
                        "ctx": {"task_text": task}})

    if "mc" in sources:
        # MC-shaped seeds carry ctx={mc_choices, gold_letter} so the letter-only
        # solo constraint can be selected.  When it isn't picked, the seed
        # runs through the normal text-answer combo path.
        #
        # danish-citizen-tests is DELIBERATELY excluded: EuroEval's KNOW
        # benchmark uses that dataset, so training on it would leak eval data.
        # See _load_mc_seeds_citizen — kept for reference but do not call.
        out.extend(_load_mc_seeds_sciq(rng, max_per_source))

    if "sciq" in sources:
        ds = load_dataset("jensjepsen/danish-sciq", "default", split="train")
        pool = [r for r in ds if (r.get("da_support") or "").strip()
                and (r.get("da_question") or "").strip()]
        pool = rng.sample(pool, min(max_per_source, len(pool)))
        for r in pool:
            task = (f"Kontekst:\n{r['da_support'].strip()}\n\n"
                    f"Spørgsmål: {r['da_question'].strip()}")
            out.append({"task": task, "source": "sciq",
                        "ctx": {"task_text": task}})

    rng.shuffle(out)
    return out


def _load_mc_seeds_citizen(rng: random.Random, cap: int) -> list[dict]:
    ds = load_dataset("alexandrainst/danish-citizen-tests", split="train")
    rows = list(ds)
    rng.shuffle(rows)
    out = []
    for r in rows[:cap]:
        # citizen tests use option_a / option_b / option_c
        letters = [L for L in "ABC" if r.get(f"option_{L.lower()}")]
        opts_text = "\n".join(f"{L}) {r[f'option_{L.lower()}']}" for L in letters)
        task = f"Spørgsmål: {r['question']}\n\n{opts_text}"
        gold = str(r["answer"]).upper().strip()
        if gold not in letters:  # skip malformed
            continue
        out.append({
            "task": task,
            "source": "mc:citizen-tests",
            "ctx": {"mc_choices": letters, "gold_letter": gold},
        })
    return out


def _load_wiki_salient_seeds_UNUSED(rng: random.Random, cap: int,
                              salience_path: str,
                              tiers: tuple[str, ...] = ("T1_universal", "T2_mainstream"),
                              intro_chars: int = 1200) -> list[dict]:
    """Wiki-topic seeds sampled ONLY from salience-filtered T1+T2 articles.

    For each picked article, uses its intro from `wikimedia/wikipedia`
    20231101.da as CONTEXT and generates a short natural-language task
    about the topic. The IF constraint rules then get appended by the
    normal pipeline.

    Task shapes drawn uniformly:
      - "Skriv en kort opsummering af følgende tekst."
      - "Skriv 3 vigtige fakta om {title}."
      - "Forklar hvad {title} er, baseret på teksten."
      - "Skriv en kort tekst om {title}."  (no ctx, general-knowledge)

    First three use context; fourth is prompt-only (topic reference).
    """
    # 1) Read salience.tsv → list of (score, pageid, title)
    picks: list[tuple[float, int, str]] = []
    with open(salience_path) as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[5] in tiers:
                picks.append((float(parts[6]), int(parts[0]), parts[1]))
    picks.sort(reverse=True)
    # Sample WITH replacement — same article can seed multiple rows with
    # different constraint combos + task shapes. This lets a modest article
    # pool (~11k T1+T2) produce 100k+ unique training rows.
    if cap:
        picks = rng.choices(picks, k=cap)
    want_titles = {t for _, _, t in picks}
    print(f"[wiki_salient] loading wikimedia/wikipedia 20231101.da "
          f"({len(want_titles)} titles)…", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.da", split="train")
    title_to_text: dict[str, str] = {}
    for row in ds:
        if row["title"] in want_titles:
            title_to_text[row["title"]] = row["text"]
            if len(title_to_text) == len(want_titles):
                break
    print(f"[wiki_salient]   matched {len(title_to_text)}/{len(want_titles)}",
          flush=True)

    def _intro(txt: str, cap_chars: int) -> str:
        paras = [p.strip() for p in txt.split("\n") if p.strip()]
        out, total = [], 0
        for p in paras:
            if total + len(p) > cap_chars:
                break
            out.append(p); total += len(p) + 1
        return "\n\n".join(out)

    out = []
    for _, _, title in picks:
        text = title_to_text.get(title)
        if not text:
            continue
        intro = _intro(text, intro_chars)
        shape = rng.random()
        if shape < 0.25:
            task = f"Tekst:\n{intro}\n\nSkriv en kort opsummering af teksten."
        elif shape < 0.50:
            task = f"Tekst:\n{intro}\n\nSkriv 3 vigtige fakta om emnet baseret på teksten."
        elif shape < 0.75:
            task = f"Tekst:\n{intro}\n\nForklar kort hvad {title} er."
        else:
            # prompt-only, uses topic as general-knowledge cue
            task = f"Skriv en kort tekst om {title}."
        out.append({"task": task, "source": f"wiki_salient:{title}",
                    "ctx": {"task_text": task}})
    return out


def _load_mc_seeds_sciq(rng: random.Random, cap: int) -> list[dict]:
    ds = load_dataset("jensjepsen/danish-sciq", "default", split="train")
    rows = [r for r in ds if (r.get("da_question") or "").strip()
            and (r.get("da_correct_answer") or "").strip()]
    rng.shuffle(rows)
    out = []
    letters = ["A", "B", "C", "D"]
    for r in rows[:cap]:
        opts = [r["da_correct_answer"], r["da_distractor1"],
                r["da_distractor2"], r["da_distractor3"]]
        if not all(opts):
            continue
        # Shuffle options per row (deterministic via row-scoped rng)
        idxs = list(range(4))
        rng.shuffle(idxs)
        gold_slot = idxs.index(0)  # slot where the correct answer landed
        gold_letter = letters[gold_slot]
        opts_text = "\n".join(f"{letters[slot]}) {opts[orig]}"
                              for slot, orig in enumerate(idxs))
        task = f"Spørgsmål: {r['da_question']}\n\n{opts_text}"
        out.append({
            "task": task,
            "source": "mc:sciq",
            "ctx": {"mc_choices": letters, "gold_letter": gold_letter,
                    "task_text": task},
        })
    return out


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# Prompt rendering
#
# We build TWO prompts per row:
#   1. gemini_prompt = canonical strict-scaffold form (max Gemini reliability)
#   2. sft_prompt    = varied-presentation form (what the trained model sees)
# Both express the same rules, so Gemini's answer satisfies both.  The varied
# form goes into the SFT training data; the strict form is throwaway.
# ────────────────────────────────────────────────────────────────────────────

# Canonical strict scaffold — always the same shape, most reliable for Gemini.
_GEMINI_TEMPLATE = """{task}

Regler (skal overholdes præcist — ellers er svaret ubrugeligt):
{rules}"""


def render_prompt_gemini(task: str, combo: list[dict]) -> str:
    if not combo:
        return task
    rules = "\n".join(f"- {r['render']}" for r in combo)
    return _GEMINI_TEMPLATE.format(task=task, rules=rules)


# ────────────────────────────────────────────────────────────────────────────
# Varied presentation for the SFT-stored prompt so the trained model learns
# to follow rules regardless of the wrapping scaffold.
# ────────────────────────────────────────────────────────────────────────────

# Preamble phrasings; None = no preamble (rules stand alone).
_PREAMBLES = [
    "Regler (skal overholdes præcist):",
    "Regler:",
    "Instruktioner:",
    "Krav til svaret:",
    "Følg disse regler:",
    "Bemærk:",
    "Vigtigt:",
    "Format:",
    "Følgende krav skal overholdes:",
    None, None,   # weight the "no preamble" case a bit — matches natural asks
]

# List styles for the rules block.
_STYLES = ["bullets", "numbered", "prose", "inline"]

# Where the rules sit relative to the task.
_POSITIONS = ["after", "before"]


# ────────────────────────────────────────────────────────────────────────────
# Surface variation — small casing/punctuation perturbations applied at low
# rate so the model doesn't memorize exact character sequences. Preserves
# semantic content (never touches quoted spans "..." or ALL-CAPS markers).
# ────────────────────────────────────────────────────────────────────────────

_QUOTED_SPAN_RE = re.compile(r'("[^"\n]*"|<<[^>\n]*>>|\[[^\]\n]+\]|\*\*[^*\n]+\*\*)')

# Alternative quote styles a Danish writer might use. Each entry = (open, close).
_QUOTE_STYLES = [
    ('"',  '"'),    # ASCII straight (baseline, most common in digital text)
    ('“',  '”'),   # curly typographic
    ('»',  '«'),   # traditional Danish print — low at start, high at end
    ('«',  '»'),   # French style
    ("'",  "'"),    # single ASCII
    ('‘',  '’'),   # single curly
]


def _maybe_swap_quote_style(span: str, rng: random.Random) -> str:
    """If `span` is an ASCII double-quoted string, sometimes rewrap it in a
    different quote style. Leaves other bracketed spans (<<...>>, [...], **...**)
    alone."""
    if not (span.startswith('"') and span.endswith('"') and len(span) >= 2):
        return span
    if rng.random() < 0.35:  # 35% of ASCII-quoted spans get rewrapped
        opener, closer = rng.choice(_QUOTE_STYLES)
        return opener + span[1:-1] + closer
    return span


def _protect_spans(text: str, rng: random.Random | None = None) -> tuple[str, list[str]]:
    """Replace quoted / bracketed spans with placeholders so transforms can't
    corrupt the parameters. If rng given, sometimes rewrap ASCII-quoted spans
    in a different quote style. Returns (protected_text, list_of_originals)."""
    protected = []
    def sub(m):
        span = m.group()
        if rng is not None:
            span = _maybe_swap_quote_style(span, rng)
        protected.append(span)
        return f"\x00{len(protected)-1}\x00"
    return _QUOTED_SPAN_RE.sub(sub, text), protected


def _restore_spans(text: str, protected: list[str]) -> str:
    for i, orig in enumerate(protected):
        text = text.replace(f"\x00{i}\x00", orig)
    return text


# Danish number words for 1-20. Model should learn either "3" or "tre".
_DA_NUMBERS = {
    1: "en", 2: "to", 3: "tre", 4: "fire", 5: "fem",
    6: "seks", 7: "syv", 8: "otte", 9: "ni", 10: "ti",
    11: "elleve", 12: "tolv", 13: "tretten", 14: "fjorten", 15: "femten",
    16: "seksten", 17: "sytten", 18: "atten", 19: "nitten", 20: "tyve",
}

# "fx" alternates — all common in Danish.
_FX_VARIANTS = ["fx", "f.eks.", "eksempelvis", "som fx", "for eksempel"]

# Dash-style equivalents.
_DASH_VARIANTS = ["—", "–", "-"]


def _surface_variate_rule(rule: str, rng: random.Random) -> str:
    """Apply small probabilistic perturbations to a rendered rule."""
    r, protected = _protect_spans(rule, rng)

    # 25% swap digits 1-20 for Danish number words (or vice versa)
    if rng.random() < 0.25:
        def sub_digit(m):
            n = int(m.group())
            if n in _DA_NUMBERS and rng.random() < 0.7:
                return _DA_NUMBERS[n]
            return m.group()
        # only swap standalone digits, not inside example-strings like "1."
        r = re.sub(r"(?<!\d)(?<![.,])\b(\d{1,2})\b(?!\.\d)", sub_digit, r)

    # 15% swap "fx" for a variant
    if rng.random() < 0.15:
        r = re.sub(r"\bfx\b", rng.choice(_FX_VARIANTS), r, count=1)

    # 12% swap em-dashes for another dash flavor
    if rng.random() < 0.12:
        target = rng.choice(_DASH_VARIANTS)
        r = r.replace("—", target)

    # 20% drop trailing period
    if rng.random() < 0.20:
        r = re.sub(r"\.\s*$", "", r)
    # 15% lowercase the first letter (fragment / colloquial)
    if rng.random() < 0.15 and r and r[0].isupper():
        r = r[0].lower() + r[1:]
    # 8% lowercase the whole rule
    if rng.random() < 0.08:
        r = r.lower()
    # 10% double a single space (typo)
    if rng.random() < 0.10:
        positions = [m.start() for m in re.finditer(r"(?<=\S) (?=\S)", r)]
        if positions:
            i = rng.choice(positions)
            r = r[:i] + "  " + r[i+1:]
    # 5% swap trailing "." for "!"
    if rng.random() < 0.05 and r.endswith("."):
        r = r[:-1] + "!"

    return _restore_spans(r, protected)


_PREAMBLE_TERMINATORS = [":", " —", " -", ""]


def _surface_variate_preamble(pre: str | None, rng: random.Random) -> str | None:
    """Small casing/punct perturbations on the preamble too."""
    if pre is None:
        return None
    p = pre
    # 15% lowercase the whole preamble
    if rng.random() < 0.15:
        p = p.lower()
    # 30% swap trailing ":" for a different terminator
    if rng.random() < 0.30 and p.endswith(":"):
        p = p[:-1] + rng.choice(_PREAMBLE_TERMINATORS)
    return p


# Bullet-marker variants for the SFT wrapper's own bullet lists (not the
# `bullet_list_n_items` constraint's own output — that has its own checker).
_WRAPPER_BULLETS = ["-", "*", "•", "–"]


def render_prompt_sft(task: str, combo: list[dict], rng: random.Random) -> str:
    """Format the base task + rendered rules with random presentation style.

    Combines a preamble phrasing, list style, position, AND small surface
    perturbations per-rule and per-preamble so the model doesn't overfit to
    exact character sequences. Inline style always concatenates rules into
    the task text with no preamble.
    """
    # Shuffle the render order — sample_combo already returns a random order,
    # but re-shuffling here makes the SFT prompt's rule order independent of
    # whatever order Gemini saw, so the model can't learn a positional bias.
    shuffled = list(combo)
    rng.shuffle(shuffled)
    rules = [_surface_variate_rule(r["render"], rng) for r in shuffled]
    if not rules:
        return task

    style = rng.choice(_STYLES)
    # Inline is a fixed shape — task text + rules concatenated in one flow.
    if style == "inline":
        return f"{task.rstrip()} " + " ".join(rules)

    if style == "bullets":
        bullet = rng.choice(_WRAPPER_BULLETS)
        block = "\n".join(f"{bullet} {r}" for r in rules)
    elif style == "numbered":
        # Vary the numbered marker format: "1." vs "1)" vs "(1)"
        num_style = rng.choice(["dot", "paren", "wrapped"])
        if num_style == "dot":
            block = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))
        elif num_style == "paren":
            block = "\n".join(f"{i}) {r}" for i, r in enumerate(rules, 1))
        else:  # wrapped
            block = "\n".join(f"({i}) {r}" for i, r in enumerate(rules, 1))
    else:  # prose
        block = " ".join(rules)

    preamble = _surface_variate_preamble(rng.choice(_PREAMBLES), rng)
    header = f"{preamble}\n{block}" if preamble else block
    position = rng.choice(_POSITIONS)
    return f"{header}\n\n{task}" if position == "before" else f"{task}\n\n{header}"


# Global counters + lock, updated by every worker coroutine.
_PASSES = 0
_FAILS = 0
_ATTEMPTS_HIST: Counter = Counter()
_PER_CONSTRAINT: dict[str, list[int]] = defaultdict(lambda: [0, 0])
_FILE_LOCK: asyncio.Lock | None = None


async def _process_seed(seed: dict, combo: list[dict],
                        gemini_prompt: str, sft_prompt: str, task_hash: str,
                        retry_k: int, fout, log_every: int, total: int, idx: int):
    """Run one seed through retries + verify + write.

    `gemini_prompt` is the canonical strict-scaffold prompt we send to
    Gemini (max reliability).  `sft_prompt` is the varied-presentation
    prompt stored in the SFT row (so the trained model learns robustness).
    Both encode the same rules; Gemini's answer satisfies both.
    """
    global _PASSES, _FAILS
    got = None
    attempts = 0
    failures_last: list[str] = []

    # Fast path: solo letter-only constraint on an MC seed. Gold letter is
    # already in the constraint's params — no need to call the LLM.
    if (len(combo) == 1 and combo[0]["name"] == "answer_only_letter"
            and "gold_letter" in combo[0]["params"]):
        got = combo[0]["params"]["gold_letter"].upper()
        _PER_CONSTRAINT["answer_only_letter"][0] += 1
        _PER_CONSTRAINT["answer_only_letter"][1] += 1
    else:
        for _ in range(retry_k):
            attempts += 1
            answer = await call_llm_async(gemini_prompt)
            if not answer:
                continue
            ok, failures = verify_all(answer, combo)
            for r in combo:
                _PER_CONSTRAINT[r["name"]][1] += 1
                if r["name"] not in failures:
                    _PER_CONSTRAINT[r["name"]][0] += 1
            if ok:
                got = answer
                break
            failures_last = failures

    _ATTEMPTS_HIST[attempts] += 1
    if got is None:
        _FAILS += 1
        return

    _PASSES += 1
    row = {
        "messages": [
            {"role": "user", "content": sft_prompt},
            {"role": "assistant", "content": got},
        ],
        "source": seed["source"],
        "constraints": [r["name"] for r in combo],
        "params": {r["name"]: r["params"] for r in combo},
        "attempts": attempts,
        "_task_hash": task_hash,
    }
    async with _FILE_LOCK:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()


async def _run(args, seeds: list[dict], seen: set, out_path: Path):
    global _FILE_LOCK
    _FILE_LOCK = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    rng = random.Random(args.seed)

    t0 = time.time()
    total = len(seeds)

    with out_path.open("a") as fout:
        async def worker(idx: int, seed: dict):
            th = str(hash(seed["task"]))
            if (seed["source"], th) in seen:
                return
            combo = sample_combo(rng, args.min_combo_size, args.max_combo_size,
                                 ctx=seed.get("ctx"))
            # Same paraphrased render goes to BOTH prompts, so a passing check
            # validates that the paraphrase preserved rule semantics.
            maybe_swap_paraphrase(combo, rng, args.paraphrase_prob)
            gemini_prompt = render_prompt_gemini(seed["task"], combo)
            sft_prompt = render_prompt_sft(seed["task"], combo, rng)
            async with sem:
                await _process_seed(seed, combo, gemini_prompt, sft_prompt, th,
                                    args.retry_k, fout, args.log_every,
                                    total, idx)
            # Progress print (safe from any worker; only prints at cadence)
            done = _PASSES + _FAILS
            if done and done % args.log_every == 0:
                elapsed = time.time() - t0
                rate = _PASSES / elapsed if elapsed else 0
                print(f"  [{done}/{total}] pass={_PASSES} fail={_FAILS}  "
                      f"{rate:.1f} row/s  attempts={dict(_ATTEMPTS_HIST)}",
                      flush=True)

        tasks = [asyncio.create_task(worker(i, s)) for i, s in enumerate(seeds, 1)]
        await asyncio.gather(*tasks)

    return time.time() - t0


def _cmd_gen(args) -> None:
    stats_path = Path(args.stats) if args.stats else Path(args.out + ".stats.json")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_cache = load_paraphrase_cache(args.paraphrase_cache)
    if args.paraphrase_cache:
        print(f"[gen] paraphrase cache: {n_cache} entries from "
              f"{args.paraphrase_cache}  (swap prob={args.paraphrase_prob})",
              flush=True)
    elif args.paraphrase_prob > 0:
        print("[gen] WARNING: --paraphrase-prob > 0 but no --paraphrase-cache; "
              "paraphrase swapping disabled", flush=True)
        args.paraphrase_prob = 0.0

    rng = random.Random(args.seed)
    print(f"loading seeds from: {args.sources}", flush=True)
    seeds = load_seeds(args.sources, args.max_per_source, rng)
    if args.limit:
        seeds = seeds[: args.limit]
    if args.n_total:
        seeds = seeds[: args.n_total]
    print(f"seed pool: {len(seeds):,}  concurrency: {args.concurrency}", flush=True)

    seen = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                r = json.loads(line)
                seen.add((r["source"], r["_task_hash"]))
        print(f"resuming — already have {len(seen):,} rows", flush=True)

    duration = asyncio.run(_run(args, seeds, seen, out_path))

    stats = {
        "total_seeds": len(seeds),
        "passed": _PASSES,
        "failed": _FAILS,
        "pass_rate": _PASSES / max(1, _PASSES + _FAILS),
        "attempts_histogram": dict(_ATTEMPTS_HIST),
        "per_constraint_pass_rate": {
            k: (v[0] / v[1] if v[1] else 0.0, v[0], v[1])
            for k, v in _PER_CONSTRAINT.items()
        },
        "model": MODEL_ID,
        "concurrency": args.concurrency,
        "paraphrase_cache": args.paraphrase_cache,
        "paraphrase_prob": args.paraphrase_prob,
        "duration_seconds": round(duration, 1),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nwrote {_PASSES} rows to {out_path}  ({duration:.0f}s)")
    print(f"stats → {stats_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ── warmup ────────────────────────────────────────────────────────────
    a = sub.add_parser("warmup",
                       help="Build LLM-paraphrase cache for constraint rules.")
    a.add_argument("--cache", required=True,
                   help="JSONL cache path (append-only, resumable).")
    a.add_argument("--target-variants", type=int, default=10,
                   help="Target paraphrases per (constraint, params) entry. "
                        "If cache already has M for an entry and M < target, "
                        "generate the delta and MERGE. Never regenerates "
                        "existing paraphrases.")
    a.add_argument("--enumerate-k", type=int, default=100,
                   help="Times to call each constraint's sample() to enumerate "
                        "unique parameter instances.")
    a.add_argument("--concurrency", type=int, default=20)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--log-every", type=int, default=25)

    # ── gen ───────────────────────────────────────────────────────────────
    g = sub.add_parser("gen", help="Generate IF training rows.")
    g.add_argument("--out", required=True)
    g.add_argument("--stats", default=None)
    g.add_argument("--sources", nargs="+",
                   default=["wiki", "sciq", "mc"],
                   choices=["wiki", "sciq", "mc"])
    g.add_argument("--max-per-source", type=int, default=2000)
    g.add_argument("--n-total", type=int, default=None)
    g.add_argument("--retry-k", type=int, default=3)
    g.add_argument("--min-combo-size", type=int, default=1)
    g.add_argument("--max-combo-size", type=int, default=5)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--concurrency", type=int, default=20,
                   help="Parallel LLM calls; 1 = sync. Gemini rate limits are "
                        "generous (~4k RPM for flash-lite), 20 is safe.")
    g.add_argument("--log-every", type=int, default=25)
    g.add_argument("--paraphrase-cache", default=None,
                   help="Path to a warmup JSONL. If unset, uses only hard-coded "
                        "render_variants.")
    g.add_argument("--paraphrase-prob", type=float, default=0.7,
                   help="Per-rule probability of swapping the hard-coded render "
                        "with a cached paraphrase.")

    # ── smoke ─────────────────────────────────────────────────────────────
    s = sub.add_parser("smoke",
                       help="Inspect cache: per-constraint variant counts + "
                            "sample paraphrases.")
    s.add_argument("--cache", required=True)
    s.add_argument("--n-samples", type=int, default=2,
                   help="How many params-instances per constraint to print.")
    s.add_argument("--n-variants", type=int, default=5,
                   help="How many variants to print per params-instance.")
    s.add_argument("--max-shown", type=int, default=10,
                   help="How many constraints to show detail for.")
    s.add_argument("--only", default=None,
                   help="Comma-separated constraint names to focus on.")
    s.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    if args.cmd == "warmup":
        asyncio.run(_warmup(args))
    elif args.cmd == "gen":
        _cmd_gen(args)
    elif args.cmd == "smoke":
        _smoke(args)


if __name__ == "__main__":
    main()
