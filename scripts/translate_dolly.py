"""Translate the Dolly-15K instruction dataset to Esperanto via Gemini.

Each example in databricks/databricks-dolly-15k has:
  instruction  — the task or question
  context      — optional passage/background (closed_qa, summarization, etc.)
  response     — human-written answer
  category     — task type (open_qa, brainstorming, summarization, ...)

We translate instruction/context/response together (per example) so the
translator sees full context and preserves cross-field references.
Category stays English (metadata, never shown to user).

Output JSONL: {messages: [user: instr+context, assistant: response], category}

Translation cache at data/dolly_eo/dolly_dict.json maps a deterministic key
(sha1 of the source triple) to the translated triple, so partial runs resume.
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def example_key(ex: dict) -> str:
    """Stable hash of (instruction, context, response) for cache lookup."""
    blob = "\x1f".join([ex.get("instruction", ""), ex.get("context", ""), ex.get("response", "")])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def translate_batch_via_gemini(client, examples: list[dict], model_name: str,
                               timeout_s: int = 120) -> dict:
    """Translate a batch of full Dolly examples. Returns {key: translated_dict}."""
    # Compose a numbered request — keep all three fields together per example
    items = []
    for i, ex in enumerate(examples):
        items.append({
            "i": i + 1,
            "instruction": ex.get("instruction", ""),
            "context": ex.get("context", ""),
            "response": ex.get("response", ""),
        })

    prompt = f"""Traduku ĉiun el la sekvaj instrukciaj ekzemploj al natura, idioma Esperanto.

Reguloj:
- Traduku ĉiujn tri kampojn (instruction, context, response) en sama ekzemplo, konservante koherecon.
- Konservu nomojn de personoj, lokoj, kompanioj, libroj, ktp. (ne traduku ilin).
- Konservu nombrojn, datojn, kaj formatadon (listoj, paragrafoj, krampoj).
- Uzu ĝustajn supersignojn (ĉ, ĝ, ĥ, ĵ, ŝ, ŭ).
- Uzu ĝustajn akuzativojn (-n) kaj plurajn formojn (-j).
- Se la "context" estas malplena, lasu ĝin malplena en la traduko.
- Konservu mallongan respondon mallonga; konservu detalan respondon detala.

Respondu kiel JSON-listo de objektoj kun kampoj "i" (indekso), "instruction", "context", "response".
Ne aldonu komentojn. Respondu NUR per la JSON.

Ekzemploj:
{json.dumps(items, ensure_ascii=False, indent=2)}"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={"http_options": {"timeout": timeout_s * 1000}},
    )
    text = response.text.strip()
    if "```" in text:
        for part in text.split("```"):
            if part.lstrip().startswith(("json", "[")):
                text = part.removeprefix("json").strip()
                break
    try:
        results = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}; head: {text[:200]}", file=sys.stderr)
        return {}

    out: dict[str, dict] = {}
    for r in results:
        idx = r.get("i")
        if idx is None or not (1 <= idx <= len(examples)):
            continue
        src = examples[idx - 1]
        out[example_key(src)] = {
            "instruction": r.get("instruction", "").strip(),
            "context":     r.get("context", "").strip(),
            "response":    r.get("response", "").strip(),
            "category":    src.get("category", ""),
        }
    return out


def to_sft(translated: dict) -> dict | None:
    """Reassemble a translated example as a single-turn SFT conversation."""
    instr = translated.get("instruction", "").strip()
    ctx = translated.get("context", "").strip()
    resp = translated.get("response", "").strip()
    if not instr or not resp:
        return None
    user = f"{instr}\n\n{ctx}" if ctx else instr
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": resp},
        ],
        "category": translated.get("category", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/dolly_eo/dolly_dict.json"))
    parser.add_argument("--out", type=Path, default=Path("data/sft/sft_dolly.jsonl"))
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N examples (for sampling)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Examples per API call (Dolly responses can be long, keep small)")
    parser.add_argument("--parallel", type=int, default=10)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview",
                        help="Match the other translate_* scripts; flash-lite is fast and cheap")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Loading databricks/databricks-dolly-15k...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    examples = [dict(ex) for ex in ds]
    if args.n:
        examples = examples[:args.n]
    print(f"  {len(examples):,} examples")

    cache: dict[str, dict] = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text())
        print(f"Cache hits: {len(cache):,}")
    todo = [ex for ex in examples if example_key(ex) not in cache]
    print(f"To translate: {len(todo):,}")

    if args.dry_run:
        print(f"\n--- Sample batch payload (first {min(args.batch_size, 3)}): ---\n")
        for ex in todo[:min(args.batch_size, 3)]:
            print(f"  category: {ex.get('category')}")
            print(f"  instruction: {ex.get('instruction')[:120]}")
            if ex.get("context"):
                print(f"  context: {ex.get('context')[:120]}...")
            print(f"  response: {ex.get('response')[:120]}")
            print()
        n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
        print(f"Would issue ~{n_batches} API calls in batches of {args.batch_size}")
        return

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY (or GOOGLE_API_KEY).", file=sys.stderr)
        sys.exit(1)
    from google import genai
    client = genai.Client(api_key=api_key)

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    print(f"Translating in {n_batches} batches of {args.batch_size}, {args.parallel}-way parallel...", flush=True)
    cache_lock = threading.Lock()
    start_t = time.time()

    def do_batch(batch):
        try:
            return translate_batch_via_gemini(client, batch, args.model)
        except Exception as e:
            try:
                return translate_batch_via_gemini(client, batch, args.model)
            except Exception as e2:
                return {"__error__": str(e2)}

    completed = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(do_batch, todo[bi * args.batch_size : (bi + 1) * args.batch_size]): bi
            for bi in range(n_batches)
        }
        for fut in as_completed(futures):
            bi = futures[fut]
            results = fut.result()
            err = results.pop("__error__", None) if isinstance(results, dict) else None
            with cache_lock:
                cache.update(results or {})
                completed += 1
                cache_size = len(cache)
                if completed % 25 == 0 or completed == n_batches:
                    args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
            elapsed = time.time() - start_t
            eta = elapsed / completed * (n_batches - completed)
            tag = f"ERR ({err})" if err else f"+{len(results)}"
            print(f"  [{completed:>4}/{n_batches}] batch {bi+1}: {tag}; cache={cache_size:,}; ETA={eta/60:.1f}m",
                  flush=True)

    args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    print(f"Translation cache: {len(cache):,} entries → {args.cache}")

    # Reassemble as SFT
    print("Reassembling Esperanto SFT...")
    written = 0
    skipped = 0
    with open(args.out, "w") as f:
        for ex in examples:
            translated = cache.get(example_key(ex))
            if not translated:
                skipped += 1
                continue
            sft = to_sft(translated)
            if not sft:
                skipped += 1
                continue
            f.write(json.dumps(sft, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written:,} SFT conversations → {args.out}  (skipped {skipped})")


if __name__ == "__main__":
    main()
