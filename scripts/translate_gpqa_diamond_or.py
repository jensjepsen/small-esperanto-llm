"""Translate GPQA-Diamond → Danish via gemini-3.1-flash-lite on OpenRouter.

Small dataset (198 rows), one row = one API call. Translates Question + 4
answer choices in a single JSON response. Preserves technical notation
(equations, units, numeric values) verbatim.

Output JSONL rows:
    {orig_idx, question_en, question_da,
     answers_en: [correct, incorrect1, incorrect2, incorrect3],
     answers_da: [correct, incorrect1, incorrect2, incorrect3],
     correct_idx: 0,  # always 0 in the translated schema (correct first)
     subdomain, domain,
     in_t, out_t, cost}
"""
from __future__ import annotations
import argparse, asyncio, json, re
from pathlib import Path
import aiohttp
from datasets import load_dataset

MODEL = "google/gemini-3.1-flash-lite"
API = "https://openrouter.ai/api/v1/chat/completions"

PROMPT = """Du er en dansk oversætter for tekniske videnskabelige spørgsmål.

Oversæt følgende engelske GPQA-spørgsmål og de 4 svarmuligheder til NATURLIG dansk.

STRENGE REGLER:
  * Bevar ALLE tal, enheder, formler, symboler og videnskabelig notation EKSAKT som i kilden (fx '10^-4 eV', 'ΔE', 'sin(θ)', 'H2O').
  * Bevar egennavne (fysikere, molekyler, sætninger) på originalt sprog eller brug etableret dansk term.
  * Alt andet tekst oversættes til flydende dansk — brug korrekt dansk faglig terminologi.
  * Svarene skal have SAMME rækkefølge som i input.
  * Output KUN gyldig JSON: {{"question": "...", "answers": ["...", "...", "...", "..."]}}
  * Ingen markdown, ingen kommentarer.

QUESTION (English):
{q}

ANSWERS (English):
0) {a0}
1) {a1}
2) {a2}
3) {a3}"""

PARSE_RE = re.compile(r"\{.*\}", re.S)


def _relax_json(s):
    """Fix bare newlines AND invalid backslash escapes (e.g. LaTeX \\vec, \\Delta)
    within JSON string values."""
    valid_esc = set('"\\/bfnrtu')
    out, in_str, esc = [], False, False
    for c in s:
        if esc:
            if in_str and c not in valid_esc:
                # Unknown escape inside a string — double the backslash so
                # json.loads accepts it verbatim (turns \\vec into \\\\vec).
                out.append("\\")  # emit an extra backslash
            out.append(c); esc = False; continue
        if c == "\\": out.append(c); esc = True; continue
        if c == '"': in_str = not in_str; out.append(c); continue
        if in_str and c in "\n\r\t":
            out.append({"\n":"\\n","\r":"\\r","\t":"\\t"}[c])
        else:
            out.append(c)
    return "".join(out)


def parse(raw):
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("\n",1)[1] if "\n" in t else t
        t = t.rsplit("```",1)[0].strip()
    m = PARSE_RE.search(t)
    if not m: return None
    js = m.group(0)
    try: obj = json.loads(js)
    except json.JSONDecodeError:
        try: obj = json.loads(_relax_json(js))
        except json.JSONDecodeError: return None
    if not isinstance(obj.get("question"), str): return None
    if not isinstance(obj.get("answers"), list) or len(obj["answers"]) != 4:
        return None
    if not all(isinstance(a, str) for a in obj["answers"]): return None
    return obj


async def translate(session, sem, key, provider, orig_idx, row):
    q = row["Question"]
    a0 = row["Correct Answer"]
    a1 = row["Incorrect Answer 1"]
    a2 = row["Incorrect Answer 2"]
    a3 = row["Incorrect Answer 3"]
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(q=q, a0=a0, a1=a1, a2=a2, a3=a3)}],
        "temperature": 0.5,
        "max_tokens": 3500,
        "provider": {"order": [provider], "allow_fallbacks": True},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://claude-code-gpqa-da", "X-Title": "DA-GPQA-Translate"}
    base = {"orig_idx": orig_idx,
            "question_en": q, "answers_en": [a0, a1, a2, a3],
            "correct_idx": 0,
            "subdomain": row.get("Subdomain"), "domain": row.get("High-level domain")}
    async with sem:
        for attempt in range(3):
            try:
                async with session.post(API, headers=headers, json=body, timeout=120) as resp:
                    data = await resp.json()
                if "choices" not in data:
                    if attempt < 2: await asyncio.sleep(2 ** attempt); continue
                    return {**base, "question_da": None, "reject": f"api:{json.dumps(data)[:180]}"}
                raw = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                in_t = int(usage.get("prompt_tokens", 0))
                out_t = int(usage.get("completion_tokens", 0))
                cost = float(usage.get("cost", 0) or 0)
                obj = parse(raw)
                if obj is None:
                    if attempt < 2: await asyncio.sleep(2 ** attempt); continue
                    return {**base, "question_da": None, "in_t": in_t, "out_t": out_t,
                            "cost": cost, "reject": "parse_fail", "raw": raw[:400]}
                return {**base, "question_da": obj["question"], "answers_da": obj["answers"],
                        "in_t": in_t, "out_t": out_t, "cost": cost}
            except Exception as e:
                if attempt < 2: await asyncio.sleep(2 ** attempt); continue
                return {**base, "question_da": None, "reject": f"exc:{str(e)[:200]}"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--key-file", type=Path, default=Path("/home/jepsen/or"))
    ap.add_argument("--provider", default="Google AI Studio")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    key = args.key_file.read_text().strip()
    print("loading Idavidrein/gpqa:gpqa_diamond...", flush=True)
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    n = min(args.limit, len(ds)) if args.limit else len(ds)
    print(f"  {n}/{len(ds)} rows to translate", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    prior_cost = 0.0
    if args.out.exists():
        for line in args.out.open():
            try:
                r = json.loads(line)
                if not r.get("reject"):
                    done.add(r["orig_idx"])
                    prior_cost += r.get("cost", 0) or 0
            except Exception: pass
        print(f"  resume: {len(done)} rows done (${prior_cost:.4f})", flush=True)

    todo = [(i, ds[i]) for i in range(n) if i not in done]
    print(f"  {len(todo)} rows to go", flush=True)
    if not todo: return

    n_ok = n_rej = 0
    tok_in = tok_out = 0
    cost = 0.0
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(args.concurrency)
        tasks = [asyncio.create_task(translate(session, sem, key, args.provider, i, r)) for i, r in todo]
        with args.out.open("a") as fout:
            for coro in asyncio.as_completed(tasks):
                r = await coro
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                if r.get("reject"): n_rej += 1
                else: n_ok += 1
                tok_in += r.get("in_t", 0)
                tok_out += r.get("out_t", 0)
                cost += r.get("cost", 0) or 0
                d = n_ok + n_rej
                if d % 20 == 0 or d == len(todo):
                    print(f"  {d}/{len(todo)}  ok={n_ok} rej={n_rej}  cost=${cost:.4f}", flush=True)

    print(f"\nDone. ok={n_ok} rej={n_rej}  cost=${cost:.4f} (+prior ${prior_cost:.4f})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
