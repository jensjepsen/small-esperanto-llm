"""Translate handcrafted ICL eval (EO) to English using v5b.

Translates each row's user prompt, gold answer, and all accepted answer
variants. Writes a parallel JSONL with the same structure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import MarianMTModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sp_tokenizer import SPMTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/mnt/data/espllm/runs/mt/eneo_v5b/final")
    ap.add_argument("--tokenizer", default="mt/data/tokenizer/spm_eneo_32k.model")
    ap.add_argument("--src", type=Path, default=Path("data/causal_corpus/eval_handcrafted_v31.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/causal_corpus/eval_handcrafted_v31_en.jsonl"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=192)
    args = ap.parse_args()

    print(f"Loading {args.checkpoint}…")
    tok = SPMTokenizer(args.tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MarianMTModel.from_pretrained(args.checkpoint).to(device).eval()
    model.generation_config.no_repeat_ngram_size = 5

    rows = [json.loads(l) for l in args.src.open()]
    print(f"  {len(rows)} rows to translate")

    # Collect all unique strings to translate
    strings = []
    str_idx = {}
    def add(s):
        if s not in str_idx:
            str_idx[s] = len(strings)
            strings.append(s)
    for r in rows:
        for m in r["messages"]:
            add(m["content"])
        for a in r["accepted_answers"]:
            add(a)
    print(f"  {len(strings)} unique strings")

    # Hand-curated lookup for short EO answer labels — the model degenerates on
    # bare 2-3 char stems even with carrier sentences, so just translate these
    # explicitly. Keys are the lowercased trimmed form (and the original casing).
    LOOKUP = {
        # numbers
        "du": "two", "tri": "three", "kvar": "four", "kvin": "five",
        "ses": "six", "sep": "seven", "ok": "eight", "naŭ": "nine", "dek": "ten",
        "unu": "one", "nul": "zero",
        # colors
        "ruĝa": "red", "blua": "blue", "verda": "green", "flava": "yellow",
        "nigra": "black", "blanka": "white", "bruna": "brown", "griza": "grey",
        "rozkolora": "pink", "oranĝa": "orange", "purpura": "purple",
        # binary states
        "fermita": "closed", "malfermita": "open",
        "plena": "full", "malplena": "empty",
        "ŝlosita": "locked", "malŝlosita": "unlocked",
        "aktiva": "active", "neaktiva": "inactive",
        "pura": "clean", "malpura": "dirty",
        "varma": "warm", "malvarma": "cold",
        "vivanta": "alive", "morta": "dead",
        # time-of-day
        "matene": "in the morning", "vespere": "in the evening", "nokte": "at night",
        "tage": "during the day", "posttagmeze": "in the afternoon",
        # misc
        "ujo": "container",
        # common nouns that appear as "La X" patterns in golds
        "instruisto": "teacher", "instruistino": "teacher",
        "knabo": "boy", "knabino": "girl",
        "avo": "grandfather", "avino": "grandmother",
        "maljunulino": "old woman", "maljunulo": "old man",
        "kuracisto": "doctor", "ĉasisto": "hunter",
        "luno": "moon", "pluvo": "rain", "rivero": "river",
        "kato": "cat", "hundo": "dog", "patro": "father", "patrino": "mother",
        "fratino": "sister", "frato": "brother",
        # locations/containers
        "monujo": "wallet", "fridujo": "fridge", "skatolo": "box", "ŝranko": "cabinet",
        # times
        "matene": "morning", "vespere": "evening",
        # everyday objects/concepts (noun stems — accusative -n / plural -j stripped at lookup)
        "lakto": "milk", "mono": "money", "ŝlosilo": "key",
        "fenestro": "window", "urbo": "city", "ĉapelo": "hat",
        "arbo": "tree", "libro": "book", "papilo": "butterfly",
        "frukto": "fruit", "ilo": "tool", "planto": "plant",
        "besto": "animal", "tero": "earth", "pordo": "door",
        "tablo": "table", "seĝo": "chair", "lito": "bed",
        # numbers — written words
        "cent": "hundred", "mil": "thousand",
        # proper names — pass through
        "zamenhof": "Zamenhof", "linus torvalds": "Linus Torvalds",
        "watson kaj crick": "Watson and Crick",
        "parizo": "Paris", "tokio": "Tokyo", "stratford": "Stratford",
        "germanio": "Germany", "vieno": "Vienna",
        "sudameriko": "South America", "egiptujo": "Egypt",
        "etiopio": "Ethiopia", "jupyter": "Jupyter",
        # elements/concepts
        "hidrogeno": "hydrogen", "oksigeno": "oxygen",
        "radiumo": "radium", "poloniumo": "polonium",
        "elefanto": "elephant", "luno": "moon",
    }
    # Prefixes that wrap a noun phrase
    EO_PREP_PREFIXES = {"al la": "to the", "per la": "with the", "en la": "in the",
                        "sur la": "on the", "sub la": "under the", "kun la": "with the"}
    EO_CONJ_PATTERN = re.compile(r"^(\S+)\s+(?:kaj|aŭ)\s+(\S+)$")
    EO_WRAP_PREFIX = "La respondo estas: "
    EN_STRIP_PATTERNS = [
        re.compile(r"^\s*the answer is[:,]?\s*", re.IGNORECASE),
    ]

    def _resolve(word: str) -> str | None:
        """Look up a single word, stripping accusative -n / plural -j endings."""
        if word in LOOKUP:
            return LOOKUP[word]
        # Strip accusative -n (e.g., "lakton" → "lakto")
        if word.endswith("n") and word[:-1] in LOOKUP:
            return LOOKUP[word[:-1]]
        # Strip plural -j (e.g., "papiloj" → "papilo")
        if word.endswith("j") and word[:-1] in LOOKUP:
            return LOOKUP[word[:-1]] + "s"     # crude pluralization
        # Strip accusative+plural -jn (e.g., "papilojn")
        if word.endswith("jn") and word[:-2] in LOOKUP:
            return LOOKUP[word[:-2]] + "s"
        return None

    def lookup_short(s: str) -> str | None:
        """Return EN translation if the string is a hand-translatable short label.

        Handles bare stems with accusative/plural inflection, '.' suffix,
        'La X' patterns, and prepositional 'prep la X' patterns.
        """
        stripped = s.strip()
        ends_period = stripped.endswith(".")
        core = stripped.rstrip(".").lower()
        # Bare word
        en = _resolve(core)
        if en is not None:
            return en + ("." if ends_period else "")
        # 'La X' → 'the X'
        m = re.match(r"^la\s+(\S+)$", core)
        if m:
            en = _resolve(m.group(1))
            if en is not None:
                return "the " + en + ("." if ends_period else "")
        # 'prep la X' → 'EN_prep X'
        for eo_p, en_p in EO_PREP_PREFIXES.items():
            if core.startswith(eo_p + " "):
                rest = core[len(eo_p) + 1:].strip()
                en = _resolve(rest)
                if en is not None:
                    return en_p + " " + en + ("." if ends_period else "")
        # 'X kaj Y' → 'X and Y' (both in lookup)
        m = EO_CONJ_PATTERN.match(core)
        if m:
            a = _resolve(m.group(1))
            b = _resolve(m.group(2))
            if a is not None and b is not None:
                return f"{a} and {b}" + ("." if ends_period else "")
        return None

    def needs_wrap(s: str) -> bool:
        return len(s.split()) < 4

    translations: dict[str, str] = {}
    # Apply lookup first (covers the bare 2-3 char stems)
    to_translate = []
    for s in strings:
        hit = lookup_short(s)
        if hit is not None:
            translations[s] = hit
        else:
            to_translate.append((s, EO_WRAP_PREFIX + s if needs_wrap(s) else s))
    print(f"  lookup hit: {len(strings) - len(to_translate)} / {len(strings)}")
    print(f"  model translates: {len(to_translate)}")

    # Batch-translate remaining via model
    with torch.no_grad():
        for i in tqdm(range(0, len(to_translate), args.batch_size), desc="translating"):
            batch = to_translate[i : i + args.batch_size]
            ids_list = [tok.encode(w, lang="en") for _, w in batch]
            be = tok.pad_batch(ids_list)
            inp = be.input_ids.to(device)
            attn = be.attention_mask.to(device)
            out = model.generate(
                input_ids=inp,
                attention_mask=attn,
                num_beams=args.num_beams,
                max_length=args.max_length,
                early_stopping=True,
                no_repeat_ngram_size=5,
            )
            for (src, w), seq in zip(batch, out):
                tx = tok.decode(seq)
                if w != src:  # was wrapped — strip prefix
                    for pat in EN_STRIP_PATTERNS:
                        m = pat.match(tx)
                        if m:
                            tx = tx[m.end():]
                            break
                translations[src] = tx.strip()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fout:
        for r in rows:
            new_row = {
                "messages": [
                    {"role": m["role"], "content": translations[m["content"]]}
                    for m in r["messages"]
                ],
                "accepted_answers": [translations[a] for a in r["accepted_answers"]],
            }
            fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
