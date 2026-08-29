"""Audit distill_alpaca_98k_sft.jsonl for garbled content."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

PATH = Path("/home/jepsen/src/espllm/data/distill_alpaca_98k_sft.jsonl")

CJK = re.compile(r"[　-鿿가-힯぀-ヿ]")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
ARABIC = re.compile(r"[؀-ۿ]")
GREEK = re.compile(r"[Ͱ-Ͽ]")
HEBREW = re.compile(r"[֐-׿]")
REPLACEMENT = re.compile(r"[�]")
PUA = re.compile(r"[-]")
CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# EO diacritics: ĉ ĝ ĥ ĵ ŝ ŭ (both cases)
EO_DIA = "ĉĈĝĜĥĤĵĴŝŜŭŬ"


def rep_ratio(text: str) -> float:
    """Ratio of most-common word to total words; high = degenerate."""
    words = text.lower().split()
    if len(words) < 20:
        return 0.0
    c = Counter(words)
    return c.most_common(1)[0][1] / len(words)


def ngram_loop(text: str) -> bool:
    """Detect degenerate n-gram loops."""
    words = text.lower().split()
    if len(words) < 40:
        return False
    for n in (3, 4, 5):
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        c = Counter(grams)
        top, freq = c.most_common(1)[0]
        if freq >= 6:
            return True
    return False


def eo_ratio(text: str) -> float:
    """Rough Esperanto-ness: fraction of chars that are ASCII letters or EO diacritics."""
    if not text:
        return 0.0
    ok = sum(1 for c in text if c.isalpha() and (c.isascii() or c in EO_DIA))
    total = sum(1 for c in text if c.isalpha())
    return ok / max(total, 1)


def classify(msg_text: str) -> list[str]:
    tags = []
    if not msg_text or not msg_text.strip():
        tags.append("empty")
        return tags
    if CJK.search(msg_text):
        tags.append("cjk")
    if CYRILLIC.search(msg_text):
        tags.append("cyrillic")
    if ARABIC.search(msg_text):
        tags.append("arabic")
    if GREEK.search(msg_text):
        tags.append("greek")
    if HEBREW.search(msg_text):
        tags.append("hebrew")
    if REPLACEMENT.search(msg_text):
        tags.append("replacement")
    if PUA.search(msg_text):
        tags.append("pua")
    if CTRL.search(msg_text):
        tags.append("ctrl")
    if rep_ratio(msg_text) > 0.25:
        tags.append("word_repeat")
    if ngram_loop(msg_text):
        tags.append("ngram_loop")
    if eo_ratio(msg_text) < 0.85:
        tags.append("low_eo_ratio")
    return tags


total = 0
tag_counts = Counter()
any_garbled = 0
examples = {t: [] for t in [
    "empty", "cjk", "cyrillic", "arabic", "greek", "hebrew",
    "replacement", "pua", "ctrl", "word_repeat", "ngram_loop", "low_eo_ratio",
]}

with PATH.open() as f:
    for line in f:
        row = json.loads(line)
        total += 1
        msgs = row.get("messages", [])
        row_tags = set()
        for m in msgs:
            content = m.get("content", "")
            for t in classify(content):
                row_tags.add(t)
                tag_counts[t] += 1
                if len(examples[t]) < 3:
                    examples[t].append((m.get("role"), content[:220]))
        if row_tags:
            any_garbled += 1

print(f"total rows: {total:,}")
print(f"rows with any garbled tag: {any_garbled:,} ({100*any_garbled/total:.2f}%)")
print()
print("per-tag counts (messages, not rows):")
for tag, cnt in tag_counts.most_common():
    print(f"  {tag:15s} {cnt:>7,}")

print("\nexamples:")
for tag, exs in examples.items():
    if not exs:
        continue
    print(f"\n[{tag}]")
    for role, txt in exs:
        print(f"  ({role}) {txt}")
