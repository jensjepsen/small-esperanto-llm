"""Stage-1 filter for v6-MT-translated content. Catches catastrophic degeneration.

Three cheap checks:
  - length_ratio < 0.5  → truncation/garbage
  - gzip ratio < 0.15   → repetition collapse compresses near-zero
  - max word freq > 5%  → one word dominating ("skapastraj skapastraj...")

Input: JSONL with `text` (or `eo_text`) field + optional `en_length` for ratio.
Output: clean.jsonl + dropped.jsonl + summary stats.
"""
import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path


def gzip_ratio(text: str) -> float:
    if not text:
        return 1.0
    raw = text.encode("utf-8")
    compressed = gzip.compress(raw, compresslevel=6)
    return len(compressed) / len(raw)


# Common EO stopwords — naturally take 10-15% of any healthy text.
# Excluded from max-freq so the metric tracks degenerate repetition, not normal grammar.
EO_STOP_FREQ = {"la", "de", "kaj", "en", "estas", "al", "ĝi", "li", "ŝi",
                "ne", "kiu", "kiuj", "estis", "per", "por", "el", "kun",
                "sed", "se", "ke", "ankaŭ", "tio", "tiu", "tiuj", "tia",
                "the", "of", "a", "an", "and", "or", "is", "in", "to"}


def max_word_freq(text: str) -> float:
    words = [w for w in re.findall(r"\w+", text.lower()) if w not in EO_STOP_FREQ]
    if not words:
        return 1.0
    counts = Counter(words)
    return counts.most_common(1)[0][1] / len(words)


def local_collapse_stats(text: str, window: int, stem_len: int,
                          diversity_thresh: float) -> tuple[float, int]:
    """Returns (bad_window_frac, max_consecutive_same_word_run).
    bad_window_frac catches diffuse rot; the run catches G.I.-Joe-style hard repeats."""
    words = [w for w in re.findall(r"\w+", text.lower()) if w not in EO_STOP_FREQ]
    if len(words) < window:
        return 0.0, 0
    stems = [w[:stem_len] for w in words]
    n_windows = len(stems) - window + 1
    bad = 0
    for i in range(n_windows):
        win = stems[i : i + window]
        if len(set(win)) / window < diversity_thresh:
            bad += 1
    best_run = cur = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            cur += 1
            if cur > best_run:
                best_run = cur
        else:
            cur = 1
    return bad / n_windows, best_run


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def trim_collapsed_tail(text: str, window: int = 40, stem_len: int = 3,
                         diversity_thresh: float = 0.45) -> tuple[str, int]:
    """v6 MT consistently rots in the last 1-10% of generated text via
    alternating same-stem tokens. Locate every word in the doc, slide a
    suffix-window from the end inward, and truncate at the rightmost char
    position whose preceding window-of-stems is clean. Falls back to whole
    doc if no such position exists past the minimum-keep threshold."""
    matches = list(_WORD_RE.finditer(text))
    if len(matches) < window * 2:
        return text, 0
    # build a parallel list of (stem, end_char_pos) for non-stopwords only
    items = [(m.group().lower()[:stem_len], m.end()) for m in matches
             if m.group().lower() not in EO_STOP_FREQ]
    if len(items) < window * 2:
        return text, 0
    # walk from end inward, stop at first suffix-window with acceptable diversity
    n = len(items)
    cut_end_char = None
    for end in range(n, window, -1):
        suffix_stems = [s for s, _ in items[end - window : end]]
        if len(set(suffix_stems)) / window >= diversity_thresh:
            cut_end_char = items[end - 1][1]
            break
    if cut_end_char is None or cut_end_char >= len(text) - 2:
        return text, 0
    # extend cut to next sentence terminator or end if close
    tail_chunk = text[cut_end_char : cut_end_char + 200]
    m = re.search(r"[.!?]", tail_chunk)
    if m:
        cut_end_char += m.end()
    trimmed = text[:cut_end_char].rstrip()
    return trimmed, len(text) - len(trimmed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL with translated docs")
    ap.add_argument("--clean-out", default=None, help="JSONL of accepted")
    ap.add_argument("--drop-out", default=None, help="JSONL of rejected")
    ap.add_argument("--text-field", default="eo_text")
    ap.add_argument("--en-len-field", default="en_length")
    ap.add_argument("--min-len-ratio", type=float, default=0.5)
    ap.add_argument("--min-gzip-ratio", type=float, default=0.15)
    ap.add_argument("--max-word-freq", type=float, default=0.15)
    # Stage 2: per-window stem diversity (catches local collapses inside
    # otherwise-good articles, e.g. G.I. Joe's military-branch-names region).
    ap.add_argument("--window-size", type=int, default=30,
                    help="non-stopword tokens per rolling window")
    ap.add_argument("--stem-len", type=int, default=4,
                    help="prefix length used as a word stem")
    ap.add_argument("--bad-window-diversity", type=float, default=0.35,
                    help="window flagged bad if distinct_stems / window_size below this")
    ap.add_argument("--max-bad-window-frac", type=float, default=0.10,
                    help="drop doc if more than this fraction of windows are bad")
    ap.add_argument("--max-word-run", type=int, default=10,
                    help="drop if any non-stopword repeats >= this many times consecutively")
    ap.add_argument("--no-trim-tail", action="store_true",
                    help="skip trailing-collapse truncation (default: on)")
    args = ap.parse_args()

    in_path = Path(args.input)
    clean_path = Path(args.clean_out or in_path.with_suffix(".clean.jsonl"))
    drop_path = Path(args.drop_out or in_path.with_suffix(".dropped.jsonl"))

    n_total = n_clean = 0
    n_trimmed = 0
    chars_trimmed = 0
    drops = Counter()
    drop_examples = []
    with in_path.open() as f, clean_path.open("w") as cf, drop_path.open("w") as df:
        for line in f:
            n_total += 1
            row = json.loads(line)
            text = row.get(args.text_field, "")
            if not args.no_trim_tail:
                trimmed, removed = trim_collapsed_tail(text)
                if removed > 0:
                    n_trimmed += 1
                    chars_trimmed += removed
                    text = trimmed
                    row[args.text_field] = trimmed
            en_len = row.get(args.en_len_field) or len(text) * 2  # heuristic

            reasons = []
            len_ratio = len(text) / max(1, en_len)
            if len_ratio < args.min_len_ratio:
                reasons.append(f"len_ratio={len_ratio:.2f}")

            gz = gzip_ratio(text)
            if gz < args.min_gzip_ratio:
                reasons.append(f"gzip_ratio={gz:.2f}")

            mwf = max_word_freq(text)
            if mwf > args.max_word_freq:
                reasons.append(f"max_word_freq={mwf:.2f}")

            bad_frac, max_run = local_collapse_stats(
                text, args.window_size, args.stem_len, args.bad_window_diversity)
            if bad_frac > args.max_bad_window_frac:
                reasons.append(f"bad_window_frac={bad_frac:.2f}")
            if max_run >= args.max_word_run:
                reasons.append(f"word_run={max_run}")

            if reasons:
                for r in reasons:
                    drops[r.split("=")[0]] += 1
                row["_drop_reasons"] = reasons
                df.write(json.dumps(row, ensure_ascii=False) + "\n")
                if len(drop_examples) < 8:
                    drop_examples.append((row.get("title", "?"), reasons,
                                          text[:120], en_len, len(text)))
            else:
                cf.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_clean += 1

    print(f"input: {n_total} rows")
    print(f"kept:  {n_clean}  ({100*n_clean/n_total:.1f}%)")
    print(f"dropped: {n_total - n_clean}  ({100*(n_total-n_clean)/n_total:.1f}%)")
    print(f"  by reason: {dict(drops)}")
    if n_trimmed:
        print(f"tail-trimmed: {n_trimmed} docs ({chars_trimmed:,} chars removed, "
              f"avg {chars_trimmed//max(1,n_trimmed)} chars/doc)")
    print(f"\ndropped examples:")
    for title, reasons, snippet, en_len, eo_len in drop_examples:
        print(f"  {title[:55]:55s}  en={en_len:>7}  eo={eo_len:>7}  reasons={reasons}")
        print(f"    snippet: {snippet}")
    print(f"\nclean → {clean_path}")
    print(f"dropped → {drop_path}")


if __name__ == "__main__":
    main()
