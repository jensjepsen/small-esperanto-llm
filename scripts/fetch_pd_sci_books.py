"""Fetch the 100 PD scientific works for v11 EN→EO MT pretrain.

Uses Gutendex (https://gutendex.com) to look up each (title, author)
pair, picks the top-matching English plaintext format, downloads, and
strips Project Gutenberg's boilerplate header/footer. Outputs:
  - /mnt/data2/sci_books/<gutenberg_id>.txt   (clean text per book)
  - /mnt/data2/sci_books/manifest.jsonl       (one row per book with metadata)

Books that 404 or don't have a plaintext format get logged but skipped
(some titles may be on the Internet Archive only).

Usage:
    uv run python scripts/fetch_pd_sci_books.py
    uv run python scripts/fetch_pd_sci_books.py --limit 5    # smoke test
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT_DIR = Path("/mnt/data2/sci_books")
USER_AGENT = "espllm-pretrain-data/1.0 (jens.jepsen@gmail.com)"

# (title, author_lastname, optional_gutenberg_id_override)
# IDs are filled in where we know them to skip the search step.
BOOKS = [
    # Evolution / Biology
    ("On the Origin of Species", "Darwin", 1228),
    ("The Descent of Man", "Darwin", 2300),
    ("The Voyage of the Beagle", "Darwin", 3704),
    ("Expression of the Emotions in Man and Animals", "Darwin", 1227),
    ("The Variation of Animals and Plants Under Domestication", "Darwin", None),
    ("The Malay Archipelago", "Wallace", 2530),
    ("Contributions to the Theory of Natural Selection", "Wallace", None),
    ("Darwinism", "Wallace", None),
    ("Man's Place in Nature", "Huxley", None),
    ("Lectures and Essays", "Huxley", None),
    ("Mental Evolution in Animals", "Romanes", None),
    ("The Principles of Biology", "Spencer", None),
    ("Hereditary Genius", "Galton", None),
    ("Materials for the Study of Variation", "Bateson", None),
    ("Species and Varieties Their Origin by Mutation", "de Vries", None),

    # Geology
    ("Principles of Geology", "Lyell", None),
    ("The Antiquity of Man", "Lyell", None),
    ("Studies on Glaciers", "Agassiz", None),
    ("Class-Book of Geology", "Geikie", None),
    ("Exploration of the Colorado River", "Powell", None),
    ("Cosmos", "Humboldt", None),
    ("The Origin of Continents and Oceans", "Wegener", None),
    ("Climate and Time", "Croll", None),

    # Physics & Cosmology
    ("Experimental Researches in Electricity", "Faraday", None),
    ("The Chemical History of a Candle", "Faraday", 14474),
    ("The Forces of Matter", "Faraday", None),
    ("Heat A Mode of Motion", "Tyndall", None),
    ("The Forms of Water", "Tyndall", None),
    ("The Glaciers of the Alps", "Tyndall", None),
    ("Popular Lectures on Scientific Subjects", "Helmholtz", None),
    ("On the Sensations of Tone", "Helmholtz", None),
    ("The Science of Mechanics", "Mach", None),
    ("Science and Hypothesis", "Poincare", None),
    ("Science and Method", "Poincare", None),
    ("Space Time and Gravitation", "Eddington", None),
    ("Stars and Atoms", "Eddington", None),
    ("Relativity The Special and General Theory", "Einstein", 30155),
    ("The Universe Around Us", "Jeans", None),

    # Chemistry
    ("Principles of Chemistry", "Mendeleev", None),
    ("On Radiant Matter", "Crookes", None),
    ("Outlines of General Chemistry", "Ostwald", None),
    ("Modern Chemistry", "Ramsay", None),
    ("Radioactive Substances", "Curie", None),
    ("Radioactivity", "Rutherford", None),

    # Astronomy
    ("Dialogue Concerning the Two Chief World Systems", "Galileo", None),
    ("Popular Astronomy", "Newcomb", None),
    ("Astronomy for Everybody", "Newcomb", None),
    ("Popular Astronomy", "Flammarion", None),
    ("The Sun's Place in Nature", "Lockyer", None),
    ("Mars and Its Canals", "Lowell", None),
    ("A Treatise on Astronomy", "Herschel", None),
    ("The Story of the Heavens", "Ball", None),

    # Psychology / Neuroscience
    ("The Principles of Psychology Volume 1", "James", None),
    ("The Principles of Psychology Volume 2", "James", None),
    ("Varieties of Religious Experience", "James", 621),
    ("Talks to Teachers on Psychology", "James", 16287),
    ("Lectures on Human and Animal Psychology", "Wundt", None),
    ("Recollections of My Life", "Cajal", None),
    ("Conditioned Reflexes", "Pavlov", None),
    ("Psychology from the Standpoint of a Behaviorist", "Watson", None),
    ("The Interpretation of Dreams", "Freud", None),
    ("Introduction to Psychoanalysis", "Freud", None),
    ("Psychology of the Unconscious", "Jung", None),
    ("The Mentality of Apes", "Köhler", None),

    # Medicine
    ("Introduction to the Study of Experimental Medicine", "Bernard", None),
    ("On the Antiseptic Principle", "Lister", None),
    ("Bodily Changes in Pain Hunger Fear and Rage", "Cannon", None),
    ("The Principles and Practice of Medicine", "Osler", None),
    ("The Integrative Action of the Nervous System", "Sherrington", None),

    # Anthropology / Social Sciences
    ("Primitive Culture", "Tylor", None),
    ("The Golden Bough", "Frazer", None),
    ("The Mind of Primitive Man", "Boas", None),
    ("The Elementary Forms of the Religious Life", "Durkheim", None),
    ("Suicide A Study in Sociology", "Durkheim", None),
    ("The Gift", "Mauss", None),
    ("The Wealth of Nations", "Smith", 3300),
    ("Capital Volume 1", "Marx", None),
    ("On Liberty", "Mill", 34901),
    ("The Principles of Sociology", "Spencer", None),

    # Philosophy of Science
    ("The Problems of Philosophy", "Russell", 5827),
    ("Our Knowledge of the External World", "Russell", None),
    ("Introduction to Mathematical Philosophy", "Russell", None),
    ("Science and the Modern World", "Whitehead", None),
    ("An Introduction to Mathematics", "Whitehead", None),
    ("Reconstruction in Philosophy", "Dewey", None),
    ("Creative Evolution", "Bergson", None),
    ("A System of Logic", "Mill", None),

    # Engineering / Tech History
    ("On the Economy of Machinery and Manufactures", "Babbage", None),
    ("Lives of the Engineers", "Smiles", None),
    ("Public Opinion", "Lippmann", None),
    ("Records of a Family of Engineers", "Stevenson", None),
    ("The Empire of Business", "Carnegie", None),

    # Math / Logic
    ("The Laws of Thought", "Boole", None),
    ("Formal Logic", "De Morgan", None),
    ("Principia Mathematica", "Russell", None),
    ("Contributions to Founding of Theory of Transfinite Numbers", "Cantor", None),
    ("Lectures on Mathematics", "Klein", None),
    ("The Queen of the Sciences", "Bell", None),
    ("Foundations of Arithmetic", "Frege", None),
    ("Arithmetices Principia", "Peano", None),
]


def http_get(url: str, accept_json: bool = False) -> bytes | dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    if accept_json:
        return json.loads(data)
    return data


def search_gutendex(title: str, author: str) -> int | None:
    """Find best matching Gutenberg ID for (title, author). Returns None if not found."""
    q = urllib.parse.quote(f"{title} {author}")
    try:
        result = http_get(f"https://gutendex.com/books?search={q}&languages=en",
                          accept_json=True)
    except Exception as e:
        print(f"  search error: {e}")
        return None
    books = result.get("results", [])
    if not books:
        return None
    # Prefer books with English plaintext format
    for b in books:
        formats = b.get("formats", {})
        for k in formats:
            if "text/plain" in k:
                return b["id"]
    return books[0]["id"]


def fetch_book(gid: int) -> str | None:
    """Download plain-text body of a Gutenberg book, stripping boilerplate."""
    # Common URL patterns; try a few
    urls = [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
    ]
    raw = None
    for url in urls:
        try:
            raw = http_get(url).decode("utf-8", errors="replace")
            break
        except Exception:
            continue
    if raw is None:
        return None

    # Strip header and footer between *** START / *** END markers
    start_re = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                          re.IGNORECASE)
    end_re = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                        re.IGNORECASE)
    m_start = start_re.search(raw)
    m_end = end_re.search(raw)
    if m_start:
        raw = raw[m_start.end():]
    if m_end:
        raw = raw[:m_end.start()]
    return raw.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="only process first N books (0 = all)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.jsonl"

    # Resume support: skip books already in manifest
    done_titles = set()
    if manifest_path.exists():
        with manifest_path.open() as f:
            for line in f:
                try:
                    done_titles.add(json.loads(line)["title"])
                except Exception:
                    pass

    books = BOOKS[: args.limit] if args.limit else BOOKS
    n_ok = n_fail = n_skip = 0

    with manifest_path.open("a") as mf:
        for i, (title, author, override_id) in enumerate(books, 1):
            if title in done_titles:
                print(f"[{i:3d}/{len(books)}] {title[:55]:55s} skip (already have)")
                n_skip += 1
                continue

            gid = override_id
            if gid is None:
                gid = search_gutendex(title, author)
                time.sleep(0.3)
            if gid is None:
                print(f"[{i:3d}/{len(books)}] {title[:55]:55s} no Gutenberg match")
                n_fail += 1
                continue

            text = fetch_book(gid)
            time.sleep(0.3)
            if not text or len(text) < 5000:
                print(f"[{i:3d}/{len(books)}] {title[:55]:55s} gid={gid} download failed")
                n_fail += 1
                continue

            out_path = OUT_DIR / f"{gid}.txt"
            out_path.write_text(text, encoding="utf-8")
            mf.write(json.dumps({
                "title": title,
                "author": author,
                "gutenberg_id": gid,
                "length": len(text),
                "path": str(out_path),
            }, ensure_ascii=False) + "\n")
            mf.flush()
            n_ok += 1
            print(f"[{i:3d}/{len(books)}] {title[:55]:55s} gid={gid:>6} "
                  f"{len(text)/1e3:6.1f}k chars  ok")

    print(f"\n=== done: {n_ok} ok, {n_fail} failed, {n_skip} skipped ===")
    print(f"output: {OUT_DIR}")


if __name__ == "__main__":
    main()
