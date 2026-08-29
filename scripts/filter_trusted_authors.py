"""Filter Gutenberg catalog by domain + trusted-author whitelist.

Supports three profiles (--profile):
  sci  — science / philosophy / medicine  (LoCC Q*/R*/BF/GN/BC/BD)
  lit  — classic literature              (LoCC PR/PS/PA/PG/PQ/PT)
  econ — economics                       (LoCC HB/HC/HD/HF)

Within each profile, only books by named foundational authors are kept.
Output: /mnt/data2/sci_books/<profile>_trusted.jsonl
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

CATALOG = Path("/mnt/data2/sci_books/gutenberg_catalog.csv")
OUT_DIR = Path("/mnt/data2/sci_books")

PROFILES = {
    "sci": {
        "loccs": {"Q", "QA", "QB", "QC", "QD", "QE", "QH", "QK", "QL", "QM",
                  "QP", "QR", "R", "RA", "RB", "RC", "RD", "BF", "GN", "BC", "BD"},
        "authors": [
            # Evolution / biology
            "Darwin", "Huxley", "Wallace", "Spencer, Herbert", "Haeckel", "Romanes",
            "Mivart", "Bateson", "Galton", "Lankester", "Fabre", "Agassiz",
            "Lyell", "Hutton", "Geikie", "Powell, John", "Humboldt", "Croll",
            "Vries, Hugo de", "Wegener",
            # Physics / astronomy / chemistry
            "Faraday", "Tyndall", "Helmholtz", "Eddington", "Einstein", "Jeans",
            "Maxwell, James", "Galilei", "Kepler", "Newton, Isaac", "Herschel",
            "Newcomb", "Flammarion", "Lockyer", "Ball, Robert", "Mach, Ernst",
            "Poincaré", "Lodge, Oliver", "Crookes", "Ostwald", "Ramsay, William",
            "Curie", "Rutherford, Ernest", "Mendeleev", "Lavoisier",
            # Math / logic
            "Boole", "De Morgan", "Cantor", "Russell, Bertrand", "Whitehead",
            "Klein, Felix", "Frege", "Peano", "Cayley", "Hilbert",
            # Psychology
            "James, William", "Wundt", "Freud", "Jung, C.", "Watson, John",
            "Pavlov", "Köhler", "Hall, G. Stanley",
            # Medicine / physiology
            "Bernard, Claude", "Lister", "Pasteur", "Cannon, Walter", "Sherrington",
            "Osler", "Cajal",
            # Philosophy
            "Mill, John", "Hume", "Locke", "Kant", "Bergson", "Dewey",
            "Comte", "Spinoza", "Hobbes",
            # Anthropology / sociology
            "Tylor", "Frazer", "Boas, Franz", "Durkheim", "Mauss", "Marx, Karl",
            # Engineering
            "Babbage", "Smiles, Samuel", "Lippmann, Walter",
        ],
    },
    "lit": {
        # PR = English, PS = American, PA = Greek/Latin classics,
        # PG = Slavic/Russian, PQ = Romance, PT = Germanic, PN = general lit
        "loccs": {"PR", "PS", "PA", "PG", "PQ", "PT", "PN", "PZ"},
        "authors": [
            # English literature (PR)
            "Dickens", "Austen", "Brontë", "Eliot, George", "Hardy, Thomas",
            "Trollope", "Stevenson, Robert", "Wilde", "Wells, H.", "Stoker",
            "Carroll, Lewis", "Doyle, Arthur", "Kipling", "Shaw, George",
            "Galsworthy", "Bennett, Arnold", "Chesterton", "Conrad, Joseph",
            "Forster, E.", "Hudson, W.", "Lawrence, D.", "Meredith, George",
            "Collins, Wilkie", "Thackeray", "Scott, Walter", "Defoe",
            "Swift, Jonathan", "Fielding, Henry", "Sterne, Laurence",
            # American literature (PS)
            "Twain", "Clemens, Samuel", "Melville", "Hawthorne", "Whitman",
            "Poe, Edgar", "James, Henry", "Wharton", "London, Jack",
            "Crane, Stephen", "Norris, Frank", "Sinclair, Upton", "Cather, Willa",
            "Howells, William", "Bierce", "Alcott", "Cooper, James Fenimore",
            "Irving, Washington", "Emerson", "Thoreau", "Dickinson, Emily",
            "Longfellow", "Lowell, James Russell", "Holmes, Oliver Wendell",
            "Garland, Hamlin", "Dreiser",
            # Greek / Latin classics (PA)
            "Plato", "Aristotle", "Cicero", "Homer", "Virgil", "Ovid",
            "Horace", "Plutarch", "Seneca", "Sophocles", "Euripides",
            "Aeschylus", "Aristophanes", "Tacitus", "Livy", "Lucretius",
            "Marcus Aurelius", "Epictetus", "Suetonius", "Caesar, Julius",
            "Xenophon", "Pliny",
            # Russian (PG)
            "Tolstoy", "Dostoyevsky", "Chekhov", "Turgenev", "Gogol",
            "Pushkin", "Lermontov",
            # French (PQ)
            "Hugo, Victor", "Dumas, Alexandre", "Balzac", "Flaubert",
            "Zola", "Maupassant", "Verne, Jules", "Stendhal", "Voltaire",
            "Rousseau", "Molière", "Racine",
            # German (PT)
            "Goethe", "Schiller", "Mann, Thomas", "Heine", "Schopenhauer",
            "Nietzsche", "Lessing", "Grimm",
            # Other notable
            "Cervantes", "Dante", "Ibsen", "Strindberg",
        ],
    },
    "econ": {
        # HB-HJ = economics, HX = socialism/communism, JC = political theory.
        # 19th c. econ vs political philosophy is a blurry line — Marx, Mill,
        # Rousseau all straddle. Bundled into one "political economy" profile.
        "loccs": {"HB", "HC", "HD", "HE", "HF", "HG", "HJ", "HX", "JC", "JA"},
        "authors": [
            # Classical economics
            "Smith, Adam", "Ricardo, David", "Malthus", "Mill, John",
            "Marshall, Alfred", "Veblen", "Jevons", "Pareto", "Wicksell",
            "Bagehot, Walter", "George, Henry", "Keynes", "Hobson, J.",
            "Cannan", "Pigou", "Edgeworth, Francis", "Walras", "Menger, Carl",
            "Bohm-Bawerk", "Böhm-Bawerk", "Mises", "Hayek", "Schumpeter",
            # Socialism / radical political economy
            "Marx, Karl", "Engels", "Lenin", "Trotsky", "Bakunin",
            "Kropotkin", "Bellamy, Edward",
            # Classical political philosophy
            "Plato", "Aristotle", "Hobbes", "Locke", "Rousseau",
            "Montesquieu", "Burke, Edmund", "Madison, James",
            "Hamilton, Alexander", "Jefferson", "Paine, Thomas",
            "Bentham", "Tocqueville",
            # Adjacent thinkers
            "Spencer, Herbert", "Carlyle, Thomas", "Ruskin",
        ],
    },
}


def in_domain(locc: str, allowed: set[str]) -> str | None:
    if not locc:
        return None
    for code in (c.strip().upper() for c in locc.split(";")):
        if not code:
            continue
        # exact 2-char prefix match
        if code[:2] in allowed:
            return code[:2]
        if code[:1] in allowed:
            return code[:1]
    return None


def author_matches(authors_field: str, trusted: list[str]) -> str | None:
    if not authors_field:
        return None
    for author in authors_field.split(";"):
        author = author.strip()
        for t in trusted:
            if "," in t:
                if author.startswith(t + ",") or author.startswith(t + " "):
                    return t
            else:
                if author.startswith(t + ", "):
                    return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=list(PROFILES.keys()))
    ap.add_argument("--per-author-cap", type=int, default=0,
                    help="max books per trusted author (0 = no cap). Within an "
                         "author, lowest Gutenberg id wins (proxy for early-added = "
                         "more popular).")
    args = ap.parse_args()
    profile = PROFILES[args.profile]
    allowed_loccs = profile["loccs"]
    trusted = profile["authors"]

    rows = list(csv.DictReader(CATALOG.open()))
    en_text = [r for r in rows
               if r.get("Language", "").lower() == "en"
               and r.get("Type", "") == "Text"]
    print(f"profile: {args.profile}")
    print(f"  loccs: {sorted(allowed_loccs)}")
    print(f"  trusted authors: {len(trusted)}")
    print(f"  english texts in catalog: {len(en_text):,}")

    kept = []
    by_author = Counter()
    by_domain = Counter()
    for r in rows:
        if r.get("Language", "").lower() != "en":
            continue
        if r.get("Type", "") != "Text":
            continue
        domain = in_domain(r.get("LoCC", ""), allowed_loccs)
        if not domain:
            continue
        match = author_matches(r.get("Authors", ""), trusted)
        if not match:
            continue
        kept.append({
            "gutenberg_id": int(r["Text#"]),
            "title": r["Title"],
            "authors": r.get("Authors", ""),
            "trusted_author": match,
            "issued": r.get("Issued", ""),
            "subjects": r.get("Subjects", ""),
            "locc": r.get("LoCC", ""),
            "bookshelves": r.get("Bookshelves", ""),
            "domain": domain,
            "profile": args.profile,
        })
        by_author[match] += 1
        by_domain[domain] += 1

    if args.per_author_cap > 0:
        by_author_books = defaultdict(list)
        for r in kept:
            by_author_books[r["trusted_author"]].append(r)
        capped = []
        for author, books in by_author_books.items():
            # ascending gid = earliest added (heuristic for popularity)
            books.sort(key=lambda r: r["gutenberg_id"])
            capped.extend(books[: args.per_author_cap])
        print(f"  per-author cap {args.per_author_cap}: {len(kept):,} → {len(capped):,}")
        kept = capped
        by_author = Counter(r["trusted_author"] for r in kept)
        by_domain = Counter(r["domain"] for r in kept)

    out_path = OUT_DIR / f"{args.profile}_trusted.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  kept: {len(kept):,}")
    print(f"\n  by LoCC:")
    for d, n in by_domain.most_common():
        print(f"    {d:>3}: {n:>4}")
    print(f"\n  top 25 authors:")
    for a, n in by_author.most_common(25):
        print(f"    {n:>3}  {a}")
    zero = [t for t in trusted if t not in by_author]
    print(f"\n  zero-hits ({len(zero)}):")
    for t in zero[:20]:
        print(f"    - {t}")
    if len(zero) > 20:
        print(f"    ... +{len(zero)-20} more")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
