"""Download and extract Tekstaro de Esperanto corpus to JSONL.

Tekstaro (tekstaro.com) is the canonical curated EO corpus — ~14.9M words
of classical/modern EO literature, journalism (incl. Monato), Zamenhof
translations, and the EO Bible. Maintained by ESF. TEI-5 XML format.

We use the `sen_streketoj` (without pedagogical hyphens) variant — hyphens
are word-division aids for learners and would corrupt tokenization.

Output: data/tekstaro/tekstaro.jsonl with rows
  {id, title, author, source_file, text, n_words}

A separate JSONL per text (instead of one giant blob) is fine — the
downstream loader concatenates anyway and downstream filters (eg.
min_article_length) operate per-document.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS = {"tei": "http://www.tei-c.org/ns/1.0"}
DEFAULT_URL = (
    "https://tekstaro.com/elshutebla/"
    "tekstaro_de_esperanto_xml_sen_streketoj.zip"
)

# Block-level tags: after their content, insert paragraph break
BLOCK_TAGS = {"p", "head", "div", "lg", "l", "ab", "sp", "stage"}


def extract_text(elem) -> str:
    """Concatenate all text under elem, preserving paragraph breaks."""
    parts: list[str] = []

    def walk(node):
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
            child_tag = child.tag.rpartition("}")[-1]
            if child_tag in BLOCK_TAGS:
                parts.append("\n\n")
            if child.tail:
                parts.append(child.tail)

    walk(elem)
    txt = "".join(parts)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r" *\n *", "\n", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def _first_text(root, paths) -> str | None:
    for p in paths:
        e = root.find(p, NS)
        if e is not None and e.text and e.text.strip():
            return e.text.strip()
    return None


def parse_one(xml_path: Path) -> dict | None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    title = _first_text(root, [".//tei:titleStmt/tei:title"]) or xml_path.stem
    author = _first_text(root, [
        ".//tei:biblStruct/tei:analytic/tei:author",
        ".//tei:biblStruct/tei:monogr/tei:author",
    ])
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return None
    text = extract_text(body)
    if not text:
        return None
    xml_id = root.attrib.get(
        "{http://www.w3.org/XML/1998/namespace}id", xml_path.stem
    )
    return {
        "id": xml_id,
        "title": title,
        "author": author,
        "source_file": xml_path.name,
        "text": text,
        "n_words": len(text.split()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--zip", default="/tmp/tekstaro.zip",
                    help="local ZIP path (downloads if missing)")
    ap.add_argument("--extract-dir", default="/tmp/tekstaro_xml")
    ap.add_argument("--out", default="data/tekstaro/tekstaro.jsonl")
    args = ap.parse_args()

    zp = Path(args.zip)
    if not zp.exists():
        print(f"downloading {args.url} -> {zp}", flush=True)
        urllib.request.urlretrieve(args.url, zp)
    else:
        print(f"reusing existing {zp} ({zp.stat().st_size / 1e6:.1f} MB)",
              flush=True)

    ed = Path(args.extract_dir)
    if not (ed / "tekstaro_de_esperanto_xml_sen_streketoj").exists():
        print(f"extracting {zp} -> {ed}", flush=True)
        ed.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(ed)

    candidates = list(ed.glob("*/tekstoj"))
    if not candidates:
        sys.exit(f"no */tekstoj/ dir under {ed}")
    tekstoj = candidates[0]

    xml_files = sorted(tekstoj.glob("*.xml"))
    print(f"parsing {len(xml_files)} XML files", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_skipped = n_words = 0
    with open(out_path, "w") as f:
        for x in xml_files:
            try:
                rec = parse_one(x)
                if rec is None:
                    n_skipped += 1
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
                n_words += rec["n_words"]
                if n_ok % 25 == 0:
                    print(f"  {n_ok}/{len(xml_files)}  words={n_words:,}",
                          flush=True)
            except ET.ParseError as e:
                print(f"  XML err {x.name}: {e}", flush=True)
                n_skipped += 1
            except Exception as e:
                print(f"  err {x.name}: {type(e).__name__}: {e}", flush=True)
                n_skipped += 1

    print(f"\nwrote {n_ok} texts ({n_words:,} words) -> {out_path}; "
          f"skipped={n_skipped}")


if __name__ == "__main__":
    main()
