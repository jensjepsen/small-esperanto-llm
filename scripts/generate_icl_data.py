"""Generate ICL (in-context learning) format-following SFT data.

Each example shows K demonstration pairs in some format, then asks the model
to fill the (K+1)-th slot. Format and content are sampled independently so the
model can't memorize format×content pairs — it has to learn the meta-skill
of "infer format from examples, fill the blank".

Two modes per example:
- recall (default ~70%): K+1 random pairs sharing the same property; the model
  must follow the format AND recall the answer for the query entity.
- shared-value (~30%): K+1 entities that share the same value for the property,
  so the answer is inferable purely from the demonstrated pattern (pure ICL).

Sources:
- Wikidata triples grouped by property
- Inverse-direction pairs (e.g., capital → country instead of country → capital)

Output: JSONL of conversations matching scripts/train_sft.py format.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

FACTOIDS_PATH = Path("/mnt/data2/wikidata5m/eo_factoids_v2/eo_factoids.jsonl")

# Properties that produce clean, query-able ICL pairs.
GOOD_PROPERTIES = {
    "okupo", "ŝtataneco", "naskiĝloko", "mortloko", "lernejo", "laborkampo",
    "ĉefurbo", "lando", "estas", "ĝenro", "sporto", "edz(in)o", "patro",
    "patrino", "infano", "frato aŭ fratino", "denaska lingvo", "estro",
    "devenlando", "membro de", "dunginto", "religio", "posteno", "kontinento",
    "ĉefa lingvo", "loĝlando", "verko", "ĉeflando", "regiono", "ĉefa temo",
    "kreinto", "subaro de", "muziko de", "kanzonisto", "esti parto de",
    "nomita laŭ", "antaŭulo", "sekvulo", "membro de sporta teamo",
}

# Properties where inverse direction is meaningful.
INVERSE_OK = {
    "ĉefurbo", "naskiĝloko", "lando", "kreinto", "verko",
    "patro", "patrino", "infano", "edz(in)o", "frato aŭ fratino",
    "antaŭulo", "sekvulo",
}

# Properties where multiple entities commonly share the same value
# (good for shared-value mode).
HIGH_REPEAT = {
    "okupo", "ŝtataneco", "lando", "ĝenro", "sporto", "religio",
    "denaska lingvo", "kontinento", "ĉefa lingvo", "loĝlando",
    "devenlando", "ĉeflando", "subaro de", "estas",
}


# ---- Index ------------------------------------------------------------

def make_index(path: Path) -> tuple[dict, dict]:
    """Build:
       by_prop: property → [(entity, value), ...]
       by_prop_value: property → value → [entities]
    """
    by_prop: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            entity = d["label"]
            if not entity or len(entity) > 60:
                continue
            for fact in d["facts"]:
                p = fact["property"]
                if p not in GOOD_PROPERTIES:
                    continue
                if fact.get("datatype") != "wikibase-item":
                    continue
                v = fact.get("value")
                if not v or len(v) > 50:
                    continue
                if "/" in v or "_" in v or "{" in v or "(" in v.split()[0:1] and ")" not in v:
                    continue
                if re.fullmatch(r"Q\d+", v):
                    continue
                by_prop[p].append((entity, v))

    # Dedupe
    for p in list(by_prop):
        by_prop[p] = list({(e, v) for e, v in by_prop[p]})

    # Build value-grouped index
    by_prop_value: dict[str, dict[str, list[str]]] = {}
    for p, pairs in by_prop.items():
        groups: dict[str, list[str]] = defaultdict(list)
        for e, v in pairs:
            groups[v].append(e)
        # Keep only values with enough entities for shared-value sampling
        by_prop_value[p] = {v: ents for v, ents in groups.items() if len(ents) >= 6}

    return by_prop, by_prop_value


# ---- Renderers --------------------------------------------------------
# Each renderer returns (user_body, assistant_template).
# assistant_template uses {ans} for the answer; "" means just emit the value.

# -- plain "X SEP Y" line-per-pair, lots of separator variants
def make_line_renderer(sep, joiner="\n"):
    def r(pairs, query):
        lines = [f"{x} {sep} {y}" for x, y in pairs]
        lines.append(f"{query} {sep}")
        return joiner.join(lines), ""
    return r

def make_inline_renderer(sep, between=" ; "):
    """X SEP Y ; X SEP Y ; X SEP ?  — single-line format"""
    def r(pairs, query):
        chunks = [f"{x} {sep} {y}" for x, y in pairs]
        chunks.append(f"{query} {sep}")
        return between.join(chunks), ""
    return r

def render_qa_tag(pairs, query):
    lines = [f"<Q>{x}</Q> <A>{y}</A>" for x, y in pairs]
    lines.append(f"<Q>{query}</Q>")
    return "\n".join(lines), "<A>{ans}</A>"

def render_bracket(pairs, query):
    lines = [f"[A] {x} [B] {y}" for x, y in pairs]
    lines.append(f"[A] {query}")
    return "\n".join(lines), "[B] {ans}"

def render_input_output(pairs, query):
    lines = []
    for x, y in pairs:
        lines.append(f"Enigo: {x}\nEligo: {y}")
    lines.append(f"Enigo: {query}\nEligo:")
    return "\n".join(lines), ""

def render_demando_respondo(pairs, query):
    lines = []
    for x, y in pairs:
        lines.append(f"Demando: {x}\nRespondo: {y}")
    lines.append(f"Demando: {query}\nRespondo:")
    return "\n".join(lines), ""

def render_numbered(pairs, query, sep=" — "):
    lines = [f"{i+1}. {x}{sep}{y}" for i, (x, y) in enumerate(pairs)]
    lines.append(f"{len(pairs)+1}. {query}{sep}")
    return "\n".join(lines), ""

def render_bullet(pairs, query, marker="-", sep=": "):
    lines = [f"{marker} {x}{sep}{y}" for x, y in pairs]
    lines.append(f"{marker} {query}{sep}")
    return "\n".join(lines), ""

def render_pipe_table(pairs, query, headers=("Demando", "Respondo")):
    head = f"| {headers[0]} | {headers[1]} |\n| --- | --- |"
    lines = [head] + [f"| {x} | {y} |" for x, y in pairs]
    lines.append(f"| {query} |")
    return "\n".join(lines), ""

def render_estas(pairs, query):
    lines = [f"{x} estas {y}." for x, y in pairs]
    lines.append(f"{query} estas")
    return "\n".join(lines), ""

def render_keyval_yaml(pairs, query):
    lines = [f"{x}: {y}" for x, y in pairs]
    lines.append(f"{query}:")
    return "\n".join(lines), ""

def render_json_ish(pairs, query):
    inner = ",\n  ".join([f'"{x}": "{y}"' for x, y in pairs] + [f'"{query}":'])
    return "{\n  " + inner, ""

def render_code_comment(pairs, query, marker="//"):
    lines = [f"{marker} {x} = {y}" for x, y in pairs]
    lines.append(f"{marker} {query} =")
    return "\n".join(lines), ""

def render_quoted(pairs, query, sep=" → "):
    lines = [f'"{x}"{sep}"{y}"' for x, y in pairs]
    lines.append(f'"{query}"{sep}"')
    return "\n".join(lines), '{ans}"'

def render_triple(pairs, query, prop="rilato"):
    lines = [f"<{x}, {prop}, {y}>" for x, y in pairs]
    lines.append(f"<{query}, {prop}, ?>")
    return "\n".join(lines), ""

def render_paren_arrow(pairs, query):
    lines = [f"({x}) ↦ ({y})" for x, y in pairs]
    lines.append(f"({query}) ↦ (")
    return "\n".join(lines), "{ans})"

def render_markdown_section(pairs, query):
    lines = []
    for x, y in pairs:
        lines.append(f"### {x}\n{y}\n")
    lines.append(f"### {query}\n")
    return "\n".join(lines), ""

def render_csv(pairs, query):
    lines = ["entity,value"]
    for x, y in pairs:
        lines.append(f"{x},{y}")
    lines.append(f"{query},")
    return "\n".join(lines), ""

def render_double_colon(pairs, query):
    lines = [f"{x} :: {y}" for x, y in pairs]
    lines.append(f"{query} ::")
    return "\n".join(lines), ""


RENDERERS = [
    # plain separator variants (line-per-pair)
    ("arrow",      make_line_renderer("→")),
    ("rightarr",   make_line_renderer("->")),
    ("doublearr",  make_line_renderer("=>")),
    ("longarr",    make_line_renderer("⟶")),
    ("eq",         make_line_renderer("=")),
    ("doubleeq",   make_line_renderer("==")),
    ("dash",       make_line_renderer("—")),
    ("colon",      make_line_renderer(":")),
    ("colon_dbl",  render_double_colon),
    # inline (single-line) variants
    ("inline_arrow_semi", make_inline_renderer("→", " ; ")),
    ("inline_eq_comma",   make_inline_renderer("=", ", ")),
    # tagged / structured
    ("qa_tag",     render_qa_tag),
    ("bracket",    render_bracket),
    ("triple",     render_triple),
    ("paren_arr",  render_paren_arrow),
    ("quoted",     render_quoted),
    # narrative
    ("estas",      render_estas),
    ("input_out",  render_input_output),
    ("demresp",    render_demando_respondo),
    # list / table
    ("numbered",   render_numbered),
    ("numbered_dot",  lambda p, q: render_numbered(p, q, ". ")),
    ("bullet_dash",   lambda p, q: render_bullet(p, q, "-", ": ")),
    ("bullet_star",   lambda p, q: render_bullet(p, q, "*", " — ")),
    ("bullet_arrow",  lambda p, q: render_bullet(p, q, "•", " → ")),
    ("pipe_table", render_pipe_table),
    # config-style
    ("yaml",       render_keyval_yaml),
    ("json",       render_json_ish),
    ("csv",        render_csv),
    # code comment
    ("comment_slash",  render_code_comment),
    ("comment_hash",   lambda p, q: render_code_comment(p, q, "#")),
    # markdown
    ("md_section", render_markdown_section),
]


# ---- Preambles --------------------------------------------------------

PREAMBLES = [
    # No preamble — pure format demonstration. Heavily weighted.
    "", "", "", "", "", "", "",
    # Direct
    "Plenigu la mankon:\n",
    "Komplete la lastan linion:\n",
    "Daŭrigu la ŝablonon:\n",
    "Sekvante la ekzemplojn:\n",
    "Surbaze de la sekvaj ekzemploj:\n",
    "Laŭ la sama formato:\n",
    "Identigu la ŝablonon kaj plenumu:\n",
    "Imitu la formaton:\n",
    "Kompletigu la sekvon:\n",
    # Conversational
    "Jen kelkaj ekzemploj. Kompletigu la lastan:\n",
    "Studu la sekvajn parojn kaj plenumu la lastan:\n",
    "Vi vidos malsupre kelkajn rilatojn. Plenumu la mankantan:\n",
    "Bonvolu daŭrigi la liston same:\n",
    "Sekvante la saman strukturon, kompletigu:\n",
    "Mi donas al vi kelkajn ekzemplojn. Plenumu la lastan:\n",
    "Jen modelo:\n",
    "Ekzemploj:\n",
    "Donitaj ekzemploj:\n",
    "Modelo:\n",
    "Provu kompletigi:\n",
    # Style hints
    "Konservante la saman aranĝon:\n",
    "Atentu la formaton:\n",
    "Same kiel supre:\n",
    "Daŭrigu en la sama maniero:\n",
    "Plenumu la ŝablonon montritan ĉi-sube:\n",
]


# ---- Sampling logic ---------------------------------------------------

def sample_recall(by_prop, rng, prop, K, invert):
    """Random K+1 pairs sharing the property — answer requires recall."""
    pool = by_prop[prop]
    if invert:
        pool = [(v, e) for e, v in pool]
    if len(pool) < K + 1:
        return None
    sample = rng.sample(pool, K + 1)
    return sample[:-1], sample[-1]


def sample_shared_value(by_prop_value, rng, prop, K):
    """K+1 entities sharing the same value — pattern-matchable answer."""
    groups = by_prop_value.get(prop, {})
    candidates = [v for v, ents in groups.items() if len(ents) >= K + 1]
    if not candidates:
        return None
    value = rng.choice(candidates)
    entities = rng.sample(groups[value], K + 1)
    pairs = [(e, value) for e in entities]
    return pairs[:-1], pairs[-1]


def make_example(by_prop, by_prop_value, rng, shared_prob=0.30):
    """Sample one ICL example."""
    K = rng.choice([2, 3, 3, 3, 4, 4, 5])
    use_shared = rng.random() < shared_prob

    if use_shared:
        # Pick uniformly from HIGH_REPEAT props that actually have shared values
        shared_props = [
            p for p in HIGH_REPEAT
            if p in by_prop_value
            and any(len(ents) >= K + 1 for ents in by_prop_value[p].values())
        ]
        if shared_props:
            prop = rng.choice(shared_props)
            result = sample_shared_value(by_prop_value, rng, prop, K)
        else:
            use_shared = False

    if not use_shared:
        # Sqrt weighting so massive props don't dominate, but still favored
        props = list(by_prop.keys())
        weights = [len(by_prop[p]) ** 0.5 for p in props]
        prop = rng.choices(props, weights=weights, k=1)[0]
        invert = prop in INVERSE_OK and rng.random() < 0.30
        result = sample_recall(by_prop, rng, prop, K, invert=invert)

    if result is None:
        return None
    shots, (query, answer) = result

    # Pick renderer
    name, renderer = rng.choice(RENDERERS)
    user_body, ans_template = renderer(shots, query)

    preamble = rng.choice(PREAMBLES)
    user_msg = preamble + user_body
    assistant_msg = ans_template.format(ans=answer) if ans_template else answer

    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sft/sft_icl.jsonl")
    parser.add_argument("--n", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=Path, default=FACTOIDS_PATH)
    parser.add_argument("--shared-prob", type=float, default=0.30,
                        help="Fraction of examples using shared-value mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print 8 sample examples instead of writing")
    args = parser.parse_args()

    print(f"Loading factoids from {args.source}...")
    by_prop, by_prop_value = make_index(args.source)
    n_pairs = sum(len(v) for v in by_prop.values())
    print(f"Loaded {n_pairs:,} (entity, value) pairs across {len(by_prop)} properties")

    rng = random.Random(args.seed)

    if args.dry_run:
        print(f"\n--- 8 sample examples (shared-prob={args.shared_prob}) ---\n")
        for i in range(8):
            ex = make_example(by_prop, by_prop_value, rng, args.shared_prob)
            if ex is None:
                continue
            print(f"=== Example {i+1} ===")
            print("USER:")
            print(ex["messages"][0]["content"])
            print("\nASSISTANT:")
            print(ex["messages"][1]["content"])
            print()
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w") as f:
        while written < args.n:
            ex = make_example(by_prop, by_prop_value, rng, args.shared_prob)
            if ex is None:
                continue
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            written += 1
            if written % 2000 == 0:
                print(f"  written {written:,}/{args.n:,}")
    print(f"\nWrote {written:,} examples to {out_path}")


if __name__ == "__main__":
    main()
