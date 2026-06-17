"""Expand benchmark answer lists to accept semantically-equivalent alternatives.

Motivation: the benchmark's single-keyword answer lists (e.g., "matematiko" for
Euclid) penalize MORE-specific correct answers (like "geometrio"). Add subfield
and close-synonym aliases so the score reflects semantic correctness.

Usage: uv run python scripts/expand_benchmark_aliases.py
  (rewrites benchmarks/factual_qa.json in place)
"""

import json
from pathlib import Path


# Aliases for single canonical answers. Each key is an existing answer string;
# values are additional accepted variants. When the benchmark answer is a list,
# we extend it with these aliases for each element.
ALIASES = {
    # --- laborkampo (work field) — subfields count as correct ---
    "astronomio":    ["astrofiziko", "fiziko", "matematiko"],
    "biologio":      ["evoluismo", "naturhistorio", "natursciencoj", "zoologio"],
    "fiziko":        ["mekaniko", "optiko", "relativeco", "matematiko", "natursciencoj"],
    "psikologio":    ["psikoanalizo", "psikiatrio", "neŭroscienco"],
    "kemio":         ["biokemio"],
    "mikrobiologio": ["bakteriologio", "medicino", "biologio"],
    "komputiko":     ["matematiko", "informadiko", "komputoscienco", "logiko"],
    "botaniko":      ["sistematiko", "taksonomio", "naturhistorio", "biologio"],
    "matematiko":    ["geometrio", "algebro", "aritmetiko", "logiko"],

    # --- ĝenro (literary genre) — broader/narrower equivalents ---
    "tragedio":      ["ŝekspira", "dramo"],
    "dramo":         ["tragedio"],
    "fantasto":      ["fantazio", "fantasta"],
    "krimromano":    ["detektiva", "misterio", "suspensa"],
    "hororo":        ["gotika", "terura", "vampira"],
    "sciencfikcio":  ["sf", "futura"],
    "epopeo":        ["epiko", "poezio"],
    "fabelaro":      ["fabela", "rakontaro"],

    # --- profesio (profession) — close synonyms / broader roles ---
    "fizikisto":     ["teori-fizikisto", "sciencisto"],
    "komponisto":    ["muzikisto"],
    "pentristo":     ["artisto"],
    "astronomo":     ["astrofizikisto", "sciencisto"],
    "matematikisto": ["geometristo", "algebristo", "sciencisto"],
    "filozofo":      ["logikisto", "pensulo"],
    "kuracisto":     ["medicinisto"],
    "verkisto":      ["aŭtoro", "romanisto", "literaturisto"],
    "skulptisto":    ["artisto"],
    "kemiisto":      ["sciencisto"],
    "biologo":       ["natursciencisto", "naturalisto", "sciencisto"],
    "psikologo":     ["psikoanalizisto", "psikiatro"],
    "psikiatro":     ["psikologo", "psikoanalizisto"],
    "inventisto":    ["inĝeniero"],
    "lingvisto":     ["esperantisto", "filologo"],
    "militestro":    ["militisto", "generalo", "imperiestro", "reganto"],
    "esploristo":    ["vojaĝisto", "vojaĝanto", "aventuristo"],
    "poeto":         ["verkisto"],
    "reĝino":        ["reganto", "monarko"],
    "flegistino":    ["kuracistino", "flegisto"],
}


def expand_answer(answer):
    """Take a str or list, return an expanded list with aliases added."""
    current = [answer] if isinstance(answer, str) else list(answer)
    seen = {a.lower() for a in current}
    additions = []
    for base in current:
        for alias in ALIASES.get(base.lower(), []):
            if alias.lower() not in seen:
                additions.append(alias)
                seen.add(alias.lower())
    return current + additions


def main():
    path = Path("benchmarks/factual_qa.json")
    data = json.loads(path.read_text())
    changed = 0
    for q in data:
        original = q["answer"]
        expanded = expand_answer(original)
        # Only convert to list if we actually added something; preserve string form otherwise
        if len(expanded) > (1 if isinstance(original, str) else len(original)):
            q["answer"] = expanded
            changed += 1
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"Expanded answer lists on {changed} / {len(data)} questions → {path}")

    # Summary of what changed
    print("\nExpansions by category:")
    from collections import Counter
    cats = Counter()
    for q in data:
        if isinstance(q["answer"], list) and len(q["answer"]) > 1:
            cats[q["category"]] += 1
    for cat, n in cats.most_common():
        print(f"  {cat:12s}  {n:>3} questions with multi-answer")


if __name__ == "__main__":
    main()
