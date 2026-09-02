"""Publish danish-textman-v2: v1 minus the extraction subtype.

`textman_extraction` is dropped, not regenerated. It used ONE fixed schema --
people/places/dates/numbers -- across all 20,018 rows, so it could not teach
"read the keys you were given", and 26% of its `numbers` were absent from the
passage. `danish-extraction-v1` supersedes it: a schema proposed per passage,
values gated as verbatim spans, and genuine abstention targets.

This IS a version bump rather than a sibling: same source, same method, same
five remaining subtypes, with a defective part removed. v2 supersedes v1.

Filtering the published rows rather than regenerating keeps the other five
subtypes byte-identical to v1, so any v33/v34 comparison stays valid -- a
regenerated corpus would differ everywhere and confound the change.
"""
from __future__ import annotations

import argparse
from collections import Counter

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

SRC = "jensjepsen/danish-textman-v1"
REPO = "jensjepsen/danish-textman-v2"
DROP = "textman_extraction"


def card(counts, subtypes):
    rows = "\n".join(f"| `{s}` | {n:,} |" for s, n in subtypes.most_common())
    splits = "\n".join(f"| `{k}` | {v:,} |" for k, v in counts.items())
    return f"""---
language:
- da
license: apache-2.0
task_categories:
- text-generation
tags:
- danish
- text-manipulation
---

# danish-textman-v2

Danish text-manipulation tasks over Wikipedia passages. Supersedes
[`{SRC}`](https://huggingface.co/datasets/{SRC}), which is identical except
that it also contained a `{DROP}` subtype.

| split | rows |
|---|---|
{splits}

| subtype | rows |
|---|---|
{rows}

## What changed

`{DROP}` is removed. It asked for entities as JSON under ONE fixed schema --
`people` / `places` / `dates` / `numbers` -- across all of its rows, so it
taught a single hardcoded key set rather than the ability to extract whatever
keys a prompt asks for. 26% of its `numbers` values did not appear in the
source passage.

Structured extraction now lives in
[`jensjepsen/danish-extraction-v1`](https://huggingface.co/datasets/jensjepsen/danish-extraction-v1),
where the schema is proposed per passage, every value is checked to be a
verbatim span, and absent fields are real abstention targets.

The five remaining subtypes are byte-identical to v1, so results comparing a
model trained on v1 with one trained on v2 isolate the removal.

## Known limitation

87% of passages carry Wikipedia navigation cruft (`Referencer`, `Eksterne
henvisninger`, `Se også`) inherited from the source dump, and 1.2% of
instruction tails contain a non-Danish word. Neither is addressed here.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ds = load_dataset(SRC)
    out, counts, subtypes = {}, {}, Counter()
    for split in ds:
        rows = [r for r in ds[split] if r.get("subtype") != DROP]
        dropped = len(ds[split]) - len(rows)
        out[split] = rows
        counts[split] = len(rows)
        subtypes.update(r["subtype"] for r in rows)
        print(f"{split}: {len(ds[split]):,} -> {len(rows):,} "
              f"(dropped {dropped:,})", flush=True)
    assert DROP not in subtypes, "extraction survived the filter"
    print(f"subtypes: {dict(subtypes.most_common())}")

    if args.dry_run:
        print("\ndry run")
        return

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    for split, rows in out.items():
        Dataset.from_list(rows).push_to_hub(
            args.repo, split=split,
            commit_message=f"{split} ({len(rows)} rows, {DROP} removed)")
        print(f"  pushed {split} ({len(rows):,})", flush=True)
    api.upload_file(path_or_fileobj=card(counts, subtypes).encode(),
                    path_in_repo="README.md", repo_id=args.repo,
                    repo_type="dataset", commit_message="dataset card")
    print(f"-> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
