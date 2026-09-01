"""TrainerCallback that runs downstream generation evals during training.

Motivation: HF Trainer's eval_loss on the held-out mix has been shown to
mislead about downstream capability for our DA SFT setup (see memory
`project-v12-best-ckpt-selection`). Running actual downstream evals
during training gives an honest capability trajectory.

Runs on each `on_evaluate` step:
  - GSM8K greedy on danish-gsm8k:test (substring match on numeric answer)
  - SciQ open-Q on danish-sciq:test (substring match on gold answer)
  - Cit-gen on danish-citizen-tests (substring match on gold option)

Adds metrics to the eval-loss dict:
  eval_downstream_gsm8k   — greedy GSM accuracy
  eval_downstream_sciq    — SciQ open-Q accuracy
  eval_downstream_citgen  — citizen-tests generative accuracy
  eval_downstream_ifeval  — IFEval-DA instruction-level strict accuracy,
                            the same definition the model cards report
  eval_downstream_icl     — exact match on UNSEEN-schema ICL rows
  eval_downstream_extraction — exact match on unseen-passage extraction rows

These show up in wandb and stdout alongside eval_loss. Overhead: ~1-2 min
per eval step with n=100 rows and bs=32 on a 5090; ~8-12 min with full set.

Top-K preservation: with top_k>0 and output_dir set, on each eval the
callback ranks the current ckpt by mean-downstream and, when the next
save fires for that step, moves checkpoint-N → best/step-N-agg-XX.XX so
it survives HF Trainer's save_total_limit rotation.
"""
from __future__ import annotations

import json
from collections import Counter
import os
import re
import shutil
import time
import unicodedata

import torch
from datasets import load_dataset
from transformers import TrainerCallback

USER, ASST, END = "<|user|>", "<|assistant|>", "<|end|>"

# ── metric helpers ──────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"####\s*(-?\d[\d,\.]*)")
_LAST_NUM_RE = re.compile(r"(-?\d[\d,]*\.?\d*)")

_DA_STOP = {"en", "et", "den", "det", "de", "at", "og", "i", "på", "af",
            "til", "for", "med", "er", "som", "der", "har"}


def _extract_num(text: str) -> str | None:
    m = _NUM_RE.search(text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = _LAST_NUM_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "").rstrip(".")
    return None


def _norm_num(s):
    if s is None:
        return None
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError, OverflowError):
        return s


def _norm_text(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(w for w in s.split() if w not in _DA_STOP)


def _matches_text(pred: str, gold: str) -> bool:
    np, ng = _norm_text(pred), _norm_text(gold)
    return bool(ng) and ng in np


# ── the callback ────────────────────────────────────────────────────────────

class DownstreamEvalCallback(TrainerCallback):
    """Runs downstream generation evals on every `on_evaluate` step and
    injects the accuracy metrics into the eval-metrics dict so HF Trainer
    logs them (and wandb picks them up)."""

    def __init__(self, tokenizer, evals=("gsm8k", "sciq", "citgen"),
                 n_per_eval=None, batch_size=32, max_new_gsm=300,
                 max_new_short=48, seed=42, top_k=0, output_dir=None):
        """n_per_eval: None or 0 = use full test set (default, no sampling bias).
        Non-zero = randomly subsample with a rotating seed per eval step so
        bias averages out across the trajectory rather than being pinned to
        one lucky/unlucky subset. Never use fixed shuffle(seed=42)+first-N
        — that repeatedly hits the same biased subset (see memory
        `project_v15_callback_subsample_bias`).

        top_k > 0 + output_dir: preserve the top-K checkpoints ranked by
        mean downstream accuracy. On save, mv checkpoint-N → best/step-N-
        agg-XX.XX so HF's save_total_limit rotation never touches them.
        Displaced ckpts are moved back to checkpoint-N so they can rotate
        normally."""
        self.tokenizer = tokenizer
        self.evals = tuple(evals)
        self.n = n_per_eval  # None or 0 = full set
        self.bs = batch_size
        self.max_new_gsm = max_new_gsm
        self.max_new_short = max_new_short
        self.seed = seed
        self._cache = {}  # eval_name → list of (prompt, gold) tuples
        self._end_id = tokenizer.convert_tokens_to_ids(END)
        self.top_k = top_k
        self.output_dir = output_dir
        self.top: list[tuple[float, int]] = []  # (agg_score, step), desc by score
        self._preserve_pending: tuple[int, float] | None = None
        self._demote_pending: list[int] = []
        # Diagnostic sub-metrics a scorer wants logged but which must NOT enter
        # the top-k aggregate: adding a breakdown should never silently
        # reweight checkpoint selection toward whichever eval reports the most
        # sub-numbers.
        self._extra_metrics: dict[str, float] = {}

    # ── dataset loaders (called lazily on first eval) ──────────────────────

    # Per-eval row caps, applied on top of --downstream-n. The four real
    # benchmarks are small enough to run whole (ifeval-da 541, citgen 720,
    # sciq 1000, gsm8k 1317 = 3,578 rows); our own icl eval_schema is 6,067,
    # which is 63% of the generation cost of a full sweep and would dominate
    # the eval budget for a split we control the size of. Capping it keeps the
    # published benchmarks unsampled, where sampling noise would actually
    # compromise a comparison.
    # extraction is capped harder than icl because its prompts are ~3x longer
    # (multi-shot, each demo carrying a full passage). At 1000 rows one eval
    # pass measured 685s wall, which across 12 eval points is ~2h17m of a ~7h30m
    # run -- the eval was costing more than the capability it measures is worth
    # mid-run. 200 rows keeps the trajectory readable; the published number
    # should come from a full-split run afterwards, not from this.
    PER_EVAL_CAP = {"icl": 1000, "extraction": 200}

    def _maybe_subsample(self, ds, step: int, name: str | None = None):
        # Effective n = tightest of the global --downstream-n and this eval's
        # cap; neither set means the full split.
        limits = [x for x in (self.n or None, self.PER_EVAL_CAP.get(name))
                  if x]
        if not limits:
            return ds  # full set
        n = min(limits)
        if n >= len(ds):
            return ds
        # Rotate seed by training step so different subsets each eval —
        # bias averages out across the trajectory.
        return ds.shuffle(seed=self.seed + step).select(range(n))

    def _load_gsm8k(self, step: int = 0):
        ds = load_dataset("jensjepsen/danish-gsm8k", "sft", split="test")
        ds = self._maybe_subsample(ds, step)
        items = []
        for r in ds:
            q = r["messages"][0]["content"]
            gold = _extract_num(r["messages"][1]["content"])
            items.append((q, gold))
        return items

    def _load_sciq(self, step: int = 0):
        ds = load_dataset("jensjepsen/danish-sciq", "default", split="test")
        ds = self._maybe_subsample(ds, step)
        return [(r["da_question"], r["da_correct_answer"]) for r in ds]

    def _load_citgen(self, step: int = 0):
        ds = load_dataset("alexandrainst/danish-citizen-tests", split="train")
        ds = self._maybe_subsample(ds, step)
        items = []
        for r in ds:
            gold_letter = r["answer"]
            gold_text = r.get(f"option_{gold_letter.lower()}")
            if not gold_text:
                continue
            items.append((r["question"], gold_text))
        return items

    def _load_piqa(self, step: int = 0):
        """Danish PIQA — 100 human-authored items, 2-option MC (A/B)."""
        ds = load_dataset("mrlbenchmarks/global-piqa-nonparallel", "dan_latn",
                          split="test")
        ds = self._maybe_subsample(ds, step)
        items = []
        for r in ds:
            opts = {"A": r["solution0"].strip(), "B": r["solution1"].strip()}
            gold = "A" if r["label"] == 0 else "B"
            items.append((r["prompt"].strip(), opts, gold))
        return items

    def _load_arc(self, step: int = 0):
        """DEPRECATED — old alexandrainst/m_arc:da:test (rough translation).
        Kept for backward compat with pre-v31 runs. New runs should use
        `arc_easy` and `arc_challenge` from jensjepsen/danish-arc."""
        ds = load_dataset("alexandrainst/m_arc", "da", split="test")
        ds = self._maybe_subsample(ds, step)
        items = []
        for r in ds:
            opts = {}
            for ll in "abcde":
                v = r.get(f"option_{ll}")
                if v: opts[ll.upper()] = v
            gold = r["answer"].upper()
            if gold not in opts: continue
            items.append((r["instruction"], opts, gold))
        return items

    def _load_arc_danish(self, cfg: str, step: int = 0):
        """Load jensjepsen/danish-arc (gemini-3.1-flash-lite translation) for
        the given config (arc_easy | arc_challenge). Test split, MC-letter."""
        ds = load_dataset("jensjepsen/danish-arc", cfg, split="test")
        ds = self._maybe_subsample(ds, step)
        items = []
        for r in ds:
            opts = {c["label"]: c["text"] for c in r["choices"]}
            gold = r["answerKey"]
            if gold not in opts: continue
            items.append((r["question"], opts, gold))
        return items

    def _load_arc_easy(self, step: int = 0):
        return self._load_arc_danish("arc_easy", step)

    def _load_arc_challenge(self, step: int = 0):
        return self._load_arc_danish("arc_challenge", step)

    def _load_gpqa(self, step: int = 0):
        """GPQA-Diamond-DA — translated 4-choice MC. Shuffled per-row for
        stable A/B/C/D positioning (correct answer's slot deterministic
        per pageid via seed)."""
        import random as _rnd
        ds = load_dataset("jensjepsen/danish-gpqa-diamond-v1", split="train")
        ds = self._maybe_subsample(ds, step)
        items = []
        for r in ds:
            answers = list(r["answers_da"])  # [correct, w1, w2, w3]
            rng = _rnd.Random(self.seed + r["orig_idx"])
            idxs = list(range(4)); rng.shuffle(idxs)
            opts = {}
            gold = None
            for slot, orig in enumerate(idxs):
                lab = "ABCD"[slot]
                opts[lab] = answers[orig]
                if orig == 0: gold = lab
            items.append((r["question_da"], opts, gold))
        return items

    def _load_textman_summary(self, step: int = 0):
        """Textman summary val — reference-based (ChrF++ vs gold summary)."""
        ds = load_dataset("jensjepsen/danish-textman-v1", split="validation")
        ds = ds.filter(lambda r: r["subtype"] == "textman_summary")
        ds = self._maybe_subsample(ds, step)
        items = []
        for r in ds:
            prompt = r["messages"][0]["content"]
            gold = r["messages"][1]["content"]
            items.append((prompt, gold))
        return items

    def _load_textman_rewrite(self, step: int = 0):
        """Textman rewrite val — reference-based (ChrF++ vs gold rewrite)."""
        ds = load_dataset("jensjepsen/danish-textman-v1", split="validation")
        ds = ds.filter(lambda r: r["subtype"] == "textman_rewrite")
        ds = self._maybe_subsample(ds, step)
        items = []
        for r in ds:
            prompt = r["messages"][0]["content"]
            gold = r["messages"][1]["content"]
            items.append((prompt, gold))
        return items

    def _load_ifeval(self, step: int = 0):
        """IFEval-DA, the benchmark the model cards already report.

        An earlier version scored held-out rows of our own IF v4 with the 46
        GRPO verifiers, as a mean fraction of constraints satisfied. That is a
        valid within-run signal but sits on a different scale from anything
        published: the v31 card records prompt-strict 21.2 / inst-strict 35.2
        on IFEval-DA, and 'mean fraction passed on our data' cannot be put in
        the same table. Scoring the benchmark itself makes the in-loop curve
        comparable to the card, to v22-avg and to the mid-run v31 checkpoint.
        """
        ds = load_dataset("danish-foundation-models/ifeval-da", split="train")
        ds = self._maybe_subsample(ds, step)
        return [(r["prompt"], r) for r in ds]

    def _load_icl(self, step: int = 0):
        """Unseen-SCHEMA rows of the ICL set. Scored by exact match on the
        parsed object, so it measures induction rather than loss."""
        ds = load_dataset("jensjepsen/danish-icl-schema-format-v3",
                          "default", split="eval_schema")
        ds = self._maybe_subsample(ds, step, "icl")
        items = []
        for r in ds:
            items.append((r["messages"][0]["content"],
                          (r["messages"][1]["content"], r["format"],
                           r["schema"], r["symbols"], r["n_fields"])))
        return items

    # Formats withheld from the extraction training split
    # (gen_extraction_da.py --held-formats). Rows using one of these are the
    # format-transfer question; the rest are the schema-transfer question.
    EXTRACTION_HELD_FORMATS = ("kv_eq", "bracket_pair", "brace_pair")

    # Extraction answers are multi-value and, for `fill`, a whole reconstructed
    # passage — far longer than the 300 tokens the GSM-shaped default allows.
    # Measured on eval_schema gold: extract median 72 / p99 564 / max 1644, fill
    # median 164 / p99 300 / max 390, so 6.2% of extract answers and 1.0% of
    # fill answers were being truncated mid-answer, failing to parse, and
    # scoring zero on length rather than on content. 640 covers p99 of both.
    EXTRACTION_MAX_NEW = 640

    def _load_extraction(self, step: int = 0):
        """Extraction rows whose schema was never trained on.

        Note what this split actually withholds. The schema is proposed per
        passage, so schema and passage are ~1:1 and the two hash partitions
        coincide: only 6 of eval_schema's 1,801 passages also occur in train.
        The metric is therefore unseen text AND unseen field set, not a clean
        schema-only ablation — which makes it the harder of the two readings,
        not a weaker one.
        """
        ds = load_dataset("jensjepsen/danish-extraction-v1",
                          "default", split="eval_schema")
        ds = self._maybe_subsample(ds, step, "extraction")
        return [(r["messages"][0]["content"],
                 (r["messages"][1]["content"], r["meta"])) for r in ds]

    def _load_citmc(self, step: int = 0):
        """Cit-MC: same source as cit-gen, formatted as labeled MC. The
        citizen-tests dataset has variable option count (2, 3, sometimes 4)
        with missing options as None — keep only present options and skip
        rows where the gold letter isn't present."""
        ds = load_dataset("alexandrainst/danish-citizen-tests", split="train")
        ds = self._maybe_subsample(ds, step)
        items = []
        for r in ds:
            gold_letter = r.get("answer")
            if not gold_letter:
                continue
            gold_letter = gold_letter.upper()
            opts = {}
            for ll in ["a", "b", "c", "d"]:
                val = r.get(f"option_{ll}")
                if val:
                    opts[ll.upper()] = val
            if len(opts) < 2 or gold_letter not in opts:
                continue
            items.append((r["question"], opts, gold_letter))
        return items

    def _get(self, name: str):
        if name not in self._cache:
            loader = getattr(self, f"_load_{name}")
            self._cache[name] = loader()
        return self._cache[name]

    # ── batched greedy generation ──────────────────────────────────────────

    def _generate(self, model, prompts: list[str], max_new: int) -> list[str]:
        tok = self.tokenizer
        # Save + set left padding for generation, restore after
        prev_side = tok.padding_side
        prev_pad = tok.pad_token
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        outs = []
        try:
            eos_ids = [tok.eos_token_id]
            if self._end_id is not None and self._end_id != tok.unk_token_id:
                eos_ids.append(self._end_id)
            for i in range(0, len(prompts), self.bs):
                batch = prompts[i:i + self.bs]
                enc = tok(batch, return_tensors="pt", padding=True,
                          add_special_tokens=False,
                          return_token_type_ids=False).to(model.device)
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        max_new_tokens=max_new, do_sample=False, num_beams=1,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id,
                        eos_token_id=eos_ids,
                        repetition_penalty=1.1,
                    )
                plen = enc["input_ids"].shape[1]
                for row in gen:
                    outs.append(tok.decode(row[plen:], skip_special_tokens=True).strip())
        finally:
            tok.padding_side = prev_side
            tok.pad_token = prev_pad
        return outs

    # ── per-eval scorers ───────────────────────────────────────────────────

    def _score_gsm8k(self, model) -> float:
        items = self._get("gsm8k")
        prompts = [f"{USER} {q} {ASST}" for q, _ in items]
        outs = self._generate(model, prompts, self.max_new_gsm)
        n_ok = sum(1 for out, (_, gold) in zip(outs, items)
                   if _norm_num(_extract_num(out)) == _norm_num(gold))
        return n_ok / len(items)

    def _score_sciq(self, model) -> float:
        items = self._get("sciq")
        prompts = [f"{USER}{q}{END}{ASST}" for q, _ in items]
        outs = self._generate(model, prompts, self.max_new_short)
        n_ok = sum(1 for out, (_, gold) in zip(outs, items)
                   if _matches_text(out, gold))
        return n_ok / len(items)

    def _score_citgen(self, model) -> float:
        items = self._get("citgen")
        prompts = [f"{USER}{q}{END}{ASST}" for q, _ in items]
        outs = self._generate(model, prompts, self.max_new_short)
        n_ok = sum(1 for out, (_, gold) in zip(outs, items)
                   if _matches_text(out, gold))
        return n_ok / len(items)

    def _score_ifeval(self, model) -> float:
        """IFEval-DA instruction-level STRICT accuracy.

        inst-strict rather than prompt-strict: prompt-strict needs every
        constraint in a row to pass, so at this model's level it is mostly
        zeros and moves too coarsely to read a training curve from. The card
        reports both (21.2 prompt / 35.2 inst for v31-avg-top3); this logs the
        one with usable resolution and the same definition.
        """
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "scripts"))
        from eval_ifeval_da import build_instructions, score_row
        items = self._get("ifeval")
        if not items:
            return 0.0
        prompts = [f"{USER}{q}{END}{ASST}" for q, _ in items]
        outs = self._generate(model, prompts, self.max_new_gsm)
        ok = tot = 0
        for out, (_, row) in zip(outs, items):
            insts = build_instructions(row)
            if not insts:
                continue
            # score_row returns (strict_flags, loose_flags) -- two values,
            # despite a docstring promising three
            strict_flags, _ = score_row(out, insts)
            ok += sum(bool(x) for x in strict_flags)
            tot += len(strict_flags)
        return ok / tot if tot else 0.0

    def _score_icl(self, model) -> float:
        """Exact match on the parsed answer, using the generator's own
        per-format parser so a flat format and JSON are scored alike."""
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "scripts"))
        from gen_icl_schema_format import canon, SYMBOLS
        items = self._get("icl")
        if not items:
            return 0.0
        prompts = [f"{USER}{q}{END}{ASST}" for q, _ in items]
        outs = self._generate(model, prompts, self.max_new_gsm)
        ok = 0
        for out, (_, (gold, fmt, schema, sym, nf)) in zip(outs, items):
            keys = (set(schema.split("|")) if sym == "none"
                    else set(SYMBOLS[sym][:nf]))
            g = canon(gold, fmt, keys)
            p = canon(out, fmt, keys)
            ok += (p is not None and p == g)
        return ok / len(items)

    @staticmethod
    def _score_fill(gold: str, pred: str, meta: dict) -> tuple[float, float]:
        """Score a reconstructed passage. Returns (placement, exact).

        `fill` no longer asks the model to recall a masked span — the spans are
        given in the prompt in shuffled order and the model puts them back. So
        the question is placement, and the metric has to be able to tell
        placement from two degenerate strategies:

          * echoing the gapped text with the markers stripped — high token
            overlap with gold, nothing actually filled. Caught by requiring no
            marker to survive.
          * echoing the supplied value list — every span present, but in the
            shuffled order they were given in, not the text's order. Caught by
            scoring the ORDER of the spans, via longest common subsequence
            against their true sequence.

        Whitespace-normalised, since gold comes from real prose and the model's
        line breaks are not the thing under test.
        """
        pairs = meta.get("fill_pairs") or []
        norm = lambda s: " ".join(s.split())                    # noqa: E731
        g, p = norm(gold), norm(pred)
        exact = float(g == p)
        if not pairs:
            return exact, exact
        # a surviving marker means the gap was never filled
        if any(norm(shown) in p for shown, _ in pairs):
            return 0.0, exact
        spans = [norm(sp) for _, sp in pairs]
        pos = [(p.find(sp), i) for i, sp in enumerate(spans) if sp and sp in p]
        pos.sort()
        seq = [i for _, i in pos]                # span indices, in PRED order
        want = list(range(len(spans)))           # span indices, in GOLD order
        # LCS: how much of the true ordering survived
        dp = [[0] * (len(want) + 1) for _ in range(len(seq) + 1)]
        for a in range(len(seq)):
            for b in range(len(want)):
                dp[a + 1][b + 1] = (dp[a][b] + 1 if seq[a] == want[b]
                                    else max(dp[a][b + 1], dp[a + 1][b]))
        order = dp[len(seq)][len(want)] / len(spans)
        # Multiplied by context fidelity, because order alone is coarse at
        # n=2-4: a shuffled echo of the supplied value list still contains a
        # long increasing subsequence by chance (0.67 on a 3-span case). The
        # echo has almost none of the surrounding prose, so token overlap
        # separates it. Multiplicative, not additive — reproducing the context
        # while placing nothing should not earn a floor.
        gt, pt = Counter(g.split()), Counter(p.split())
        inter = sum((gt & pt).values())
        tf1 = (0.0 if not inter else
               2 * inter / (sum(gt.values()) + sum(pt.values())))
        return order * tf1, exact

    @staticmethod
    def _parse_fill(text: str) -> list | None:
        """`fill` answers are always `<marker> = <span>` lines, regardless of
        the row's `format` — build_fill() overwrites the rendered answer. So
        meta['format'] must NOT be used to parse them; it describes the
        extract-shaped answer that was discarded.

        Returns an ORDERED LIST, not a dict. Only ~60% of fill rows use
        numbered markers; the rest repeat one marker (`…`, `___`) for every
        gap, so a dict keyed by marker silently collapses them — measured 39.8%
        of eval_schema fill rows collapsing and 33.0% of all gap lines dropped,
        leaving the score to rest on whichever gap happened to be last. The
        gold round-trip check could not catch this because gold and prediction
        collapse identically; it reads as a pass either way.
        """
        out = []
        for line in text.strip().splitlines():
            if " = " not in line:
                continue
            k, v = line.split(" = ", 1)
            out.append((k.strip(), v.strip()))
        return out or None

    @staticmethod
    def _pair_f1(pred: set, gold: set) -> float:
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0
        tp = len(pred & gold)
        if not tp:
            return 0.0
        p, r = tp / len(pred), tp / len(gold)
        return 2 * p * r / (p + r)

    def _score_extraction(self, model) -> float:
        """Field-level pair-F1 on the parsed answer, per-task parser.

        The headline is F1, not exact match. Exact match on a multi-field
        object over unseen text has no resolution at this sample size: three
        consecutive evals read 0.8% / 0.5% / 0.0%, which is 8 / 1 / 0 hits and
        indistinguishable from noise, while hand-written probes at the same
        checkpoint showed the model extracting verbatim spans and abstaining
        correctly on absent fields. One wrong span in one field scored those
        rows zero. F1 over (key, value) pairs is the same measure
        rl_rewards.reward_structured uses for GRPO, so the training signal and
        the eval agree.

        Exact match, parse rate and the per-cell splits are still reported as
        sub-metrics — they diagnose *why* a number moved (format rendering vs
        field routing vs extraction), which one aggregate cannot.
        """
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "scripts"))
        from gen_icl_schema_format import NULL, SYMBOLS, canon
        items = self._get("extraction")
        if not items:
            return 0.0
        prompts = [f"{USER}{q}{END}{ASST}" for q, _ in items]
        outs = self._generate(model, prompts, self.EXTRACTION_MAX_NEW)
        f1s, exact, parsed = [], [], []
        by: dict[str, list[float]] = {}
        for out, (_, (gold, meta)) in zip(outs, items):
            task, fmt = meta["task"], meta["format"]
            if task == "fill":
                f1, ex = self._score_fill(gold, out, meta)
                f1s.append(f1)
                exact.append(ex)
                parsed.append(1.0)   # free-text reconstruction always "parses"
                held = fmt in self.EXTRACTION_HELD_FORMATS
                by.setdefault(f"fill/{'unseen-fmt' if held else 'seen-fmt'}",
                              []).append(f1)
                continue
            else:
                keys = (list(meta["schema"].split("|"))
                        if meta["symbols"] == "none"
                        else list(SYMBOLS[meta["symbols"]][:meta["n_fields"]]))
                g, p = canon(gold, fmt, keys), canon(out, fmt, keys)
                # NULL pairs are dropped: crediting "absent field left absent"
                # as a matched pair inflates F1 on rows that are mostly
                # abstentions, which is the opposite of what we want to measure
                gp = {(k, v) for k, vs in (g or {}).items()
                      for v in vs if v != NULL}
                pp = {(k, v) for k, vs in (p or {}).items()
                      for v in vs if v != NULL}
            f1 = self._pair_f1(pp, gp)
            f1s.append(f1)
            exact.append(float(p is not None and p == g))
            parsed.append(float(p is not None))
            held = fmt in self.EXTRACTION_HELD_FORMATS
            by.setdefault(f"{task}/{'unseen-fmt' if held else 'seen-fmt'}",
                          []).append(f1)
        mean = lambda v: sum(v) / len(v) if v else 0.0  # noqa: E731
        parts = "  ".join(f"{k} F1 {100*mean(v):.1f}% (n={len(v)})"
                          for k, v in sorted(by.items()))
        print(f"  [downstream] extraction F1 by cell: {parts}", flush=True)
        print(f"  [downstream]   exact={100*mean(exact):.1f}%  "
              f"parsed={100*mean(parsed):.1f}%", flush=True)
        self._extra_metrics.update({
            "eval_downstream_extraction_exact": mean(exact),
            "eval_downstream_extraction_parse_rate": mean(parsed),
            **{f"eval_downstream_extraction_f1_{k.replace('/', '_')}": mean(v)
               for k, v in by.items()},
        })
        return mean(f1s)

    def _score_citmc(self, model) -> float:
        """MC on citizen-tests. Uses the wiki-mc-letters prompt shape.
        opts is a dict of {label:text} with only present options (usually
        A/B or A/B/C, occasionally A/B/C/D). Scored by first present-letter
        in emission, case-insensitive."""
        items = self._get("citmc")
        if not items:
            return 0.0
        prompts = []
        for q, opts, _ in items:
            opts_str = "\n".join(f"{lab}) {opts[lab]}" for lab in sorted(opts))
            body = (f"{q}\n\n{opts_str}\n\n"
                    f"Svar med bogstavet på det korrekte svar.")
            prompts.append(f"{USER}{body}{END}{ASST}")
        outs = self._generate(model, prompts, 8)
        n_ok = 0
        for out, (_, opts, gold) in zip(outs, items):
            present = "".join(sorted(opts))  # e.g. "ABC"
            # Anchor to word boundaries so "match" doesn't match "A" etc.
            # Case-insensitive matches "b) foo" or "Svar: B".
            m = re.search(rf"\b[{present}]\b", out, re.IGNORECASE)
            if m and m.group(0).upper() == gold:
                n_ok += 1
        return n_ok / len(items)

    def _score_mc_letter(self, model, items, max_new: int = 8) -> float:
        """Shared MC-letter scorer used by piqa/arc/gpqa (and citmc-style
        emit-single-letter tasks). items = list of (q, opts_dict, gold_letter).
        Batched via self._generate."""
        if not items:
            return 0.0
        prompts = []
        for q, opts, _ in items:
            opts_str = "\n".join(f"{lab}) {opts[lab]}" for lab in sorted(opts))
            body = (f"{q}\n\n{opts_str}\n\n"
                    f"Svar med bogstavet på det korrekte svar.")
            prompts.append(f"{USER}{body}{END}{ASST}")
        outs = self._generate(model, prompts, max_new)
        n_ok = 0
        for out, (_, opts, gold) in zip(outs, items):
            present = "".join(sorted(opts))
            m = re.search(rf"\b[{present}]\b", out, re.IGNORECASE)
            if m and m.group(0).upper() == gold:
                n_ok += 1
        return n_ok / len(items)

    def _score_piqa(self, model) -> float:
        return self._score_mc_letter(model, self._get("piqa"))

    def _score_arc(self, model) -> float:
        return self._score_mc_letter(model, self._get("arc"))

    def _score_arc_easy(self, model) -> float:
        return self._score_mc_letter(model, self._get("arc_easy"))

    def _score_arc_challenge(self, model) -> float:
        return self._score_mc_letter(model, self._get("arc_challenge"))

    def _score_gpqa(self, model) -> float:
        return self._score_mc_letter(model, self._get("gpqa"))

    def _score_chrf(self, model, name: str, max_new: int) -> float:
        """ChrF++ vs single reference. Returns 0..1 (matches accuracy metrics
        for consistent aggregation — the callback pipeline *100s to percent)."""
        from sacrebleu.metrics import CHRF
        items = self._get(name)
        prompts = [q for q, _ in items]
        golds = [g for _, g in items]
        outs = self._generate(model, prompts, max_new)
        chrf = CHRF(word_order=2)  # ChrF++
        return chrf.corpus_score(outs, [golds]).score / 100.0

    def _score_textman_summary(self, model) -> float:
        return self._score_chrf(model, "textman_summary", max_new=200)

    def _score_textman_rewrite(self, model) -> float:
        return self._score_chrf(model, "textman_rewrite", max_new=512)

    # ── HF Trainer hook ────────────────────────────────────────────────────

    def on_evaluate(self, args, state, control, model=None, metrics=None,
                    **kwargs):
        if model is None:
            return control
        model.eval()
        t0 = time.time()
        downstream_metrics = {}
        self._extra_metrics = {}
        for name in self.evals:
            score = getattr(self, f"_score_{name}")(model)
            key = f"eval_downstream_{name}"
            downstream_metrics[key] = score
            if metrics is not None:
                metrics[key] = score  # for HF logging on same-step
            print(f"  [downstream] {name}: {100*score:.1f}%", flush=True)
        # Meta-metric: if both arc splits ran, log their mean.
        if ("arc_easy" in self.evals) and ("arc_challenge" in self.evals):
            arc_mean = (downstream_metrics["eval_downstream_arc_easy"]
                        + downstream_metrics["eval_downstream_arc_challenge"]) / 2
            downstream_metrics["eval_downstream_arc_mean"] = arc_mean
            if metrics is not None:
                metrics["eval_downstream_arc_mean"] = arc_mean
            print(f"  [downstream] arc_mean: {100*arc_mean:.1f}%", flush=True)
        elapsed = time.time() - t0
        n_label = "full" if not self.n else str(self.n)
        print(f"  [downstream] {len(self.evals)} evals in {elapsed:.0f}s "
              f"(n={n_label} each, bs={self.bs})", flush=True)
        # Explicit wandb.log() — Trainer's built-in log already fired for
        # the eval metrics dict BEFORE on_evaluate ran, so mutating `metrics`
        # doesn't reach wandb. Push our downstream metrics directly at the
        # current global_step.
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({**downstream_metrics, **self._extra_metrics},
                          step=state.global_step)
        except ImportError:
            pass
        if metrics is not None:
            metrics.update(self._extra_metrics)

        # Top-K bookkeeping: decide if this ckpt deserves preservation.
        # NOTE ordering: HF Trainer flow at an eval+save step is
        #   on_evaluate → _save_checkpoint (save+rotate) → on_save
        # We handle demotions HERE (before rotation) so a demoted ckpt
        # moved back to checkpoint-N is eligible for the imminent rotation.
        # Preservation of the current step is deferred to on_save because
        # checkpoint-<current> doesn't exist yet at eval time.
        if self.top_k > 0 and self.output_dir and downstream_metrics:
            agg = sum(downstream_metrics.values()) / len(downstream_metrics)
            step = state.global_step
            all_scores = self.top + [(agg, step)]
            all_scores.sort(key=lambda x: (-x[0], -x[1]))
            new_top = all_scores[:self.top_k]
            new_steps = {s for _, s in new_top}
            old_steps = {s for _, s in self.top}
            demoted = old_steps - new_steps
            self.top = new_top
            self._write_best_json(agg_metric_name="mean_downstream")
            if step in new_steps:
                self._preserve_pending = (step, agg)
                print(f"  [downstream] step-{step} enters top-{self.top_k} "
                      f"(agg={agg:.3f})", flush=True)
            # Demote now, before rotation runs during _save_checkpoint.
            for dstep in sorted(demoted):
                self._demote(dstep)
        return control

    def on_save(self, args, state, control, **kwargs):
        """Post-rotation. Move newly-top-K current ckpt to best/ so future
        rotations don't touch it."""
        if self.top_k <= 0 or not self.output_dir:
            return control
        if self._preserve_pending is not None:
            step, agg = self._preserve_pending
            self._preserve_pending = None
            src = os.path.join(self.output_dir, f"checkpoint-{step}")
            dst = os.path.join(self.output_dir, "best",
                               f"step-{step}-agg-{agg:.3f}")
            if os.path.isdir(src) and not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    shutil.move(src, dst)
                    print(f"  [downstream] preserved: {src} → {dst}",
                          flush=True)
                except Exception as e:
                    print(f"  [downstream] preserve failed for step-{step}: "
                          f"{e}", flush=True)
        return control

    def _demote(self, step: int):
        """Move best/step-N-* back to checkpoint-N so rotation can prune it."""
        best_dir = os.path.join(self.output_dir, "best")
        if not os.path.isdir(best_dir):
            return
        matches = [d for d in os.listdir(best_dir)
                   if d.startswith(f"step-{step}-")]
        for d in matches:
            src = os.path.join(best_dir, d)
            dst = os.path.join(self.output_dir, f"checkpoint-{step}")
            if os.path.exists(dst):
                shutil.rmtree(src)
                continue
            try:
                shutil.move(src, dst)
                print(f"  [downstream] demoted: {src} → {dst}", flush=True)
            except Exception as e:
                print(f"  [downstream] demote failed for step-{step}: {e}",
                      flush=True)

    def _write_best_json(self, agg_metric_name: str = "mean_downstream"):
        path = os.path.join(self.output_dir, "best_ckpts.json")
        payload = {
            "metric": agg_metric_name,
            "evals": list(self.evals),
            "top": [{"step": s, "agg": a} for a, s in self.top],
        }
        try:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"  [downstream] failed to write {path}: {e}", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        if self.top_k <= 0 or not self.top:
            return control
        print("\n  [downstream] top-K by mean-downstream:", flush=True)
        for i, (agg, step) in enumerate(self.top, 1):
            print(f"    {i}. step-{step}  agg={agg:.3f}", flush=True)
        return control
