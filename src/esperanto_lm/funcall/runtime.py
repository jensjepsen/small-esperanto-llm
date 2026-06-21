"""Tool-call inference runtime.

Inference loop:
  1. generate until `<|/tool_call|>` or `<|end|>`
  2. if `<|tool_call|>EXPR<|/tool_call|>` emitted, extract EXPR, evaluate
     it with sympy, inject `<|tool_result|>VAL<|/tool_result|>`, continue
  3. else if `<|end|>`, finish

Safety: sympy.sympify is used instead of `eval()` — no arbitrary code
execution, handles `^` as exponent, deep parens, rationals.
"""
from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from esperanto_lm.data import _morpheme_preprocess

from .tokens import (
    USER, ASST, END,
    TOOL_CALL_OPEN as TC_O,
    TOOL_CALL_CLOSE as TC_C,
    TOOL_RESULT_OPEN as TR_O,
    TOOL_RESULT_CLOSE as TR_C,
    SPECIAL_TOKENS,
)

__all__ = [
    "ToolInferenceRunner",
    "safe_eval",
    "normalize_numbers",
    "morpheme_chat_text",
]


def normalize_numbers(text: str) -> str:
    """Strip thousands-comma separators so the morpheme tokenizer doesn't
    split a magnitude across separator tokens that the model often drops.

    Matches ``\\d,\\d{3}`` (digit + comma + exactly 3 digits, not
    followed by another digit) and removes the comma. Loops to handle
    nested groupings like ``1,234,567``. European decimals like
    ``0,5`` or ``1,5`` are NOT touched (only 1 digit after comma →
    no match).

        "$80,000"     -> "$80000"
        "1,234,567"   -> "1234567"
        "0,5"         -> "0,5"        (decimal, unchanged)
        "$1,234.56"   -> "$1234.56"
    """
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", text)
    return text


def morpheme_chat_text(text: str) -> str:
    """Morpheme-preprocess a chat string, protecting special tokens.

    Splits on any special token, leaves them intact, runs
    ``_morpheme_preprocess`` on the content parts. Same algorithm as
    scripts/train_sft.py uses during tokenization so the same string is
    seen at train and inference time.
    """
    pat = "(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")"
    return " ".join(
        p if p in SPECIAL_TOKENS else _morpheme_preprocess(p)
        for p in re.split(pat, text)
    )


def safe_eval(expr: str) -> str:
    """Evaluate an arithmetic expression with sympy. Returns string result
    or 'ERROR'.

    Handles morpheme-tokenizer whitespace artifacts before parsing:

      "60 * . 6"   -> 36
      "1 . 5 + 2"  -> 3.5
      "- 3"        -> -3
      "2 ^ 5"      -> 32
      "1/2 + 1/4"  -> 0.75
    """
    cleaned = re.sub(r"[^\d+\-*/.()^ ]", "", expr.strip())
    if not cleaned:
        return "ERROR"
    cleaned = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", cleaned)
    cleaned = re.sub(r"(?<!\d)\.\s+(\d)", r".\1", cleaned)
    cleaned = re.sub(r"(\d)\s+\.(?!\d)", r"\1.", cleaned)
    cleaned = re.sub(r"(^|[+\-*/(])\s*-\s+(\d)", r"\1-\2", cleaned)
    cleaned = cleaned.replace(" ", "")
    try:
        from sympy import sympify
        val = sympify(cleaned, evaluate=True)
        f = float(val)
        if f == int(f):
            return str(int(f))
        s = str(f).rstrip("0").rstrip(".")
        return s if s else "0"
    except Exception:
        return "ERROR"


class ToolInferenceRunner:
    """Greedy generate-eval-inject loop over a model trained on the funcall
    chat format.
    """

    def __init__(
        self,
        ckpt: str,
        tokenizer_dir: str = "tokenizer_morpheme",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        self.tok.add_special_tokens(
            {"additional_special_tokens": list(SPECIAL_TOKENS)}
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=dtype
        ).to(device).eval()
        self.model.resize_token_embeddings(len(self.tok))
        self.device = device
        self.end_id = self.tok.convert_tokens_to_ids(END)
        self.tc_o_id = self.tok.convert_tokens_to_ids(TC_O)
        self.tc_c_id = self.tok.convert_tokens_to_ids(TC_C)
        self.tr_o_id = self.tok.convert_tokens_to_ids(TR_O)
        self.tr_c_id = self.tok.convert_tokens_to_ids(TR_C)

    def _decode(self, ids: list[int]) -> str:
        toks = self.tok.convert_ids_to_tokens(ids)
        out = []
        for t in toks:
            if t.startswith("<w"):
                out.append(" ")
            else:
                out.append(t)
        return "".join(out).strip()

    def chat(
        self,
        prompt: str,
        max_hops: int = 6,
        max_new_per_hop: int = 200,
        verbose: bool = False,
    ) -> dict:
        """Run the tool loop on a user prompt. Returns ``{text, trace}``.

        - ``text`` is the model's full rendered output including injected
          tool results.
        - ``trace`` is a list of ``('gen', text)`` and
          ``('tool', expr, result)`` tuples per hop.
        """
        prompt = normalize_numbers(prompt)
        body = morpheme_chat_text(f"{USER} {prompt} {ASST}")
        ids = self.tok(body, add_special_tokens=False).input_ids
        all_generated_ids: list[int] = []
        trace: list[tuple] = []
        seen_calls: set[tuple[str, str]] = set()  # detect repeats

        for hop in range(max_hops):
            in_ids = torch.tensor([ids], device=self.device)
            with torch.no_grad():
                out = self.model.generate(
                    in_ids,
                    max_new_tokens=max_new_per_hop,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id or self.end_id,
                    eos_token_id=[self.end_id, self.tc_c_id],
                )
            new_ids = out[0][len(ids):].tolist()
            all_generated_ids.extend(new_ids)
            ids = ids + new_ids
            decoded_new = self._decode(new_ids)
            trace.append(("gen", decoded_new))
            if verbose:
                print(f"[hop {hop}] gen: {decoded_new}")

            if new_ids and new_ids[-1] == self.tc_c_id:
                if self.tc_o_id not in new_ids:
                    if verbose:
                        print(f"[hop {hop}] tool_call_close without open — bailing")
                    break
                open_pos = len(new_ids) - 1 - new_ids[::-1].index(self.tc_o_id)
                expr_ids = new_ids[open_pos + 1 : -1]
                expr = self._decode(expr_ids)
                result = safe_eval(expr)
                if verbose:
                    print(f"[hop {hop}] EXPR={expr!r} RESULT={result!r}")
                trace.append(("tool", expr, result))
                call_sig = (expr.strip(), result.strip())
                if call_sig in seen_calls:
                    if verbose:
                        print(f"[hop {hop}] repeat call — stopping")
                    break
                seen_calls.add(call_sig)
                inject = morpheme_chat_text(f"{TR_O}{result}{TR_C}")
                inject_ids = self.tok(inject, add_special_tokens=False).input_ids
                ids = ids + inject_ids
                all_generated_ids.extend(inject_ids)
                continue

            # Hit <|end|> or maxed out
            break

        return {"text": self._decode(all_generated_ids), "trace": trace}
