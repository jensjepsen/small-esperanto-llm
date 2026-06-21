"""Tool-call ("funcall") training + inference for the EO student model.

A small model can't reliably do multi-digit arithmetic, but it can learn
to emit a calculation as a tool call and let an executor do the math.
This module provides:

- Token constants for chat + tool turns (shared with scripts/train_sft.py)
- A converter that rewrites verbal-CoT math datasets (GSM8K, orca-math)
  into multi-turn `<|tool_call|>EXPR<|/tool_call|> <|tool_result|>VAL<|/tool_result|>`
  sequences for SFT training
- A runtime that wraps a trained model in a generate→eval→inject loop,
  with sympy-backed safe_eval, thousands-comma normalization, and
  repeat-call detection
"""
from .tokens import (
    USER, ASST, END,
    TOOL_CALL_OPEN, TOOL_CALL_CLOSE,
    TOOL_RESULT_OPEN, TOOL_RESULT_CLOSE,
    SPECIAL_TOKENS,
)
from .runtime import (
    ToolInferenceRunner,
    safe_eval,
    normalize_numbers,
    morpheme_chat_text,
)
from .converter import (
    convert_gsm_answer,
    convert_orca_answer,
    verify_arith,
)

__all__ = [
    "USER", "ASST", "END",
    "TOOL_CALL_OPEN", "TOOL_CALL_CLOSE",
    "TOOL_RESULT_OPEN", "TOOL_RESULT_CLOSE",
    "SPECIAL_TOKENS",
    "ToolInferenceRunner",
    "safe_eval",
    "normalize_numbers",
    "morpheme_chat_text",
    "convert_gsm_answer",
    "convert_orca_answer",
    "verify_arith",
]
