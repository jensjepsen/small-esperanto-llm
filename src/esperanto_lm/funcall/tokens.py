"""Special tokens for chat + tool-call format.

Mirrors the constants in scripts/train_sft.py — both are kept in sync.
Tool tokens are added by train_sft.py via `add_special_tokens` so they
tokenize to single IDs; the inference runtime adds them again at probe
time (idempotent — `add_special_tokens` is a no-op for tokens already in
the vocab).
"""

USER = "<|user|>"
ASST = "<|assistant|>"
END = "<|end|>"

TOOL_CALL_OPEN = "<|tool_call|>"
TOOL_CALL_CLOSE = "<|/tool_call|>"
TOOL_RESULT_OPEN = "<|tool_result|>"
TOOL_RESULT_CLOSE = "<|/tool_result|>"

SPECIAL_TOKENS = [
    USER, ASST, END,
    TOOL_CALL_OPEN, TOOL_CALL_CLOSE,
    TOOL_RESULT_OPEN, TOOL_RESULT_CLOSE,
]
