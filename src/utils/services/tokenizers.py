"""Token counting helper used for cost instrumentation.

Uses tiktoken's `cl100k_base` encoding as a proxy for NVIDIA's actual
embedding tokenizer. Typically within ~10% of the true count — accurate
enough for week-over-week cost trending, not for hard token-budget
enforcement. See docs/OBSERVABILITY.md §3.2 for the design rationale.
"""

import tiktoken

_ENCODER: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    """Return the token count for `text`.

    Returns at least 1 even for empty input so the result can be safely
    used in Langfuse `usage_details`, which expects positive integers.
    """
    return max(1, len(_get_encoder().encode(text)))
