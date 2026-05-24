"""Shared retry helper for transient upstream failures.

Wraps a zero-arg callable with exponential backoff. Used on every
external call on the `/chat` read path (NVIDIA chat completion, NVIDIA
embedding, NVIDIA rerank) so a single 502/503 from the upstream load
balancer doesn't surface as a user-facing 5xx.

Mirrors the pattern from `_add_texts_with_retry` in milvus_store.py
which protects the write/ingest path. The shape is intentionally the
same — one retry policy, one log line format, easy to reason about.

Retries on broad `Exception` because the NVIDIA SDK raises a bare
exception with the status code in the message (e.g. "[502] Bad
Gateway"). Parsing that message to filter "transient vs not" is
brittle. Treating every failure as potentially-transient is cheap:
worst case a real bug surfaces after `max_attempts` retries instead
of immediately.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from src.utils.services.logger_config import logger

T = TypeVar("T")

# Defaults are tuned for the read path: short total wait budget so the
# user isn't hanging for a minute. The ingest path (write) uses 5/30
# because there's no human waiting.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_SLEEP_S = 15


def call_with_retry(
    fn: Callable[[], T],
    *,
    op_name: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_sleep_s: int = DEFAULT_MAX_SLEEP_S,
) -> T:
    """Call `fn` with exponential backoff (2s, 4s, 8s, ... capped at `max_sleep_s`).

    Returns the function's value on first success. Re-raises the last
    exception if all attempts fail. Each non-final failure logs at
    WARNING level so a tail of the app logs shows real-time retry
    activity without needing to dig into Langfuse.

    `op_name` identifies the call site in log lines — keep it short and
    snake_case (e.g. "nim_chat_completion", "embed_query", "rerank").
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt == max_attempts:
                break
            sleep_s = min(2 ** attempt, max_sleep_s)
            logger.warning(
                "%s attempt %d/%d failed (%s); retrying in %ds",
                op_name, attempt, max_attempts, e, sleep_s,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc
