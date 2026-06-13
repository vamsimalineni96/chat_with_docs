"""In-memory store for pending HITL approval tokens.

When the action agent detects a destructive tool call (e.g. create_refund),
it returns a pending_approval dict instead of executing. The chat service
stores the paused graph state here, keyed by an opaque token. When the user
clicks Approve/Reject in the UI, the /approve endpoint retrieves the state,
injects the decision, and resumes the graph.

TTL: tokens expire after APPROVAL_TTL_SECONDS to prevent stale approvals
from accumulating. Default: 10 minutes.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

APPROVAL_TTL_SECONDS = 600  # 10 minutes

_store: dict[str, dict[str, Any]] = {}


def create_token(paused_state: dict[str, Any]) -> str:
    """Store paused graph state and return an opaque approval token."""
    token = secrets.token_urlsafe(32)
    _store[token] = {
        "state": paused_state,
        "created_at": time.monotonic(),
    }
    return token


def consume_token(token: str) -> dict[str, Any] | None:
    """Retrieve and remove the paused state for a token.

    Returns None if the token is unknown or expired.
    """
    entry = _store.pop(token, None)
    if entry is None:
        return None
    age = time.monotonic() - entry["created_at"]
    if age > APPROVAL_TTL_SECONDS:
        return None
    return entry["state"]


def pending_count() -> int:
    """Number of tokens currently waiting — useful for health checks."""
    return len(_store)
