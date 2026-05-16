"""
Langfuse observability — central wiring.

Targets Langfuse Python SDK v3.x (OpenTelemetry-based). The 3.x API:
  - `@observe` is a top-level import: `from langfuse import observe`
  - `CallbackHandler` lives in `langfuse.langchain`
  - Trace/span updaters live on the singleton client: `Langfuse.update_current_trace`
    and `Langfuse.update_current_span`

Exposes:
  - `observe`: decorator. No-ops when Langfuse is disabled or import fails.
  - `langfuse_callback()`: returns a LangChain CallbackHandler or None.
  - `update_current_trace(...)`: attach metadata/tags/user/session to the active
    trace; no-op when disabled.
  - `update_current_observation(...)`: attach input/output/metadata to the active
    span; no-op when disabled.
  - `flush()`: force-flush. Safe to call when disabled.
"""

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any

from src.utils import config
from src.utils.services.logger_config import logger

_enabled: bool = False
_client = None
_observe_decorator: Callable | None = None


def _init() -> None:
    global _enabled, _client, _observe_decorator

    if not config.LANGFUSE_ENABLED:
        logger.info("Langfuse disabled via LANGFUSE_ENABLED=false")
        return

    if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
        logger.warning(
            "Langfuse enabled but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
            "are missing — tracing will be skipped."
        )
        return

    try:
        from langfuse import Langfuse
        from langfuse import observe as _observe

        _client = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
            flush_at=config.LANGFUSE_FLUSH_AT,
            flush_interval=config.LANGFUSE_FLUSH_INTERVAL,
        )
        _observe_decorator = _observe
        _enabled = True
        logger.info("Langfuse client initialized (host=%s)", config.LANGFUSE_HOST)
    except Exception as e:
        logger.exception("Failed to initialize Langfuse — tracing disabled: %s", e)
        _enabled = False
        _client = None
        _observe_decorator = None


_init()


def is_enabled() -> bool:
    return _enabled


def _make_passthrough(fn: Callable) -> Callable:
    """Return an async-aware no-op wrapper that preserves coroutine-ness."""
    if asyncio.iscoroutinefunction(fn):
        @wraps(fn)
        async def async_passthrough(*args, **kwargs):
            return await fn(*args, **kwargs)
        return async_passthrough

    @wraps(fn)
    def sync_passthrough(*args, **kwargs):
        return fn(*args, **kwargs)
    return sync_passthrough


def observe(*dargs, **dkwargs):
    """
    Wrapper around `langfuse.observe` that no-ops when Langfuse is disabled.
    Supports `@observe` (no parens) and `@observe(name=..., as_type=...)`.

    The no-op path preserves async/sync nature of the wrapped function.
    """
    # Bare `@observe` (no parens) — single callable positional arg.
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        fn = dargs[0]
        if _enabled and _observe_decorator is not None:
            return _observe_decorator(fn)
        return _make_passthrough(fn)

    # `@observe(...)` — return a decorator.
    def _decorator(fn: Callable) -> Callable:
        if _enabled and _observe_decorator is not None:
            return _observe_decorator(*dargs, **dkwargs)(fn)
        return _make_passthrough(fn)

    return _decorator


def langfuse_callback():
    """
    Return a `langfuse.langchain.CallbackHandler` for use in LCEL chains via
    `chain.invoke(..., config={"callbacks": [cb]})`. Returns None when disabled.
    """
    if not _enabled:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        # 3.x picks up auth from the initialized singleton; no args needed.
        return CallbackHandler()
    except Exception as e:
        logger.warning("Failed to build Langfuse CallbackHandler (non-fatal): %s", e)
        return None


def update_current_trace(
    user_id: str | None = None,
    session_id: str | None = None,
    name: str | None = None,
    tags: list | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Attach trace-level fields (user_id, session_id, tags, metadata, name) to
    the currently-active OTel span. Langfuse 4.x exposes these via OTel
    attribute keys defined in `LangfuseOtelSpanAttributes`.
    """
    if not _enabled:
        return
    try:
        import json

        from langfuse import LangfuseOtelSpanAttributes as Attr
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return

        if user_id is not None:
            span.set_attribute(Attr.TRACE_USER_ID, str(user_id))
        if session_id is not None:
            span.set_attribute(Attr.TRACE_SESSION_ID, str(session_id))
        if name is not None:
            span.set_attribute(Attr.TRACE_NAME, name)
        if tags:
            # OTel attributes accept list-of-strings; Langfuse parses it as tags.
            span.set_attribute(Attr.TRACE_TAGS, [str(t) for t in tags])
        if metadata:
            # Metadata is stored as a JSON-encoded string at the trace level.
            span.set_attribute(Attr.TRACE_METADATA, json.dumps(metadata, default=str))
    except Exception as e:
        logger.debug("Langfuse update_current_trace failed (non-fatal): %s", e)


def update_current_observation(
    input: Any = None,
    output: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Attach input/output/metadata to the currently-active span."""
    if not _enabled or _client is None:
        return
    try:
        _client.update_current_span(
            input=input,
            output=output,
            metadata=metadata,
        )
    except Exception as e:
        logger.debug("Langfuse update_current_span failed (non-fatal): %s", e)


def update_current_generation(
    model: str | None = None,
    input: Any = None,
    output: Any = None,
    metadata: dict[str, Any] | None = None,
    usage_details: dict[str, int] | None = None,
    cost_details: dict[str, float] | None = None,
    model_parameters: dict[str, Any] | None = None,
) -> None:
    """
    Update the currently-active generation/embedding observation with model
    metadata so it shows up in Langfuse's Model Usage dashboard.

    Must be called from inside a function decorated with `@observe(as_type=
    "generation" | "embedding")`.
    """
    if not _enabled or _client is None:
        return
    try:
        _client.update_current_generation(
            model=model,
            input=input,
            output=output,
            metadata=metadata,
            usage_details=usage_details,
            cost_details=cost_details,
            model_parameters=model_parameters,
        )
    except Exception as e:
        logger.debug("Langfuse update_current_generation failed (non-fatal): %s", e)


def flush() -> None:
    """Flush queued events. Safe to call when disabled."""
    if _client is not None:
        try:
            _client.flush()
        except Exception as e:
            logger.debug("Langfuse flush failed (non-fatal): %s", e)
