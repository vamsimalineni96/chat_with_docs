"""Unit tests for the observability helper module.

These pin the contract of the thin wrappers we put in front of Langfuse —
both the no-op behavior when Langfuse is disabled (the default in CI) and
the args we forward to the client when it is enabled.

The Langfuse client itself is mocked, so these run fast and offline in the
lightweight unit-tests CI job. They are *not* a substitute for an end-to-end
smoke test (does a trace actually appear in the Langfuse UI?) — that step
stays manual and is run pre-merge for any change that touches `observability.py`
or its callers.

The OTel-attribute test pins the v3-style approach that we deliberately
kept on v4 (see observability.py docstring and docs/OBSERVABILITY.md §3.1).
If we later migrate to v4's `propagate_attributes()` idiom, that test will
need to be rewritten to assert the new call shape.
"""

import asyncio
from unittest.mock import MagicMock

from src.utils import observability as obs

# ---------- Disabled-path: the default when LANGFUSE_ENABLED is unset ----------


def test_is_enabled_returns_false_when_module_state_says_so(monkeypatch):
    monkeypatch.setattr(obs, "_enabled", False)
    monkeypatch.setattr(obs, "_client", None)
    assert obs.is_enabled() is False


def test_observe_bare_passthrough_when_disabled(monkeypatch):
    """`@observe` (no parens) returns the original function's semantics."""
    monkeypatch.setattr(obs, "_enabled", False)
    monkeypatch.setattr(obs, "_observe_decorator", None)

    @obs.observe
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_observe_with_args_passthrough_when_disabled(monkeypatch):
    """`@observe(name=..., as_type=...)` also no-ops cleanly."""
    monkeypatch.setattr(obs, "_enabled", False)
    monkeypatch.setattr(obs, "_observe_decorator", None)

    @obs.observe(name="my_op", as_type="span")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_observe_preserves_async_nature_when_disabled(monkeypatch):
    """Regression guard: wrapping an async fn must keep it awaitable.
    A naive passthrough that calls the fn synchronously would silently
    return a coroutine instead of awaiting it."""
    monkeypatch.setattr(obs, "_enabled", False)
    monkeypatch.setattr(obs, "_observe_decorator", None)

    @obs.observe
    async def add_async(a, b):
        return a + b

    assert asyncio.iscoroutinefunction(add_async)
    assert asyncio.run(add_async(2, 3)) == 5


def test_langfuse_callback_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(obs, "_enabled", False)
    assert obs.langfuse_callback() is None


def test_update_current_trace_is_safe_when_disabled(monkeypatch):
    monkeypatch.setattr(obs, "_enabled", False)
    # Must not raise even with rich kwargs.
    obs.update_current_trace(
        user_id="u", session_id="s", name="n", tags=["t"], metadata={"k": "v"}
    )


def test_update_current_observation_is_safe_when_disabled(monkeypatch):
    monkeypatch.setattr(obs, "_enabled", False)
    monkeypatch.setattr(obs, "_client", None)
    obs.update_current_observation(input={"q": "hi"}, output={"a": "hi"})


def test_update_current_generation_is_safe_when_disabled(monkeypatch):
    monkeypatch.setattr(obs, "_enabled", False)
    monkeypatch.setattr(obs, "_client", None)
    obs.update_current_generation(model="m", usage_details={"input": 5, "total": 5})


def test_flush_is_safe_when_disabled(monkeypatch):
    monkeypatch.setattr(obs, "_client", None)
    obs.flush()  # no raise


# ---------- Enabled-path: a MagicMock stands in for the real client ----------


def test_update_current_observation_forwards_args_to_client(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(obs, "_enabled", True)
    monkeypatch.setattr(obs, "_client", mock_client)

    obs.update_current_observation(
        input={"q": "x"}, output={"a": "y"}, metadata={"k": "v"}
    )

    mock_client.update_current_span.assert_called_once_with(
        input={"q": "x"}, output={"a": "y"}, metadata={"k": "v"}
    )


def test_update_current_generation_forwards_args_to_client(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(obs, "_enabled", True)
    monkeypatch.setattr(obs, "_client", mock_client)

    obs.update_current_generation(
        model="nv-embed",
        input="hello",
        output={"embedding_dim": 1024},
        usage_details={"input": 1, "total": 1},
        metadata={"input_type": "query"},
    )

    mock_client.update_current_generation.assert_called_once()
    kwargs = mock_client.update_current_generation.call_args.kwargs
    assert kwargs["model"] == "nv-embed"
    assert kwargs["usage_details"] == {"input": 1, "total": 1}
    assert kwargs["metadata"] == {"input_type": "query"}


def test_flush_calls_client_flush_when_client_present(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(obs, "_client", mock_client)
    obs.flush()
    mock_client.flush.assert_called_once()


def test_update_current_trace_sets_expected_otel_attributes(monkeypatch):
    """Pins our chosen approach for trace-level attributes: each non-None
    kwarg becomes one OTel `span.set_attribute(...)` call. This works on
    both v3 and v4 — `LangfuseOtelSpanAttributes` is present in both.
    """
    monkeypatch.setattr(obs, "_enabled", True)

    mock_span = MagicMock()
    mock_span.is_recording.return_value = True

    from opentelemetry import trace

    monkeypatch.setattr(trace, "get_current_span", lambda: mock_span)

    obs.update_current_trace(
        user_id="user-123",
        session_id="sess-abc",
        name="rag_output",
        tags=["prompt:v2", "rag-path"],
        metadata={"question": "hi"},
    )

    # All five kwargs supplied -> five OTel attribute writes.
    assert mock_span.set_attribute.call_count == 5


def test_update_current_observation_coerces_metadata_to_strings(monkeypatch):
    """v4 validates metadata as `dict[str, str]` and drops oversized or
    non-string values with a warning. The wrapper coerces proactively so
    callers can pass natural Python types (bools, ints) without warnings."""
    mock_client = MagicMock()
    monkeypatch.setattr(obs, "_enabled", True)
    monkeypatch.setattr(obs, "_client", mock_client)

    obs.update_current_observation(
        metadata={"cache_enabled": True, "retry_count": 3, "note": "x" * 500},
    )

    kwargs = mock_client.update_current_span.call_args.kwargs
    md = kwargs["metadata"]
    assert md == {
        "cache_enabled": "True",
        "retry_count": "3",
        "note": "x" * 200,  # truncated to ≤200 chars
    }


def test_update_current_generation_coerces_metadata_to_strings(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(obs, "_enabled", True)
    monkeypatch.setattr(obs, "_client", mock_client)

    obs.update_current_generation(
        model="nv-embed",
        metadata={"chunk_size": 800, "input_type": "passage"},
        usage_details={"input": 100, "total": 100},
    )

    kwargs = mock_client.update_current_generation.call_args.kwargs
    assert kwargs["metadata"] == {"chunk_size": "800", "input_type": "passage"}
