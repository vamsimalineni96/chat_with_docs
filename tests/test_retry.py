"""Behavior contract for the shared retry helper.

Pure tests — no network, no real LLM, no real Langfuse. `time.sleep` is
monkeypatched to zero so the test suite stays fast.
"""

import pytest

from src.utils.services.retry import call_with_retry


def test_returns_value_on_first_success():
    """Happy path — fn succeeds immediately, no retry needed."""
    assert call_with_retry(lambda: 42, op_name="test") == 42


def test_recovers_after_n_transient_failures(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient 502")
        return "ok"

    assert call_with_retry(flaky, op_name="test", max_attempts=5) == "ok"
    assert attempts["n"] == 3


def test_raises_last_exception_after_exhaustion(monkeypatch):
    """The exception from the FINAL attempt should propagate, with its message."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    def always_fails():
        raise RuntimeError("perma")

    with pytest.raises(RuntimeError, match="perma"):
        call_with_retry(always_fails, op_name="test", max_attempts=2)


def test_respects_custom_max_attempts(monkeypatch):
    """fn should be called exactly `max_attempts` times when it never succeeds."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    attempts = {"n": 0}

    def counter():
        attempts["n"] += 1
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        call_with_retry(counter, op_name="test", max_attempts=5)
    assert attempts["n"] == 5


def test_sleep_durations_cap_at_max_sleep_s(monkeypatch):
    """Exponential 2,4,8,... should clip to max_sleep_s once it exceeds the cap."""
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    def fails():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        call_with_retry(fails, op_name="test", max_attempts=10, max_sleep_s=4)
    # 9 sleeps for 10 attempts. All within the cap, and the cap is actually hit.
    assert len(sleeps) == 9
    assert max(sleeps) == 4
    assert all(s <= 4 for s in sleeps)
