"""Assertion engine: compare a Case.expected block against a /chat response.

Each `assert_*` function returns an AssertionOutcome — either passed or
failed with a human-readable reason. A case's overall result is the
conjunction of all its applicable assertions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cases import Case, Expected, ExpectedHITL, ExpectedToolCall


@dataclass
class AssertionOutcome:
    dimension: str       # "intent" | "tool_calls" | "hitl" | "http_status"
    passed: bool
    reason: str = ""     # only set on failure


@dataclass
class CaseResult:
    case_id: str
    question: str
    description: str
    outcomes: list[AssertionOutcome] = field(default_factory=list)
    error: str | None = None             # set when the chat call itself failed
    raw_response: dict[str, Any] | None = None
    http_status: int = 0

    @property
    def passed(self) -> bool:
        return self.error is None and all(o.passed for o in self.outcomes)

    def failures(self) -> list[AssertionOutcome]:
        return [o for o in self.outcomes if not o.passed]


def _dict_subset(needle: dict[str, Any], haystack: dict[str, Any]) -> tuple[bool, str]:
    """Return (ok, reason). Strings compared case-insensitively (substring); other types == ."""
    for k, expected in needle.items():
        if k not in haystack:
            return False, f"missing key {k!r}"
        actual = haystack[k]
        if isinstance(expected, str) and isinstance(actual, str):
            if expected.lower() not in actual.lower():
                return False, f"{k}: expected substring {expected!r}, got {actual!r}"
        else:
            if actual != expected:
                return False, f"{k}: expected {expected!r}, got {actual!r}"
    return True, ""


def assert_intent(expected: str, response: dict[str, Any]) -> AssertionOutcome:
    actual = ((response.get("debug") or {}).get("intent")) or ""
    if actual == expected:
        return AssertionOutcome("intent", True)
    return AssertionOutcome(
        "intent", False, f"expected intent={expected!r}, got {actual!r}"
    )


def assert_tool_calls(
    expected_calls: list[ExpectedToolCall],
    response: dict[str, Any],
) -> AssertionOutcome:
    actual_calls = ((response.get("debug") or {}).get("tool_calls")) or []

    # Match by ordered prefix: each expected call must appear in order, but
    # the actual list can have additional calls interleaved/after.
    actual_idx = 0
    for i, expected in enumerate(expected_calls):
        matched = False
        while actual_idx < len(actual_calls):
            ac = actual_calls[actual_idx]
            actual_idx += 1
            if ac.get("name") != expected.name:
                continue
            ok, reason = _dict_subset(expected.args_contain, ac.get("args") or {})
            if ok:
                matched = True
                break
            # name matched but args didn't — keep scanning, the agent might
            # have called the same tool twice with different args.
        if not matched:
            actual_names = [c.get("name") for c in actual_calls]
            return AssertionOutcome(
                "tool_calls",
                False,
                f"expected call #{i+1} {expected.name}({expected.args_contain}) "
                f"not found in actual calls {actual_names}",
            )
    return AssertionOutcome("tool_calls", True)


def assert_hitl(expected: ExpectedHITL, response: dict[str, Any]) -> AssertionOutcome:
    pending = response.get("pending_approval")
    actually_paused = pending is not None

    if expected.paused is not None and expected.paused != actually_paused:
        return AssertionOutcome(
            "hitl",
            False,
            f"expected paused={expected.paused}, got paused={actually_paused}",
        )

    if expected.kind is not None:
        if not actually_paused:
            return AssertionOutcome(
                "hitl", False, f"expected kind={expected.kind!r}, got no pause"
            )
        actual_kind = (pending or {}).get("kind", "approval")
        if actual_kind != expected.kind:
            return AssertionOutcome(
                "hitl", False, f"expected kind={expected.kind!r}, got {actual_kind!r}"
            )

    if expected.min_candidates is not None:
        candidates = (pending or {}).get("candidates") or []
        if len(candidates) < expected.min_candidates:
            return AssertionOutcome(
                "hitl",
                False,
                f"expected >= {expected.min_candidates} candidates, "
                f"got {len(candidates)}",
            )

    return AssertionOutcome("hitl", True)


def assert_http_status(expected: int, actual: int) -> AssertionOutcome:
    if expected == actual:
        return AssertionOutcome("http_status", True)
    return AssertionOutcome(
        "http_status", False, f"expected {expected}, got {actual}"
    )


def evaluate_case(
    case: Case,
    response: dict[str, Any] | None,
    http_status: int,
    error: str | None = None,
) -> CaseResult:
    """Run every applicable assertion on a case and aggregate outcomes."""
    result = CaseResult(
        case_id=case.id,
        question=case.question,
        description=case.description,
        raw_response=response,
        http_status=http_status,
        error=error,
    )

    if error is not None or response is None:
        # No response → no assertions to run beyond http_status if expected
        if case.expected.http_status is not None:
            result.outcomes.append(
                assert_http_status(case.expected.http_status, http_status)
            )
        return result

    exp: Expected = case.expected
    if exp.http_status is not None:
        result.outcomes.append(assert_http_status(exp.http_status, http_status))
    if exp.intent is not None:
        result.outcomes.append(assert_intent(exp.intent, response))
    if exp.tool_calls is not None:
        result.outcomes.append(assert_tool_calls(exp.tool_calls, response))
    if exp.hitl is not None:
        result.outcomes.append(assert_hitl(exp.hitl, response))
    return result
