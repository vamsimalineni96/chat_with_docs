"""Agentic eval runner.

Usage:

  # Validate cases load cleanly (no network, no LLM):
  python -m evals.agentic.run_eval --validate-only

  # Full run against the live /chat endpoint:
  python -m evals.agentic.run_eval \\
      --output docs/eval-reports/agentic_$(date +%Y-%m-%d).md

Each case spins up a fresh conversation (no conversation_id sent), so cases
are fully isolated from each other.

The eval makes assertions on routing (intent), tool selection (tool_calls),
and HITL behaviour (pause kind + candidate count). The chat call uses
debug=true so the response includes these decisions inline; no Langfuse
SDK dependency.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .assertions import CaseResult, evaluate_case
from .cases import Case, load_all_cases
from .reporter import render_markdown

DEFAULT_API_BASE = "http://localhost:8000"
# Mirrors ui.py — the actual ShopCo corpus collection. Override via --collection-name.
DEFAULT_COLLECTION = "shopco_docs"
DEFAULT_TIMEOUT_S = 180


def _default_chat_call(
    question: str,
    *,
    api_base: str,
    user_id: str,
    collection_name: str,
    timeout_s: int,
) -> tuple[dict[str, Any] | None, int, str | None]:
    """POST to /chat with debug=true. Returns (json, status, error).

    On HTTPError, still returns whatever JSON came back (FastAPI sends
    structured error bodies). On network failure, returns (None, 0, err).
    """
    import requests  # noqa: PLC0415

    payload = {
        "user_external_id": user_id,
        "question": question,
        "collection_name": collection_name,
        "debug": True,
    }
    try:
        response = requests.post(
            f"{api_base.rstrip('/')}/chat",
            json=payload,
            timeout=timeout_s,
        )
    except requests.RequestException as e:
        return None, 0, f"network error: {e}"

    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400:
        return body, response.status_code, None
    return body, response.status_code, None


def evaluate_one(
    case: Case,
    *,
    chat_caller: Callable[[str, str], tuple[dict[str, Any] | None, int, str | None]],
) -> CaseResult:
    """Run one case against the chat caller and evaluate assertions.

    chat_caller signature: (question, user_id) -> (body, http_status, error)
    """
    # Unique user_id per case keeps conversations isolated even when run
    # against an instance with stale state — no risk of locking conflicts.
    user_id = f"eval-agentic-{case.id}-{uuid.uuid4().hex[:6]}"
    body, status, err = chat_caller(case.question, user_id)
    return evaluate_case(case, body, status, error=err)


def run_eval(
    cases: list[Case],
    *,
    chat_caller: Callable[[str, str], tuple[dict[str, Any] | None, int, str | None]],
    on_result: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        result = evaluate_one(case, chat_caller=chat_caller)
        if on_result:
            on_result(result)
        results.append(result)
    return results


def _format_progress(r: CaseResult) -> str:
    if r.error:
        return f"[{r.case_id}] ⚠️  ERROR: {r.error}"
    if r.passed:
        return f"[{r.case_id}] ✅ ({len(r.outcomes)} assertions)"
    fails = ", ".join(o.dimension for o in r.failures())
    return f"[{r.case_id}] ❌ failed: {fails}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and summarize the cases without running the eval.",
    )
    parser.add_argument(
        "--api-base", default=DEFAULT_API_BASE,
        help=f"FastAPI /chat base URL (default: {DEFAULT_API_BASE}).",
    )
    parser.add_argument(
        "--collection-name", default=DEFAULT_COLLECTION,
        help=f"Milvus collection (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--timeout-s", type=int, default=DEFAULT_TIMEOUT_S,
        help="Per-request /chat timeout in seconds.",
    )
    parser.add_argument(
        "--case-id", default=None,
        help="Run only the case with this id (smoke test single behaviour).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Markdown report path. Required outside --validate-only mode.",
    )
    parser.add_argument(
        "--latest", type=Path,
        default=Path("docs/eval-reports/agentic_latest.md"),
        help="Stable 'latest' path; written alongside --output.",
    )
    args = parser.parse_args(argv)

    try:
        cases = load_all_cases()
    except (FileNotFoundError, ValueError) as e:
        print(f"failed to load cases: {e}", file=sys.stderr)
        return 1

    if args.case_id:
        cases = [c for c in cases if c.id == args.case_id]
        if not cases:
            print(f"no case found with id={args.case_id!r}", file=sys.stderr)
            return 1

    if args.validate_only:
        print(f"Loaded {len(cases)} cases:")
        for c in cases:
            assertions = [
                d for d in ("intent", "tool_calls", "hitl", "http_status")
                if getattr(c.expected, d) is not None
            ]
            print(f"  - {c.id}: asserts {assertions}")
        return 0

    if args.output is None:
        parser.error("--output is required unless --validate-only is set")

    print(f"Running {len(cases)} cases against {args.api_base}...")

    def chat_caller(
        question: str, user_id: str,
    ) -> tuple[dict[str, Any] | None, int, str | None]:
        return _default_chat_call(
            question,
            api_base=args.api_base,
            user_id=user_id,
            collection_name=args.collection_name,
            timeout_s=args.timeout_s,
        )

    results = run_eval(
        cases,
        chat_caller=chat_caller,
        on_result=lambda r: print(_format_progress(r), flush=True),
    )

    passed = sum(1 for r in results if r.passed)
    print(f"\n{passed}/{len(results)} cases passed")

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    md = render_markdown(results, generated_at=generated_at, api_base=args.api_base)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(f"Report written to {args.output}")

    if args.latest:
        args.latest.parent.mkdir(parents=True, exist_ok=True)
        args.latest.write_text(md)
        print(f"Also wrote {args.latest}")

    # Exit non-zero if any case failed — useful when wired into CI
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
