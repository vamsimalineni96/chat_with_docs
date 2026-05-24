"""Eval driver CLI.

Two modes:

  # 1. Validate the qa_set (cheap, no LLM, no network):
  python -m evals.quality.run_eval --validate-only

  # 2. Full end-to-end run against the live /chat endpoint + judge LLM:
  python -m evals.quality.run_eval --output docs/eval-reports/eval_$(date +%Y-%m-%d).md

Full-run mode requires:
  - The FastAPI app running locally (default: http://localhost:8000)
  - NVIDIA_API_KEY in env (for the judge LLM)

Each question opens a fresh conversation (no shared history). Errors on a
single question are recorded as that row's error sentinel rather than
aborting the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .judge import JudgeResult, judge
from .metrics import keyword_recall_at_k, reciprocal_rank
from .reporter import render_markdown

load_dotenv()

# Module-relative default — keeps the CLI working from any cwd
# without forcing the caller to chdir into the repo root.
_DEFAULT_QA_SET = Path(__file__).parent / "qa_set.jsonl"

REQUIRED_FIELDS = {
    "id",
    "question",
    "expected_keywords_in_answer",
    "expected_keywords_in_top_chunks",
    "book",
    "category",
}

DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_COLLECTION = "docs"
DEFAULT_USER_ID = "eval-runner"
DEFAULT_TIMEOUT_S = 120
DEFAULT_GENERATOR_MODEL = "google/gemma-4-31b-it"  # mirrors NVIDIA_LLM_MODEL in .env
RECALL_K = 5


# ---------------------------------------------------------------------------
# Dataset I/O (carried over from #7a)
# ---------------------------------------------------------------------------


def load_qa_set(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL Q&A file. Raises on the first malformed line."""
    entries: list[dict[str, Any]] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"qa_set line {lineno}: invalid JSON — {e}") from e
            missing = REQUIRED_FIELDS - obj.keys()
            if missing:
                raise ValueError(
                    f"qa_set line {lineno} (id={obj.get('id')!r}): missing "
                    f"required fields {sorted(missing)}"
                )
            entries.append(obj)
    return entries


def summarize(entries: list[dict[str, Any]]) -> str:
    """Human-readable summary of the loaded dataset."""
    by_book = Counter(e["book"] for e in entries)
    by_category = Counter(e["category"] for e in entries)
    avg_answer_kws = (
        sum(len(e["expected_keywords_in_answer"]) for e in entries) / len(entries)
        if entries
        else 0.0
    )
    avg_chunk_kws = (
        sum(len(e["expected_keywords_in_top_chunks"]) for e in entries) / len(entries)
        if entries
        else 0.0
    )
    lines = [
        f"Total Q&A pairs: {len(entries)}",
        "",
        "By book:",
        *(f"  - {book}: {count}" for book, count in sorted(by_book.items())),
        "",
        "By category:",
        *(
            f"  - {cat}: {count}"
            for cat, count in sorted(by_category.items(), key=lambda kv: -kv[1])
        ),
        "",
        f"Avg expected_keywords_in_answer per entry:     {avg_answer_kws:.1f}",
        f"Avg expected_keywords_in_top_chunks per entry: {avg_chunk_kws:.1f}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# /chat HTTP caller — module-level shim so tests can substitute a stub.
# ---------------------------------------------------------------------------


def _default_chat_call(
    question: str,
    *,
    api_base: str,
    user_id: str,
    collection_name: str,
    timeout_s: int,
) -> dict[str, Any]:
    """POST to the FastAPI /chat endpoint with debug=true.

    Deferred import of `requests` keeps `import evals.quality.run_eval` cheap
    (no requests needed when tests substitute a stub caller).
    """
    import requests  # noqa: PLC0415

    payload = {
        "user_external_id": user_id,
        "question": question,
        "collection_name": collection_name,
        "debug": True,
    }
    response = requests.post(
        f"{api_base.rstrip('/')}/chat",
        json=payload,
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Per-question orchestration — testable in isolation.
# ---------------------------------------------------------------------------


def _extract_context_for_judge(
    debug_payload: dict[str, Any] | None,
    top_n: int = RECALL_K,
) -> str:
    """Concatenate the top-N reranked chunks (or retrieved, if rerank
    didn't run) into a single string the judge can read.
    """
    if not debug_payload:
        return ""
    chunks = debug_payload.get("reranked_top_k") or debug_payload.get(
        "retrieved_chunks"
    ) or []
    parts = [(c.get("text") or "") for c in chunks[:top_n]]
    return "\n\n".join(parts)


def evaluate_one(
    qa: dict[str, Any],
    *,
    chat_caller: Callable[[str], dict[str, Any]],
    judge_caller: Callable[[str, str, str, list[str]], JudgeResult],
) -> dict[str, Any]:
    """Run one Q&A through chat + metrics + judge. Returns an eval-row dict.

    A failure in either chat or judge produces a row with zeroed scores and
    an `error` field — the row still appears in the report (counted under
    Failures) rather than crashing the whole eval.
    """
    base_row: dict[str, Any] = {
        "id": qa["id"],
        "question": qa["question"],
        "book": qa["book"],
        "category": qa["category"],
        "answer": "",
        "recall_at_5": 0.0,
        "reciprocal_rank": 0.0,
        "judge": {
            "groundedness": 0,
            "accuracy": 0,
            "completeness": 0,
            "reasoning": "",
        },
        "latency_ms": 0.0,
        "error": None,
    }

    try:
        response = chat_caller(qa["question"])
    except Exception as e:
        base_row["error"] = f"chat call failed: {e}"
        return base_row

    debug = response.get("debug") or {}
    retrieved = debug.get("retrieved_chunks") or []
    answer = response.get("answer", "")
    latency_ms = float((debug.get("timings_ms") or {}).get("total", 0.0))

    expected_chunk_kws = qa["expected_keywords_in_top_chunks"]
    base_row.update(
        {
            "answer": answer,
            "recall_at_5": keyword_recall_at_k(
                expected_chunk_kws, retrieved, k=RECALL_K
            ),
            "reciprocal_rank": reciprocal_rank(expected_chunk_kws, retrieved),
            "latency_ms": latency_ms,
        }
    )

    try:
        jr = judge_caller(
            qa["question"],
            answer,
            _extract_context_for_judge(debug),
            qa["expected_keywords_in_answer"],
        )
    except Exception as e:
        base_row["error"] = f"judge call failed: {e}"
        return base_row

    base_row["judge"] = {
        "groundedness": jr.groundedness,
        "accuracy": jr.accuracy,
        "completeness": jr.completeness,
        "reasoning": jr.reasoning,
    }
    if jr.error:
        base_row["error"] = f"judge returned error: {jr.error}"
    return base_row


def run_eval(
    qa_entries: list[dict[str, Any]],
    *,
    chat_caller: Callable[[str], dict[str, Any]],
    judge_caller: Callable[[str, str, str, list[str]], JudgeResult],
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Sequential per-question evaluation. Returns a list of eval rows.

    `on_row` is called with each completed row — useful for printing
    progress while a long run is in flight.
    """
    rows: list[dict[str, Any]] = []
    for qa in qa_entries:
        row = evaluate_one(qa, chat_caller=chat_caller, judge_caller=judge_caller)
        if on_row is not None:
            on_row(row)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_progress(row: dict[str, Any]) -> str:
    j = row["judge"]
    err = f" ERR={row['error']!r}" if row.get("error") else ""
    return (
        f"[{row['id']}] recall@{RECALL_K}={row['recall_at_5']:.2f} "
        f"g={j['groundedness']} a={j['accuracy']} c={j['completeness']} "
        f"latency={row['latency_ms']:.0f}ms{err}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-set", type=Path, default=_DEFAULT_QA_SET)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and summarize the qa_set without running the eval.",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"FastAPI /chat base URL (default: {DEFAULT_API_BASE}).",
    )
    parser.add_argument(
        "--collection-name", default=DEFAULT_COLLECTION,
        help=f"Milvus collection (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--user-id", default=DEFAULT_USER_ID,
        help=f"user_external_id sent to /chat (default: {DEFAULT_USER_ID}).",
    )
    parser.add_argument(
        "--timeout-s", type=int, default=DEFAULT_TIMEOUT_S,
        help="Per-request /chat timeout in seconds.",
    )
    parser.add_argument(
        "--max-questions", type=int, default=None,
        help="Cap on questions to run — useful for a quick smoke test.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Markdown report path. Required in full-run mode.",
    )
    parser.add_argument(
        "--latest", type=Path, default=Path("docs/eval-reports/latest.md"),
        help="Stable 'latest' symlink-style path; written alongside --output.",
    )
    parser.add_argument(
        "--generator-model", default=DEFAULT_GENERATOR_MODEL,
        help="Label only — used in the report header.",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Override the judge model. Default: from env or judge.py default.",
    )
    args = parser.parse_args(argv)

    try:
        entries = load_qa_set(args.qa_set)
    except (FileNotFoundError, ValueError) as e:
        print(f"qa_set load failed: {e}", file=sys.stderr)
        return 1

    if args.validate_only:
        print(summarize(entries))
        return 0

    if args.output is None:
        parser.error("--output is required when not in --validate-only mode")

    if args.max_questions:
        entries = entries[: args.max_questions]

    print(f"Running eval against {args.api_base} on {len(entries)} questions...")

    def chat_caller(question: str) -> dict[str, Any]:
        return _default_chat_call(
            question,
            api_base=args.api_base,
            user_id=args.user_id,
            collection_name=args.collection_name,
            timeout_s=args.timeout_s,
        )

    def judge_caller(
        question: str, answer: str, context: str, kws: list[str]
    ) -> JudgeResult:
        return judge(question, answer, context, kws, model=args.judge_model)

    rows = run_eval(
        entries,
        chat_caller=chat_caller,
        judge_caller=judge_caller,
        on_row=lambda r: print(_format_progress(r), flush=True),
    )

    judge_label = args.judge_model or "meta/llama-3.3-70b-instruct (default)"
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    md = render_markdown(
        rows,
        generated_at=generated_at,
        generator_model=args.generator_model,
        judge_model=judge_label,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(f"\nReport written to {args.output}")

    if args.latest:
        args.latest.parent.mkdir(parents=True, exist_ok=True)
        args.latest.write_text(md)
        print(f"Also wrote {args.latest}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
