"""Eval driver CLI.

Sub-PR #7a (this file's current state): just loads and validates
`eval/qa_set.jsonl` and prints a summary. No LLM calls, no Langfuse,
no `/chat` requests. The intent is to lock down the dataset contract
before wiring up the more expensive pieces.

Sub-PR #7c will extend `main()` to actually call `/chat` for each
question, capture retrieved chunks, compute metrics, invoke the judge,
and write a report via reporter.render_markdown.

Usage (current):
    python -m eval.run_eval --qa-set eval/qa_set.jsonl --validate-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "id",
    "question",
    "expected_keywords_in_answer",
    "expected_keywords_in_top_chunks",
    "book",
    "category",
}


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-set",
        type=Path,
        default=Path("eval/qa_set.jsonl"),
        help="Path to the JSONL Q&A dataset.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Currently the only supported mode. Sub-PR #7c will add a "
             "full eval-run mode that actually queries /chat and the judge.",
    )
    args = parser.parse_args(argv)

    try:
        entries = load_qa_set(args.qa_set)
    except (FileNotFoundError, ValueError) as e:
        print(f"qa_set load failed: {e}", file=sys.stderr)
        return 1

    print(summarize(entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
