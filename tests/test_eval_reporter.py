"""Behavior contract for the markdown reporter.

Drives off `tests/fixtures/eval_results.json`. Assertions target
qualitative invariants (expected sections present, failures section
flags the low-scoring row, model names appear) rather than exact
substring matches that would shatter on a small format tweak.
"""

import json
from pathlib import Path

from eval.reporter import render_markdown

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "eval_results.json"


def _load_rows():
    with open(FIXTURE_PATH) as fh:
        return json.load(fh)["rows"]


def test_reporter_includes_expected_sections():
    md = render_markdown(
        _load_rows(),
        generated_at="2026-05-19 04:00 UTC",
        generator_model="google/gemma-4-31b-it",
        judge_model="meta/llama-3.3-70b-instruct",
    )
    for section in (
        "# Eval run",
        "## Aggregate",
        "## By book",
        "## By category",
        "## Per question",
        "## Failures",
    ):
        assert section in md, f"missing section {section!r}"


def test_reporter_lists_both_models_in_header():
    md = render_markdown(
        _load_rows(),
        generated_at="2026-05-19",
        generator_model="google/gemma-4-31b-it",
        judge_model="meta/llama-3.3-70b-instruct",
    )
    assert "google/gemma-4-31b-it" in md
    assert "meta/llama-3.3-70b-instruct" in md


def test_reporter_flags_low_scoring_row_in_failures():
    md = render_markdown(
        _load_rows(),
        generated_at="2026-05-19",
        generator_model="g",
        judge_model="j",
    )
    failures_section = md.split("## Failures")[1]
    # hp7-002 has recall_at_5=0.2 (< 0.5) and completeness=2 (< 3) in the
    # fixture — it should appear in the failures block.
    assert "hp7-002" in failures_section
    # And the strong rows should NOT appear there.
    assert "hp4-003" not in failures_section


def test_reporter_handles_empty_rows():
    md = render_markdown(
        [],
        generated_at="2026-05-19",
        generator_model="g",
        judge_model="j",
    )
    assert "## Aggregate" in md
    assert "## Failures (0)" in md
