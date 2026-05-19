"""Behavior contract for the cost-report aggregator.

All tests target the pure functions — no Langfuse SDK, no network. The
fixture-based path is the same path CI uses, so anything that passes here
will also pass in CI's `unit-tests` job.
"""

import json
from pathlib import Path

from scripts.cost_report import (
    aggregate,
    classify_stage,
    compute_cost,
    render_markdown,
    summarize_by_stage,
    summarize_by_task,
    task_type_from_tags,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "observations.json"


def _load_fixture() -> tuple[list[dict], list[dict]]:
    with open(FIXTURE_PATH) as fh:
        data = json.load(fh)
    return data["traces"], data["observations"]


# ----------------------------------------------------------------------------
# task_type_from_tags
# ----------------------------------------------------------------------------


def test_task_type_recognizes_cache_and_rag():
    assert task_type_from_tags(["prompt:v2", "cache-path", "normal"]) == "cache-hit"
    assert task_type_from_tags(["prompt:v2", "rag-path", "normal"]) == "rag-full"


def test_task_type_unknown_when_no_path_tag():
    assert task_type_from_tags(["prompt:v2", "normal"]) == "unknown"
    assert task_type_from_tags([]) == "unknown"


# ----------------------------------------------------------------------------
# classify_stage
# ----------------------------------------------------------------------------


def test_classify_stage_known_names():
    assert classify_stage("embed_query") == "embedding"
    assert classify_stage("embed_passage_batch") == "embedding"
    assert classify_stage("rerank") == "rerank"
    assert classify_stage("cache_lookup") == "cache_lookup"
    assert classify_stage("ChatNVIDIA") == "llm"


def test_classify_stage_unknown_falls_through_to_other():
    assert classify_stage("totally_made_up_span_name") == "other"


# ----------------------------------------------------------------------------
# compute_cost
# ----------------------------------------------------------------------------


def test_compute_cost_llm_uses_token_rates():
    # 1000 input @ $0.0006/1K + 500 output @ $0.0006/1K = 0.0006 + 0.0003 = 0.0009
    cost = compute_cost(
        stage="llm",
        model="meta/llama-3.1-70b-instruct",
        usage={"input": 1000, "output": 500},
    )
    assert abs(cost - 0.0009) < 1e-9


def test_compute_cost_rerank_uses_per_call_price():
    # Rerank ignores tokens; price comes from RERANK_PRICE_PER_CALL_USD.
    cost = compute_cost(
        stage="rerank",
        model="nvidia/nv-rerankqa-mistral-4b-v3",
        usage={"input": 999, "output": 999},
    )
    assert abs(cost - 0.00010) < 1e-9


def test_compute_cost_unknown_model_returns_zero():
    cost = compute_cost(
        stage="llm",
        model="some/unknown-model",
        usage={"input": 1000, "output": 1000},
    )
    assert cost == 0.0


# ----------------------------------------------------------------------------
# aggregate + summarize
# ----------------------------------------------------------------------------


def test_aggregate_returns_one_row_per_observation():
    traces, observations = _load_fixture()
    rows = aggregate(traces, observations)
    assert len(rows) == len(observations)


def test_summarize_by_task_partitions_cache_vs_rag():
    traces, observations = _load_fixture()
    rows = aggregate(traces, observations)
    by_task = summarize_by_task(rows)
    task_names = {r["task_type"] for r in by_task}
    assert {"cache-hit", "rag-full"}.issubset(task_names)

    # Cache hits should be dramatically cheaper per task than full RAG —
    # this is the headline insight Pooja's framework cares about, so we
    # pin it as a contract.
    cache = next(r for r in by_task if r["task_type"] == "cache-hit")
    rag = next(r for r in by_task if r["task_type"] == "rag-full")
    assert cache["avg_cost_per_task_usd"] < rag["avg_cost_per_task_usd"]


def test_summarize_by_stage_llm_dominates_spend():
    traces, observations = _load_fixture()
    rows = aggregate(traces, observations)
    by_stage = summarize_by_stage(rows)
    # LLM is the most expensive stage in any realistic RAG workload;
    # if this flips, something is suspicious in the pricing table.
    top = by_stage[0]
    assert top["stage"] == "llm"
    assert top["share_of_spend_pct"] > 50.0


# ----------------------------------------------------------------------------
# render_markdown — smoke test
# ----------------------------------------------------------------------------


def test_render_markdown_includes_expected_sections():
    traces, observations = _load_fixture()
    rows = aggregate(traces, observations)
    md = render_markdown(
        rows,
        source_label="fixture",
        window_label="fixture-defined",
        generated_at="2026-05-17 10:00 UTC",
    )
    for marker in ("# Cost report", "## Aggregate", "## By task type", "## By pipeline stage"):
        assert marker in md, f"missing section {marker!r} in rendered report"
