"""Behavior contract for the latency aggregator.

Pure-function tests against tests/fixtures/observations_latency.json — no
Langfuse, no network. Same shape as tests/test_cost_report.py.
"""

import json
from pathlib import Path

from scripts.latency_report import (
    aggregate,
    percentiles,
    render_markdown,
    slow_traces,
    summarize_by_stage,
    summarize_by_task,
)

FIXTURE = Path(__file__).parent / "fixtures" / "observations_latency.json"


def _load() -> tuple[list[dict], list[dict]]:
    with open(FIXTURE) as fh:
        d = json.load(fh)
    return d["traces"], d["observations"]


# --- percentiles ------------------------------------------------------------


def test_percentiles_basic_distribution():
    p = percentiles([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # Nearest-rank with N=10: p50 → index 4, p95 → index 9, p99 → index 9.
    assert p["p50"] == 50
    assert p["p95"] == 100
    assert p["p99"] == 100


def test_percentiles_empty_returns_zeros():
    assert percentiles([]) == {"p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_percentiles_single_value_collapses_all_to_that_value():
    p = percentiles([42.0])
    assert p["p50"] == p["p95"] == p["p99"] == 42.0


# --- aggregate --------------------------------------------------------------


def test_aggregate_emits_one_row_per_observation():
    traces, observations = _load()
    rows, _ = aggregate(traces, observations)
    assert len(rows) == len(observations)


def test_aggregate_classifies_all_pipeline_stages():
    traces, observations = _load()
    rows, _ = aggregate(traces, observations)
    stages = {r.stage for r in rows}
    # Cache hits exercise embedding + cache_lookup; RAG exercises rerank + llm.
    assert {"embedding", "cache_lookup", "rerank", "llm"} <= stages


def test_aggregate_prefers_trace_latency_ms_when_provided():
    traces, observations = _load()
    _, trace_total = aggregate(traces, observations)
    # Fixture sets these explicitly — verbatim, not summed.
    assert trace_total["trace-rag-3"] == 22480.0
    assert trace_total["trace-cache-1"] == 118.0


def test_aggregate_falls_back_to_sum_when_trace_latency_missing():
    traces = [{"id": "t1", "tags": ["rag-path"], "timestamp": "2026-05-23T00:00:00Z"}]
    observations = [
        {"id": "o1", "trace_id": "t1", "name": "embed_query",  "duration_ms": 50},
        {"id": "o2", "trace_id": "t1", "name": "ChatNVIDIA",   "duration_ms": 1500},
    ]
    _, trace_total = aggregate(traces, observations)
    assert trace_total["t1"] == 1550.0


def test_aggregate_maps_unknown_trace_id_to_unknown_task():
    traces: list[dict] = []  # No tags lookup possible.
    observations = [
        {"id": "o1", "trace_id": "orphan", "name": "embed_query", "duration_ms": 10},
    ]
    rows, _ = aggregate(traces, observations)
    assert rows[0].task_type == "unknown"


# --- summarize_by_task ------------------------------------------------------


def test_summarize_by_task_separates_cache_from_rag():
    traces, observations = _load()
    rows, trace_total = aggregate(traces, observations)
    by_task = {row["task_type"]: row for row in summarize_by_task(rows, trace_total)}

    assert "cache-hit" in by_task and "rag-full" in by_task
    # The whole point of breaking out by task: cache p95 << rag p95.
    assert by_task["cache-hit"]["p95_ms"] < by_task["rag-full"]["p95_ms"]
    assert by_task["cache-hit"]["count"] == 2
    assert by_task["rag-full"]["count"] == 3


# --- summarize_by_stage -----------------------------------------------------


def test_summarize_by_stage_llm_dominates_total_time():
    traces, observations = _load()
    rows, _ = aggregate(traces, observations)
    by_stage = {row["stage"]: row for row in summarize_by_stage(rows)}
    # LLM stage should eat the vast majority of time in any sane RAG pipeline.
    assert by_stage["llm"]["share_of_time_pct"] > 80.0


def test_summarize_by_stage_includes_every_stage_present():
    traces, observations = _load()
    rows, _ = aggregate(traces, observations)
    by_stage = {row["stage"] for row in summarize_by_stage(rows)}
    assert {"embedding", "cache_lookup", "rerank", "llm"} <= by_stage


# --- slow_traces ------------------------------------------------------------


def test_slow_traces_returns_sorted_top_n():
    traces, observations = _load()
    rows, trace_total = aggregate(traces, observations)
    slow = slow_traces(rows, trace_total, n=3)
    assert len(slow) == 3
    # trace-rag-3 is the slowest in the fixture (22.5s).
    assert slow[0]["trace_id"] == "trace-rag-3"
    # Descending by total_ms.
    assert slow[0]["total_ms"] >= slow[1]["total_ms"] >= slow[2]["total_ms"]


def test_slow_traces_caps_at_n_even_when_fewer_traces_exist():
    traces, observations = _load()
    rows, trace_total = aggregate(traces, observations)
    slow = slow_traces(rows, trace_total, n=100)
    # Fixture has 5 traces.
    assert len(slow) == 5


# --- render_markdown --------------------------------------------------------


def test_render_markdown_has_expected_sections():
    traces, observations = _load()
    rows, trace_total = aggregate(traces, observations)
    md = render_markdown(
        rows,
        trace_total,
        source_label="fixture (test)",
        window_label="test",
        generated_at="2026-05-23 00:00 UTC",
    )
    for section in (
        "# Latency report",
        "## Aggregate",
        "## By task type",
        "## By pipeline stage",
        "## Slowest traces",
    ):
        assert section in md, f"missing section {section!r}"


def test_render_markdown_handles_empty_input():
    md = render_markdown(
        [],
        {},
        source_label="empty",
        window_label="none",
        generated_at="2026-05-23 00:00 UTC",
    )
    # All sections still render; numbers collapse to zero.
    assert "## Aggregate" in md
    assert "0 ms" in md
