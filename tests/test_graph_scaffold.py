"""Behavior contract for the LangGraph scaffold.

The scaffold has a single `rag` node that delegates to the existing
RAG callable. These tests exercise the graph with a stub RAG function
so we never touch the network, the LLM, or Milvus.

What we want to pin here:
- The graph compiles without error.
- It calls the rag function with the correct kwargs (mapped from
  state fields, including the `debug_flag` → `debug` rename).
- It surfaces the RAG output back into state under the renamed keys
  (`debug_info` instead of `debug`).
- Optional fields (`heuristics`) flow through cleanly.

Tests use `build_chat_graph(rag_fn=stub)` rather than the cached
`get_chat_graph()` so each test gets an isolated graph instance bound
to its own stub.
"""

from __future__ import annotations

from typing import Any

from src.agents.graph import ChatGraphState, build_chat_graph


def _stub_rag_result(**overrides: Any) -> dict[str, Any]:
    """Mirror the shape of `answer_question`'s return dict."""
    base: dict[str, Any] = {
        "answer": "Cedric was a Hufflepuff champion who was killed.",
        "t_milvus_start": 100.0,
        "t_milvus_end": 100.5,
        "t_llm_start": 100.6,
        "t_llm_end": 101.5,
        "debug": None,
        "heuristics": {
            "overall_passed": True,
            "refusal_check_passed": True,
            "citation_check_passed": True,
            "length_check_passed": True,
            "failed_checks": [],
        },
    }
    base.update(overrides)
    return base


def _baseline_state() -> ChatGraphState:
    return {
        "question": "Who is Cedric Diggory?",
        "query_vec": [0.1, 0.2, 0.3],
        "collection_name": "docs",
        "history": [],
        "debug_flag": False,
    }


def test_graph_compiles_and_invokes_with_stub():
    """Smoke test: the graph compiles and an invoke roundtrips."""
    graph = build_chat_graph(rag_fn=lambda **_: _stub_rag_result())
    final = graph.invoke(_baseline_state())
    assert final["answer"] == "Cedric was a Hufflepuff champion who was killed."
    assert final["t_milvus_start"] == 100.0
    assert final["t_llm_end"] == 101.5
    assert final["heuristics"]["overall_passed"] is True


def test_graph_forwards_debug_flag_to_rag_callable():
    """The `debug_flag` field in state must reach the RAG fn as `debug`."""
    seen: dict[str, Any] = {}

    def _capturing_rag(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return _stub_rag_result()

    graph = build_chat_graph(rag_fn=_capturing_rag)
    graph.invoke({**_baseline_state(), "debug_flag": True})
    assert seen["debug"] is True
    assert seen["question"] == "Who is Cedric Diggory?"
    assert seen["collection_name"] == "docs"


def test_graph_remaps_debug_output_to_debug_info():
    """RAG's `debug` payload becomes state's `debug_info` so it doesn't
    collide with the input-side `debug_flag` field.
    """
    debug_payload = {"retrieved_chunks": [{"text": "..."}], "timings_ms": {"total": 950}}
    graph = build_chat_graph(
        rag_fn=lambda **_: _stub_rag_result(debug=debug_payload)
    )
    final = graph.invoke({**_baseline_state(), "debug_flag": True})
    assert final["debug_info"] == debug_payload
    # And the rename is one-directional: the input flag stays in state too.
    assert final["debug_flag"] is True


def test_graph_propagates_heuristics_report():
    """The heuristic report must roundtrip — chat_service depends on it
    to tag the Langfuse trace at the root span.
    """
    failing_report = {
        "overall_passed": False,
        "refusal_check_passed": False,
        "citation_check_passed": True,
        "length_check_passed": True,
        "failed_checks": ["refusal"],
    }
    graph = build_chat_graph(
        rag_fn=lambda **_: _stub_rag_result(heuristics=failing_report)
    )
    final = graph.invoke(_baseline_state())
    assert final["heuristics"] == failing_report


def test_graph_tolerates_missing_optional_fields():
    """Some RAG return paths (e.g. cache hits in a future refactor) might
    omit `heuristics` entirely. The graph must not blow up.
    """
    minimal = {
        "answer": "ok",
        "t_milvus_start": 0.0,
        "t_milvus_end": 0.0,
        "t_llm_start": 0.0,
        "t_llm_end": 0.0,
    }
    graph = build_chat_graph(rag_fn=lambda **_: minimal)
    final = graph.invoke(_baseline_state())
    assert final["answer"] == "ok"
    assert final.get("heuristics") is None
    assert final.get("debug_info") is None
