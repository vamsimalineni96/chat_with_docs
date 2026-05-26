"""Behavior contract for the multi-node chat graph.

Every test stubs out the entire node set so we never touch the real
RAG pipeline, the LLM, or Milvus. The graph's job is *orchestration*
— routing, state threading, conditional edges — and that's what we
exercise here.

What we want to pin:
- The happy path runs all four nodes in order (retrieve → rerank →
  generate → postprocess) and END state has the expected keys.
- The conditional edge after `retrieve` skips rerank+generate when no
  chunks came back, going through `canned_no_retrieval` instead.
- Each node only sees the state fields it should: retrieve reads
  inputs, rerank reads `retrieved`, generate reads `top_chunks`,
  postprocess reads `answer`.
- `debug_flag` propagates through to wherever the real nodes would
  look for it.

Tests pass `*_fn=stub` per node to short-circuit. Defaults (which
lazy-import the langchain stack) are never exercised here — that's
intentional, so this file can collect cleanly in the minimal CI env.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.agents.graph import (
    CANNED_NO_RETRIEVAL_ANSWER,
    CANNED_OUT_OF_SCOPE_ANSWER,
    ChatGraphState,
    build_chat_graph,
)

# ---------------------------------------------------------------------------
# Stub nodes
# ---------------------------------------------------------------------------


def _stub_classify_in_corpus(state: ChatGraphState) -> dict[str, Any]:
    """Default classifier stub — routes everything to the RAG path."""
    return {"intent": "in_corpus", "intent_reasoning": "stub"}


def _stub_classify_out_of_scope(state: ChatGraphState) -> dict[str, Any]:
    return {"intent": "out_of_scope", "intent_reasoning": "stub out of scope"}


def _stub_classify_tool_call(state: ChatGraphState) -> dict[str, Any]:
    return {"intent": "tool_call", "intent_reasoning": "stub tool call"}


def _stub_call_mcp_tool(state: ChatGraphState) -> dict[str, Any]:
    """Stand-in for the MCP ReAct sub-agent.

    Emits the same shape as the real `_default_call_mcp_tool_node`:
    an answer string, a tool_calls record, zeroed milvus timings,
    and `retrieved=[]` so postprocess's citation check has something
    to read.
    """
    return {
        "answer": "Order ORD-1001 is shipped via FedEx, ETA 2026-05-27.",
        "tool_calls": [
            {"name": "get_order_status", "args": {"order_id": "ORD-1001"}}
        ],
        "t_milvus_start": 0.0,
        "t_milvus_end": 0.0,
        "t_llm_start": 200.0,
        "t_llm_end": 201.0,
        "retrieved": [],
    }


def _stub_retrieve(state: ChatGraphState) -> dict[str, Any]:
    """Return one matching chunk so the conditional routes to rerank."""
    return {
        "retrieved": [{"text": "Cedric Diggory was a Hufflepuff champion.", "id": "c1"}],
        "t_milvus_start": 100.0,
        "t_milvus_end": 100.5,
        "debug_info": {} if state.get("debug_flag") else None,
    }


def _stub_retrieve_empty(state: ChatGraphState) -> dict[str, Any]:
    """Empty retrieval — the conditional should route to canned_no_retrieval."""
    return {
        "retrieved": [],
        "t_milvus_start": 100.0,
        "t_milvus_end": 100.5,
        "debug_info": {} if state.get("debug_flag") else None,
    }


def _stub_rerank(state: ChatGraphState) -> dict[str, Any]:
    return {
        "top_chunks": state["retrieved"][:1],
        "t_rerank_start": 100.5,
        "t_rerank_end": 100.6,
    }


def _stub_generate(state: ChatGraphState) -> dict[str, Any]:
    return {
        "answer": "Cedric was a Hufflepuff champion who was killed.",
        "t_llm_start": 100.6,
        "t_llm_end": 101.5,
    }


def _stub_postprocess(state: ChatGraphState) -> dict[str, Any]:
    """Trivially marks the answer as passing — orchestration test, not the
    real heuristic logic.
    """
    return {
        "heuristics": {
            "overall_passed": True,
            "refusal_check_passed": True,
            "citation_check_passed": True,
            "length_check_passed": True,
            "failed_checks": [],
        }
    }


def _all_stubs() -> dict[str, Any]:
    return {
        "classify_intent_fn": _stub_classify_in_corpus,
        "retrieve_fn": _stub_retrieve,
        "rerank_fn": _stub_rerank,
        "generate_fn": _stub_generate,
        "postprocess_fn": _stub_postprocess,
    }


def _baseline_state() -> ChatGraphState:
    return {
        "question": "Who is Cedric Diggory?",
        "query_vec": [0.1, 0.2, 0.3],
        "collection_name": "docs",
        "history": [],
        "debug_flag": False,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_runs_all_five_nodes():
    """classify → retrieve → rerank → generate → postprocess; END state has full shape."""
    graph = build_chat_graph(**_all_stubs())
    final = asyncio.run(graph.ainvoke(_baseline_state()))

    assert final["intent"] == "in_corpus"
    assert final["retrieved"] == [
        {"text": "Cedric Diggory was a Hufflepuff champion.", "id": "c1"}
    ]
    assert final["top_chunks"] == final["retrieved"]
    assert final["answer"] == "Cedric was a Hufflepuff champion who was killed."
    assert final["t_milvus_end"] == 100.5
    assert final["t_llm_end"] == 101.5
    assert final["heuristics"]["overall_passed"] is True


def test_node_execution_order():
    """Full in-corpus path: classify → retrieve → rerank → generate → postprocess."""
    visited: list[str] = []

    def _track(name: str, stub):
        def _wrapped(state: ChatGraphState) -> dict[str, Any]:
            visited.append(name)
            return stub(state)

        return _wrapped

    graph = build_chat_graph(
        classify_intent_fn=_track("classify_intent", _stub_classify_in_corpus),
        retrieve_fn=_track("retrieve", _stub_retrieve),
        rerank_fn=_track("rerank", _stub_rerank),
        generate_fn=_track("generate", _stub_generate),
        postprocess_fn=_track("postprocess", _stub_postprocess),
    )
    asyncio.run(graph.ainvoke(_baseline_state()))
    assert visited == [
        "classify_intent",
        "retrieve",
        "rerank",
        "generate",
        "postprocess",
    ]


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def test_empty_retrieval_routes_to_canned_no_retrieval_path():
    """No chunks back from retrieve → skip rerank + generate, hit
    canned_no_retrieval. The classifier itself said in_corpus (the
    common case where the question *looks* like it should be in the
    docs but retrieval came up empty).
    """
    rerank_called: list[bool] = []
    generate_called: list[bool] = []

    def _rerank_should_not_run(state: ChatGraphState) -> dict[str, Any]:
        rerank_called.append(True)
        return _stub_rerank(state)

    def _generate_should_not_run(state: ChatGraphState) -> dict[str, Any]:
        generate_called.append(True)
        return _stub_generate(state)

    graph = build_chat_graph(
        classify_intent_fn=_stub_classify_in_corpus,
        retrieve_fn=_stub_retrieve_empty,
        rerank_fn=_rerank_should_not_run,
        generate_fn=_generate_should_not_run,
        postprocess_fn=_stub_postprocess,
    )
    final = asyncio.run(graph.ainvoke(_baseline_state()))

    assert rerank_called == []
    assert generate_called == []
    assert final["answer"] == CANNED_NO_RETRIEVAL_ANSWER
    # The canned path reuses milvus timings for the LLM slot so
    # chat_service's metrics log doesn't need to special-case it.
    assert final["t_llm_start"] == 100.0
    assert final["t_llm_end"] == 100.5
    assert final["heuristics"]["overall_passed"] is True


def test_out_of_scope_classifier_short_circuits_before_retrieve():
    """Classifier returns out_of_scope → skip retrieve / rerank /
    generate entirely. Saves the full RAG cost on questions we
    know are bogus.
    """
    retrieve_called: list[bool] = []
    rerank_called: list[bool] = []
    generate_called: list[bool] = []

    def _retrieve_should_not_run(state: ChatGraphState) -> dict[str, Any]:
        retrieve_called.append(True)
        return _stub_retrieve(state)

    def _rerank_should_not_run(state: ChatGraphState) -> dict[str, Any]:
        rerank_called.append(True)
        return _stub_rerank(state)

    def _generate_should_not_run(state: ChatGraphState) -> dict[str, Any]:
        generate_called.append(True)
        return _stub_generate(state)

    graph = build_chat_graph(
        classify_intent_fn=_stub_classify_out_of_scope,
        retrieve_fn=_retrieve_should_not_run,
        rerank_fn=_rerank_should_not_run,
        generate_fn=_generate_should_not_run,
        postprocess_fn=_stub_postprocess,
    )
    final = asyncio.run(graph.ainvoke(_baseline_state()))

    assert retrieve_called == []
    assert rerank_called == []
    assert generate_called == []
    assert final["intent"] == "out_of_scope"
    assert final["answer"] == CANNED_OUT_OF_SCOPE_ANSWER
    # Timings stamped to 0 for the short-circuit so chat_service's
    # metrics log shows the cost saving.
    assert final["t_milvus_end"] == 0.0
    assert final["t_llm_end"] == 0.0
    # Postprocess STILL runs (heuristics on every answer that leaves
    # the graph — same invariant as the canned_no_retrieval path).
    assert final["heuristics"]["overall_passed"] is True


def test_tool_call_intent_routes_to_mcp_tool_path():
    """Classifier returns tool_call → skip retrieve / rerank / generate
    entirely and run the MCP ReAct sub-agent stub. Postprocess still
    runs heuristics on whatever answer the sub-agent produced.
    """
    retrieve_called: list[bool] = []
    rerank_called: list[bool] = []
    generate_called: list[bool] = []

    def _retrieve_should_not_run(state: ChatGraphState) -> dict[str, Any]:
        retrieve_called.append(True)
        return _stub_retrieve(state)

    def _rerank_should_not_run(state: ChatGraphState) -> dict[str, Any]:
        rerank_called.append(True)
        return _stub_rerank(state)

    def _generate_should_not_run(state: ChatGraphState) -> dict[str, Any]:
        generate_called.append(True)
        return _stub_generate(state)

    graph = build_chat_graph(
        classify_intent_fn=_stub_classify_tool_call,
        retrieve_fn=_retrieve_should_not_run,
        rerank_fn=_rerank_should_not_run,
        generate_fn=_generate_should_not_run,
        call_mcp_tool_fn=_stub_call_mcp_tool,
        postprocess_fn=_stub_postprocess,
    )
    final = asyncio.run(graph.ainvoke(_baseline_state()))

    assert retrieve_called == []
    assert rerank_called == []
    assert generate_called == []
    assert final["intent"] == "tool_call"
    assert final["answer"].startswith("Order ORD-1001 is shipped")
    assert final["tool_calls"] == [
        {"name": "get_order_status", "args": {"order_id": "ORD-1001"}}
    ]
    # Postprocess still ran — the heuristic invariant holds on every
    # terminal path, MCP branch included.
    assert final["heuristics"]["overall_passed"] is True


def test_intent_unknown_falls_through_to_rag():
    """Anything other than the explicit "out_of_scope" verdict routes
    to RAG. This guards against a bad future intent value (e.g. an
    unrecognised category) silently breaking the request — the safe
    bias is to run RAG and let the refusal heuristic catch it
    downstream if the answer turns out bad.
    """

    def _classify_weird(state: ChatGraphState) -> dict[str, Any]:
        return {"intent": "unknown_category", "intent_reasoning": "stub"}

    graph = build_chat_graph(
        classify_intent_fn=_classify_weird,
        retrieve_fn=_stub_retrieve,
        rerank_fn=_stub_rerank,
        generate_fn=_stub_generate,
        postprocess_fn=_stub_postprocess,
    )
    final = asyncio.run(graph.ainvoke(_baseline_state()))
    # The generate node ran; we got a real answer, not the canned one.
    assert final["answer"] == "Cedric was a Hufflepuff champion who was killed."


def test_all_four_terminal_paths_run_postprocess():
    """Sanity: heuristics fire on every answer leaving the graph,
    regardless of which branch produced it.
    """
    for stubs in (
        # full RAG path
        _all_stubs(),
        # retrieve-empty path
        {**_all_stubs(), "retrieve_fn": _stub_retrieve_empty},
        # out-of-scope path
        {**_all_stubs(), "classify_intent_fn": _stub_classify_out_of_scope},
        # tool_call path (added in PR #5)
        {
            **_all_stubs(),
            "classify_intent_fn": _stub_classify_tool_call,
            "call_mcp_tool_fn": _stub_call_mcp_tool,
        },
    ):
        postprocess_called: list[bool] = []

        # Default-arg binding instead of free-var closure capture — the
        # for-loop body would otherwise re-bind `postprocess_called` on
        # each iteration and ruff B023 (rightly) flags it as a latent bug.
        def _track(state, _called=postprocess_called):
            _called.append(True)
            return _stub_postprocess(state)

        graph = build_chat_graph(**{**stubs, "postprocess_fn": _track})
        asyncio.run(graph.ainvoke(_baseline_state()))
        assert postprocess_called == [True]


# ---------------------------------------------------------------------------
# State threading
# ---------------------------------------------------------------------------


def test_debug_flag_propagates_through_to_debug_info():
    """When debug_flag=True, retrieve initializes debug_info={}, and downstream
    nodes can write into it. End-state debug_info contains entries that the
    nodes added along the way.
    """

    def _retrieve_writing_debug(state: ChatGraphState) -> dict[str, Any]:
        debug_info = {} if state.get("debug_flag") else None
        if debug_info is not None:
            debug_info["retrieved_chunks"] = ["chunk_a"]
        return {
            "retrieved": [{"text": "...", "id": "c1"}],
            "t_milvus_start": 100.0,
            "t_milvus_end": 100.5,
            "debug_info": debug_info,
        }

    def _rerank_writing_debug(state: ChatGraphState) -> dict[str, Any]:
        if state.get("debug_info") is not None:
            state["debug_info"]["reranked_top_k"] = ["chunk_a"]
        return {
            "top_chunks": state["retrieved"][:1],
            "t_rerank_start": 100.5,
            "t_rerank_end": 100.6,
        }

    graph = build_chat_graph(
        classify_intent_fn=_stub_classify_in_corpus,
        retrieve_fn=_retrieve_writing_debug,
        rerank_fn=_rerank_writing_debug,
        generate_fn=_stub_generate,
        postprocess_fn=_stub_postprocess,
    )
    final = asyncio.run(graph.ainvoke({**_baseline_state(), "debug_flag": True}))

    assert final["debug_info"] is not None
    assert final["debug_info"]["retrieved_chunks"] == ["chunk_a"]
    assert final["debug_info"]["reranked_top_k"] == ["chunk_a"]


def test_top_chunks_threads_from_rerank_to_generate():
    """generate must see what rerank produced, not the pre-rerank retrieved
    set. We assert by mutation: rerank emits a marker; generate reads it.
    """
    sentinel = [{"text": "post-rerank marker", "id": "marker"}]

    def _rerank_emits_marker(state: ChatGraphState) -> dict[str, Any]:
        return {
            "top_chunks": sentinel,
            "t_rerank_start": 100.5,
            "t_rerank_end": 100.6,
        }

    seen: list[Any] = []

    def _generate_capturing_top_chunks(state: ChatGraphState) -> dict[str, Any]:
        seen.append(state["top_chunks"])
        return _stub_generate(state)

    graph = build_chat_graph(
        classify_intent_fn=_stub_classify_in_corpus,
        retrieve_fn=_stub_retrieve,
        rerank_fn=_rerank_emits_marker,
        generate_fn=_generate_capturing_top_chunks,
        postprocess_fn=_stub_postprocess,
    )
    asyncio.run(graph.ainvoke(_baseline_state()))
    assert seen == [sentinel]
