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

from typing import Any

from src.agents.graph import ChatGraphState, build_chat_graph

# ---------------------------------------------------------------------------
# Stub nodes
# ---------------------------------------------------------------------------


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


def test_happy_path_runs_all_four_nodes():
    """retrieve → rerank → generate → postprocess; END state has full shape."""
    graph = build_chat_graph(**_all_stubs())
    final = graph.invoke(_baseline_state())

    assert final["retrieved"] == [
        {"text": "Cedric Diggory was a Hufflepuff champion.", "id": "c1"}
    ]
    assert final["top_chunks"] == final["retrieved"]
    assert final["answer"] == "Cedric was a Hufflepuff champion who was killed."
    assert final["t_milvus_end"] == 100.5
    assert final["t_llm_end"] == 101.5
    assert final["heuristics"]["overall_passed"] is True


def test_node_execution_order():
    """The conditional path is retrieve → rerank → generate → postprocess.

    We track visits via a sentinel list so we can assert *order*, not
    just *which* nodes ran.
    """
    visited: list[str] = []

    def _track(name: str, stub):
        def _wrapped(state: ChatGraphState) -> dict[str, Any]:
            visited.append(name)
            return stub(state)

        return _wrapped

    graph = build_chat_graph(
        retrieve_fn=_track("retrieve", _stub_retrieve),
        rerank_fn=_track("rerank", _stub_rerank),
        generate_fn=_track("generate", _stub_generate),
        postprocess_fn=_track("postprocess", _stub_postprocess),
    )
    graph.invoke(_baseline_state())
    assert visited == ["retrieve", "rerank", "generate", "postprocess"]


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def test_empty_retrieval_routes_to_canned_path():
    """No chunks → skip rerank + generate, go through canned_no_retrieval.

    The canned answer must come from the real CANNED_NO_RETRIEVAL_ANSWER
    constant in rag_pipeline.py — we don't stub canned_no_retrieval so
    the default fires.
    """
    from src.utils.rag_pipeline import CANNED_NO_RETRIEVAL_ANSWER

    rerank_called: list[bool] = []
    generate_called: list[bool] = []

    def _rerank_should_not_run(state: ChatGraphState) -> dict[str, Any]:
        rerank_called.append(True)
        return _stub_rerank(state)

    def _generate_should_not_run(state: ChatGraphState) -> dict[str, Any]:
        generate_called.append(True)
        return _stub_generate(state)

    graph = build_chat_graph(
        retrieve_fn=_stub_retrieve_empty,
        rerank_fn=_rerank_should_not_run,
        generate_fn=_generate_should_not_run,
        postprocess_fn=_stub_postprocess,
    )
    final = graph.invoke(_baseline_state())

    assert rerank_called == []
    assert generate_called == []
    assert final["answer"] == CANNED_NO_RETRIEVAL_ANSWER
    # The canned path reuses milvus timings for the LLM slot so
    # chat_service's metrics log doesn't need to special-case it.
    assert final["t_llm_start"] == 100.0
    assert final["t_llm_end"] == 100.5
    # Postprocess still ran (the heuristic check on the canned response is
    # what makes the refusal show up as `heuristic_failed:refusal` in
    # Langfuse).
    assert final["heuristics"]["overall_passed"] is True


def test_canned_path_also_runs_postprocess():
    """Both branches converge on postprocess. Verify with a sentinel."""
    postprocess_called: list[bool] = []

    def _postprocess_with_sentinel(state: ChatGraphState) -> dict[str, Any]:
        postprocess_called.append(True)
        return _stub_postprocess(state)

    graph = build_chat_graph(
        retrieve_fn=_stub_retrieve_empty,
        postprocess_fn=_postprocess_with_sentinel,
    )
    graph.invoke(_baseline_state())
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
        retrieve_fn=_retrieve_writing_debug,
        rerank_fn=_rerank_writing_debug,
        generate_fn=_stub_generate,
        postprocess_fn=_stub_postprocess,
    )
    final = graph.invoke({**_baseline_state(), "debug_flag": True})

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
        retrieve_fn=_stub_retrieve,
        rerank_fn=_rerank_emits_marker,
        generate_fn=_generate_capturing_top_chunks,
        postprocess_fn=_stub_postprocess,
    )
    graph.invoke(_baseline_state())
    assert seen == [sentinel]
