"""Behavior contract for the multi-agent chat graph.

Every test stubs out the entire node set so we never touch the real
RAG pipeline, the LLM, or Milvus. The graph's job is *orchestration*
— routing, state threading, parallel execution — and that's what we
exercise here.

New topology (feature/multi_agent):

    supervisor → research | action | parallel_agents | canned_out_of_scope
    parallel_agents → aggregate
    all paths → postprocess → END
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.agents.graph import (
    CANNED_OUT_OF_SCOPE_ANSWER,
    ChatGraphState,
    build_chat_graph,
)

# ---------------------------------------------------------------------------
# Supervisor stubs — control which path the graph takes
# ---------------------------------------------------------------------------

def _stub_supervisor_research(state: ChatGraphState) -> dict[str, Any]:
    return {
        "supervisor_plan": "research",
        "supervisor_reasoning": "stub: needs documents",
        "intent": "research",
        "intent_reasoning": "stub",
        "debug_info": {} if state.get("debug_flag") else None,
    }


def _stub_supervisor_action(state: ChatGraphState) -> dict[str, Any]:
    return {
        "supervisor_plan": "action",
        "supervisor_reasoning": "stub: needs live tools",
        "intent": "tool_call",
        "intent_reasoning": "stub",
        "debug_info": None,
    }


def _stub_supervisor_both(state: ChatGraphState) -> dict[str, Any]:
    return {
        "supervisor_plan": "both",
        "supervisor_reasoning": "stub: needs both agents",
        "intent": "both",
        "intent_reasoning": "stub",
        "debug_info": None,
    }


def _stub_supervisor_out_of_scope(state: ChatGraphState) -> dict[str, Any]:
    return {
        "supervisor_plan": "out_of_scope",
        "supervisor_reasoning": "stub: unrelated question",
        "intent": "out_of_scope",
        "intent_reasoning": "stub",
        "debug_info": None,
    }


# ---------------------------------------------------------------------------
# Agent stubs
# ---------------------------------------------------------------------------

def _stub_research(state: ChatGraphState) -> dict[str, Any]:
    return {
        "answer": "Cedric Diggory was a Hufflepuff champion.",
        "retrieved": [{"text": "Cedric Diggory was a Hufflepuff champion.", "id": "c1"}],
        "t_milvus_start": 100.0,
        "t_milvus_end": 100.5,
        "t_llm_start": 100.6,
        "t_llm_end": 101.5,
        "tool_calls": [],
        "tool_failure_reason": None,
    }


def _stub_action(state: ChatGraphState) -> dict[str, Any]:
    return {
        "answer": "Order ORD-1001 is shipped via FedEx.",
        "tool_calls": [{"name": "get_order_status", "args": {"order_id": "ORD-1001"}}],
        "tool_failure_reason": None,
        "t_milvus_start": 0.0,
        "t_milvus_end": 0.0,
        "t_llm_start": 200.0,
        "t_llm_end": 201.0,
        "retrieved": [],
    }


def _stub_parallel_agents(state: ChatGraphState) -> dict[str, Any]:
    return {
        "research_answer": "Wireless Headphones has 42 units in stock.",
        "action_answer": "Invoice in_xxx created for Jane Smith ($149.99).",
        "retrieved": [{"text": "Wireless Headphones stock info.", "id": "c2"}],
        "tool_calls": [{"name": "create_invoice", "args": {}}],
        "tool_failure_reason": None,
        "t_milvus_start": 100.0,
        "t_milvus_end": 100.5,
        "t_llm_start": 100.6,
        "t_llm_end": 201.0,
    }


def _stub_aggregate(state: ChatGraphState) -> dict[str, Any]:
    return {
        "answer": (
            "Wireless Headphones has 42 units in stock. "
            "I've also created invoice in_xxx for Jane Smith at $149.99."
        ),
        "t_llm_start": 201.0,
        "t_llm_end": 202.0,
    }


def _stub_postprocess(state: ChatGraphState) -> dict[str, Any]:
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
        "supervisor_fn": _stub_supervisor_research,
        "research_fn": _stub_research,
        "action_fn": _stub_action,
        "parallel_agents_fn": _stub_parallel_agents,
        "aggregate_fn": _stub_aggregate,
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
# Happy path — research plan
# ---------------------------------------------------------------------------


def test_happy_path_runs_all_five_nodes():
    """supervisor → research → postprocess; END state has expected shape."""
    graph = build_chat_graph(**_all_stubs())
    final = asyncio.run(graph.ainvoke(_baseline_state()))

    assert final["supervisor_plan"] == "research"
    assert final["retrieved"] == [
        {"text": "Cedric Diggory was a Hufflepuff champion.", "id": "c1"}
    ]
    assert final["answer"] == "Cedric Diggory was a Hufflepuff champion."
    assert final["t_milvus_end"] == 100.5
    assert final["t_llm_end"] == 101.5
    assert final["heuristics"]["overall_passed"] is True


def test_node_execution_order():
    """Research path: supervisor → research → postprocess."""
    visited: list[str] = []

    def _track(name: str, stub):
        def _wrapped(state: ChatGraphState) -> dict[str, Any]:
            visited.append(name)
            return stub(state)
        return _wrapped

    graph = build_chat_graph(
        supervisor_fn=_track("supervisor", _stub_supervisor_research),
        research_fn=_track("research", _stub_research),
        action_fn=_track("action", _stub_action),
        parallel_agents_fn=_track("parallel_agents", _stub_parallel_agents),
        aggregate_fn=_track("aggregate", _stub_aggregate),
        postprocess_fn=_track("postprocess", _stub_postprocess),
    )
    asyncio.run(graph.ainvoke(_baseline_state()))
    assert visited == ["supervisor", "research", "postprocess"]


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


def test_out_of_scope_supervisor_short_circuits():
    """supervisor=out_of_scope → skip all agents, return canned answer."""
    research_called: list[bool] = []
    action_called: list[bool] = []

    def _research_should_not_run(state):
        research_called.append(True)
        return _stub_research(state)

    def _action_should_not_run(state):
        action_called.append(True)
        return _stub_action(state)

    graph = build_chat_graph(
        supervisor_fn=_stub_supervisor_out_of_scope,
        research_fn=_research_should_not_run,
        action_fn=_action_should_not_run,
        parallel_agents_fn=_stub_parallel_agents,
        aggregate_fn=_stub_aggregate,
        postprocess_fn=_stub_postprocess,
    )
    final = asyncio.run(graph.ainvoke(_baseline_state()))

    assert research_called == []
    assert action_called == []
    assert final["supervisor_plan"] == "out_of_scope"
    assert final["answer"] == CANNED_OUT_OF_SCOPE_ANSWER
    assert final["t_milvus_end"] == 0.0
    assert final["t_llm_end"] == 0.0
    assert final["heuristics"]["overall_passed"] is True


def test_action_plan_routes_to_action_agent():
    """supervisor=action → skip research, run action agent only."""
    research_called: list[bool] = []

    def _research_should_not_run(state):
        research_called.append(True)
        return _stub_research(state)

    graph = build_chat_graph(
        supervisor_fn=_stub_supervisor_action,
        research_fn=_research_should_not_run,
        action_fn=_stub_action,
        parallel_agents_fn=_stub_parallel_agents,
        aggregate_fn=_stub_aggregate,
        postprocess_fn=_stub_postprocess,
    )
    final = asyncio.run(graph.ainvoke(_baseline_state()))

    assert research_called == []
    assert final["supervisor_plan"] == "action"
    assert final["answer"].startswith("Order ORD-1001 is shipped")
    assert final["tool_calls"] == [
        {"name": "get_order_status", "args": {"order_id": "ORD-1001"}}
    ]
    assert final["heuristics"]["overall_passed"] is True


def test_tool_call_intent_routes_to_mcp_tool_path():
    """Alias test: action plan sets intent=tool_call for Langfuse tagging."""
    graph = build_chat_graph(**{**_all_stubs(), "supervisor_fn": _stub_supervisor_action})
    final = asyncio.run(graph.ainvoke(_baseline_state()))
    assert final["intent"] == "tool_call"


def test_intent_unknown_falls_through_to_rag():
    """Unknown supervisor_plan defaults to research path."""

    def _classify_weird(state: ChatGraphState) -> dict[str, Any]:
        return {
            "supervisor_plan": "unknown_category",
            "supervisor_reasoning": "stub",
            "intent": "unknown_category",
            "intent_reasoning": "stub",
            "debug_info": None,
        }

    graph = build_chat_graph(
        **{**_all_stubs(), "supervisor_fn": _classify_weird}
    )
    final = asyncio.run(graph.ainvoke(_baseline_state()))
    assert final["answer"] == "Cedric Diggory was a Hufflepuff champion."


# ---------------------------------------------------------------------------
# "both" plan — parallel execution + aggregation
# ---------------------------------------------------------------------------


def test_both_plan_runs_parallel_agents_and_aggregate():
    """supervisor=both → parallel_agents → aggregate → postprocess."""
    visited: list[str] = []

    def _track(name, stub):
        def _w(state):
            visited.append(name)
            return stub(state)
        return _w

    graph = build_chat_graph(
        supervisor_fn=_track("supervisor", _stub_supervisor_both),
        research_fn=_track("research", _stub_research),
        action_fn=_track("action", _stub_action),
        parallel_agents_fn=_track("parallel_agents", _stub_parallel_agents),
        aggregate_fn=_track("aggregate", _stub_aggregate),
        postprocess_fn=_track("postprocess", _stub_postprocess),
    )
    asyncio.run(graph.ainvoke(_baseline_state()))
    assert visited == ["supervisor", "parallel_agents", "aggregate", "postprocess"]


def test_both_plan_state_has_both_answers():
    """After parallel_agents, state carries research_answer AND action_answer."""
    graph = build_chat_graph(**{**_all_stubs(), "supervisor_fn": _stub_supervisor_both})
    final = asyncio.run(graph.ainvoke(_baseline_state()))

    assert final["research_answer"] == "Wireless Headphones has 42 units in stock."
    assert final["action_answer"] == "Invoice in_xxx created for Jane Smith ($149.99)."
    assert "42 units" in final["answer"]
    assert "in_xxx" in final["answer"]


def test_both_plan_does_not_run_single_agent_nodes():
    """When plan=both, the individual research and action nodes don't run."""
    research_called: list[bool] = []
    action_called: list[bool] = []

    def _research_should_not_run(state):
        research_called.append(True)
        return _stub_research(state)

    def _action_should_not_run(state):
        action_called.append(True)
        return _stub_action(state)

    graph = build_chat_graph(
        supervisor_fn=_stub_supervisor_both,
        research_fn=_research_should_not_run,
        action_fn=_action_should_not_run,
        parallel_agents_fn=_stub_parallel_agents,
        aggregate_fn=_stub_aggregate,
        postprocess_fn=_stub_postprocess,
    )
    asyncio.run(graph.ainvoke(_baseline_state()))
    assert research_called == []
    assert action_called == []


# ---------------------------------------------------------------------------
# Postprocess invariant — runs on every terminal path
# ---------------------------------------------------------------------------


def test_all_paths_run_postprocess():
    """Heuristics fire on every answer leaving the graph."""
    for supervisor_stub, _extra in [
        (_stub_supervisor_research, {}),
        (_stub_supervisor_action, {}),
        (_stub_supervisor_both, {}),
        (_stub_supervisor_out_of_scope, {}),
    ]:
        postprocess_called: list[bool] = []

        def _track(state, _called=postprocess_called):
            _called.append(True)
            return _stub_postprocess(state)

        graph = build_chat_graph(
            **{**_all_stubs(), "supervisor_fn": supervisor_stub, "postprocess_fn": _track}
        )
        asyncio.run(graph.ainvoke(_baseline_state()))
        assert postprocess_called == [True], f"postprocess not called for {supervisor_stub.__name__}"


# ---------------------------------------------------------------------------
# State threading
# ---------------------------------------------------------------------------


def test_debug_flag_propagates_through_to_debug_info():
    """When debug_flag=True, supervisor initializes debug_info={}."""

    def _supervisor_with_debug(state: ChatGraphState) -> dict[str, Any]:
        return {
            "supervisor_plan": "research",
            "supervisor_reasoning": "stub",
            "intent": "research",
            "intent_reasoning": "stub",
            "debug_info": {} if state.get("debug_flag") else None,
        }

    def _research_writing_debug(state: ChatGraphState) -> dict[str, Any]:
        result = _stub_research(state)
        if state.get("debug_info") is not None:
            state["debug_info"]["retrieved_chunks"] = ["chunk_a"]
        return result

    graph = build_chat_graph(
        supervisor_fn=_supervisor_with_debug,
        research_fn=_research_writing_debug,
        action_fn=_stub_action,
        parallel_agents_fn=_stub_parallel_agents,
        aggregate_fn=_stub_aggregate,
        postprocess_fn=_stub_postprocess,
    )
    final = asyncio.run(graph.ainvoke({**_baseline_state(), "debug_flag": True}))
    assert final["debug_info"] is not None
    assert final["debug_info"]["retrieved_chunks"] == ["chunk_a"]


def test_top_chunks_threads_from_rerank_to_generate():
    """Research node receives query_vec and collection_name from state."""
    seen_question: list[str] = []

    def _research_capturing_state(state: ChatGraphState) -> dict[str, Any]:
        seen_question.append(state["question"])
        return _stub_research(state)

    graph = build_chat_graph(
        **{**_all_stubs(), "research_fn": _research_capturing_state}
    )
    asyncio.run(graph.ainvoke(_baseline_state()))
    assert seen_question == ["Who is Cedric Diggory?"]
