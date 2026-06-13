"""Multi-agent LangGraph state machine for the chat pipeline.

Graph topology:

                       START
                         │
                         ▼
                      supervisor          (LLM: 4 plans)
                         │
          ┌──────────────┼──────────────┬──────────────┐
          │              │              │               │
          ▼              ▼              ▼               ▼
       research        action         both         out_of_scope
      (RAG agent)   (ReAct MCP)   (parallel)      (canned)
          │              │              │               │
          │              │              ▼               │
          │              │           aggregate          │
          │              │              │               │
          └──────────────┴──────────────┴───────────────┘
                                   │
                                   ▼
                              postprocess
                                   │
                                   ▼
                                  END

Supervisor plans:
  research     → research_node runs the full RAG pipeline
  action       → action_node runs the MCP ReAct agent
  both         → parallel_agents_node runs both concurrently via
                 asyncio.gather, then aggregate_node synthesises
  out_of_scope → canned_out_of_scope short-circuits

Each default node lazy-imports its implementation to keep this module
loadable in the minimal CI test environment.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.utils.errors import NodeError
from src.utils.observability import observe

logger = logging.getLogger(__name__)

CANNED_OUT_OF_SCOPE_ANSWER = (
    "That question looks like it's outside the scope of the documents "
    "I've been given. I can only answer things grounded in the indexed "
    "material — try rephrasing in that direction, or upload a document "
    "that covers the topic."
)


class ChatGraphState(TypedDict, total=False):
    """State threaded through every node."""

    # --- inputs ---------------------------------------------------------------
    question: str
    query_vec: list[float]
    collection_name: str
    history: list[dict[str, Any]]
    debug_flag: bool

    # --- debug payload --------------------------------------------------------
    debug_info: dict[str, Any] | None

    # --- written by supervisor ------------------------------------------------
    supervisor_plan: str       # "research" | "action" | "both" | "out_of_scope"
    supervisor_reasoning: str

    # kept for backwards-compat with chat_service.py tag logic
    intent: str
    intent_reasoning: str

    # --- written by research_node / parallel_agents_node ---------------------
    retrieved: list[dict[str, Any]]
    t_milvus_start: float
    t_milvus_end: float

    # --- written by action_node / parallel_agents_node -----------------------
    tool_calls: list[dict[str, Any]]
    tool_failure_reason: str | None

    # --- written by research_node, action_node, or aggregate_node -----------
    answer: str
    t_llm_start: float
    t_llm_end: float

    # --- intermediate answers for the "both" path ----------------------------
    research_answer: str | None
    action_answer: str | None

    # --- written by postprocess ----------------------------------------------
    heuristics: dict[str, Any] | None

    # --- human-in-the-loop ---------------------------------------------------
    # Set by action_node when a destructive tool (e.g. create_refund) is about
    # to be called. The graph pauses and surfaces this to the user for approval.
    # Shape: {"tool": str, "args": dict, "display": str} | None
    pending_approval: dict[str, Any] | None
    # Set by the /approve endpoint before graph resumes: "approved" | "rejected"
    approval_decision: str | None


NodeFn = (
    Callable[[ChatGraphState], dict[str, Any]]
    | Callable[[ChatGraphState], Awaitable[dict[str, Any]]]
)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def _default_supervisor_node(state: ChatGraphState) -> dict[str, Any]:
    from src.agents.supervisor import run_supervisor  # noqa: PLC0415

    try:
        result = run_supervisor(
            state["question"],
            history=state.get("history", []),
        )
    except Exception as e:
        raise NodeError("supervisor", e) from e

    plan = result.plan
    return {
        "supervisor_plan": plan,
        "supervisor_reasoning": result.reasoning,
        # mirror into intent/intent_reasoning so chat_service tagging works
        "intent": "tool_call" if plan == "action" else plan,
        "intent_reasoning": result.reasoning,
        "debug_info": {} if state.get("debug_flag") else None,
    }


async def _default_research_node(state: ChatGraphState) -> dict[str, Any]:
    from src.agents.research_agent import run_research_agent  # noqa: PLC0415

    result = await run_research_agent(
        question=state["question"],
        query_vec=state["query_vec"],
        collection_name=state["collection_name"],
        history=state.get("history", []),
        debug_info=state.get("debug_info"),
    )
    return {
        "answer": result["answer"],
        "retrieved": result["retrieved"],
        "t_milvus_start": result["t_milvus_start"],
        "t_milvus_end": result["t_milvus_end"],
        "t_llm_start": result["t_llm_start"],
        "t_llm_end": result["t_llm_end"],
        "tool_calls": [],
        "tool_failure_reason": None,
    }


async def _default_action_node(state: ChatGraphState) -> dict[str, Any]:
    from src.agents.tool_node import run_tool_agent  # noqa: PLC0415

    # If a previous approval_gate pass set approval_decision, inject it so
    # run_tool_agent can skip the confirmation step and call the tool directly.
    result = await run_tool_agent(
        state["question"],
        history=state.get("history", []),
        approval_decision=state.get("approval_decision"),
    )
    return {
        "answer": result["answer"],
        "tool_calls": result.get("tool_calls", []),
        "tool_failure_reason": result.get("tool_failure_reason"),
        "pending_approval": result.get("pending_approval"),
        "t_milvus_start": 0.0,
        "t_milvus_end": 0.0,
        "t_llm_start": result.get("t_llm_start", 0.0),
        "t_llm_end": result.get("t_llm_end", 0.0),
        "retrieved": [],
    }


async def _default_parallel_agents_node(state: ChatGraphState) -> dict[str, Any]:
    """Run research + action agents concurrently, store both answers.

    asyncio.gather fires both agents simultaneously — the total wall time
    is max(research_time, action_time) instead of the sum.
    """
    import asyncio  # noqa: PLC0415

    from src.agents.research_agent import run_research_agent  # noqa: PLC0415
    from src.agents.tool_node import run_tool_agent  # noqa: PLC0415

    research_coro = run_research_agent(
        question=state["question"],
        query_vec=state["query_vec"],
        collection_name=state["collection_name"],
        history=state.get("history", []),
        debug_info=state.get("debug_info"),
    )
    action_coro = run_tool_agent(
        state["question"],
        history=state.get("history", []),
    )

    research_result, action_result = await asyncio.gather(
        research_coro, action_coro
    )

    return {
        "research_answer": research_result["answer"],
        "action_answer": action_result["answer"],
        "retrieved": research_result["retrieved"],
        "tool_calls": action_result.get("tool_calls", []),
        "tool_failure_reason": action_result.get("tool_failure_reason"),
        "t_milvus_start": research_result["t_milvus_start"],
        "t_milvus_end": research_result["t_milvus_end"],
        "t_llm_start": min(
            research_result["t_llm_start"], action_result.get("t_llm_start", float("inf"))
        ),
        "t_llm_end": max(
            research_result["t_llm_end"], action_result.get("t_llm_end", 0.0)
        ),
    }


@observe(name="aggregate")
async def _default_aggregate_node(state: ChatGraphState) -> dict[str, Any]:
    """Synthesise research + action answers into one user-facing reply."""
    import asyncio  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import yaml  # noqa: PLC0415
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415
    from langchain_nvidia_ai_endpoints import ChatNVIDIA  # noqa: PLC0415

    prompt_path = Path(__file__).parent / "prompts" / "aggregator.yaml"
    prompts = yaml.safe_load(prompt_path.read_text())

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    llm = ChatNVIDIA(
        model=os.environ.get("NVIDIA_LLM_MODEL", "meta/llama-3.1-70b-instruct"),
        api_key=api_key,
        temperature=0.1,
        max_tokens=512,
    )

    user_prompt = prompts["user_prompt"].format(
        question=state["question"],
        research_answer=state.get("research_answer", ""),
        action_answer=state.get("action_answer", ""),
    )

    t_start = time.perf_counter()
    response = await asyncio.to_thread(
        llm.invoke,
        [
            SystemMessage(content=prompts["system_prompt"]),
            HumanMessage(content=user_prompt),
        ],
    )
    t_end = time.perf_counter()

    return {
        "answer": response.content,
        "t_llm_start": t_start,
        "t_llm_end": t_end,
    }


def _default_canned_out_of_scope_node(state: ChatGraphState) -> dict[str, Any]:
    return {
        "answer": CANNED_OUT_OF_SCOPE_ANSWER,
        "t_milvus_start": 0.0,
        "t_milvus_end": 0.0,
        "t_llm_start": 0.0,
        "t_llm_end": 0.0,
        "retrieved": [],
        "tool_calls": [],
        "tool_failure_reason": None,
    }


def _default_postprocess_node(state: ChatGraphState) -> dict[str, Any]:
    from src.utils.rag_pipeline import compute_heuristics_for_answer  # noqa: PLC0415

    try:
        report = compute_heuristics_for_answer(
            state["answer"],
            retrieved_chunks=state.get("retrieved", []),
            debug_info=state.get("debug_info"),
        )
    except Exception as e:
        raise NodeError("postprocess", e) from e
    return {"heuristics": report}


def _default_approval_gate_node(state: ChatGraphState) -> dict[str, Any]:
    """Pass-through node after action/parallel_agents.

    If the action agent detected a destructive tool and set pending_approval,
    this node does nothing — postprocess will surface it to the caller.
    If approval_decision is already set (resume path), clear it so the action
    node re-runs cleanly without re-triggering approval.
    """
    if state.get("pending_approval"):
        # Paused — postprocess will pick up pending_approval and surface it
        return {}
    # Resume path: clear the decision so it doesn't affect future turns
    return {"approval_decision": None}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_supervisor(state: ChatGraphState) -> str:
    plan = state.get("supervisor_plan", "research")
    if plan == "action":
        return "action"
    if plan == "both":
        return "both"
    if plan == "out_of_scope":
        return "out_of_scope"
    return "research"


def _route_after_approval_gate(state: ChatGraphState) -> str:
    """If pending_approval is set, skip postprocess and go straight to END
    so the caller receives the approval request immediately.
    Otherwise continue to postprocess as normal.
    """
    if state.get("pending_approval"):
        return "pause"
    return "continue"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_chat_graph(
    *,
    supervisor_fn: NodeFn | None = None,
    research_fn: NodeFn | None = None,
    action_fn: NodeFn | None = None,
    parallel_agents_fn: NodeFn | None = None,
    aggregate_fn: NodeFn | None = None,
    canned_out_of_scope_fn: NodeFn | None = None,
    approval_gate_fn: NodeFn | None = None,
    postprocess_fn: NodeFn | None = None,
) -> CompiledStateGraph:
    g: StateGraph = StateGraph(ChatGraphState)

    g.add_node("supervisor", supervisor_fn or _default_supervisor_node)
    g.add_node("research", research_fn or _default_research_node)
    g.add_node("action", action_fn or _default_action_node)
    g.add_node("parallel_agents", parallel_agents_fn or _default_parallel_agents_node)
    g.add_node("aggregate", aggregate_fn or _default_aggregate_node)
    g.add_node("canned_out_of_scope", canned_out_of_scope_fn or _default_canned_out_of_scope_node)
    g.add_node("approval_gate", approval_gate_fn or _default_approval_gate_node)
    g.add_node("postprocess", postprocess_fn or _default_postprocess_node)

    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "research": "research",
            "action": "action",
            "both": "parallel_agents",
            "out_of_scope": "canned_out_of_scope",
        },
    )

    g.add_edge("research", "postprocess")
    # action → approval_gate → (postprocess | END depending on pending_approval)
    g.add_edge("action", "approval_gate")
    g.add_conditional_edges(
        "approval_gate",
        _route_after_approval_gate,
        {"continue": "postprocess", "pause": END},
    )
    g.add_edge("parallel_agents", "aggregate")
    g.add_edge("aggregate", "postprocess")
    g.add_edge("canned_out_of_scope", "postprocess")
    g.add_edge("postprocess", END)

    return g.compile()


_cached_graph: CompiledStateGraph | None = None


def get_chat_graph() -> CompiledStateGraph:
    global _cached_graph
    if _cached_graph is None:
        _cached_graph = build_chat_graph()
    return _cached_graph
