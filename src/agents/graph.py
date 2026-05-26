"""LangGraph state machine for the chat pipeline.

Graph topology:

                       START
                         │
                         ▼
                   classify_intent      (LLM router: 3 intents)
                         │
            ┌────────────┼────────────┐
            │            │            │
            ▼            ▼            ▼
        retrieve   call_mcp_tool  canned_out_of_scope
            │            │            │
         ┌──┴──┐          │            │
         │     │          │            │
         ▼     ▼          │            │
      rerank  canned_no_retrieval      │
         │       │        │            │
         ▼       │        │            │
      generate   │        │            │
         │       │        │            │
         └───┬───┴────────┴────────────┘
             ▼
         postprocess (heuristics)
             │
             ▼
            END

The classifier in front of `retrieve` is the first agent decision:
- `in_corpus`    → run RAG normally
- `tool_call`    → spawn the MCP ReAct sub-agent (live shopping data)
- `out_of_scope` → short-circuit before burning Milvus + rerank + LLM

The second conditional (after `retrieve`) handles the case where the
classifier let a question through but Milvus returned nothing anyway
— same canned-no-retrieval path as before.

`postprocess` runs heuristics on all four terminal paths so every
answer carries a heuristic report.

Each default node lazy-imports its langchain-dependent
implementation. Tests pass per-node stubs through `build_chat_graph`
to short-circuit any subset of nodes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.utils.errors import NodeError

# Surfaced to the caller when retrieval returns nothing. Defined here
# (rather than in rag_pipeline) so the canned-path node doesn't have
# to import rag_pipeline, which in turn pulls in the langchain stack
# — important for keeping this module loadable in the minimal CI
# test env.
CANNED_NO_RETRIEVAL_ANSWER = (
    "I couldn't find anything in the indexed document that touches on that. "
    "Could you try rephrasing, or asking about a different topic from the book?"
)

# Surfaced when the classifier decides the question is outside the
# corpus domain. Phrased to invite a retry / upload rather than just
# slamming the door — the classifier can be wrong, and we don't want
# the user to think the bot is broken.
CANNED_OUT_OF_SCOPE_ANSWER = (
    "That question looks like it's outside the scope of the documents "
    "I've been given. I can only answer things grounded in the indexed "
    "material — try rephrasing in that direction, or upload a document "
    "that covers the topic."
)


class ChatGraphState(TypedDict, total=False):
    """State threaded through every node.

    `total=False`: callers populate only the input fields; nodes fill
    the rest as they execute. LangGraph merges each node's return
    dict into the state.
    """

    # --- inputs (set by chat_service before invoke) ------------------------
    question: str
    query_vec: list[float]
    collection_name: str
    history: list[dict[str, Any]]
    debug_flag: bool

    # --- shared / accumulated debug payload --------------------------------
    debug_info: dict[str, Any] | None

    # --- written by classify_intent ---------------------------------------
    intent: str  # one of intent_classifier.VALID_INTENTS
    intent_reasoning: str

    # --- written by retrieve ----------------------------------------------
    retrieved: list[dict[str, Any]]
    t_milvus_start: float
    t_milvus_end: float

    # --- written by rerank ------------------------------------------------
    top_chunks: list[dict[str, Any]]
    t_rerank_start: float
    t_rerank_end: float

    # --- written by generate (or canned_no_retrieval / call_mcp_tool) -----
    answer: str
    t_llm_start: float
    t_llm_end: float

    # --- written by call_mcp_tool only ------------------------------------
    # List of {"name": str, "args": dict} records, one per tool the ReAct
    # sub-agent invoked. Empty list on the failure path. Surfaced in the
    # chat trace so we can see which MCP tools the agent reached for.
    tool_calls: list[dict[str, Any]]
    # One of "no_tools" | "no_api_key" | "agent_error" | "no_messages" when
    # the sub-agent failed; None on success. Surfaced as mcp_failure:<reason>
    # in Langfuse so you can distinguish MCP server down from API key missing.
    tool_failure_reason: str | None

    # --- written by postprocess -------------------------------------------
    heuristics: dict[str, Any] | None


# Node-function signatures: take the current state, return a partial
# state dict that LangGraph merges in. Async nodes are supported —
# LangGraph handles both when the graph is invoked via `ainvoke`.
NodeFn = (
    Callable[[ChatGraphState], dict[str, Any]]
    | Callable[[ChatGraphState], Awaitable[dict[str, Any]]]
)


# ---------------------------------------------------------------------------
# Default node implementations.
#
# These are thin wrappers that lift fields out of state, call the
# `@observe`-decorated stage functions in rag_pipeline, and return the
# fields the stage produced. The wrappers also pre-populate the
# shared `debug_info` dict on the first node that runs so subsequent
# nodes can write into it.
# ---------------------------------------------------------------------------


def _default_classify_intent_node(state: ChatGraphState) -> dict[str, Any]:
    from src.agents.intent_classifier import classify_intent  # noqa: PLC0415

    try:
        result = classify_intent(state["question"])
    except Exception as e:
        raise NodeError("classify_intent", e) from e
    return {
        "intent": result.intent,
        "intent_reasoning": result.reasoning,
    }


def _default_retrieve_node(state: ChatGraphState) -> dict[str, Any]:
    from src.utils.rag_pipeline import retrieve_chunks  # noqa: PLC0415

    debug_info: dict[str, Any] | None = (
        {} if state.get("debug_flag") else None
    )
    result = retrieve_chunks(
        question=state["question"],
        collection_name=state["collection_name"],
        debug_info=debug_info,
    )
    return {
        "retrieved": result["retrieved"],
        "t_milvus_start": result["t_milvus_start"],
        "t_milvus_end": result["t_milvus_end"],
        "debug_info": debug_info,
    }


def _default_rerank_node(state: ChatGraphState) -> dict[str, Any]:
    from src.utils.rag_pipeline import rerank_chunks  # noqa: PLC0415

    try:
        result = rerank_chunks(
            question=state["question"],
            retrieved=state["retrieved"],
            debug_info=state.get("debug_info"),
        )
    except Exception as e:
        raise NodeError("rerank", e) from e
    return {
        "top_chunks": result["top_chunks"],
        "t_rerank_start": result["t_rerank_start"],
        "t_rerank_end": result["t_rerank_end"],
    }


def _default_generate_node(state: ChatGraphState) -> dict[str, Any]:
    from src.utils.rag_pipeline import generate_answer  # noqa: PLC0415

    try:
        result = generate_answer(
            question=state["question"],
            retrieved=state["retrieved"],
            top_chunks=state["top_chunks"],
            history=state["history"],
            query_vec=state["query_vec"],
            debug_info=state.get("debug_info"),
        )
    except Exception as e:
        raise NodeError("generate", e) from e
    return {
        "answer": result["answer"],
        "t_llm_start": result["t_llm_start"],
        "t_llm_end": result["t_llm_end"],
    }


def _default_canned_no_retrieval_node(state: ChatGraphState) -> dict[str, Any]:
    """Short-circuit path when retrieval returned nothing.

    Reuses the milvus timings for t_llm_start/end so chat_service's
    metrics log doesn't have to special-case this branch. The refusal
    heuristic in postprocess will (correctly) flag this response as a
    refusal — that's the intended trace signal.

    Note: this node intentionally does NOT import from rag_pipeline.
    Keeping it self-contained means the canned path is loadable in
    the minimal CI test env without the langchain stack.
    """
    t_milvus_end = state.get("t_milvus_end", 0.0)
    t_milvus_start = state.get("t_milvus_start", t_milvus_end)
    return {
        "answer": CANNED_NO_RETRIEVAL_ANSWER,
        "t_llm_start": t_milvus_start,
        "t_llm_end": t_milvus_end,
    }


def _default_canned_out_of_scope_node(state: ChatGraphState) -> dict[str, Any]:
    """Short-circuit path when the classifier flags the question as
    outside the corpus domain.

    Like canned_no_retrieval, this node is self-contained — no
    imports — so the path is loadable in the minimal CI test env.
    Timing fields are stamped to 0 since neither Milvus nor the
    generator LLM was touched; chat_service's metrics log will just
    show this branch as effectively instant.
    """
    return {
        "answer": CANNED_OUT_OF_SCOPE_ANSWER,
        "t_milvus_start": 0.0,
        "t_milvus_end": 0.0,
        "t_llm_start": 0.0,
        "t_llm_end": 0.0,
        # Postprocess reads `retrieved` for the citation heuristic; keep
        # the field present-but-empty so it doesn't KeyError.
        "retrieved": [],
    }


async def _default_call_mcp_tool_node(state: ChatGraphState) -> dict[str, Any]:
    """Run the MCP ReAct sub-agent on the user's question.

    Spawns the wrapped `create_react_agent` (cached at module level
    inside `tool_node`), which discovers MCP tools, calls them, and
    synthesizes a user-facing answer in one ReAct loop. On failure
    (no tools / LLM down / agent error), tool_node returns the
    canned `TOOL_FAILURE_ANSWER` so the `answer` field is always
    populated — same invariant as the other canned paths.

    Milvus timings are stamped to 0 (this branch never touches
    Milvus) so chat_service's metrics log shows the cost saving
    relative to the RAG path. `retrieved` is set to [] so postprocess
    can still run citation checks without KeyError.
    """
    from src.agents.tool_node import run_tool_agent  # noqa: PLC0415

    result = await run_tool_agent(state["question"])
    return {
        "answer": result["answer"],
        "tool_calls": result.get("tool_calls", []),
        "tool_failure_reason": result.get("tool_failure_reason"),
        "t_milvus_start": 0.0,
        "t_milvus_end": 0.0,
        "t_llm_start": result.get("t_llm_start", 0.0),
        "t_llm_end": result.get("t_llm_end", 0.0),
        "retrieved": [],
    }


def _default_postprocess_node(state: ChatGraphState) -> dict[str, Any]:
    from src.utils.rag_pipeline import compute_heuristics_for_answer  # noqa: PLC0415

    # Heuristics use the pre-rerank retrieved chunks as their grounding
    # source — same as the pre-LangGraph behavior — because the citation
    # check just needs SOME context to compare against, and the broader
    # set is more forgiving.
    try:
        report = compute_heuristics_for_answer(
            state["answer"],
            retrieved_chunks=state.get("retrieved", []),
            debug_info=state.get("debug_info"),
        )
    except Exception as e:
        raise NodeError("postprocess", e) from e
    return {"heuristics": report}


# ---------------------------------------------------------------------------
# Conditional edge — routes after retrieve based on whether anything came back.
# ---------------------------------------------------------------------------


def _route_after_classify(state: ChatGraphState) -> str:
    """First conditional in the graph — drives the agentic routing.

    Three explicit branches; everything else falls through to RAG.
    The fall-through is a deliberate safety bias — a false-positive
    on out_of_scope or tool_call would refuse / mishandle a legitimate
    question, whereas falling through to RAG at worst wastes a
    retrieval. The classifier's own failure-mode fallback is also
    `in_corpus`, so this branch's default is doubly safe.
    """
    intent = state.get("intent")
    if intent == "tool_call":
        return "tool_call"
    if intent == "out_of_scope":
        return "out_of_scope"
    return "continue"


def _route_after_retrieve(state: ChatGraphState) -> str:
    """Either continue down the LLM path, or short-circuit to the canned reply."""
    return "continue" if state.get("retrieved") else "abort"


# ---------------------------------------------------------------------------
# Graph construction.
# ---------------------------------------------------------------------------


def build_chat_graph(
    *,
    classify_intent_fn: NodeFn | None = None,
    retrieve_fn: NodeFn | None = None,
    rerank_fn: NodeFn | None = None,
    generate_fn: NodeFn | None = None,
    call_mcp_tool_fn: NodeFn | None = None,
    canned_no_retrieval_fn: NodeFn | None = None,
    canned_out_of_scope_fn: NodeFn | None = None,
    postprocess_fn: NodeFn | None = None,
) -> CompiledStateGraph:
    """Build and compile the chat graph.

    Each node has an optional injection point for tests — pass a stub
    callable for any node you want to short-circuit. Defaults wire up
    to the real implementations (imported lazily inside each default
    to keep the test CI install line minimal).
    """
    g: StateGraph = StateGraph(ChatGraphState)
    g.add_node(
        "classify_intent", classify_intent_fn or _default_classify_intent_node
    )
    g.add_node("retrieve", retrieve_fn or _default_retrieve_node)
    g.add_node("rerank", rerank_fn or _default_rerank_node)
    g.add_node("generate", generate_fn or _default_generate_node)
    g.add_node(
        "call_mcp_tool", call_mcp_tool_fn or _default_call_mcp_tool_node
    )
    g.add_node(
        "canned_no_retrieval",
        canned_no_retrieval_fn or _default_canned_no_retrieval_node,
    )
    g.add_node(
        "canned_out_of_scope",
        canned_out_of_scope_fn or _default_canned_out_of_scope_node,
    )
    g.add_node("postprocess", postprocess_fn or _default_postprocess_node)

    g.set_entry_point("classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "continue": "retrieve",
            "tool_call": "call_mcp_tool",
            "out_of_scope": "canned_out_of_scope",
        },
    )
    g.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"continue": "rerank", "abort": "canned_no_retrieval"},
    )
    g.add_edge("rerank", "generate")
    g.add_edge("generate", "postprocess")
    g.add_edge("call_mcp_tool", "postprocess")
    g.add_edge("canned_no_retrieval", "postprocess")
    g.add_edge("canned_out_of_scope", "postprocess")
    g.add_edge("postprocess", END)

    return g.compile()


_cached_graph: CompiledStateGraph | None = None


def get_chat_graph() -> CompiledStateGraph:
    """Return the process-wide compiled chat graph.

    Compilation is one-shot — the graph object is stateless across
    invocations; per-request state lives in the dict passed to
    `.invoke()`.
    """
    global _cached_graph
    if _cached_graph is None:
        _cached_graph = build_chat_graph()
    return _cached_graph
