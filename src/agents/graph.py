"""LangGraph state machine for the chat pipeline.

Graph topology:

         START
           │
           ▼
       retrieve
           │
       ┌───┴───┐  (conditional edge: did we get any chunks?)
       │       │
       ▼       ▼
   rerank    canned_no_retrieval
       │       │
       ▼       │
   generate    │
       │       │
       └───┬───┘
           ▼
       postprocess (heuristics)
           │
           ▼
          END

Why these splits:

- Each node maps to one observable stage in the Langfuse trace tree —
  pre-LangGraph everything nested inside a single `answer_question`
  span; now `retrieve`, `rerank`, and `generate` are siblings under
  `rag_output`. Easier to slice timings, latencies, and errors by
  stage in dashboards.
- The conditional after `retrieve` lets us short-circuit to a canned
  refusal *without* burning a rerank + LLM call when Milvus returned
  nothing. That used to be an `if not retrieved: return` inside
  `answer_question`; it's now a routing decision.
- `postprocess` runs the heuristic checks on both paths so every
  answer that leaves the graph carries a heuristic report.

Each default node implementation is imported lazily so the test CI
install line (no langchain stack) can still load this module and
exercise the graph with stubs. The same pattern as PR #1 — see
`build_chat_graph`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

# Surfaced to the caller when retrieval returns nothing. Defined here
# (rather than in rag_pipeline) so the canned-path node doesn't have
# to import rag_pipeline, which in turn pulls in the langchain stack
# — important for keeping this module loadable in the minimal CI
# test env.
CANNED_NO_RETRIEVAL_ANSWER = (
    "I couldn't find anything in the indexed document that touches on that. "
    "Could you try rephrasing, or asking about a different topic from the book?"
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

    # --- written by retrieve ----------------------------------------------
    retrieved: list[dict[str, Any]]
    t_milvus_start: float
    t_milvus_end: float

    # --- written by rerank ------------------------------------------------
    top_chunks: list[dict[str, Any]]
    t_rerank_start: float
    t_rerank_end: float

    # --- written by generate (or canned_no_retrieval) ---------------------
    answer: str
    t_llm_start: float
    t_llm_end: float

    # --- written by postprocess -------------------------------------------
    heuristics: dict[str, Any] | None


# Node-function signatures: take the current state, return a partial
# state dict that LangGraph merges in.
NodeFn = Callable[[ChatGraphState], dict[str, Any]]


# ---------------------------------------------------------------------------
# Default node implementations.
#
# These are thin wrappers that lift fields out of state, call the
# `@observe`-decorated stage functions in rag_pipeline, and return the
# fields the stage produced. The wrappers also pre-populate the
# shared `debug_info` dict on the first node that runs so subsequent
# nodes can write into it.
# ---------------------------------------------------------------------------


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

    result = rerank_chunks(
        question=state["question"],
        retrieved=state["retrieved"],
        debug_info=state.get("debug_info"),
    )
    return {
        "top_chunks": result["top_chunks"],
        "t_rerank_start": result["t_rerank_start"],
        "t_rerank_end": result["t_rerank_end"],
    }


def _default_generate_node(state: ChatGraphState) -> dict[str, Any]:
    from src.utils.rag_pipeline import generate_answer  # noqa: PLC0415

    result = generate_answer(
        question=state["question"],
        retrieved=state["retrieved"],
        top_chunks=state["top_chunks"],
        history=state["history"],
        query_vec=state["query_vec"],
        debug_info=state.get("debug_info"),
    )
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


def _default_postprocess_node(state: ChatGraphState) -> dict[str, Any]:
    from src.utils.rag_pipeline import compute_heuristics_for_answer  # noqa: PLC0415

    # Heuristics use the pre-rerank retrieved chunks as their grounding
    # source — same as the pre-LangGraph behavior — because the citation
    # check just needs SOME context to compare against, and the broader
    # set is more forgiving.
    report = compute_heuristics_for_answer(
        state["answer"],
        retrieved_chunks=state.get("retrieved", []),
        debug_info=state.get("debug_info"),
    )
    return {"heuristics": report}


# ---------------------------------------------------------------------------
# Conditional edge — routes after retrieve based on whether anything came back.
# ---------------------------------------------------------------------------


def _route_after_retrieve(state: ChatGraphState) -> str:
    """Either continue down the LLM path, or short-circuit to the canned reply."""
    return "continue" if state.get("retrieved") else "abort"


# ---------------------------------------------------------------------------
# Graph construction.
# ---------------------------------------------------------------------------


def build_chat_graph(
    *,
    retrieve_fn: NodeFn | None = None,
    rerank_fn: NodeFn | None = None,
    generate_fn: NodeFn | None = None,
    canned_no_retrieval_fn: NodeFn | None = None,
    postprocess_fn: NodeFn | None = None,
) -> CompiledStateGraph:
    """Build and compile the chat graph.

    Each node has an optional injection point for tests — pass a stub
    callable for any node you want to short-circuit. Defaults wire up
    to the real `rag_pipeline` functions (imported lazily inside each
    default to keep the test CI install line minimal).
    """
    g: StateGraph = StateGraph(ChatGraphState)
    g.add_node("retrieve", retrieve_fn or _default_retrieve_node)
    g.add_node("rerank", rerank_fn or _default_rerank_node)
    g.add_node("generate", generate_fn or _default_generate_node)
    g.add_node(
        "canned_no_retrieval",
        canned_no_retrieval_fn or _default_canned_no_retrieval_node,
    )
    g.add_node("postprocess", postprocess_fn or _default_postprocess_node)

    g.set_entry_point("retrieve")
    g.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"continue": "rerank", "abort": "canned_no_retrieval"},
    )
    g.add_edge("rerank", "generate")
    g.add_edge("generate", "postprocess")
    g.add_edge("canned_no_retrieval", "postprocess")
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
