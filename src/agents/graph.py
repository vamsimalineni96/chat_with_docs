"""LangGraph scaffold for the chat pipeline.

This is the first step of the LangGraph migration. The graph here has
exactly one node that delegates to the existing `answer_question` RAG
pipeline — zero behavior change versus calling `answer_question`
directly. The point of this PR is to establish the plumbing
(`StateGraph`, state type, node function, compiled-graph cache) so
that subsequent PRs can break the linear pipeline into proper graph
nodes (retrieve → rerank → generate → ...) and add new ones
(classify_intent, call_mcp_tool) without simultaneously rewriting the
RAG internals.

The state schema below intentionally mirrors the inputs and outputs of
`answer_question` 1:1, with one rename: `debug` (the input bool flag)
becomes `debug_flag` inside the state, so it doesn't collide with the
`debug_info` output payload. Callers swap one field name; nothing
else changes.

Observability: `answer_question` keeps its `@observe` decorator, so
the Langfuse trace tree is identical to pre-LangGraph. We do NOT
decorate the graph node itself yet — that would add an extra span
between `rag_output` and `answer_question` with no information, just
visual noise. When we split the pipeline into multiple nodes (PR #2),
each node will get its own `@observe` and the trace tree will gain
real structure.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.utils.rag_pipeline import answer_question


class ChatGraphState(TypedDict, total=False):
    """State threaded through the chat graph.

    `total=False` so callers can pass only the input fields; the rag
    node fills the rest. LangGraph merges node return values into the
    state by default, so each node only needs to return the keys it
    writes.
    """

    # --- inputs (set by the caller, never mutated by nodes) ----------------
    question: str
    query_vec: list[float]
    collection_name: str
    history: list[dict[str, Any]]
    debug_flag: bool  # renamed from `debug` to avoid colliding with debug_info

    # --- outputs (filled by nodes) -----------------------------------------
    answer: str
    t_milvus_start: float
    t_milvus_end: float
    t_llm_start: float
    t_llm_end: float
    debug_info: dict[str, Any] | None
    heuristics: dict[str, Any] | None


# Type alias for the rag callable so tests can substitute a stub.
RagCallable = Callable[..., dict[str, Any]]


def _make_rag_node(rag_fn: RagCallable):
    """Build a node closure over the RAG callable.

    Returning a closure (rather than a top-level function bound to the
    real `answer_question`) lets tests inject a stub via
    `build_chat_graph(rag_fn=...)` without monkeypatching.
    """

    def _run_rag(state: ChatGraphState) -> dict[str, Any]:
        result = rag_fn(
            question=state["question"],
            query_vec=state["query_vec"],
            collection_name=state["collection_name"],
            history=state["history"],
            debug=state.get("debug_flag", False),
        )
        return {
            "answer": result["answer"],
            "t_milvus_start": result["t_milvus_start"],
            "t_milvus_end": result["t_milvus_end"],
            "t_llm_start": result["t_llm_start"],
            "t_llm_end": result["t_llm_end"],
            "debug_info": result.get("debug"),
            "heuristics": result.get("heuristics"),
        }

    return _run_rag


def build_chat_graph(rag_fn: RagCallable = answer_question) -> CompiledStateGraph:
    """Construct and compile the chat graph.

    Public-but-rarely-called: production code goes through
    `get_chat_graph()` which caches the compiled instance. Tests call
    this directly so they can pass `rag_fn=stub`.
    """
    g: StateGraph = StateGraph(ChatGraphState)
    g.add_node("rag", _make_rag_node(rag_fn))
    g.set_entry_point("rag")
    g.add_edge("rag", END)
    return g.compile()


_cached_graph: CompiledStateGraph | None = None


def get_chat_graph() -> CompiledStateGraph:
    """Return the process-wide compiled chat graph.

    Compilation is cheap but not free — keep one instance for the
    lifetime of the process. The graph object is stateless across
    invocations; per-request state lives in the dict passed to
    `.invoke()`.
    """
    global _cached_graph
    if _cached_graph is None:
        _cached_graph = build_chat_graph()
    return _cached_graph
