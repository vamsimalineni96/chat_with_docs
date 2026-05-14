"""Agentic RAG pipeline (LangGraph).

Replaces the previous fixed `retrieve → rerank → assemble → generate` chain.
The LLM now drives the loop: it sees a `search_chunks` tool and decides when
(and how many times) to retrieve.

Graph shape:

      ┌─────────────────────────────┐
      │                             │
      ▼                             │
    agent ── tool_calls? ── tools ──┘
      │           no
      ▼
     END

`agent_node` calls the LLM with tools bound. If the response has tool_calls
we route to `tools` (LangGraph's built-in `ToolNode`), which executes them
and returns ToolMessages. Those get appended to the message list and we loop
back to `agent`. When the LLM returns a plain answer (no tool_calls), we hit
END.

Bounded by `MAX_AGENT_ITERATIONS` to avoid runaway loops.
"""

import time
from typing import List, Dict, Any, Optional, TypedDict, Annotated

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.utils import config
from src.utils.errors import InferenceError
from src.utils.observability import observe, langfuse_callback
from src.utils.services.chunk_ranking import NVidiaReranker
from src.utils.services.inference import NIMClient
from src.utils.services.logger_config import logger
from src.utils.services.milvus_store import MilvusStoreHandler, get_cache_store
from src.utils.tools import build_search_chunks_tool


MAX_AGENT_ITERATIONS = 5


def make_tool_node(tools: List):
    """Build a tool-executor node.

    Looks at the last AIMessage in `state["messages"]`, executes every
    tool_call it carries, and returns a list of ToolMessages that
    `add_messages` will append to the running message trail.

    Equivalent to `langgraph.prebuilt.ToolNode([...])` but written inline so
    we don't depend on the prebuilt package (its version pinning is fussy).
    """
    tool_map = {t.name: t for t in tools}

    def tool_node(state: "AgentState") -> Dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None) or []
        outputs: List[ToolMessage] = []
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = (tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})) or {}
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)

            tool = tool_map.get(name)
            if tool is None:
                content = f"Error: unknown tool {name!r}"
                logger.warning("LLM requested unknown tool %r", name)
            else:
                try:
                    result = tool.invoke(args)
                    content = result if isinstance(result, str) else str(result)
                except Exception as e:
                    logger.exception("Tool %s raised — returning error message to LLM", name)
                    content = f"Error calling {name}: {e}"
            outputs.append(ToolMessage(content=content, tool_call_id=tc_id, name=name))
        return {"messages": outputs}

    return tool_node


def format_history_for_prompt(history: List[Dict], max_turns: int = config.HISTORY_MAX_TURNS) -> str:
    """Render history as plain text. Retained for debug capture and any
    external callers that still want a single string."""
    if not history:
        return "None"
    trimmed = history[-max_turns:]
    lines = []
    for msg in trimmed:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "Assistant"
        elif role == "system":
            prefix = "System"
        else:
            prefix = role or "Unknown"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def history_to_messages(
    history: List[Dict],
    max_turns: int = config.HISTORY_MAX_TURNS,
) -> List[BaseMessage]:
    """Convert stored chat history rows to LangChain BaseMessages."""
    if not history:
        return []
    # max_turns is "exchanges"; multiply by 2 to cover user+assistant pairs.
    trimmed = history[-(max_turns * 2):]
    out: List[BaseMessage] = []
    for m in trimmed:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


class AgentState(TypedDict, total=False):
    """State for the agentic loop.

    `messages` uses LangGraph's `add_messages` reducer so node returns of
    `{"messages": [new_msg]}` are *appended* to the running list rather than
    replacing it. Other fields use the default replace reducer.

    `timings` and `debug` are mutable dicts owned by the caller — nodes
    mutate them in place so the caller can read final values after invoke.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    timings: Dict[str, float]
    debug: Optional[Dict[str, Any]]
    iterations: int


def build_agent_graph(llm_with_tools, tool_node):
    """Compile the agent ↔ tools loop graph."""

    def agent_node(state: AgentState) -> Dict[str, Any]:
        timings = state["timings"]
        iter_n = state.get("iterations", 0)

        t_start = time.perf_counter()
        timings.setdefault("t_llm_start", t_start)

        response = llm_with_tools.invoke(state["messages"])
        timings["t_llm_end"] = time.perf_counter()

        if state.get("debug") is not None:
            dbg = state["debug"]
            steps = dbg.setdefault("agent_steps", [])
            steps.append(
                {
                    "iteration": iter_n,
                    "tool_calls": getattr(response, "tool_calls", []) or [],
                    "content_preview": (getattr(response, "content", "") or "")[:300],
                }
            )

        return {"messages": [response], "iterations": iter_n + 1}

    def should_continue(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return "end"
        if state.get("iterations", 0) >= MAX_AGENT_ITERATIONS:
            logger.warning(
                "Agent hit iteration cap (%d) — forcing END with last response.",
                MAX_AGENT_ITERATIONS,
            )
            return "end"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


@observe(name="answer_question")
def answer_question(
    question: str,
    query_vec: List[float],
    collection_name: str,
    history: List[Dict],
    debug: bool = False,
) -> Dict[str, Any]:
    """Agentic RAG orchestration.

    The LLM is given the `search_chunks` tool and drives the loop: it can
    retrieve zero, one, or multiple times before producing the final answer.
    Bounded by `MAX_AGENT_ITERATIONS`.

    Returns a dict with answer, stage timings, and (when debug=True) the
    full message trail + per-step tool_calls.

    Raises:
        InferenceError: for unrecoverable LLM / graph failures.
    """
    milvus_store = MilvusStoreHandler(collection_name=collection_name)
    reranker = NVidiaReranker()
    nim_client = NIMClient()

    debug_info: Optional[Dict[str, Any]] = {} if debug else None

    # Build the tool, bind it to the LLM, wrap in a ToolNode.
    search_chunks = build_search_chunks_tool(milvus_store, reranker)
    tools = [search_chunks]
    llm_with_tools = nim_client.llm.bind_tools(tools)
    tool_node = make_tool_node(tools)

    graph = build_agent_graph(llm_with_tools, tool_node)

    # Initial messages: system + prior turns + current question.
    history_msgs = history_to_messages(history)
    initial_messages: List[BaseMessage] = [
        SystemMessage(content=nim_client.system_prompt),
        *history_msgs,
        HumanMessage(content=question),
    ]

    timings: Dict[str, float] = {}
    initial_state: AgentState = {
        "messages": initial_messages,
        "timings": timings,
        "debug": debug_info,
        "iterations": 0,
    }

    cb = langfuse_callback()
    invoke_config = {"callbacks": [cb]} if cb else {}

    t_graph_start = time.perf_counter()
    try:
        logger.info(
            "Invoking agentic RAG graph (max %d iterations)",
            MAX_AGENT_ITERATIONS,
        )
        final_state = graph.invoke(initial_state, config=invoke_config)
    except InferenceError:
        raise
    except Exception as e:
        logger.exception("Unexpected error from agentic RAG graph: %s", e)
        raise InferenceError("Unexpected error from agentic RAG graph.") from e
    t_graph_end = time.perf_counter()

    # The final answer is the content of the last AIMessage in the trail.
    final_msg = final_state["messages"][-1]
    answer = (getattr(final_msg, "content", None) or "").strip()

    # Did any retrieval actually happen? Look for ToolMessages.
    retrieved_any = any(isinstance(m, ToolMessage) for m in final_state["messages"])
    iterations_used = final_state.get("iterations", 0)

    if debug_info is not None:
        debug_info["total_iterations"] = iterations_used
        debug_info["retrieved_any"] = retrieved_any
        debug_info["final_messages"] = [
            {
                "type": getattr(m, "type", "unknown"),
                "content": (getattr(m, "content", "") or "")[:500],
                "tool_calls": getattr(m, "tool_calls", None) or [],
                "tool_call_id": getattr(m, "tool_call_id", None),
                "name": getattr(m, "name", None),
            }
            for m in final_state["messages"]
        ]

    # Cache write — only when retrieval happened and we're not in debug.
    # In agentic mode we don't track exact chunk IDs across multiple tool
    # calls, so context_chunk_ids stays empty; the cache key is question +
    # model + prompt_version anyway.
    if config.TOGGLE_CACHE and not debug and retrieved_any and answer:
        try:
            get_cache_store().put_entry(
                question_text=question,
                query_vec=query_vec,
                answer_text=answer,
                context_chunk_ids=[],
                model_name=config.LLM_MODEL,
                prompt_version=config.PROMPT_VERSION,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )
            logger.info("Stored Q/A pair in semantic cache.")
        except Exception as e:
            logger.exception("Cache write failed (non-fatal): %s", e)

    # Timing breakdown: agentic mode interleaves LLM and retrieval, so the
    # old "milvus_ms vs llm_ms" split no longer cleanly applies. We report
    # the whole graph as LLM time and leave milvus stamps as zero-duration
    # markers so the upstream metrics logger keeps working.
    t_llm_start = timings.get("t_llm_start", t_graph_start)
    t_llm_end = timings.get("t_llm_end", t_graph_end)
    t_milvus_start = t_llm_start
    t_milvus_end = t_llm_start

    return {
        "answer": answer,
        "t_milvus_start": t_milvus_start,
        "t_milvus_end": t_milvus_end,
        "t_llm_start": t_llm_start,
        "t_llm_end": t_llm_end,
        "debug": debug_info,
    }
