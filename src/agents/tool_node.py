"""ReAct sub-agent that handles the `tool_call` branch of the chat graph.

When the classifier emits `intent: tool_call`, the graph routes here.
This module wraps `langgraph.prebuilt.create_react_agent` over the
MCP tools discovered by `src.agents.mcp_client`. The ReAct loop
itself does the work: the LLM sees the discovered tools, decides
which one(s) to call, sees the result, and synthesizes a final
user-facing answer — all in a single `.invoke()`.

Why a sub-graph instead of a single LLM call:
- Multiple tools per question come for free ("Is SKU-001 in stock
  AND what's the return window for electronics?" → two tool calls).
- Adding a new MCP tool means editing `mcp_servers/` — nothing in
  this file or the classifier prompt needs to change.
- Matches the standard LangGraph+MCP pattern that's easy to point
  at in an interview.

Failure policy: if no MCP tools are discovered, or the React agent
errors out, return a canned apology string and an empty tool-call
list. The chat graph's postprocess step still runs heuristics on
the canned answer — same invariant as the other short-circuit
paths (refusal heuristic will (correctly) flag it).

Imports of langchain / langgraph.prebuilt / langchain_nvidia_ai_endpoints
are deferred inside `_build_agent` so this module can be imported
in the minimal CI test env that doesn't install the langchain
stack — tests inject a fake agent via the `agent=` parameter.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TOOL_AGENT_MODEL = os.environ.get(
    "TOOL_AGENT_MODEL", "meta/llama-3.1-70b-instruct"
)

# Surfaced when the sub-agent can't run (no tools discovered, LLM
# unreachable, ReAct loop blew up). Phrased to invite a retry rather
# than just slamming the door — same tone as the other canned paths.
TOOL_FAILURE_ANSWER = (
    "I tried to look that up using my live tools, but the lookup didn't "
    "complete. Please try again in a moment, or rephrase your question."
)


# Module-level cache. The React agent is stateless across invocations
# (per-call state lives in the messages list we pass), so one
# process-wide instance is safe and avoids re-discovering MCP tools on
# every request.
_cached_agent: Any | None = None


def _build_agent(model: str) -> Any:
    """Construct a fresh React agent bound to the current MCP tools.

    Imports are deferred so the test env (which mocks the agent via DI)
    never needs langchain installed. `asyncio.run` is safe here because
    the calling node is sync — there's no enclosing event loop in the
    FastAPI request path or in the test suite.
    """
    from langchain_nvidia_ai_endpoints import ChatNVIDIA  # noqa: PLC0415
    from langgraph.prebuilt import create_react_agent  # noqa: PLC0415

    from src.agents.mcp_client import get_available_tools  # noqa: PLC0415

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")

    tools = asyncio.run(get_available_tools())
    if not tools:
        raise RuntimeError(
            "no MCP tools discovered — is the shopping_support server reachable?"
        )

    llm = ChatNVIDIA(
        model=model,
        api_key=api_key,
        temperature=0.0,  # deterministic tool selection
    )
    return create_react_agent(llm, tools)


def get_agent(model: str | None = None) -> Any:
    """Return the process-wide React agent, building it on first call."""
    global _cached_agent
    if _cached_agent is None:
        _cached_agent = _build_agent(model or DEFAULT_TOOL_AGENT_MODEL)
    return _cached_agent


def reset_cache() -> None:
    """Clear the cached agent. Used by tests and by any future
    'reconnect to MCP after restart' code.
    """
    global _cached_agent
    _cached_agent = None


def _extract_tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
    """Pull tool-call records out of the ReAct message history.

    Each call is logged as `{name, args}` so the chat trace / debug
    payload can show "the agent called get_order_status with
    {order_id: ORD-1001}" without us having to fish through the raw
    message objects.
    """
    calls: list[dict[str, Any]] = []
    for m in messages:
        tcs = getattr(m, "tool_calls", None) or []
        for tc in tcs:
            calls.append(
                {
                    "name": tc.get("name"),
                    "args": tc.get("args"),
                }
            )
    return calls


def run_tool_agent(question: str, *, agent: Any | None = None) -> dict[str, Any]:
    """Run the React agent on `question` and return a node-shaped dict.

    Returns:
        {
          "answer": str,                      # always populated
          "tool_calls": list[dict],           # [] on failure
          "t_llm_start": float,               # perf_counter at invoke
          "t_llm_end": float,                 # perf_counter at return
          "error": str | None,                # populated on failure path
        }

    Never raises. The graph's postprocess step is what surfaces an
    "I couldn't help" response to the user via the refusal heuristic.
    `agent=` is the DI seam tests use to inject a fake agent.
    """
    from langchain_core.messages import HumanMessage  # noqa: PLC0415

    t_start = time.perf_counter()
    try:
        if agent is None:
            agent = get_agent()
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
    except Exception as e:
        logger.warning("Tool sub-agent failed: %s", e, exc_info=True)
        t_end = time.perf_counter()
        return {
            "answer": TOOL_FAILURE_ANSWER,
            "tool_calls": [],
            "t_llm_start": t_start,
            "t_llm_end": t_end,
            "error": str(e),
        }

    t_end = time.perf_counter()
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        logger.warning("Tool sub-agent returned no messages")
        return {
            "answer": TOOL_FAILURE_ANSWER,
            "tool_calls": [],
            "t_llm_start": t_start,
            "t_llm_end": t_end,
            "error": "agent returned empty message list",
        }

    final = messages[-1]
    answer = getattr(final, "content", None) or str(final)
    return {
        "answer": answer,
        "tool_calls": _extract_tool_calls(messages),
        "t_llm_start": t_start,
        "t_llm_end": t_end,
        "error": None,
    }
