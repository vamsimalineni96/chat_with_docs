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
import concurrent.futures
import contextvars
import logging
import os
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from src.utils.observability import langfuse_callback, observe

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Single worker thread that owns a fresh asyncio loop. Reused across
# requests so we don't pay thread-spawn cost on every tool_call. One
# worker is enough — each call blocks the caller until done, and we
# don't try to parallelise tool invocations.
_LOOP_THREAD: concurrent.futures.ThreadPoolExecutor | None = None


def _get_loop_thread() -> concurrent.futures.ThreadPoolExecutor:
    global _LOOP_THREAD
    if _LOOP_THREAD is None:
        _LOOP_THREAD = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tool-node-loop"
        )
    return _LOOP_THREAD


def _run_async_blocking(
    coro_factory: Callable[[], Coroutine[Any, Any, _T]],
) -> _T:
    """Run a fresh coroutine to completion from a sync caller.

    FastAPI request handlers are `async def`, so by the time we land
    inside `graph.invoke -> _default_call_mcp_tool_node -> here`,
    there's already a running event loop in the calling thread. Plain
    `asyncio.run(...)` would raise "cannot be called from a running
    event loop". This helper detects that case and hops to a worker
    thread that owns its own loop. From a pure-sync caller (tests,
    scripts) it just uses `asyncio.run` on the current thread.

    `coro_factory` must be zero-arg returning a fresh coroutine — a
    coroutine instance can't be shared across event loops, so the
    worker thread builds its own.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop on this thread; safe to run directly.
        return asyncio.run(coro_factory())
    # Copy the calling thread's contextvars (which include the OTel /
    # Langfuse span context the `@observe` decorator pushed) so child
    # spans created inside the React loop — ChatNVIDIA, each MCP tool —
    # attach to the live trace instead of being orphaned. Without this,
    # the worker thread starts with empty contextvars and the trace
    # tree truncates at `call_mcp_tool`.
    ctx = contextvars.copy_context()
    fut = _get_loop_thread().submit(
        lambda: ctx.run(asyncio.run, coro_factory())
    )
    return fut.result()

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
    never needs langchain installed. MCP discovery is async and we may
    be inside FastAPI's request loop, so we route the discovery through
    `_run_async_blocking` rather than calling `asyncio.run` directly.
    """
    from langchain_nvidia_ai_endpoints import ChatNVIDIA  # noqa: PLC0415
    from langgraph.prebuilt import create_react_agent  # noqa: PLC0415

    from src.agents.mcp_client import get_available_tools  # noqa: PLC0415

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")

    tools = _run_async_blocking(get_available_tools)
    if not tools:
        raise RuntimeError(
            "no MCP tools discovered — is the shopping_support server reachable?"
        )

    llm = ChatNVIDIA(
        model=model,
        api_key=api_key,
        temperature=0.0,  # deterministic tool selection
    )
    # Note on parallel tool calls: some NIM models (notably the
    # `meta/llama-3.x-instruct` deployments) reject assistant messages
    # with multiple tool calls — the API returns 400 "this model only
    # supports single tool-calls at once". We tried
    # `.bind(parallel_tool_calls=False)` to force sequential calls;
    # NIM's Llama deployments don't honor that OpenAI-API kwarg, so the
    # only working fix was to switch `TOOL_AGENT_MODEL` to a model that
    # natively supports parallel calls (Mistral / Mixtral families
    # generally do).
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


@observe(name="call_mcp_tool")
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

    We call `agent.ainvoke` (async) through `_run_async_blocking` rather
    than the sync `.invoke`. The React agent dispatches to async MCP
    tools internally; using `.invoke` from inside FastAPI's running
    loop would hit the same "cannot run asyncio.run from a running
    loop" failure that `_build_agent` had on the first call.
    """
    from langchain_core.messages import HumanMessage  # noqa: PLC0415

    t_start = time.perf_counter()
    try:
        if agent is None:
            agent = get_agent()

        # Pass the Langfuse callback into the React agent's invoke so
        # ChatNVIDIA + each MCP tool show up as nested spans under
        # `call_mcp_tool` — same pattern the LCEL chain in `generate`
        # uses. `langfuse_callback()` returns None when tracing is
        # disabled, in which case we just omit the config entirely.
        cb = langfuse_callback()
        invoke_config: dict[str, Any] | None = (
            {"callbacks": [cb]} if cb is not None else None
        )

        async def _invoke():
            if invoke_config is not None:
                return await agent.ainvoke(
                    {"messages": [HumanMessage(content=question)]},
                    config=invoke_config,
                )
            return await agent.ainvoke(
                {"messages": [HumanMessage(content=question)]}
            )

        result = _run_async_blocking(_invoke)
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
