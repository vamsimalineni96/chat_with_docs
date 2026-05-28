"""ReAct sub-agent that handles the `tool_call` branch of the chat graph.

When the classifier emits `intent: tool_call`, the graph routes here.
This module wraps `langgraph.prebuilt.create_react_agent` over the
MCP tools discovered by `src.agents.mcp_client`. The ReAct loop
itself does the work: the LLM sees the discovered tools, decides
which one(s) to call, sees the result, and synthesizes a final
user-facing answer — all in a single `ainvoke`.

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

import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml

from src.utils.observability import langfuse_callback, observe, update_current_observation

logger = logging.getLogger(__name__)

DEFAULT_TOOL_AGENT_MODEL = os.environ.get(
    "TOOL_AGENT_MODEL", "meta/llama-3.1-70b-instruct"
)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "tool_agent.yaml"
_PROMPTS: dict[str, str] | None = None


def _load_prompts() -> dict[str, str]:
    global _PROMPTS
    if _PROMPTS is None:
        with open(_PROMPT_PATH) as fh:
            loaded = yaml.safe_load(fh)
        for required in ("system_prompt", "user_prompt"):
            if required not in loaded:
                raise RuntimeError(
                    f"tool_agent prompts file missing required key: {required}"
                )
        _PROMPTS = loaded
    return _PROMPTS


def _build_user_message(question: str) -> str:
    return _load_prompts()["user_prompt"].format(question=question)

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


async def _build_agent(model: str) -> Any:
    """Construct a fresh React agent bound to the current MCP tools.

    Imports are deferred so the test env (which mocks the agent via DI)
    never needs langchain installed.
    """
    from langchain_nvidia_ai_endpoints import ChatNVIDIA  # noqa: PLC0415
    from langgraph.prebuilt import create_react_agent  # noqa: PLC0415

    from src.agents.mcp_client import get_available_tools  # noqa: PLC0415

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")

    tools = await get_available_tools()
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
    system_prompt = _load_prompts()["system_prompt"]
    return create_react_agent(llm, tools, prompt=system_prompt)


async def get_agent(model: str | None = None) -> Any:
    """Return the process-wide React agent, building it on first call."""
    global _cached_agent
    if _cached_agent is None:
        _cached_agent = await _build_agent(model or DEFAULT_TOOL_AGENT_MODEL)
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
async def run_tool_agent(
    question: str,
    *,
    history: list[dict] | None = None,
    agent: Any | None = None,
) -> dict[str, Any]:
    """Run the React agent on `question` and return a node-shaped dict.

    `history` is the prior conversation turns as a list of
    {"role": "user"|"assistant", "content": str} dicts (same shape as
    ChatGraphState["history"]). When provided, the turns are prepended to
    the message list so the ReAct agent can resolve pronouns and follow-up
    questions ("what's the return policy for *that* order?").

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
    from langchain_core.messages import AIMessage, HumanMessage  # noqa: PLC0415

    t_start = time.perf_counter()
    try:
        if agent is None:
            agent = await get_agent()

        # Convert prior turns to LangChain message objects. System messages
        # are skipped — the ReAct agent already has its own system prompt and
        # a second one would conflict.
        history_msgs: list[Any] = []
        for turn in history or []:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                history_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                history_msgs.append(AIMessage(content=content))

        # Stamp the span input so Langfuse shows the full context that
        # was handed to the ReAct loop: question, turn count, and each
        # prior turn (content truncated to 300 chars so the UI stays
        # readable on long conversations).
        update_current_observation(
            input={
                "question": question,
                "history_turns": len(history_msgs),
                "history": [
                    {
                        "role": turn.get("role"),
                        "content": (turn.get("content") or "")[:300],
                    }
                    for turn in (history or [])
                ],
            }
        )

        # Pass the Langfuse callback into the React agent's invoke so
        # ChatNVIDIA + each MCP tool show up as nested spans under
        # `call_mcp_tool`. OTel span context (which carries the active
        # Langfuse trace) propagates naturally through `await` in the
        # same event loop — no manual context copying needed.
        cb = langfuse_callback()
        invoke_state = {"messages": [HumanMessage(content=_build_user_message(question))]}
        if cb is not None:
            result = await agent.ainvoke(invoke_state, config={"callbacks": [cb]})
        else:
            result = await agent.ainvoke(invoke_state)
    except Exception as e:
        logger.warning("Tool sub-agent failed: %s", e, exc_info=True)
        t_end = time.perf_counter()
        error_str = str(e)
        if "no MCP tools" in error_str:
            failure_reason = "no_tools"
        elif "NVIDIA_API_KEY" in error_str:
            failure_reason = "no_api_key"
        else:
            failure_reason = "agent_error"
        return {
            "answer": TOOL_FAILURE_ANSWER,
            "tool_calls": [],
            "tool_failure_reason": failure_reason,
            "t_llm_start": t_start,
            "t_llm_end": t_end,
            "error": error_str,
        }

    t_end = time.perf_counter()
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        logger.warning("Tool sub-agent returned no messages")
        return {
            "answer": TOOL_FAILURE_ANSWER,
            "tool_calls": [],
            "tool_failure_reason": "no_messages",
            "t_llm_start": t_start,
            "t_llm_end": t_end,
            "error": "agent returned empty message list",
        }

    final = messages[-1]
    answer = getattr(final, "content", None) or ""
    if not answer:
        # Model returned an empty final message — can happen when the tool
        # schema payload is too large for the model to synthesize a reply.
        logger.warning("Tool sub-agent returned empty content in final message")
        return {
            "answer": TOOL_FAILURE_ANSWER,
            "tool_calls": _extract_tool_calls(messages),
            "tool_failure_reason": "empty_content",
            "t_llm_start": t_start,
            "t_llm_end": t_end,
            "error": "agent returned empty content",
        }
    return {
        "answer": answer,
        "tool_calls": _extract_tool_calls(messages),
        "tool_failure_reason": None,
        "t_llm_start": t_start,
        "t_llm_end": t_end,
        "error": None,
    }
