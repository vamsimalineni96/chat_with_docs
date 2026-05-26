"""Behavior contract for the MCP ReAct sub-agent wrapper.

`tool_node.run_tool_agent` is the seam tests target. We never build a
real `create_react_agent` here — the `agent=` DI parameter lets us
inject a fake whose `.ainvoke(...)` returns a canned message list, so
no langchain / langgraph.prebuilt / NVIDIA HTTP traffic ever fires.

The wrapper's job is to:
  - call the agent and unwrap the final message's content
  - record each tool call the agent made (for trace + debug)
  - never raise — return the canned failure answer on any exception
  - stamp timings around the invocation

That's what we pin below.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from src.agents import tool_node


@pytest.fixture(autouse=True)
def reset_tool_node_cache():
    """Each test starts with no cached agent."""
    tool_node.reset_cache()
    yield
    tool_node.reset_cache()


def _msg(content: str = "", tool_calls: list[dict[str, Any]] | None = None):
    """Stand-in for a LangChain Message. Only `content` and `tool_calls`
    are read by `_extract_tool_calls` / `run_tool_agent`.
    """
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


class _FakeAgent:
    """Mimics the surface of a `create_react_agent` compiled graph:
    `await .ainvoke({"messages": [...]})` returns `{"messages": [...]}`
    where the last message is the synthesized answer.
    """

    def __init__(self, messages: list[Any]):
        self._messages = messages
        self.invoke_calls: list[dict[str, Any]] = []

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # Accept the optional Langfuse callback config for signature parity
        # but don't assert on it — these tests target the wrapper, not tracing.
        self.invoke_calls.append(state)
        return {"messages": self._messages}


class _RaisingAgent:
    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        raise RuntimeError("simulated agent blow-up")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_tool_agent_returns_final_message_content():
    agent = _FakeAgent(
        [
            _msg(content="user question"),
            _msg(
                content="",
                tool_calls=[
                    {"name": "get_order_status", "args": {"order_id": "ORD-1001"}}
                ],
            ),
            _msg(content="tool result was: shipped"),
            _msg(content="Your order ORD-1001 is shipped via FedEx."),
        ]
    )
    result = asyncio.run(
        tool_node.run_tool_agent("Where is order ORD-1001?", agent=agent)
    )

    assert result["answer"] == "Your order ORD-1001 is shipped via FedEx."
    assert result["error"] is None
    # Timing fields are present and ordered.
    assert result["t_llm_end"] >= result["t_llm_start"]


def test_run_tool_agent_extracts_all_tool_calls():
    """When the ReAct loop chains multiple tools (e.g., "is SKU-001 in
    stock AND what's the return policy for electronics?"), each call
    is captured separately in the trace record.
    """
    agent = _FakeAgent(
        [
            _msg(
                content="",
                tool_calls=[
                    {"name": "check_inventory", "args": {"sku": "SKU-001"}}
                ],
            ),
            _msg(content="inventory result"),
            _msg(
                content="",
                tool_calls=[
                    {
                        "name": "get_return_policy_window",
                        "args": {"category": "electronics"},
                    }
                ],
            ),
            _msg(content="policy result"),
            _msg(content="In stock; 30-day return window for electronics."),
        ]
    )
    result = asyncio.run(tool_node.run_tool_agent("compound question", agent=agent))
    assert result["tool_calls"] == [
        {"name": "check_inventory", "args": {"sku": "SKU-001"}},
        {
            "name": "get_return_policy_window",
            "args": {"category": "electronics"},
        },
    ]


def test_run_tool_agent_forwards_question_as_human_message():
    """The wrapper builds a HumanMessage from the question string and
    passes it as the only entry in `messages` — pinning this contract
    so a future refactor doesn't silently drop the question.
    """
    agent = _FakeAgent([_msg(content="ok")])
    asyncio.run(tool_node.run_tool_agent("hello tools", agent=agent))

    assert len(agent.invoke_calls) == 1
    sent = agent.invoke_calls[0]["messages"]
    assert len(sent) == 1
    # The HumanMessage content is the user's question.
    assert sent[0].content == "hello tools"


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_run_tool_agent_returns_canned_answer_on_agent_error():
    """Any exception from the React agent is swallowed and a canned
    apology returned, so the graph's postprocess step always has an
    `answer` to inspect.
    """
    result = asyncio.run(tool_node.run_tool_agent("anything", agent=_RaisingAgent()))
    assert result["answer"] == tool_node.TOOL_FAILURE_ANSWER
    assert result["tool_calls"] == []
    assert result["error"] is not None
    assert "blow-up" in result["error"]


def test_run_tool_agent_handles_empty_message_list():
    """Defensive: an agent that returns no messages shouldn't crash —
    same failure shape as a raised exception.
    """
    agent = _FakeAgent([])
    result = asyncio.run(tool_node.run_tool_agent("anything", agent=agent))
    assert result["answer"] == tool_node.TOOL_FAILURE_ANSWER
    assert result["tool_calls"] == []
    assert result["error"] is not None


def test_run_tool_agent_falls_back_when_build_agent_fails(monkeypatch):
    """No `agent=` injected and `_build_agent` raises (e.g., NVIDIA key
    missing or MCP discovery returned []) → fallback path fires
    without bothering the caller. Tests this exercises the cache path
    too: a failed build does not poison the cache (the next call gets
    a fresh attempt).
    """

    async def _boom(model):
        raise RuntimeError("NVIDIA_API_KEY not set")

    monkeypatch.setattr(tool_node, "_build_agent", _boom)
    result = asyncio.run(tool_node.run_tool_agent("anything"))
    assert result["answer"] == tool_node.TOOL_FAILURE_ANSWER
    assert "NVIDIA_API_KEY" in result["error"]
    # Cache was not populated by the failed build → reset_cache is a no-op.
    assert tool_node._cached_agent is None


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_get_agent_caches_after_first_build(monkeypatch):
    """`_build_agent` should only run once across multiple `get_agent`
    calls in the same process.
    """
    build_count = {"n": 0}

    async def _fake_build(model):
        build_count["n"] += 1
        return _FakeAgent([_msg(content="cached")])

    monkeypatch.setattr(tool_node, "_build_agent", _fake_build)
    asyncio.run(tool_node.get_agent())
    asyncio.run(tool_node.get_agent())
    asyncio.run(tool_node.get_agent())
    assert build_count["n"] == 1
