"""Behavior contract for the MCP client wrapper.

We never spawn the real MCP subprocess in these tests — that's
slow, flaky, and not what we're testing. The wrapper's job is to
discover, cache, dispatch, and fail gracefully; we exercise each of
those by monkeypatching `_client` with a stub that returns
controlled tool lists.

Tests use plain `asyncio.run()` to drive the async calls — no
pytest-asyncio dependency needed, and no fixture machinery to
introduce.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents import mcp_client


@pytest.fixture(autouse=True)
def reset_mcp_cache():
    """Each test starts with no cached client + no cached tools."""
    mcp_client.reset_cache()
    yield
    mcp_client.reset_cache()


def _fake_tool(name: str, return_value: Any = "ok"):
    """Build a stand-in for a BaseTool. Only the attributes the wrapper
    actually touches (`name`, `ainvoke`) need to be present.
    """
    tool = MagicMock()
    tool.name = name
    tool.ainvoke = AsyncMock(return_value=return_value)
    return tool


def _install_fake_client(monkeypatch, tools, *, get_tools_raises=None):
    """Replace the module-level client with one that returns the given
    tools (or raises). Returns the fake client so tests can assert on
    call counts.
    """
    fake = MagicMock()
    if get_tools_raises is not None:
        fake.get_tools = AsyncMock(side_effect=get_tools_raises)
    else:
        fake.get_tools = AsyncMock(return_value=tools)
    monkeypatch.setattr(mcp_client, "get_client", lambda: fake)
    return fake


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_get_available_tools_returns_discovered_tools(monkeypatch):
    _install_fake_client(
        monkeypatch,
        [_fake_tool("get_order_status"), _fake_tool("check_inventory")],
    )
    tools = asyncio.run(mcp_client.get_available_tools())
    assert [t.name for t in tools] == ["get_order_status", "check_inventory"]


def test_get_available_tools_caches_after_first_call(monkeypatch):
    fake = _install_fake_client(monkeypatch, [_fake_tool("only_one")])
    asyncio.run(mcp_client.get_available_tools())
    asyncio.run(mcp_client.get_available_tools())
    assert fake.get_tools.await_count == 1


def test_get_available_tools_returns_empty_list_on_failure(monkeypatch):
    """Server down → empty list + logged warning. Never raises."""
    _install_fake_client(
        monkeypatch,
        [],
        get_tools_raises=RuntimeError("MCP server unreachable"),
    )
    tools = asyncio.run(mcp_client.get_available_tools())
    assert tools == []


def test_get_available_tools_does_not_cache_a_failure(monkeypatch):
    """A failed discovery shouldn't poison the cache — a retry should
    actually retry, not return the cached empty list forever.
    """
    fake = _install_fake_client(
        monkeypatch, [], get_tools_raises=RuntimeError("temporary")
    )
    asyncio.run(mcp_client.get_available_tools())
    # Swap to a successful discovery without resetting cache manually.
    fake.get_tools = AsyncMock(return_value=[_fake_tool("recovered")])
    tools = asyncio.run(mcp_client.get_available_tools())
    assert [t.name for t in tools] == ["recovered"]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def test_call_tool_dispatches_to_named_tool(monkeypatch):
    order_tool = _fake_tool(
        "get_order_status", return_value={"status": "shipped"}
    )
    inv_tool = _fake_tool("check_inventory", return_value={"in_stock": 5})
    _install_fake_client(monkeypatch, [order_tool, inv_tool])

    result = asyncio.run(
        mcp_client.call_tool("get_order_status", {"order_id": "ORD-1001"})
    )

    assert result == {"status": "shipped"}
    order_tool.ainvoke.assert_awaited_once_with({"order_id": "ORD-1001"})
    inv_tool.ainvoke.assert_not_awaited()


def test_call_tool_raises_for_unknown_name(monkeypatch):
    _install_fake_client(monkeypatch, [_fake_tool("only_one_tool")])
    with pytest.raises(ValueError, match="Tool 'does_not_exist'"):
        asyncio.run(mcp_client.call_tool("does_not_exist", {}))


def test_call_tool_propagates_tool_exceptions(monkeypatch):
    """If the tool itself errors (e.g., bad args reaching the MCP server),
    the wrapper does NOT swallow it — the caller decides whether to
    retry, fall back, or surface to the user.
    """
    bad_tool = _fake_tool("flaky_tool")
    bad_tool.ainvoke = AsyncMock(side_effect=RuntimeError("tool blew up"))
    _install_fake_client(monkeypatch, [bad_tool])

    with pytest.raises(RuntimeError, match="tool blew up"):
        asyncio.run(mcp_client.call_tool("flaky_tool", {"x": 1}))


# ---------------------------------------------------------------------------
# Config plumbing — pure functions, no async needed
# ---------------------------------------------------------------------------


def test_resolve_server_path_default(monkeypatch):
    """Default path resolves to an absolute path under the repo root."""
    monkeypatch.delenv("MCP_SHOPPING_SERVER_PATH", raising=False)
    path = mcp_client._resolve_server_path()
    assert path.endswith("mcp_servers/shopping_support.py")
    # Absolute so launching from any cwd works.
    assert path.startswith("/")


def test_resolve_server_path_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_SHOPPING_SERVER_PATH", str(tmp_path / "custom.py"))
    assert mcp_client._resolve_server_path() == str(tmp_path / "custom.py")


def test_build_server_config_uses_stdio_transport():
    """Config shape is the contract with langchain-mcp-adapters — pin it
    so an accidental rename can't silently break the connection.
    """
    cfg = mcp_client._build_server_config()
    assert "shopping_support" in cfg
    server = cfg["shopping_support"]
    assert server["transport"] == "stdio"
    assert server["command"].endswith("python") or server[
        "command"
    ].endswith("python3") or "python" in server["command"]
    assert server["args"][0].endswith("shopping_support.py")
