"""MCP client wrapper for the chat agent.

Discovers and invokes tools exposed by MCP servers. Currently
configured to spawn the in-repo mock shopping-support server as a
stdio subprocess; swap in additional / remote servers by editing
`_build_server_config` (or, later, lifting that config to YAML/env).

Design notes:

- The underlying `MultiServerMCPClient` is async-only. PR #5 wires
  this into a LangGraph node where async is natural; for now both
  functions here are `async def`.
- Tool discovery is cached at module level on first success. A real
  deployment would want a refresh / liveness check, but for a
  portfolio demo "one shot per process" is fine.
- Failure to reach the MCP server returns an empty tool list and
  logs a warning rather than raising. The agent (PR #5) treats
  "no tools available" as a routing signal — it'll just stay on the
  RAG path until the server comes back up.

PR #4 deliberately does NOT touch the graph. This file exists in
isolation; the only callers in this PR are tests and a manual smoke
script. PR #5 adds the `call_mcp_tool` graph node that uses these
functions.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

# Default path to the mock server (relative to repo root). Override
# via env var to point at a different MCP server script — useful
# when testing real backends instead of the demo mock.
_DEFAULT_SERVER_PATH = "mcp_servers/shopping_support.py"


def _resolve_server_path() -> str:
    """Return an absolute path to the MCP server script.

    Resolves relative paths against the project root (two levels up
    from this file: `src/agents/mcp_client.py` → repo root) so the
    server can be launched regardless of where the FastAPI app was
    started from.
    """
    raw = os.environ.get("MCP_SHOPPING_SERVER_PATH", _DEFAULT_SERVER_PATH)
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    project_root = Path(__file__).resolve().parents[2]
    return str(project_root / path)


def _build_server_config() -> dict[str, Any]:
    """Build the MultiServerMCPClient config dict.

    Always includes the local shopping_support mock (zero-dependency
    demo fallback). When STRIPE_SECRET_KEY is present in the
    environment, the real Stripe MCP server is added alongside it —
    its tools are merged into the same flat list that the ReAct agent
    sees, so the agent can answer both mock-order questions and real
    Stripe queries in a single turn.

    This is the key MCP multi-server pattern: one client, N servers,
    one flat tool list — the agent never knows or cares which server
    a tool came from.
    """
    config: dict[str, Any] = {
        "shopping_support": {
            "command": sys.executable,
            "args": [_resolve_server_path()],
            "transport": "stdio",
        }
    }

    stripe_key = os.environ.get("STRIPE_SECRET_KEY")
    if stripe_key:
        # Use our own Python Stripe MCP server (mcp_servers/stripe_support.py)
        # instead of the npx @stripe/mcp package. The npx package exposes a
        # catch-all `stripe_api_execute` tool with a massive schema that causes
        # Llama-family models to time out. Our Python server has 6 focused tools
        # with small schemas — same pattern as shopping_support.py.
        # STRIPE_SECRET_KEY is inherited naturally since this is a Python
        # subprocess in the same process environment.
        project_root = Path(__file__).resolve().parents[2]
        stripe_server_path = str(project_root / "mcp_servers" / "stripe_support.py")
        config["stripe"] = {
            "command": sys.executable,
            "args": [stripe_server_path],
            "transport": "stdio",
            "env": {**os.environ, "STRIPE_SECRET_KEY": stripe_key},
        }
        logger.info("Stripe MCP server enabled (STRIPE_SECRET_KEY is set)")
    else:
        logger.info(
            "Stripe MCP server disabled — set STRIPE_SECRET_KEY to enable real Stripe tools"
        )

    return config


# Module-level caches. `get_client()` / `get_available_tools()` are
# the only readers; `reset_cache()` is the only writer outside the
# normal flow (called from tests, and from any future "reconnect on
# failure" code).
_client: MultiServerMCPClient | None = None
_cached_tools: list[BaseTool] | None = None


def get_client() -> MultiServerMCPClient:
    """Return the process-wide MCP client.

    Construction is cheap (no network until `get_tools` is awaited),
    so the cache exists mostly for symmetry with `_cached_tools` and
    to keep callers from rebuilding the config dict each call.
    """
    global _client
    if _client is None:
        _client = MultiServerMCPClient(_build_server_config())
    return _client


async def get_available_tools() -> list[BaseTool]:
    """Discover tools from all configured MCP servers.

    Cached after the first successful discovery. On failure (server
    not running, stdio handshake error, etc.), returns an empty list
    and logs a warning — graceful degradation lets the rest of the
    app run when the MCP server is down.
    """
    global _cached_tools
    if _cached_tools is not None:
        return _cached_tools
    try:
        client = get_client()
        tools = await client.get_tools()
        _cached_tools = list(tools)
        logger.info(
            "Discovered %d MCP tool(s): %s",
            len(_cached_tools),
            [t.name for t in _cached_tools],
        )
        return _cached_tools
    except Exception as e:
        logger.warning(
            "Failed to discover MCP tools (continuing with empty tool set): %s",
            e,
        )
        return []


async def call_tool(name: str, args: dict[str, Any]) -> Any:
    """Invoke a tool by name with the given args dict.

    Raises ValueError if no tool with that name was discovered.
    Exceptions raised by the tool itself propagate as-is — the
    caller (PR #5's graph node) decides how to surface them.
    """
    tools = await get_available_tools()
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        raise ValueError(
            f"Tool {name!r} not found. Available: {[t.name for t in tools]}"
        )
    return await tool.ainvoke(args)


def reset_cache() -> None:
    """Clear the cached client + tools.

    Used by tests to isolate state. Also useful if you ever want to
    force a re-discovery (e.g., the MCP server was restarted with
    new tools) without bouncing the FastAPI process.
    """
    global _client, _cached_tools
    _client = None
    _cached_tools = None
