"""Mock MCP server for the customer-support demo.

Exposes three tools a typical shopping-app chatbot would need —
order lookup, inventory check, return-policy lookup — backed by
hand-rolled dicts so the demo has zero infrastructure dependencies.
Edit FAKE_ORDERS / FAKE_INVENTORY / RETURN_POLICY_DAYS to exercise
specific failure paths.

Run directly to start the server on stdio transport:

    python mcp_servers/shopping_support.py

The agent (src/agents/mcp_client.py) spawns this script as a
subprocess on first use; no separate process management needed.
"""

from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("shopping_support")


# --- Fake data store ---------------------------------------------------------
# Frozen at module load. Mutable in tests if needed (just re-import).

FAKE_ORDERS: dict[str, dict[str, Any]] = {
    "ORD-1001": {
        "status": "shipped",
        "last_update": "2026-05-22T14:30:00Z",
        "delivery_eta": "2026-05-27",
        "carrier": "FedEx",
        "tracking": "1Z9999999999999999",
    },
    "ORD-1002": {
        "status": "processing",
        "last_update": "2026-05-23T09:15:00Z",
        "delivery_eta": "2026-05-29",
        "carrier": None,
        "tracking": None,
    },
    "ORD-1003": {
        "status": "delivered",
        "last_update": "2026-05-20T16:45:00Z",
        "delivery_eta": "2026-05-20",
        "carrier": "UPS",
        "tracking": "1Z8888888888888888",
    },
}

FAKE_INVENTORY: dict[str, dict[str, Any]] = {
    "SKU-001": {"name": "Wireless Headphones", "in_stock": 42, "warehouses": 3},
    "SKU-002": {"name": "USB-C Cable 6ft", "in_stock": 0, "warehouses": 0},
    "SKU-003": {"name": "Mechanical Keyboard", "in_stock": 17, "warehouses": 2},
}

RETURN_POLICY_DAYS: dict[str, int] = {
    "electronics": 30,
    "apparel": 60,
    "books": 14,
    "perishables": 0,
}


# --- Tools -------------------------------------------------------------------
#
# Each @mcp.tool() docstring is the description the LLM sees during tool
# selection — keep it imperative and specific, no marketing copy. The
# function signature (with type hints) doubles as the args schema.


@mcp.tool()
def get_order_status(order_id: str) -> dict[str, Any]:
    """Look up the current status of an order by its ID.

    Returns shipping status, last update timestamp, delivery ETA,
    and carrier/tracking info when available. Order IDs follow the
    pattern ORD-NNNN.
    """
    order = FAKE_ORDERS.get(order_id)
    if order is None:
        return {"error": f"Order {order_id} not found", "order_id": order_id}
    return {"order_id": order_id, **order}


@mcp.tool()
def check_inventory(sku: str) -> dict[str, Any]:
    """Check current stock for a product SKU across all warehouses.

    Returns product name, total in_stock count, and number of
    warehouses with stock. SKUs follow the pattern SKU-NNN.
    """
    item = FAKE_INVENTORY.get(sku)
    if item is None:
        return {"error": f"SKU {sku} not found in inventory", "sku": sku}
    return {"sku": sku, **item}


@mcp.tool()
def get_return_policy_window(category: str) -> dict[str, Any]:
    """Get the number of days within which an item from this category
    can be returned for a full refund.

    Valid categories: electronics, apparel, books, perishables.
    A return_window_days of 0 means the category is non-returnable.
    """
    days = RETURN_POLICY_DAYS.get(category.lower())
    if days is None:
        return {
            "error": (
                f"Unknown category {category!r}. "
                f"Known: {list(RETURN_POLICY_DAYS.keys())}"
            ),
            "category": category,
        }
    return {"category": category.lower(), "return_window_days": days}


if __name__ == "__main__":
    mcp.run()
