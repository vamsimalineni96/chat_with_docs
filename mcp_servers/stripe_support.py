"""Stripe MCP server for the customer-support demo.

Exposes simple, focused tools that wrap the Stripe Python SDK.
Same pattern as shopping_support.py — small tool schemas so the
ReAct model can select tools quickly without hitting context limits.

Requires: pip install stripe
Requires: STRIPE_SECRET_KEY env var (rk_test_... for test mode)

Run directly to start the server on stdio transport:

    python mcp_servers/stripe_support.py
"""

import os
from typing import Any

import stripe
from mcp.server.fastmcp import FastMCP

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")

mcp = FastMCP("stripe_support")


# --- Tools -------------------------------------------------------------------


@mcp.tool()
def create_customer(name: str, email: str) -> dict[str, Any]:
    """Create a new Stripe customer with the given name and email address.

    Use this when a user asks to register, add, or create a customer account.
    Returns the new customer's ID, name, and email.
    """
    if not stripe.api_key:
        return {"error": "Stripe API key not configured"}
    try:
        customer = stripe.Customer.create(name=name, email=email)
        return {"id": customer.id, "name": customer.name, "email": customer.email, "created": customer.created}
    except stripe.StripeError as e:
        return {"error": str(e)}


@mcp.tool()
def list_customers(email: str | None = None, limit: int = 10) -> dict[str, Any]:
    """List Stripe customers, optionally filtered by exact email address.

    Use this to look up whether a customer exists, or to list recent customers.
    Returns id, name, email for each customer.
    """
    if not stripe.api_key:
        return {"error": "Stripe API key not configured"}
    try:
        params: dict[str, Any] = {"limit": limit}
        if email:
            params["email"] = email
        result = stripe.Customer.list(**params)
        return {
            "customers": [
                {"id": c.id, "name": c.name, "email": c.email}
                for c in result.data
            ],
            "count": len(result.data),
        }
    except stripe.StripeError as e:
        return {"error": str(e)}


@mcp.tool()
def list_payment_intents(customer_id: str | None = None, limit: int = 10) -> dict[str, Any]:
    """List recent Stripe payment intents, optionally filtered by customer ID.

    Use this when a user asks about their payments, charges, or billing history.
    Payment intent IDs follow the pattern pi_...
    Returns id, amount (in cents), currency, status, and created timestamp.
    """
    if not stripe.api_key:
        return {"error": "Stripe API key not configured"}
    try:
        params: dict[str, Any] = {"limit": limit}
        if customer_id:
            params["customer"] = customer_id
        result = stripe.PaymentIntent.list(**params)
        return {
            "payment_intents": [
                {
                    "id": p.id,
                    "amount": p.amount,
                    "currency": p.currency,
                    "status": p.status,
                    "created": p.created,
                }
                for p in result.data
            ],
            "count": len(result.data),
        }
    except stripe.StripeError as e:
        return {"error": str(e)}


@mcp.tool()
def create_product(name: str, description: str | None = None) -> dict[str, Any]:
    """Create a new Stripe product with the given name and optional description.

    Use this when a user asks to add or register a product in the catalog.
    Returns the new product's ID and name.
    """
    if not stripe.api_key:
        return {"error": "Stripe API key not configured"}
    try:
        params: dict[str, Any] = {"name": name}
        if description:
            params["description"] = description
        product = stripe.Product.create(**params)
        return {"id": product.id, "name": product.name, "description": product.description}
    except stripe.StripeError as e:
        return {"error": str(e)}


@mcp.tool()
def list_products(limit: int = 10) -> dict[str, Any]:
    """List products in the Stripe catalog.

    Use this when a user asks what products are available in the system.
    Returns id, name, and description for each product.
    """
    if not stripe.api_key:
        return {"error": "Stripe API key not configured"}
    try:
        result = stripe.Product.list(limit=limit, active=True)
        return {
            "products": [
                {"id": p.id, "name": p.name, "description": p.description}
                for p in result.data
            ],
            "count": len(result.data),
        }
    except stripe.StripeError as e:
        return {"error": str(e)}


@mcp.tool()
def create_refund(payment_intent_id: str, reason: str | None = None) -> dict[str, Any]:
    """Issue a full refund for a payment intent.

    Use this ONLY after confirming the payment intent ID with the user.
    Refunds cannot be undone. Valid reasons: duplicate, fraudulent, requested_by_customer.
    Returns refund ID and status.
    """
    if not stripe.api_key:
        return {"error": "Stripe API key not configured"}
    try:
        params: dict[str, Any] = {"payment_intent": payment_intent_id}
        if reason:
            params["reason"] = reason
        refund = stripe.Refund.create(**params)
        return {"id": refund.id, "status": refund.status, "amount": refund.amount}
    except stripe.StripeError as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()
