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
def list_customers(email: str | None = None, name: str | None = None, limit: int = 10) -> dict[str, Any]:
    """List Stripe customers, optionally filtered by email or name.

    Use this to look up whether a customer exists, find a customer by name,
    or list recent customers. Returns id, name, email for each customer.

    Provide email for exact match. Provide name to search by full or partial name.
    Leave both empty to list the most recent customers.
    """
    if not stripe.api_key:
        return {"error": "Stripe API key not configured"}
    try:
        if name:
            result = stripe.Customer.search(query=f'name:"{name}"', limit=limit)
            customers = result.data
        else:
            params: dict[str, Any] = {"limit": limit}
            if email:
                params["email"] = email
            result = stripe.Customer.list(**params)
            customers = result.data
        return {
            "customers": [
                {"id": c.id, "name": c.name, "email": c.email}
                for c in customers
            ],
            "count": len(customers),
        }
    except stripe.StripeError as e:
        return {"error": str(e)}


@mcp.tool()
def get_customer_payments(name_or_email: str, limit: int = 5) -> dict[str, Any]:
    """Look up a customer by name or email and return their recent payment history.

    Use this whenever a user mentions a customer name or email in the context
    of payments, charges, refunds, or billing history. This tool handles the
    customer lookup and payment retrieval in one step — do NOT call
    list_customers and list_payment_intents separately.

    Returns customer id, name, email, and their recent payment intents with
    id, amount, currency, status, and description.
    """
    if not stripe.api_key:
        return {"error": "Stripe API key not configured"}
    try:
        # Try email match first, then name search
        if "@" in name_or_email:
            result = stripe.Customer.list(email=name_or_email, limit=1)
            customers = result.data
        else:
            result = stripe.Customer.search(query=f'name:"{name_or_email}"', limit=1)
            customers = result.data

        if not customers:
            return {"error": f"No customer found matching '{name_or_email}'"}

        customer = customers[0]
        payments = stripe.PaymentIntent.list(customer=customer.id, limit=limit)
        return {
            "customer": {
                "id": customer.id,
                "name": customer.name,
                "email": customer.email,
            },
            "payment_intents": [
                {
                    "id": p.id,
                    "amount_cents": p.amount,
                    "currency": p.currency,
                    "status": p.status,
                    "description": p.description,
                }
                for p in payments.data
            ],
            "count": len(payments.data),
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
    """List products in the Stripe catalog including their prices.

    Use this when a user asks what products are available, what something
    costs, or for product + price information. Returns id, name, description,
    and the default price (amount in cents and currency) for each product.
    """
    if not stripe.api_key:
        return {"error": "Stripe API key not configured"}
    try:
        result = stripe.Product.list(limit=limit, active=True, expand=["data.default_price"])
        products = []
        for p in result.data:
            entry: dict[str, Any] = {"id": p.id, "name": p.name, "description": p.description}
            dp = p.default_price
            if dp and hasattr(dp, "unit_amount"):
                entry["price_cents"] = dp.unit_amount
                entry["currency"] = dp.currency
            else:
                # fall back to fetching the most recent price for this product
                prices = stripe.Price.list(product=p.id, active=True, limit=1)
                if prices.data:
                    entry["price_cents"] = prices.data[0].unit_amount
                    entry["currency"] = prices.data[0].currency
            products.append(entry)
        return {"products": products, "count": len(products)}
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


@mcp.tool()
def create_invoice(name_or_email: str, product_name: str, amount_cents: int, currency: str = "usd") -> dict[str, Any]:
    """Create and finalize a Stripe invoice for a customer.

    Accepts customer name (e.g. "Jane Smith") or email address.
    Adds the product as a line item at the given amount, then finalizes
    the invoice so it is ready to send.

    amount_cents: price in cents (e.g. 7999 for $79.99)
    currency: ISO currency code, default "usd"

    Returns invoice ID, status, amount due, and a hosted invoice URL.
    """
    if not stripe.api_key:
        return {"error": "Stripe API key not configured"}
    try:
        if "@" in name_or_email:
            customers = stripe.Customer.list(email=name_or_email, limit=1).data
        else:
            customers = stripe.Customer.search(query=f'name:"{name_or_email}"', limit=1).data
        if not customers:
            return {"error": f"No customer found matching '{name_or_email}'"}
        customer_id = customers[0].id

        invoice = stripe.Invoice.create(customer=customer_id)
        stripe.InvoiceItem.create(
            customer=customer_id,
            amount=amount_cents,
            currency=currency,
            description=product_name,
            invoice=invoice.id,
        )
        finalized = stripe.Invoice.finalize_invoice(invoice.id)
        return {
            "invoice_id": finalized.id,
            "status": finalized.status,
            "amount_due": finalized.amount_due,
            "currency": finalized.currency,
            "hosted_invoice_url": finalized.hosted_invoice_url,
        }
    except stripe.StripeError as e:
        return {"error": str(e)}


@mcp.tool()
def list_invoices(name_or_email: str, limit: int = 5) -> dict[str, Any]:
    """List recent invoices for a customer identified by name or email.

    Accepts either a full name (e.g. "Jane Smith") or an email address.
    Returns invoice ID, status, amount due, currency, and a hosted URL
    the customer can use to view or pay the invoice.
    """
    if not stripe.api_key:
        return {"error": "Stripe API key not configured"}
    try:
        if "@" in name_or_email:
            result = stripe.Customer.list(email=name_or_email, limit=1)
            customers = result.data
        else:
            result = stripe.Customer.search(query=f'name:"{name_or_email}"', limit=1)
            customers = result.data

        if not customers:
            return {"error": f"No customer found matching '{name_or_email}'"}

        customer_id = customers[0].id
        invoices = stripe.Invoice.list(customer=customer_id, limit=limit)
        return {
            "customer": {"id": customer_id, "name": customers[0].name, "email": customers[0].email},
            "invoices": [
                {
                    "id": inv.id,
                    "status": inv.status,
                    "amount_due": inv.amount_due,
                    "currency": inv.currency,
                    "hosted_invoice_url": inv.hosted_invoice_url,
                }
                for inv in invoices.data
            ],
            "count": len(invoices.data),
        }
    except stripe.StripeError as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stripe MCP server")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse"],
        help="stdio for subprocess mode (default), sse for standalone HTTP service",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (SSE mode only)")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind (SSE mode only)")
    args = parser.parse_args()

    if args.transport == "sse":
        # Set host/port directly on the settings object — env vars are read
        # at FastMCP instantiation time so os.environ changes are too late.
        # Also open allowed_hosts so Docker bridge network requests aren't blocked.
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        # Disable DNS rebinding protection — this is an internal service
        # called by the app, never by a browser, so the protection is
        # unnecessary and blocks Docker bridge network requests.
        mcp.settings.transport_security.enable_dns_rebinding_protection = False
        mcp.run(transport="sse")
    else:
        mcp.run()
