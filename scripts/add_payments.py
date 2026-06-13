"""Add fresh refundable payments to existing seeded customers.

Useful between demo runs so each refund attempt has a payment that
hasn't been refunded yet.

Run from repo root with a full secret key (rk_test_ doesn't allow
payment confirmation):

    STRIPE_SECRET_KEY=sk_test_... python scripts/add_payments.py

Optional: pass --per-customer N to control how many payments to add
per existing customer (default: 3).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

import stripe

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
if not stripe.api_key:
    print("ERROR: STRIPE_SECRET_KEY not set.")
    sys.exit(1)

if not stripe.api_key.startswith(("rk_test_", "sk_test_")):
    print("ERROR: This script only runs against test keys.")
    sys.exit(1)


# Pool of mock purchases — cycled per customer
_PURCHASES = [
    (7999, "Mechanical Keyboard"),
    (14999, "Wireless Headphones"),
    (1999, "USB-C Cable 6ft"),
    (4999, "Mouse Pad XXL"),
    (12999, "Webcam 4K"),
    (3499, "Phone Stand"),
    (8999, "Bluetooth Speaker"),
    (2499, "Notebook Set"),
]


def add_payment(customer: stripe.Customer, amount_cents: int, description: str) -> stripe.PaymentIntent | None:
    """Create a confirmed PaymentIntent for the customer using a test card token."""
    try:
        pi = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency="usd",
            customer=customer.id,
            payment_method="pm_card_visa",
            description=description,
            confirm=True,
            automatic_payment_methods={"enabled": True, "allow_redirects": "never"},
        )
        print(f"  ✅ {customer.name}: ${amount_cents / 100:.2f} for {description} → {pi.id}")
        return pi
    except stripe.StripeError as e:
        print(f"  ❌ {customer.name}: ${amount_cents / 100:.2f} failed — {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Add fresh payments for refund testing")
    parser.add_argument("--per-customer", type=int, default=3,
                        help="Number of payments to add per existing customer (default: 3)")
    args = parser.parse_args()

    print(f"\n── Adding {args.per_customer} payments per existing customer ──\n")

    customers = stripe.Customer.list(limit=100).data
    if not customers:
        print("No customers found. Run scripts/seed_stripe.py first.")
        sys.exit(1)

    added = 0
    for customer in customers:
        if not customer.name:
            continue  # skip anonymous customers
        print(f"\n{customer.name} ({customer.email}):")
        for i in range(args.per_customer):
            amount, desc = _PURCHASES[(added + i) % len(_PURCHASES)]
            pi = add_payment(customer, amount, desc)
            if pi:
                added += 1

    print(f"\n── Done — added {added} new payments ──")
    print("\nList them with: \"What did <customer> pay for recently?\"")
    print("Refund one with: \"Refund <customer>'s last payment\"")


if __name__ == "__main__":
    main()
