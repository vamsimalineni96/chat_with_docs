"""Seed the Stripe test sandbox with realistic customer-support demo data.

Creates:
  - 3 customers  (John Doe, Jane Smith, Bob Wilson)
  - 3 products   (Mechanical Keyboard, Wireless Headphones, USB-C Cable 6ft)
  - 2 confirmed payments  (John → keyboard, Jane → headphones)
  - 1 declined  payment   (Bob → cable, card declined)
  - 1 open invoice        (Jane → second keyboard order billed monthly)

Run from repo root:
    python scripts/seed_stripe.py

Requires STRIPE_SECRET_KEY in environment (or .env file).
Safe to re-run — checks for existing customers by email before creating.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env so STRIPE_SECRET_KEY is available when run locally
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

import stripe

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
if not stripe.api_key:
    print("ERROR: STRIPE_SECRET_KEY not set. Add it to .env or export it.")
    sys.exit(1)

if not stripe.api_key.startswith(("rk_test_", "sk_test_")):
    print("ERROR: This script only runs against test keys (rk_test_ or sk_test_).")
    print("       Never seed with a live key.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_or_create_customer(name: str, email: str) -> stripe.Customer:
    existing = stripe.Customer.list(email=email, limit=1)
    if existing.data:
        print(f"  customer exists: {email}")
        return existing.data[0]
    c = stripe.Customer.create(name=name, email=email)
    print(f"  created customer: {name} ({email}) → {c.id}")
    return c


def get_or_create_product(name: str, description: str) -> stripe.Product:
    results = stripe.Product.search(query=f'name:"{name}"', limit=1)
    if results.data:
        print(f"  product exists: {name}")
        return results.data[0]
    p = stripe.Product.create(name=name, description=description)
    print(f"  created product: {name} → {p.id}")
    return p


def create_payment(
    customer: stripe.Customer,
    amount_cents: int,
    description: str,
    pm_token: str = "pm_card_visa",
) -> stripe.PaymentIntent:
    # Use Stripe's built-in test PaymentMethod tokens — no raw card data needed.
    # Common tokens:
    #   pm_card_visa                          → succeeds
    #   pm_card_chargeDeclinedInsufficientFunds → declined (insufficient funds)
    #   pm_card_chargeDeclined                → generic decline
    pi = stripe.PaymentIntent.create(
        amount=amount_cents,
        currency="usd",
        customer=customer.id,
        payment_method=pm_token,
        description=description,
        confirm=True,
        automatic_payment_methods={"enabled": True, "allow_redirects": "never"},
    )
    print(f"  payment {pi.status}: {description} (${amount_cents / 100:.2f}) → {pi.id}")
    return pi


def create_invoice_for(
    customer: stripe.Customer,
    description: str,
    amount_cents: int,
) -> stripe.Invoice:
    invoice = stripe.Invoice.create(customer=customer.id)
    stripe.InvoiceItem.create(
        customer=customer.id,
        amount=amount_cents,
        currency="usd",
        description=description,
        invoice=invoice.id,
    )
    finalized = stripe.Invoice.finalize_invoice(invoice.id)
    print(f"  invoice created: {description} (${amount_cents / 100:.2f}) → {finalized.id}")
    return finalized


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n── Customers ──────────────────────────────────────────────────")
    john = get_or_create_customer("John Doe", "john.doe@example.com")
    jane = get_or_create_customer("Jane Smith", "jane.smith@example.com")
    bob  = get_or_create_customer("Bob Wilson", "bob.wilson@example.com")

    print("\n── Products ───────────────────────────────────────────────────")
    keyboard    = get_or_create_product("Mechanical Keyboard", "High-performance mechanical keyboard with RGB backlighting")
    headphones  = get_or_create_product("Wireless Headphones", "Noise-cancelling over-ear wireless headphones")
    cable       = get_or_create_product("USB-C Cable 6ft",     "Braided USB-C to USB-C cable, 6 feet, 100W charging")

    print("\n── Payments ───────────────────────────────────────────────────")
    # John: successful keyboard purchase
    create_payment(john, 7999, f"{keyboard.name} purchase")

    # Jane: successful headphones purchase
    create_payment(jane, 14999, f"{headphones.name} purchase")

    # Bob: declined payment (insufficient funds)
    try:
        create_payment(bob, 1999, f"{cable.name} purchase",
                       pm_token="pm_card_chargeDeclinedInsufficientFunds")
    except stripe.CardError as e:
        print(f"  payment declined (expected): {e.user_message} → Bob's {cable.name}")

    print("\n── Invoices ───────────────────────────────────────────────────")
    # Jane: open invoice for a second keyboard order
    create_invoice_for(jane, f"{keyboard.name} — monthly billing", 7999)

    print("\n── Done ────────────────────────────────────────────────────────")
    print("Seed complete. Test conversation starters:")
    print("  'I'm john.doe@example.com, what did I pay for recently?'")
    print("  'Jane Smith wants a refund for her headphones'")
    print("  'Bob Wilson says his payment failed — what happened?'")
    print("  'Show me Jane's open invoices'")
    print("  'Is the Wireless Headphones in stock and what does it cost in Stripe?'")


if __name__ == "__main__":
    main()
