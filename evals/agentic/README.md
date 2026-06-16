# Agentic eval

Assertion-based eval for the multi-agent chatbot. Scores **routing**, **tool
selection**, and **HITL** behaviour — orthogonal to `evals/quality/`, which
scores final-answer prose.

## What gets asserted

Each case is a YAML file under [`cases/`](cases/). The runner sends the
case's question to the live `/chat` endpoint with `debug=true`, then checks
the response against the `expected` block:

| Dimension | Source | What's checked |
|---|---|---|
| `intent` | `response.debug.intent` | Supervisor routed to the right path (`research` / `tool_call` / `both` / `out_of_scope`) |
| `tool_calls` | `response.debug.tool_calls` | The expected MCP tools were called, in order, with matching args (subset/substring match) |
| `hitl` | `response.pending_approval` | Graph paused (or didn't) for the right kind, with enough candidates |
| `http_status` | HTTP code | Endpoint returned the expected status |

Any missing `expected.*` key = "don't assert this dimension."

## Case schema

```yaml
id: hitl-refund-disambig-webcam            # unique, kebab-case
description: |
  Multi-match by product name must pause with kind="disambig".
question: "Refund Bob Wilson for the webcam."
expected:
  intent: tool_call
  tool_calls:
    - name: get_customer_payments
      args_contain:                         # subset; strings = case-insensitive substring
        name_or_email: bob
        product_filter: webcam
  hitl:
    paused: true
    kind: disambig                          # "approval" or "disambig"
    min_candidates: 2
  http_status: 200
```

## Run

Validate cases load (cheap, no network):
```bash
python -m evals.agentic.run_eval --validate-only
```

Full run against the live app:
```bash
python -m evals.agentic.run_eval \
    --output docs/eval-reports/agentic_$(date +%Y-%m-%d).md
```

Single case (smoke test one behaviour):
```bash
python -m evals.agentic.run_eval --case-id hitl-refund-disambig-webcam \
    --output /tmp/one.md
```

Exits with code 2 if any case failed — wire into CI when ready.

## Prerequisites

- App running locally (`uvicorn app:app` on `:8000` by default)
- Stripe MCP up (`docker compose up -d stripe-mcp`)
- Test data seeded:
  ```bash
  STRIPE_SECRET_KEY=sk_test_... python scripts/seed_stripe.py
  STRIPE_SECRET_KEY=sk_test_... python scripts/add_payments.py    # for disambig cases
  ```

The disambig case (`hitl-refund-disambig-webcam`) expects Bob Wilson to have
≥ 2 refundable Webcam payments. Re-run `add_payments.py` if a previous run
refunded one of them.

## Why assertion-based, not LLM judge?

LLM-as-judge is the right tool for **answer quality** (groundedness, accuracy,
completeness) — see [`evals/quality/`](../quality/). It's the wrong tool for
**routing and tool selection**: those have deterministic correct answers,
judges add variance and cost, and a regression on routing should be a hard
red signal, not a soft score.

Final-answer prose is not asserted here. If the agent routes correctly and
calls the right tools, the answer's wording is the quality eval's domain.
