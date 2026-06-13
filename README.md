# Multi-agent customer-support chatbot

[![CI](https://github.com/vamsimalineni96/chat_with_docs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/vamsimalineni96/chat_with_docs/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)

A production-pattern agentic chatbot that grounds answers in a document corpus
(RAG) **and** takes real action against live Stripe APIs — with an architecturally
unbypassable human-in-the-loop guarding every destructive action.

Built on LangGraph (supervisor + ReAct sub-agents), MCP tool servers, Milvus
hybrid retrieval, FastAPI, and Streamlit. Every retrieval, tool call, and
approval decision is traced in Langfuse.

## What it does

An intent classifier ([`src/agents/intent_classifier.py`](src/agents/intent_classifier.py))
labels each question, and a supervisor ([`src/agents/supervisor.py`](src/agents/supervisor.py))
routes it down one of four paths:

| Intent | Path | What it does |
|---|---|---|
| `research` | RAG | Hybrid retrieval (dense + BM25 → RRF) over the doc corpus, cross-encoder rerank, LLM answer |
| `action` | Tool ReAct agent | Calls MCP tools to *do* something — look up customers, list payments, create invoices, request refunds |
| `both` | RAG + Tool, then aggregator | Runs both branches in parallel, fuses answers |
| `out_of_scope` | Canned refusal | Heuristic-gated short-circuit |

Topology lives in [`src/agents/graph.py`](src/agents/graph.py). Every node is its
own `@observe`'d Langfuse span — one trace per request, end-to-end.

## The action path: MCP + Stripe

```
User question
      │
      ▼
intent classifier ──► "action" / "both" ──► ReAct tool agent
                                                  │
                          ┌───────────────────────┴────────────────────┐
                          ▼                                             ▼
               shopping_support.py                           stripe_support.py
               (fake orders / inventory)                     (real Stripe API)
                    stdio subprocess                          SSE Docker container
```

The `MultiServerMCPClient` in [`src/agents/mcp_client.py`](src/agents/mcp_client.py)
connects to both servers simultaneously and merges their tools into one flat list.
The ReAct agent sees all tools and picks the right one — it never knows or cares
which server a tool came from.

### Why a custom MCP server for Stripe?

Stripe is not an MCP server — it's a payments API. [`mcp_servers/stripe_support.py`](mcp_servers/stripe_support.py)
is an MCP server that **we own and run**, sitting between the agent and Stripe's REST API.

We tried Stripe's official `@stripe/mcp` npm package first. It exposed 25+ generic
tools with enormous schemas — the Llama model timed out trying to reason about
all of them. The custom server has 8 focused tools with small, clear docstrings;
the model picks the right one instantly.

**Rule:** official MCP servers are great for broad integrations. For production
use cases you almost always end up writing your own — specific tool names, small
schemas the model can reason about quickly, and behavior you control.

### stdio vs SSE (dev vs prod)

| | stdio | SSE |
|---|---|---|
| Transport | stdin/stdout pipe | HTTP over network |
| Process | subprocess of the app | standalone Docker container |
| Use for | local dev | production |
| Stripe key | in app env | only on the MCP container; app never sees it |

In production mode, the app sets `STRIPE_MCP_URL=http://stripe-mcp:8001/sse`. The
secret key is isolated to the MCP container — a useful blast-radius reduction.

## Human-in-the-Loop (HITL) for destructive actions

Destructive Stripe actions (refunds) never execute through the agent. The trust
boundary is architectural — even a smarter model swapped in tomorrow cannot
unilaterally issue a refund.

```
User: "refund Bob for the webcam"
      │
      ▼
ReAct agent → create_refund(pi_id)
                  │
                  ▼
        Tool returns {requires_confirmation: True, ...}    ← never actually refunds
                  │
                  ▼
        Graph PAUSES, mints an approval token
                  │
                  ▼
        UI surfaces an Approve/Reject card
                  │
                  ▼
        POST /approve → stripe.Refund.create(...)          ← runs OUT-of-band
                                                              via direct SDK
```

Two design decisions hold this together:

1. **The destructive tool can't execute.** [`mcp_servers/stripe_support.py`](mcp_servers/stripe_support.py)'s
   `create_refund` always returns a confirmation request — it never calls
   `stripe.Refund.create`. The Stripe SDK is only invoked from
   [`app.py:_execute_approved_refund`](app.py), reached only via the `/approve`
   endpoint after a valid token is consumed. The agent has no path to that SDK call.

2. **Disambiguation is server-side, not LLM judgment.** When the user says
   "refund the webcam" and two webcam payments exist, the MCP server itself
   (not the LLM) counts matches and returns `requires_disambig: True` with the
   candidate list. The UI surfaces a pick-one card; selecting it triggers
   `/approve` with the chosen `payment_intent_id`. LLMs decide *what to do*;
   deterministic code decides *what can be done*.

Key files: [`src/agents/graph.py`](src/agents/graph.py) (approval gate node),
[`src/agents/tool_node.py`](src/agents/tool_node.py) (`_check_pending_approval`
HITL detector), [`src/utils/services/approval_store.py`](src/utils/services/approval_store.py)
(token store with 10-min TTL), [`ui.py`](ui.py) (Approve/Reject + disambig cards).

Every pause and decision is tagged in Langfuse (`hitl:pending`,
`hitl_decision:approved|rejected`, `hitl_kind:approval|disambig`) and tied
back to the original chat trace via shared `session_id`.

## Production hygiene

This started as a portfolio piece exercising the three-pillar observability
framework articulated by [Pooja Palod in the *Data Journey* series](https://datajourney24.substack.com/)
— **Cost**, **Quality**, **Latency** — each with a measurement script, a CI/cron
path, and dated reports committed back to the repo. **The reports are the work.**

| Pillar | What it answers | How |
|---|---|---|
| **Cost** | What does a task cost? | [`evals/cost/cost_report.py`](evals/cost/cost_report.py) → dated [`docs/reports/cost_*.md`](docs/reports/) (nightly cron) |
| **Quality** | Are answers grounded, accurate, complete? | [`evals/quality/run_eval.py`](evals/quality/run_eval.py) + LLM judge → dated [`docs/eval-reports/eval_*.md`](docs/eval-reports/). Schema-gated on every PR. |
| **Latency** | Where is time spent? Where are the tails? | [`evals/latency/latency_report.py`](evals/latency/latency_report.py) → dated [`docs/reports/latency_*.md`](docs/reports/) (nightly cron) |

Plus two reliability behaviours under the latency pillar:

- **Retries with exponential backoff on every NVIDIA call** — [`src/utils/services/retry.py`](src/utils/services/retry.py)
  (read path) and [`src/utils/services/milvus_store.py`](src/utils/services/milvus_store.py)
  (ingest). One transient 502/503 no longer surfaces as a 5xx.
- **Idempotent ingestion** — deterministic per-page `doc_id` + delete-before-insert
  means re-running an ingest after a mid-stream failure doesn't accumulate
  duplicate chunks in Milvus.

Design rationale: [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md). Build log:
[`docs/PROGRESS.md`](docs/PROGRESS.md). Latest-numbers index:
[`docs/dashboards/index.md`](docs/dashboards/index.md).

## Stack

- **API**: FastAPI ([`app.py`](app.py)) — chat, async PDF ingestion, approval, admin.
- **UI**: Streamlit ([`ui.py`](ui.py)) — multi-user chat, HITL cards, debug panel.
- **Agents**: LangGraph supervisor + ReAct sub-agents, NVIDIA AI endpoints via LangChain.
- **Tools**: MCP servers — `stripe_support.py` (real Stripe), `shopping_support.py` (fake fixtures).
- **Vector store**: Milvus 2.5 — hybrid dense + BM25, RRF-fused, cross-encoder reranked.
- **State**: Postgres (users / conversations / messages), Redis (per-conversation lock + approval tokens).
- **Observability**: Langfuse v4 — traces, prompts, timings, HITL tags.

## Layout

- [`app.py`](app.py), [`ui.py`](ui.py) — FastAPI routes + Streamlit UI.
- [`src/agents/`](src/agents/) — supervisor, intent classifier, RAG node, tool node, graph topology, prompts.
- [`src/utils/`](src/utils/) — embedder, Milvus store, conversation store, Redis lock, approval token store, retry helper, chat orchestration.
- [`mcp_servers/`](mcp_servers/) — Stripe + shopping support MCP servers (FastMCP, dual stdio/SSE).
- [`evals/`](evals/) — Q&A dataset, retrieval metrics, LLM judge, cost/latency aggregators.
- [`scripts/`](scripts/) — Stripe seeding, payment top-up, locust workload.
- [`docs/`](docs/) — observability design, progress log, dated reports.
- [`tests/`](tests/) — unit tests covering the graph, intent classifier, MCP client, tool node, retry pipeline (no real LLM, no network).
- [`docker-compose.yml`](docker-compose.yml) — Milvus (+etcd, MinIO), Postgres, Redis, Langfuse, Stripe MCP container.

## Run

```bash
docker compose up -d                          # Milvus, Postgres, Redis, Langfuse, Stripe MCP
pip install -r requirements.txt
uvicorn app:app --reload                      # API on :8000
streamlit run ui.py                           # UI on :8501
```

Required env (see [`.env.example`](.env.example)): `NVIDIA_API_KEY`, `STRIPE_SECRET_KEY`,
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.

**Seed Stripe test data:**
```bash
STRIPE_SECRET_KEY=sk_test_... python scripts/seed_stripe.py        # customers, products, payments
STRIPE_SECRET_KEY=sk_test_... python scripts/add_payments.py       # extra payments for disambig testing
```

**Try these conversations:**
```
"I'm john.doe@example.com, what did I pay for recently?"
"Jane Smith wants a refund for her headphones purchase"        # HITL: Approve/Reject card
"Refund Bob for the webcam"                                    # HITL: disambig (multiple webcam charges)
"Bob Wilson says his payment failed — what happened?"
"Is the Wireless Headphones in stock and what does it cost in Stripe?"
```

## Generating reports locally

```bash
python -m evals.quality.run_eval --output docs/eval-reports/eval_$(date +%Y-%m-%d).md
python -m evals.cost.cost_report --source live --days 7 --output docs/reports/cost_$(date +%Y-%m-%d).md
python -m evals.latency.latency_report --source live --days 7 --output docs/reports/latency_$(date +%Y-%m-%d).md
```

Cost and latency also run nightly via GitHub Actions — see [`.github/workflows/`](.github/workflows/).

## Key endpoints

- `POST /chat` — multi-turn chat, per-conversation Redis lock, semantic cache + RAG/action routing.
- `POST /approve` — consume an approval token, execute or reject the paused destructive action.
- `POST /upload_pdf_async` — async ingestion with retry, resumable via `start_page=N`, idempotent on re-run; poll `GET /task_status/{task_id}`.
- `GET /list_conversations`, `GET /list_messages` — history for a `user_external_id`.
- Admin: `/clear_post_gres`, `/clear_cache`, `/clear_milvus`, `/view_milvus_store`, `/debug_database`.
