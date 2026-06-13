# Chat-with-docs!

[![CI](https://github.com/vamsimalineni96/chat_with_docs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/vamsimalineni96/chat_with_docs/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)

A RAG service over a document corpus (Milvus hybrid retrieval → cross-encoder rerank → NVIDIA NIM chat completion, behind a FastAPI app), built primarily as a portfolio piece demonstrating the three-pillar observability framework articulated by [Pooja Palod in the *Data Journey* series](https://datajourney24.substack.com/) — Cost, Quality, and Latency.

The PDF corpus under `pdfs/` (Harry Potter 4 and 7) is a placeholder used to exercise the pipeline end-to-end while the observability surface is built out.

## What makes this repo interesting

Three pillars, each with a measurement script, a CI/cron path, and dated artifacts committed back to the repo. **The reports are the work.**

| Pillar | What it answers | How |
|---|---|---|
| **Cost** | What does a task cost? | [`evals/cost/cost_report.py`](evals/cost/cost_report.py) → dated [`docs/reports/cost_*.md`](docs/reports/) (nightly cron) |
| **Quality** | Are answers grounded, accurate, and complete? | [`evals/quality/run_eval.py`](evals/quality/run_eval.py) + LLM judge → dated [`docs/eval-reports/eval_*.md`](docs/eval-reports/). Schema gated on every PR. |
| **Latency** | Where is time being spent? Where are the tails? | [`evals/latency/latency_report.py`](evals/latency/latency_report.py) → dated [`docs/reports/latency_*.md`](docs/reports/) (nightly cron) |

The story of how each pillar was built, in order: [`docs/PROGRESS.md`](docs/PROGRESS.md). The design rationale and gap audit: [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md). The latest-numbers index: [`docs/dashboards/index.md`](docs/dashboards/index.md).

Two reliability behaviours layered under the latency pillar:

- **Retries with exponential backoff on every NVIDIA call** — read path (`/chat`) via [`src/utils/services/retry.py`](src/utils/services/retry.py) and write path (PDF ingest) via [`src/utils/services/milvus_store.py`](src/utils/services/milvus_store.py). One transient 502/503 from the upstream load balancer no longer surfaces as a 5xx.
- **Idempotent ingestion** — deterministic per-page `doc_id` + delete-before-insert means re-running an ingest after a mid-stream failure doesn't accumulate duplicate chunks in Milvus.

## Stack

- **API**: FastAPI ([`app.py`](app.py)) — chat, async PDF ingestion, admin endpoints.
- **UI**: Streamlit ([`ui.py`](ui.py)) — multi-user chat with a debug panel (retrieval, rerank, prompt, timings).
- **Vector store**: Milvus 2.5 with hybrid retrieval (dense embeddings + BM25, fused via RRF).
- **Reranker**: cross-encoder rerank over the fused candidates.
- **Embeddings / LLM**: NVIDIA AI endpoints via LangChain.
- **State**: Postgres (users / conversations / messages), Redis (per-conversation lock + semantic cache key).
- **Observability**: Langfuse v4 — traces, prompts, timings. Data shaped into the dated reports above.

## Layout

- [`app.py`](app.py), [`ui.py`](ui.py) — FastAPI routes + Streamlit UI.
- [`src/`](src/) — services: embedder, PDF parser, Milvus store, conversation store, Redis lock, chat orchestration, retry helper, prompts.
- [`eval/`](eval/) — Q&A dataset, retrieval metrics, LLM judge, end-to-end runner.
- [`scripts/`](scripts/) — cost + latency aggregators, pricing table, locust workload.
- [`docs/`](docs/) — observability design, progress log, dashboards index, dated reports.
- [`tests/`](tests/) — 82 unit tests covering the cost/latency/eval/retry pipelines (no real LLM, no network).
- [`docker-compose.yml`](docker-compose.yml) — Milvus (+etcd, MinIO), Postgres, Redis.

## Run

```bash
docker compose up -d                          # Milvus, Postgres, Redis
pip install -r requirements.txt
uvicorn app:app --reload                      # API on :8000
streamlit run ui.py                           # UI on :8501
```

Langfuse: configured against Langfuse Cloud — set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` in `.env`. See [`.env.example`](.env.example).

## Generating reports locally

```bash
# Quality — runs the 18-question eval against the live app and writes a dated report.
python -m evals.quality.run_eval --output docs/eval-reports/eval_$(date +%Y-%m-%d).md

# Cost — pulls last 7 days of Langfuse traces, attributes by task + stage.
python -m evals.cost.cost_report --source live --days 7 \
  --output docs/reports/cost_$(date +%Y-%m-%d).md

# Latency — pulls last 7 days of Langfuse traces, computes p50/p95/p99.
python -m evals.latency.latency_report --source live --days 7 \
  --output docs/reports/latency_$(date +%Y-%m-%d).md
```

Both cost and latency also run as nightly GitHub Actions workflows; see [`.github/workflows/`](.github/workflows/).

## Multi-agent + Stripe MCP Integration

The chat graph is a multi-agent supervisor that routes each question to the right
specialist, then optionally takes real action against Stripe.

### Multi-agent orchestration

An intent classifier ([`src/agents/intent_classifier.py`](src/agents/intent_classifier.py))
labels every question, and a supervisor ([`src/agents/supervisor.py`](src/agents/supervisor.py))
routes it down one of four paths:

| Intent | Path | What it does |
|---|---|---|
| `research` | RAG | Hybrid retrieval over the doc corpus, rerank, LLM answer |
| `action` | Tool ReAct agent | Calls MCP tools (Stripe, shopping) to *do* something |
| `both` | RAG + Tool, then aggregator | Runs both branches in parallel, fuses answers |
| `out_of_scope` | Canned refusal | Heuristic-gated short-circuit |

The full topology lives in [`src/agents/graph.py`](src/agents/graph.py). Each branch
is its own `@observe`'d Langfuse span so the routing decision and every downstream
call show up under one trace.

### How the action path fits together

```
User question
      │
      ▼
intent classifier  ──► "action" / "both" ──► ReAct tool agent
                                                    │
                          ┌─────────────────────────┴──────────────────┐
                          ▼                                             ▼
               shopping_support.py                           stripe_support.py
               (fake orders / inventory)                     (real Stripe API)
                    stdio subprocess                          SSE Docker container
```

The `MultiServerMCPClient` in [`src/agents/mcp_client.py`](src/agents/mcp_client.py)
connects to both servers simultaneously and merges their tools into one flat list.
The ReAct agent sees all tools and picks the right one — it never knows or cares
which server a tool came from.

### What is `stripe_support.py`?

**Stripe is not an MCP server.** Stripe is just a payments company with a REST API.

`mcp_servers/stripe_support.py` is an MCP server that **you own and run**. It sits
in the middle and translates between the two worlds:

```
ReAct agent  →  MCP tool call (list_customers)
                      ↓
             stripe_support.py   ← YOUR MCP server (translator)
                      ↓
             GET api.stripe.com/v1/customers   ← Stripe REST API
```

It lives in `mcp_servers/` because it IS an MCP server — one you built.

### stdio vs SSE

These are two ways the app and the MCP server talk to each other.

| | stdio | SSE |
|---|---|---|
| How | stdin/stdout (pipe) | HTTP over network |
| Lives | subprocess inside the app process | separate Docker container |
| When it dies | with the app | independently |
| Use for | local dev | production |

**Dev mode** (`STRIPE_SECRET_KEY` set, no `STRIPE_MCP_URL`):
The app spawns `stripe_support.py` as a child process. Simple, no Docker needed.

**Production mode** (`STRIPE_MCP_URL` set):
`stripe_support.py` runs as a standalone Docker container. The app connects to it
over HTTP. Multiple app instances can share one MCP container. The Stripe key lives
only on the MCP container — the app never sees it.

### When to write your own MCP server

| Situation | What to do |
|---|---|
| The company published an official MCP server and it works well | Use it directly (Option 2) |
| Community built one and it's focused enough | Use it (modelcontextprotocol.io/servers) |
| No server exists, or the official one is too bloated / slow | Write your own |

We tried Stripe's official `@stripe/mcp` npm package first. It exposed 25+ generic
tools with enormous schemas — the Llama model timed out trying to reason about all
of them. Our custom `stripe_support.py` has 8 focused tools with small, clear
docstrings. The model picks the right one instantly.

**Rule:** official servers are great for broad integrations. For production use cases
you almost always end up writing your own because you need specific behavior,
specific tool names, and small schemas the model can reason about quickly.

### Human-in-the-Loop (HITL) for destructive actions

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
   [`app.py:_execute_approved_refund`](app.py), reached only via the `/approve` endpoint after a
   valid token is consumed. The agent has no path to that SDK call.

2. **Disambiguation is server-side, not LLM judgment.** When the user says
   "refund the webcam" and two webcam payments exist, the MCP server itself
   (not the LLM) counts matches and returns `requires_disambig: True` with the
   candidate list. The UI surfaces a pick-one card; selecting it triggers
   `/approve` with the chosen `payment_intent_id`. LLMs decide what to do
   ("which customer? which product?"); deterministic code decides what *can* be
   done.

Key files: [`src/agents/graph.py`](src/agents/graph.py) (approval gate node and
routing), [`src/agents/tool_node.py`](src/agents/tool_node.py) (`_check_pending_approval`
HITL detector), [`src/utils/services/approval_store.py`](src/utils/services/approval_store.py)
(token store with 10-min TTL), [`ui.py`](ui.py) (Approve/Reject and disambig cards).

Every pause and decision is tagged in Langfuse (`hitl:pending`,
`hitl_decision:approved|rejected`, `hitl_kind:approval|disambig`) so the full
flow shows up as a discoverable trace under the same `session_id` as the chat.

### Running the Stripe MCP service

**Dev (stdio):**
```bash
# Just set the key in .env — the app spawns the server automatically
STRIPE_SECRET_KEY=rk_test_...
```

**Production (SSE Docker container):**
```bash
# Build once
docker build -t stripe-mcp -f mcp_servers/Dockerfile mcp_servers/

# Start (reads STRIPE_SECRET_KEY from .env automatically)
./mcp_servers/run.sh

# Point the app at it
STRIPE_MCP_URL=http://localhost:8001/sse
```

### Seeding test data

```bash
# Uses sk_test_ (full secret key) — needed to confirm payments in test mode
STRIPE_SECRET_KEY=sk_test_... python scripts/seed_stripe.py
```

Creates: 3 customers (John Doe, Jane Smith, Bob Wilson), 3 products, 2 confirmed
payments, 1 declined payment (Bob), 1 open invoice.

### Test conversations

```
"I'm john.doe@example.com, what did I pay for recently?"
"Jane Smith wants a refund for her headphones purchase"        # HITL: Approve/Reject card
"Refund Bob for the webcam"                                    # HITL: disambig (after seeding multiple webcam charges)
"Bob Wilson says his payment failed — what happened?"
"Show me Jane's open invoices"
"Is the Wireless Headphones in stock and what does it cost in Stripe?"
```

Top up additional payments for disambig testing:
```bash
STRIPE_SECRET_KEY=sk_test_... python scripts/add_payments.py
```

---

## Key endpoints

- `POST /chat` — multi-turn chat, per-conversation Redis lock, semantic cache + RAG fallback.
- `POST /upload_pdf_async` — async ingestion with retry, resumable via `start_page=N`, idempotent on re-run; poll `GET /task_status/{task_id}`.
- `GET /list_conversations`, `GET /list_messages` — history for a `user_external_id`.
- Admin: `/clear_post_gres`, `/clear_cache`, `/clear_milvus`, `/view_milvus_store`, `/debug_database`.
