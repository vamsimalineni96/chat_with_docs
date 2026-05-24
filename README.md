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

## Key endpoints

- `POST /chat` — multi-turn chat, per-conversation Redis lock, semantic cache + RAG fallback.
- `POST /upload_pdf_async` — async ingestion with retry, resumable via `start_page=N`, idempotent on re-run; poll `GET /task_status/{task_id}`.
- `GET /list_conversations`, `GET /list_messages` — history for a `user_external_id`.
- Admin: `/clear_post_gres`, `/clear_cache`, `/clear_milvus`, `/view_milvus_store`, `/debug_database`.
