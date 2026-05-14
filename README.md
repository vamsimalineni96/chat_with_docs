# Agentic RAG Chatbot — LangGraph + NVIDIA NIM + Milvus

Agentic RAG over a single indexed document. LangGraph state machine (rewriter → agent → tools → verifier) with hybrid (dense + BM25) Milvus retrieval, NVIDIA Llama 3.3 70B as the agent, self-hosted Langfuse observability, FastAPI backend, and a Streamlit UI.

## Tech stack

| Layer | Tools |
|---|---|
| Orchestration | LangGraph 0.4.x |
| LLM | NVIDIA NIM — Llama 3.3 70B Instruct |
| Embeddings | NVIDIA `nv-embedqa-e5-v5` |
| Reranker | NVIDIA `nv-rerank-qa-mistral-4b:1` |
| Vector store | Milvus 2.5.4 — hybrid dense + BM25 with RRF |
| Conversation store | PostgreSQL via SQLAlchemy |
| Concurrency control | Redis distributed locks (per-conversation) |
| Observability | Langfuse 4.x (self-hosted) |
| API | FastAPI |
| UI | Streamlit |
| Load testing | Locust |
| Infrastructure | Docker Compose |

## Prerequisites

- Docker + Docker Compose
- Python 3.12
- NVIDIA API key — sign up at [build.nvidia.com](https://build.nvidia.com)

## Setup

```bash
git clone <repo-url> && cd BTX-BPD-Bodycam-Search

# 1. Configure secrets
cp .env.example .env   # fill in NVIDIA_API_KEY

# 2. Bring up the stack (Milvus + Postgres + Redis + Langfuse)
docker compose up -d

# 3. Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. Drop a PDF into pdfs/ and index it
curl -X POST "http://localhost:8000/upload_pdf_async?pdf_name=your.pdf&collection_name=docs"

# 5. Run the API + UI
uvicorn app:app --reload     # terminal 1
streamlit run ui.py          # terminal 2
```

Open <http://localhost:8501>, enter a user ID, ask a question.

## Service endpoints

| Service | URL | Credentials |
|---|---|---|
| FastAPI | <http://localhost:8000> | — |
| Streamlit UI | <http://localhost:8501> | — |
| Langfuse UI | <http://localhost:3000> | `admin@local.dev` / `admin1234` |
| Milvus | localhost:19530 | — |
| Postgres (chat) | localhost:5432 | `chatuser` / `chatpass` / `chatdb` |
| Postgres (langfuse) | localhost:5433 | `langfuse` / `langfuse` / `langfuse` |
| Redis (chat) | localhost:6379 | — |
| Redis (langfuse) | localhost:6380 | password `langfuseredis` |

## Data ingestion

PDFs only. Place the file in `pdfs/`, then trigger async indexing:

```bash
curl -X POST "http://localhost:8000/upload_pdf_async?pdf_name=hp4.pdf&collection_name=docs"
# → {"task_id": "<uuid>"}

curl "http://localhost:8000/task_status/<task_id>"   # poll until "Complete"
```

Pipeline: PyMuPDF extracts text and skips TOC / page numbers / cross-page sentence splits → `RecursiveCharacterTextSplitter` chunks at `CHUNK_SIZE`/`CHUNK_OVERLAP` → each chunk is inserted into Milvus with a dense embedding (NVIDIA) and a BM25 sparse vector (computed server-side).

A 300-page PDF takes ~1-2 minutes under NVIDIA's 40 RPM cap. Re-ingesting the same file does *not* dedupe — drop the collection first if you want a clean reindex:

```bash
curl -X POST "http://localhost:8000/clear_milvus?name=docs"
```

## Load testing

```bash
locust -f scripts/locust_workload.py --host http://localhost:8000 \
       --users 8 --spawn-rate 1 -t 10m --headless
```

Default config stays under NVIDIA NIM's 40 RPM cap. Tunables via env vars: `LOCUST_USER_POOL_SIZE`, `LOCUST_MAX_CONVOS_PER_USER`, `LOCUST_DEBUG_PROB`.

## Useful scripts

Run from the project root so `src/` is importable.

```bash
python scripts/debug_agent.py "your question"   # run the agent loop with per-node output
python scripts/visualize_graph.py               # regenerate graph.mmd and graph.png
python scripts/eval.py                          # run the evaluation harness (see below)
```

## Evaluation

`scripts/eval.py` runs a 30-case test set through the full pipeline and reports hard numbers — accuracy, retrieval recall, verifier behavior, latency. This is what lets claims like *"the verifier catches X% of unsupported answers"* be backed by data instead of vibes.

The 30 cases in [eval_dataset.jsonl](eval_dataset.jsonl) cover five categories the agentic graph should handle differently:

| Tag | Count | What it tests |
|---|---:|---|
| `factual` | 10 | Single-hop questions about indexed content. Expect 1 tool call, verifier `ok`. |
| `comparative` (multi-hop) | 5 | Questions like *"compare X and Y"*. Expect 2 parallel tool calls. |
| `followup` (pronoun) | 5 | Each carries a 2-turn chat history; the rewriter should resolve `"he"` / `"it"`. |
| `off_topic` | 5 | Questions not about the document. Expect honest refusal, **not** fabrication. |
| `adversarial` (no_info) | 5 | Questions that *look* like they're about the document but the answer isn't in it. Expect refusal; punish hallucination. |

```bash
# Full run — ~5 minutes against the indexed corpus
python scripts/eval.py

# Filter to a subset
python scripts/eval.py --tags factual followup

# Smoke-test on the first 5 cases
python scripts/eval.py --limit 5

# Save full results for diffing across pipeline changes
python scripts/eval.py --output eval_results.json
```

### What gets measured

| Metric | Definition |
|---|---|
| **Accuracy** | Fraction graded `CORRECT` by an LLM-judge (Llama 3.3 70B, separate call, strict 4-class rubric: `CORRECT` / `PARTIAL` / `WRONG` / `REFUSED`). |
| **Retrieval recall** | Per case, fraction of `expected_substrings` found in any of the agent's retrieved passages. Substring matching is case-insensitive, multi-passage. |
| **Verifier catch rate** | Of all answers graded `WRONG` or `PARTIAL`, what fraction did the verifier flag (`retry` or `exhausted`)? Higher = the verifier is doing its job. |
| **Verifier precision** | Of all flagged answers, what fraction were actually wrong? Lower = verifier is over-cautious. |
| **Latency p50 / p95** | Wall-clock per question, end-to-end. |
| **Avg iterations / corrections / tool calls** | Cost breakdown — spot workloads where the agent is doing too much. |
| **Per-tag breakdown** | Same metrics split by category, so you can see e.g. `comparative` questions averaging 2.0 tool calls vs `factual` averaging 1.0. |

### Sample output

```
[ 1/30] f01    'What are wand cores typically made of in the wizardin'   CORRECT  recall=1.00 verdict=ok       3.7s
[ 2/30] f02    "Who is Cedric Diggory's father?"                          CORRECT  recall=1.00 verdict=ok       2.9s
…

Done in 412.3s (13.7s avg per case)

─── Summary ───
{
  "n": 30,
  "accuracy": 0.833,
  "grade_breakdown": { "CORRECT": 25, "PARTIAL": 3, "WRONG": 1, "REFUSED": 1 },
  "avg_recall": 0.78,
  "p50_latency_s": 4.1,
  "p95_latency_s": 14.7,
  "avg_agent_iterations": 2.3,
  "avg_corrections": 0.13,
  "avg_tool_calls": 1.4,
  "verifier_catch_rate": 0.75,
  "verifier_precision": 0.50,
  "by_tag": {
    "factual":      { "n": 10, "accuracy": 0.90, "avg_recall": 0.95 },
    "comparative":  { "n":  5, "accuracy": 0.80, "avg_recall": 0.85 },
    "followup":     { "n":  5, "accuracy": 0.80, "avg_recall": 0.90 },
    "off_topic":    { "n":  5, "accuracy": 1.00, "avg_recall": 1.00 },
    "adversarial":  { "n":  5, "accuracy": 0.60, "avg_recall": 1.00 }
  }
}
```

*(Sample values are illustrative — your run against your indexed corpus and your model will produce real numbers.)*

### Caveats worth knowing

- **The judge is the same model as the generator.** Llama 3.3 70B grades itself. This biases scores upward; treat results as **relative** comparisons (does verifier-on > verifier-off?) rather than absolute truth.
- **Retrieval recall is binary-per-substring**, not formal IR `recall@k`. Without ground-truth chunk IDs, substring presence is the practical proxy.
- **The eval is corpus-specific.** [eval_dataset.jsonl](eval_dataset.jsonl) is for the HP4 (Goblet of Fire) corpus. Swap it for your own questions when you index a different document.

## Configuration

All knobs are in `.env` with sensible defaults. Notable ones:

| Variable | Default | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | *(required)* | NVIDIA NIM auth |
| `MILVUS_COLLECTION_NAME` | `docs` | Document collection |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | PDF chunking |
| `RETRIEVE_K` / `TOP_K` | `40` / `5` | Wide retrieve, narrow rerank |
| `TOGGLE_CACHE` | `false` | Semantic Q/A cache (Milvus collection) |
| `LANGFUSE_ENABLED` | `true` | Set `false` to skip tracing |
| `PROMPT_VERSION` | `v2` | Cache key + Langfuse tag |

## Project layout

```
.
├── app.py                       # FastAPI routes
├── ui.py                        # Streamlit chat UI
├── docker-compose.yml           # Milvus + Postgres + Redis + Langfuse
├── clickhouse_config/           # ClickHouse Keeper config (Langfuse)
├── pdfs/                        # Drop PDFs here for ingestion
├── eval_dataset.jsonl           # 30 ground-truth Q&A cases for the evaluation harness
├── scripts/
│   ├── debug_agent.py           # Terminal agent runner with per-node output
│   ├── visualize_graph.py       # Generate graph.mmd / graph.png
│   ├── locust_workload.py       # Multi-user load test
│   └── eval.py                  # Evaluation harness — accuracy / recall / verifier metrics
└── src/
    ├── prompts/prompt.yaml      # System prompt
    └── utils/
        ├── rag_pipeline.py      # LangGraph agentic loop
        ├── tools.py             # search_chunks tool
        ├── observability.py     # Langfuse wrapper
        ├── chat/chat_service.py # cache + RAG orchestration
        └── services/
            ├── milvus_store.py  # Hybrid retrieval + cache store
            ├── embedder.py      # NVIDIAEmbeddings wrapper
            ├── inference.py     # ChatNVIDIA wrapper
            └── chunk_ranking.py # NVIDIARerank wrapper
```

## Resetting state

```bash
# Wipe chat history (keeps Milvus collections)
curl -X POST http://localhost:8000/clear_post_gres

# Drop a Milvus collection
curl -X POST "http://localhost:8000/clear_milvus?name=docs"

# Drop the semantic cache collection
curl -X POST http://localhost:8000/clear_cache

# Nuke everything (containers + volumes)
docker compose down -v && rm -rf milvus_data etcd_data minio_data
```
