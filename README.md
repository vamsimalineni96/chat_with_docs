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
locust -f mt_rag_locust.py --host http://localhost:8000 \
       --users 8 --spawn-rate 1 -t 10m --headless
```

Default config stays under NVIDIA NIM's 40 RPM cap. Tunables via env vars: `LOCUST_USER_POOL_SIZE`, `LOCUST_MAX_CONVOS_PER_USER`, `LOCUST_DEBUG_PROB`.

## Useful scripts

```bash
python debug_agent.py "your question"   # run the agent loop with per-node output
python visualize_graph.py               # regenerate graph.mmd and graph.png
```

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
├── debug_agent.py               # Terminal debug runner
├── visualize_graph.py           # Generate graph.png / graph.mmd
├── mt_rag_locust.py             # Load test
├── docker-compose.yml           # Milvus + Postgres + Redis + Langfuse
├── clickhouse_config/           # ClickHouse Keeper config (Langfuse)
├── src/
│   ├── prompts/prompt.yaml      # System prompt
│   └── utils/
│       ├── rag_pipeline.py      # LangGraph agentic loop
│       ├── tools.py             # search_chunks tool
│       ├── observability.py     # Langfuse wrapper
│       ├── chat/chat_service.py # cache + RAG orchestration
│       └── services/
│           ├── milvus_store.py  # Hybrid retrieval + cache store
│           ├── embedder.py      # NVIDIAEmbeddings wrapper
│           ├── inference.py     # ChatNVIDIA wrapper
│           └── chunk_ranking.py # NVIDIARerank wrapper
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
