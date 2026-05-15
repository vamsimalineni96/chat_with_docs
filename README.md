# BTX-BPD-Bodycam-Search

RAG over a document corpus, built as the search/summarization backbone. The current `pdfs/` corpus is a placeholder (Harry Potter books 4 and 7) used to validate the pipeline end-to-end; swap in the real documents to point the system at bodycam transcripts.

## Stack

- **API**: FastAPI ([app.py](app.py)) — chat, async PDF ingestion, admin endpoints.
- **UI**: Streamlit ([ui.py](ui.py)) — multi-user chat with a debug panel (retrieval, rerank, prompt, timings).
- **Vector store**: Milvus 2.5 with hybrid retrieval (dense embeddings + BM25, fused via RRF).
- **Reranker**: cross-encoder rerank over the fused candidates.
- **Embeddings / LLM**: NVIDIA AI endpoints via LangChain.
- **State**: Postgres (users / conversations / messages), Redis (per-conversation lock + semantic cache key).
- **Observability**: self-hosted Langfuse v3 (traces, prompts, timings).

## Layout

- [app.py](app.py) — FastAPI routes.
- [ui.py](ui.py) — Streamlit chat UI.
- [src/](src/) — services: embedder, PDF parser, Milvus store, conversation store, Redis lock, chat orchestration, prompts.
- [scripts/locust_workload.py](scripts/locust_workload.py) — load test.
- [docker-compose.yml](docker-compose.yml) — Milvus (+etcd, MinIO), Postgres, Redis, Langfuse stack.
- [pdfs/](pdfs/) — input documents to ingest.

## Run

```bash
docker compose up -d                          # Milvus, Postgres, Redis, Langfuse
pip install -r requirements.txt
uvicorn app:app --reload                      # API on :8000
streamlit run ui.py                           # UI on :8501
```

Langfuse UI: http://localhost:3000 (`admin@local.dev` / `admin1234`).

## Key endpoints

- `POST /chat` — multi-turn chat, per-conversation Redis lock, semantic cache + RAG fallback.
- `POST /upload_pdf_async` — kicks off background ingestion; poll `GET /task_status/{task_id}`.
- `GET /list_conversations`, `GET /list_messages` — history for a `user_external_id`.
- Admin: `/clear_post_gres`, `/clear_cache`, `/clear_milvus`, `/view_milvus_store`, `/debug_database`.
