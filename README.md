# Agentic RAG Chatbot — LangGraph + NVIDIA NIM + Milvus

A production-style **agentic RAG** system: an LLM that decides *when* to retrieve, *what* to retrieve, and *whether its own answer is trustworthy* — instead of a fixed `retrieve → generate` pipe. Built as a LangGraph state machine with hybrid (dense + BM25) Milvus retrieval, NVIDIA Llama 3.3 70B for both tool-calling and grading, self-hosted Langfuse observability, and a multi-user Streamlit front end backed by FastAPI, Postgres, and Redis.

![Graph](graph.png)

## What makes this interesting

Not just *"vector search + LLM"*. The agent runs a four-node graph with two cycles:

```
rewriter ──► agent ◄────────────────────────┐
              │                             │
              │ tool_calls? ──► tools ──────┤
              │ no                          │
              ▼                             │
          verifier ── grounded? ── yes/cap ──► END
              │                             │
              │ no, with critique injected  │
              └─────────────────────────────┘
```

| Node | What it does | Why it's here |
|---|---|---|
| **rewriter** | LLM call (no tools). Resolves pronouns and context references against chat history (`"he"` → `"Harry Potter"`, `"the second one"` → `"the second task"`). | Follow-up questions get sharper retrieval. |
| **agent** | LLM call **with tools bound** (`ChatNVIDIA.bind_tools([...])`). Decides whether to retrieve, how to phrase the query, and how many parallel searches to fire. | The LLM does query decomposition itself — *"compare X and Y"* becomes two parallel `tool_call`s. |
| **tools** | Custom dispatcher (`make_tool_node`). Executes every tool_call in the last AIMessage, returns `ToolMessage`s with `tool_call_id` linkage. | Replaces `langgraph.prebuilt.ToolNode` to sidestep version-pinning issues. |
| **verifier** | LLM call (no tools). Grades the candidate answer against retrieved passages for grounding. Returns `GROUNDED: yes/no` + critique. | Catches hallucination. If unsupported claims are found, the loop re-enters the agent with a `SystemMessage` critique up to `MAX_CORRECTIONS=2` times. |

Both cycles (`agent ↔ tools`, `agent ↔ verifier`) are **impossible in a plain LCEL chain** — they're the whole reason LangGraph exists.

## Tech stack

| Layer | Tools |
|---|---|
| **Orchestration** | LangGraph 0.4.x (StateGraph with TypedDict state + `add_messages` reducer) |
| **LLM** | NVIDIA NIM — Llama 3.3 70B Instruct (via `langchain-nvidia-ai-endpoints`) |
| **Embeddings** | NVIDIA `nv-embedqa-e5-v5` |
| **Reranker** | NVIDIA `nv-rerank-qa-mistral-4b:1` |
| **Vector store** | Milvus 2.5.4 — hybrid dense + BM25 sparse via Reciprocal Rank Fusion |
| **Conversation store** | PostgreSQL (users, conversations, messages) via SQLAlchemy |
| **Concurrency control** | Redis distributed locks (per-conversation, prevents interleaved writes) |
| **Observability** | Langfuse 4.x (self-hosted, OpenTelemetry-based) — every node, every tool call, every LLM invocation traced |
| **API** | FastAPI (async) |
| **UI** | Streamlit (multi-user, multi-session, debug-mode inspector) |
| **Load testing** | Locust (multi-persona, configurable RPM under NIM's 40 RPM cap) |
| **Infrastructure** | Docker Compose (entire stack, including Langfuse's PostgreSQL + ClickHouse + MinIO + Redis) |

## Engineering decisions worth highlighting

These are the calls that distinguish this from a tutorial-grade RAG bot:

1. **Hybrid retrieval over pure semantic.** Dense + BM25 with RRF fusion (Milvus 2.5's `BM25BuiltInFunction`). Dense embeddings miss exact names ("Cedric Diggory"), proper nouns, and quote substrings. BM25 catches them. RRF fuses both rankings without tuning weights.

2. **Tool name decoupled from prompt.** The system prompt describes *behavior* ("use the search tool when content needs to be looked up"), never the tool's *name*. Tool naming comes entirely from the LangChain `@tool`-derived schema injected via `bind_tools()`. Renaming `search_chunks` → `search_book` requires zero prompt changes — the LLM picks up the new name from the schema.

3. **Verifier failure-open.** If the verifier's LLM call fails or returns unparseable text, treat as `grounded=True`. Prevents the graph from looping forever on infrastructure hiccups. Trade-off: a few hallucinations may slip through during outages.

4. **Tool-name source of truth = Python function name.** `@tool`-decorated function's `__name__` is the only place the tool name lives. The LangChain schema, ToolMessage `.name`, and `tool_map` lookup key all derive from it. No string duplication, no rename drift.

5. **Closure-based dependency injection for tools.** `build_search_chunks_tool(milvus_store, reranker)` returns a factory-built tool with services captured in a closure. The LLM only sees `(query, top_k)` in the schema; the heavy service objects live in the closure. This lets each request bind its own collection without globals.

6. **Per-request tool construction.** Tools are built inside `answer_question`, not at module load. Different users can hit different Milvus collections in the same process without state leaking.

7. **Defensive tool-call dispatcher.** Unknown tool names → return an error string as `ToolMessage`, don't crash. Tool raises an exception → wrap as error string. The LLM sees the failure on the next iteration and usually recovers (corrects a hallucinated tool name, retries with simpler args). Single failures don't tear down the graph.

8. **Bounded loops everywhere.** `MAX_AGENT_ITERATIONS=8` (total agent LLM calls), `MAX_CORRECTIONS=2` (verifier-triggered retries). The agent can't infinite-loop on a stubborn tool failure; the verifier can't infinite-loop on a stubborn ungrounded answer.

9. **Cache uses semantic similarity in Milvus, not exact string match.** Cosine-similar questions (≥0.9 by default) hit the same cached answer. Cache is gated by `(model_name, prompt_version)` so a prompt update invalidates the cache automatically.

10. **Langfuse trace topology mirrors the graph.** Every node is its own span; every LLM call is its own generation; every tool call is its own span. Picking through a 5-iteration agentic run in the Langfuse UI is straightforward because the trace tree is the actual call tree.

## Verified behaviors

Smoke-tested end-to-end with mocked services:

| Scenario | Behavior |
|---|---|
| Simple single-topic question | Rewriter no-ops (no history) → agent retrieves once → verifier passes → 1 tool call, 2 iterations |
| Comparative question (*"compare X and Y"*) | Agent emits **two parallel `tool_call`s** in one AIMessage; tool node executes both; agent synthesizes both passages into one answer |
| Follow-up with pronoun (*"what did **he** use…"*) | Rewriter resolves *"he"* → *"Harry"*, agent searches with the resolved phrasing |
| Ungrounded answer | Verifier emits `GROUNDED: no` with critique → loop back to agent with `SystemMessage` hint → agent revises |
| Persistently ungrounded | Hits `MAX_CORRECTIONS=2`, returns last attempt instead of looping forever |
| Off-topic question | Agent retrieves, gets no useful passages, honestly says *"I can't find this in the document"*; verifier accepts this as grounded |

## Running it

### Prerequisites

- Docker + Docker Compose
- Python 3.12
- NVIDIA API key — sign up at [build.nvidia.com](https://build.nvidia.com)

### Setup

```bash
git clone <repo-url> && cd BTX-BPD-Bodycam-Search

# 1. Configure secrets
cp .env.example .env   # then fill in NVIDIA_API_KEY

# 2. Bring up the stack (Milvus + Postgres + Redis + Langfuse)
docker compose up -d

# 3. Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. Index a document
curl -X POST "http://localhost:8000/upload_pdf_async?pdf_name=your.pdf&collection_name=docs"

# 5. Run the API + UI
uvicorn app:app --reload     # in one terminal
streamlit run ui.py          # in another
```

Open `http://localhost:8501`, enter a user ID, ask a question.

### Observability

- **Langfuse UI**: <http://localhost:3000> (login: `admin@local.dev` / `admin1234`). Every request produces a trace tree showing rewriter → agent → tool → verifier with full inputs/outputs and token usage per generation.
- **Streamlit debug mode**: toggle "🐛 Debug mode" in the sidebar to bypass cache and surface intermediate state (agent steps, verifier checks, message trail) in an expander.
- **Terminal debug script**: `python debug_agent.py "your question"` runs the agent loop end-to-end with verbose per-node output.

### Load testing

```bash
locust -f mt_rag_locust.py --host=http://localhost:8000
```

Multi-persona load against `/chat` with configurable RPM. The default config (20 personas, 10-15s think time) stays under the NIM 40 RPM cap.

### Visualize the graph

```bash
python visualize_graph.py
```

Generates `graph.mmd` (Mermaid source) and `graph.png` (rendered) for documentation or slides. Paste the Mermaid into <https://mermaid.live> for an interactive editor.

## Project layout

```
.
├── app.py                       # FastAPI routes (/chat, /list_*, /upload_pdf_async, …)
├── ui.py                        # Streamlit chat UI
├── debug_agent.py               # Terminal debug script — stream graph events
├── visualize_graph.py           # Generate graph.png / graph.mmd
├── mt_rag_locust.py             # Load test
├── docker-compose.yml           # Milvus + Postgres + Redis + Langfuse stack
├── src/
│   ├── prompts/prompt.yaml      # System prompt (tool-name agnostic)
│   └── utils/
│       ├── rag_pipeline.py      # ★ The agentic graph — rewriter + agent + tools + verifier
│       ├── tools.py             # search_chunks tool (LangChain @tool)
│       ├── observability.py     # Langfuse wrapper (@observe, update_current_*, callbacks)
│       ├── chat/chat_service.py # cache lookup + rag orchestration
│       └── services/
│           ├── milvus_store.py  # Hybrid retrieval + Q/A cache store
│           ├── embedder.py      # NVIDIAEmbeddings wrapper
│           ├── inference.py     # ChatNVIDIA wrapper, exposes .llm + .system_prompt
│           └── chunk_ranking.py # NVIDIARerank wrapper
```

The core of the system fits in [`src/utils/rag_pipeline.py`](src/utils/rag_pipeline.py) — read it top-to-bottom.

## What I learned building this

- **LangGraph is the right abstraction once you need cycles.** LCEL is great for linear pipelines; the moment you want self-correction or agentic retrieval, you spend more code working around LCEL's DAG-only constraints than you save by avoiding LangGraph.
- **Tool-calling is just structured generation.** The LLM emits a JSON object instead of text; LangChain parses it into `AIMessage.tool_calls`. There's no magic — but the trained behavior of *deciding when and how to split a comparative query into parallel tool calls* is genuinely useful and not something you'd want to reimplement.
- **Observability is not optional for agentic systems.** With multiple LLM calls per request (rewriter + agent + verifier + retries), a 5-second latency could be hiding any of six things. Langfuse spans answer "which exact stage is slow?" in seconds — without them, you're guessing.
- **Hallucination shows up in subtle ways.** The verifier loop's most common catch isn't the model inventing characters; it's adding plausible-sounding details that *almost* match the passages but aren't quite supported. A verifier with a strict grounding prompt catches more than I expected.
- **Don't hardcode names in prompts.** The single largest source of "why doesn't it work" pain was the prompt referring to `search_chunks` while the function was renamed to `search_book`. The fix — making the prompt name-agnostic and letting the schema be the single source of truth — eliminated a whole class of bugs.

## Possible extensions

- **Streaming tokens to the UI** for ChatGPT-style live typing (LangGraph supports `stream_mode=["updates", "messages"]`).
- **Multi-collection routing** — a router node that picks among indexed corpora based on the question.
- **Human-in-the-loop approval** before generation, using LangGraph's checkpointer + `interrupt`.
- **MCP server wrapper** — expose `search_chunks` as a Model Context Protocol server so Claude Desktop / other MCP clients can query the corpus.
