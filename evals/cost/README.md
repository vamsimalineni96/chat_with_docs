# Cost pillar tooling

Aggregates per-trace token usage from Langfuse, multiplies by per-model prices, and renders a markdown cost report partitioned by task type (`cache-hit` vs `rag-full`) and pipeline stage (`embedding`, `rerank`, `llm`).

| File | Purpose |
|---|---|
| [`cost_report.py`](cost_report.py) | Pure aggregation pipeline + Langfuse fetch + CLI. Run via `python -m evals.cost.cost_report`. |
| [`pricing.py`](pricing.py) | `MODEL_PRICES` table (USD per 1M input/output tokens) + rerank per-call price. Edit when contract rates change. |

Application-side token counting lives in [`src/utils/services/tokenizers.py`](../../src/utils/services/tokenizers.py) — it runs inside the request path so Langfuse spans carry token counts even for stages where the provider doesn't return usage.

See [docs/OBSERVABILITY.md §3.2](../../docs/OBSERVABILITY.md) for the framework rationale; sample report in [docs/reports/cost_sample.md](../../docs/reports/cost_sample.md).
