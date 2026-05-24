# Latency pillar tooling

Aggregates Langfuse trace timings into p50/p95/p99 by task type and pipeline stage, plus a slowest-traces top-10. Same shape as the cost report and reuses its stage-classification helpers so a span called `rerank` always lands in the same bucket across both reports.

| File | Purpose |
|---|---|
| [`latency_report.py`](latency_report.py) | Pure aggregation pipeline + Langfuse fetch + CLI. Run via `python -m evals.latency.latency_report`. |

Application-side latency instrumentation (per-stage timings emitted to Langfuse, request-path retries with backoff) lives in [`src/utils/services/`](../../src/utils/services/) — see [`retry.py`](../../src/utils/services/retry.py) and the per-stage `t_*_start`/`t_*_end` keys threaded through [`src/utils/rag_pipeline.py`](../../src/utils/rag_pipeline.py).

See [docs/OBSERVABILITY.md §3.4](../../docs/OBSERVABILITY.md) for the framework rationale; sample report in [docs/reports/latency_sample.md](../../docs/reports/latency_sample.md).
