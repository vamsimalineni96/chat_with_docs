# Latency report — 2026-05-24 05:50 UTC

- **Source:** fixture (tests/fixtures/observations_latency.json)
- **Window:** fixture-defined
- **Generator:** [evals/latency/latency_report.py](../../evals/latency/latency_report.py)

## Aggregate

| Metric | Value |
|---|---|
| Total traces | 5 |
| p50 total latency | 5320 ms |
| p95 total latency | 22480 ms |
| p99 total latency | 22480 ms |

## By task type

| Task | Trace count | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| `rag-full` | 3 | 8750 ms | 22480 ms | 22480 ms |
| `cache-hit` | 2 | 118 ms | 145 ms | 145 ms |

## By pipeline stage

| Stage | Obs count | p50 | p95 | p99 | Share of time |
|---|---:|---:|---:|---:|---:|
| `llm` | 3 | 8170 ms | 21810 ms | 21810 ms | 94.9% |
| `rerank` | 3 | 425 ms | 502 ms | 502 ms | 3.4% |
| `embedding` | 5 | 115 ms | 148 ms | 148 ms | 1.6% |
| `cache_lookup` | 2 | 23 ms | 28 ms | 28 ms | 0.1% |

## Slowest traces (top 10)

| Trace ID | Task | Total |
|---|---|---:|
| `trace-rag-3` | `rag-full` | 22480 ms |
| `trace-rag-2` | `rag-full` | 8750 ms |
| `trace-rag-1` | `rag-full` | 5320 ms |
| `trace-cache-2` | `cache-hit` | 145 ms |
| `trace-cache-1` | `cache-hit` | 118 ms |

## Notes

- Percentiles use nearest-rank with no interpolation. With small N (<100), p99 is effectively max.
- Trace total is wall-clock from Langfuse when available, else the sum of observation durations (may over-count if spans overlap).
- Stage classification is shared with the cost report — see `evals.cost.cost_report.classify_stage`.
