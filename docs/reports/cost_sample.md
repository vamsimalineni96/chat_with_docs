# Cost report — 2026-05-19 04:52 UTC

- **Source:** fixture (tests/fixtures/observations.json)
- **Window:** fixture-defined
- **Generator:** [evals/cost/cost_report.py](../../evals/cost/cost_report.py)

## Aggregate

| Metric | Value |
|---|---|
| Total traces | 5 |
| Total spend (USD) | $0.002782 |
| Cost per successful task (avg) | $0.000556 |

## By task type

| Task | Trace count | Total $ | $/task |
|---|---:|---:|---:|
| `rag-full` | 3 | $0.002779 | $0.000926 |
| `cache-hit` | 2 | $0.000003 | $0.000002 |

## By pipeline stage

| Stage | Total $ | Share | Input tokens | Output tokens |
|---|---:|---:|---:|---:|
| `llm` | $0.002472 | 88.9% | 2,870 | 1,250 |
| `rerank` | $0.000300 | 10.8% | 75 | 75 |
| `embedding` | $0.000010 | 0.3% | 97 | 0 |

## Notes

- Prices are sourced from [evals/cost/pricing.py](../../evals/cost/pricing.py). Replace with real contract rates before relying on absolute USD figures.
- Unpriced models (not in the table) currently contribute $0; they still appear in token counts.
- Rerank is priced per-call (NVIDIA hosted endpoint), not per token.
