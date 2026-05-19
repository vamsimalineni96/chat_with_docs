# Eval run — 2026-05-19 04:00 UTC (from fixture)

- **Generator:** `google/gemma-4-31b-it`
- **Judge:** `meta/llama-3.3-70b-instruct` (different model family — see [docs/PROGRESS.md](../PROGRESS.md))
- **Q&A count:** 5

## Aggregate

| Recall@5 | MRR | Groundedness | Accuracy | Completeness | p50 latency | p95 latency |
|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.70 | 4.80/5 | 4.40/5 | 4.20/5 | 1050ms | 1200ms |

## By book

| Book | Count | Recall@5 | MRR | Ground | Acc | Comp |
|---|---:|---:|---:|---:|---:|---:|
| `hp4` | 2 | 1.00 | 0.75 | 5.00 | 5.00 | 4.50 |
| `hp7` | 3 | 0.67 | 0.67 | 4.67 | 4.00 | 4.00 |

## By category

| Category | Count | Recall@5 | Ground | Acc | Comp |
|---|---:|---:|---:|---:|---:|
| `character` | 2 | 1.00 | 5.00 | 5.00 | 4.50 |
| `magic` | 2 | 0.90 | 5.00 | 4.50 | 5.00 |
| `plot` | 1 | 0.20 | 4.00 | 3.00 | 2.00 |

## Per question

| ID | Book | Category | Recall@5 | Ground | Acc | Comp | Latency |
|---|---|---|---:|---:|---:|---:|---:|
| `hp4-001` | hp4 | character | 1.00 | 5 | 5 | 4 | 1200ms |
| `hp4-003` | hp4 | magic | 1.00 | 5 | 5 | 5 | 950ms |
| `hp7-001` | hp7 | magic | 0.80 | 5 | 4 | 5 | 1450ms |
| `hp7-002` | hp7 | plot | 0.20 | 4 | 3 | 2 | 1100ms |
| `hp7-004` | hp7 | character | 1.00 | 5 | 5 | 5 | 1050ms |

## Failures (1)

Rows where retrieval recall@5 < 50% or any judge sub-score < 3/5.

### `hp7-002` (hp7, plot)

**Question:** What is the Battle of the Seven Potters?

**Answer:** I don't have enough context to answer this question accurately. The retrieved passages focused on Polyjuice Potion usage in a different scene.

**Scores:** recall@5=0.20, ground=4, acc=3, comp=2

**Judge reasoning:** Honest refusal but caused by weak retrieval — the right chunks were not surfaced.

