# Eval run — 2026-05-24 10:04 UTC

- **Generator:** `google/gemma-4-31b-it`
- **Judge:** `meta/llama-3.3-70b-instruct (default)` (different model family — see [docs/PROGRESS.md](../PROGRESS.md))
- **Q&A count:** 3

## Aggregate

| Recall@5 | MRR | Groundedness | Accuracy | Completeness | p50 latency | p95 latency |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | 5.00/5 | 5.00/5 | 3.67/5 | 21974ms | 34496ms |

## By book

| Book | Count | Recall@5 | MRR | Ground | Acc | Comp |
|---|---:|---:|---:|---:|---:|---:|
| `hp4` | 3 | 1.00 | 1.00 | 5.00 | 5.00 | 3.67 |

## By category

| Category | Count | Recall@5 | Ground | Acc | Comp |
|---|---:|---:|---:|---:|---:|
| `character` | 1 | 1.00 | 5.00 | 5.00 | 5.00 |
| `magic` | 1 | 1.00 | 5.00 | 5.00 | 5.00 |
| `plot` | 1 | 1.00 | 5.00 | 5.00 | 1.00 |

## Per question

| ID | Book | Category | Recall@5 | Ground | Acc | Comp | Latency |
|---|---|---|---:|---:|---:|---:|---:|
| `hp4-001` | hp4 | character | 1.00 | 5 | 5 | 5 | 21974ms |
| `hp4-002` | hp4 | plot | 1.00 | 5 | 5 | 1 | 34496ms |
| `hp4-003` | hp4 | magic | 1.00 | 5 | 5 | 5 | 47031ms |

## Failures (1)

Rows where retrieval recall@5 < 50% or any judge sub-score < 3/5.

### `hp4-002` (hp4, plot)

**Question:** What are the three tasks of the Triwizard Tournament in book 4?

**Answer:** The context doesn't specify the three tasks of the Triwizard Tournament, only that there will be three tasks spaced throughout the school year, testing the champions' magical prowess, daring, powers of deduction, and ability to cope with danger. It does mention that the third task takes place in the evening, but the details of each task are not provided.

**Scores:** recall@5=1.00, ground=5, acc=5, comp=1

**Judge reasoning:** The answer is grounded and accurate because it correctly states that the context does not specify the three tasks, but it lacks completeness as it does not mention any of the expected keywords like dragon, lake, maze, or merpeople, which are implied to be part of the tasks.

