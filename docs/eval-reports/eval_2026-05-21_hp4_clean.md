# Eval run — 2026-05-21 06:02 UTC

- **Generator:** `google/gemma-4-31b-it`
- **Judge:** `meta/llama-3.3-70b-instruct (default)` (different model family — see [docs/PROGRESS.md](../PROGRESS.md))
- **Q&A count:** 9

## Aggregate

| Recall@5 | MRR | Groundedness | Accuracy | Completeness | p50 latency | p95 latency |
|---:|---:|---:|---:|---:|---:|---:|
| 0.83 | 0.94 | 4.67/5 | 4.78/5 | 3.44/5 | 9836ms | 17611ms |

## By book

| Book | Count | Recall@5 | MRR | Ground | Acc | Comp |
|---|---:|---:|---:|---:|---:|---:|
| `hp4` | 9 | 0.83 | 0.94 | 4.67 | 4.78 | 3.44 |

## By category

| Category | Count | Recall@5 | Ground | Acc | Comp |
|---|---:|---:|---:|---:|---:|
| `character` | 2 | 0.75 | 4.50 | 5.00 | 4.00 |
| `factual` | 1 | 1.00 | 4.00 | 4.00 | 3.00 |
| `magic` | 2 | 1.00 | 5.00 | 5.00 | 4.50 |
| `plot` | 3 | 0.67 | 4.67 | 4.67 | 2.33 |
| `reasoning` | 1 | 1.00 | 5.00 | 5.00 | 4.00 |

## Per question

| ID | Book | Category | Recall@5 | Ground | Acc | Comp | Latency |
|---|---|---|---:|---:|---:|---:|---:|
| `hp4-001` | hp4 | character | 1.00 | 5 | 5 | 5 | 17179ms |
| `hp4-002` | hp4 | plot | 1.00 | 5 | 5 | 1 | 9404ms |
| `hp4-003` | hp4 | magic | 1.00 | 5 | 5 | 5 | 10592ms |
| `hp4-004` | hp4 | plot | 0.50 | 4 | 4 | 2 | 9261ms |
| `hp4-005` | hp4 | character | 0.50 | 4 | 5 | 3 | 17611ms |
| `hp4-006` | hp4 | factual | 1.00 | 4 | 4 | 3 | 19584ms |
| `hp4-007` | hp4 | magic | 1.00 | 5 | 5 | 4 | 6475ms |
| `hp4-008` | hp4 | plot | 0.50 | 5 | 5 | 4 | 9836ms |
| `hp4-009` | hp4 | reasoning | 1.00 | 5 | 5 | 4 | 12885ms |

## Failures (2)

Rows where retrieval recall@5 < 50% or any judge sub-score < 3/5.

### `hp4-002` (hp4, plot)

**Question:** What are the three tasks of the Triwizard Tournament in book 4?

**Answer:** The context doesn't specify the three tasks of the Triwizard Tournament, only that there will be three tasks spaced throughout the school year, testing the champions' magical prowess, daring, powers of deduction, and ability to cope with danger. It does mention that the third task takes place in the evening, but the details of each task are not provided.

**Scores:** recall@5=1.00, ground=5, acc=5, comp=1

**Judge reasoning:** The answer is grounded and accurate because it correctly states that the context does not specify the three tasks, but it lacks completeness as it does not mention any of the expected keywords such as dragon, lake, maze, or merpeople, which are implied to be part of the tasks.

### `hp4-004` (hp4, plot)

**Question:** How does Voldemort return to a physical body at the end of book 4?

**Answer:** Voldemort returns to a physical body at the end of book 4 with the help of Wormtail, who follows his instructions to obtain the essential ingredients for his rebirth. Although the exact process is not fully detailed in the provided context, it's mentioned that Wormtail's actions allow Voldemort to inhabit a rudimentary, weak body of his own, which he can use until he can gather the necessary ingredients for a true rebirth. The context does not specify what these ingredients are or the full extent of the process, but it's clear that Wormtail plays a crucial role in Voldemort's return to a physical form.

**Scores:** recall@5=0.50, ground=4, acc=4, comp=2

**Judge reasoning:** The answer is grounded in the context as it mentions Wormtail's role in Voldemort's return, but it lacks completeness as it doesn't touch on the expected keywords like Pettigrew, ritual, bone, flesh, and blood, which are crucial for understanding the process of Voldemort's rebirth.

