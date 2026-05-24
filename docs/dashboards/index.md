# Observability index

The "latest-numbers" view of the three pillars. One section per pillar, each pointing at the most recent dated report plus a one-line read.

Design rationale and gap audit: [`../OBSERVABILITY.md`](../OBSERVABILITY.md). Build log: [`../PROGRESS.md`](../PROGRESS.md).

---

## Cost

> *What does a task cost?*

- **Latest:** [`../reports/cost_2026-05-19.md`](../reports/cost_2026-05-19.md)
- **Sample (fixture):** [`../reports/cost_sample.md`](../reports/cost_sample.md)
- **Source script:** [`../../scripts/cost_report.py`](../../scripts/cost_report.py)
- **Cron:** [`../../.github/workflows/cost-report-nightly.yml`](../../.github/workflows/cost-report-nightly.yml) — 06:00 UTC daily (currently paused via the GitHub UI while in dev; re-enable in *Actions → Cost report (nightly) → Enable workflow*)

Headline from the latest run: 3 RAG-full traces over the 7-day window, **~$0.0001 per task** at current pricing. The `cache-hit` task type wasn't exercised in this window — when it is, expect $/task to drop by 2–3 orders of magnitude (the cache short-circuits the LLM stage entirely).

How to read a cost report: see [`../reports/README.md#how-to-read-a-cost-report`](../reports/README.md#how-to-read-a-cost-report).

---

## Quality

> *Are answers grounded, accurate, and complete?*

- **Latest (clean post-ingest-fix baseline):** [`../eval-reports/eval_2026-05-21_hp4_clean.md`](../eval-reports/eval_2026-05-21_hp4_clean.md)
- **First baseline (revealed an HP7 ingestion gap):** [`../eval-reports/eval_2026-05-20.md`](../eval-reports/eval_2026-05-20.md)
- **Sample (fixture):** [`../eval-reports/eval_sample.md`](../eval-reports/eval_sample.md)
- **Source script:** [`../../eval/run_eval.py`](../../eval/run_eval.py)
- **CI gate:** dataset shape is validated on every PR — see the `eval-dataset-validate` job in [`../../.github/workflows/ci.yml`](../../.github/workflows/ci.yml). The full eval is run manually against a live app, not on PRs.

Headline from the clean baseline (9 HP4 questions, gemma-4-31b generator, llama-3.3-70b judge): **recall@5 0.83, groundedness 4.67/5, accuracy 4.78/5, completeness 3.44/5.** Zero 5xx errors, zero rate-limit retries.

The eval has earned its keep twice already:
1. The first baseline ([`eval_2026-05-20.md`](../eval-reports/eval_2026-05-20.md)) revealed HP7 retrieval was effectively broken — root cause: PDF was never ingested. Surfaced a gap that wasn't visible from manual chat testing.
2. Inside the same run, two HP7 questions hit `502 Bad Gateway` from NVIDIA's load balancer and two judge calls hit `429 Too Many Requests` — the source signal that motivated PR #10 (retries on the read path) and the ingest-fix PR (retries on the write path).

How to read an eval report: see [`../eval-reports/README.md`](../eval-reports/README.md).

---

## Latency

> *Where is time being spent? Where are the tails?*

- **Latest (sample only — live cron paused):** [`../reports/latency_sample.md`](../reports/latency_sample.md)
- **Source script:** [`../../scripts/latency_report.py`](../../scripts/latency_report.py)
- **Cron:** [`../../.github/workflows/latency-report-nightly.yml`](../../.github/workflows/latency-report-nightly.yml) — 06:30 UTC daily (currently paused via the GitHub UI while in dev)

Headline from the sample fixture (5 traces): **LLM stage eats 94.9% of total request time.** Cache-hit traces complete in ~120ms vs RAG-full traces at 5–22 seconds — two orders of magnitude. Per-stage breakdown:

| Stage | p95 | Share of total time |
|---|---:|---:|
| `llm` | 21810 ms | 94.9% |
| `rerank` | 502 ms | 3.4% |
| `embedding` | 148 ms | 1.6% |
| `cache_lookup` | 28 ms | 0.1% |

This points the eye exactly where any latency-reduction work should land: the LLM stage. Embedding and rerank optimisations would be rounding error against the LLM tail.

How to read a latency report: see [`../reports/README.md#how-to-read-a-latency-report`](../reports/README.md#how-to-read-a-latency-report).

---

## Reliability primitives (under Latency)

Two behaviours that make the headline metrics achievable under transient upstream failure:

- **Retries** — every NVIDIA call on the `/chat` read path is wrapped in [`call_with_retry`](../../src/utils/services/retry.py) (3 attempts, exponential backoff, 15s cap, WARNING log per attempt). The ingest path has its own variant in [`milvus_store._add_texts_with_retry`](../../src/utils/services/milvus_store.py) (5 attempts, 30s cap — more aggressive because there's no human waiting).
- **Idempotent ingestion** — deterministic per-page `doc_id` + delete-before-insert. A re-run after a mid-stream failure replaces existing chunks rather than accumulating duplicates. Re-runnable safely without operator coordination.

What's *deferred* and tracked under [`../OBSERVABILITY.md`](../OBSERVABILITY.md) §3.4:

- Circuit breakers per dependency (NVIDIA, Milvus, Postgres, Redis).
- Fallback model — secondary NVIDIA endpoint when the primary fails its retry budget.
- Alerting on the percentile thresholds (retrieval p99 > 800ms, end-to-end p95 > 8s) once a few weeks of nightly data are accumulated.

---

## What this page is for

If you're new to the repo and want the operational snapshot rather than the source tour, this is the page. Click into any dated report for the per-task / per-stage breakdown; click into [`../OBSERVABILITY.md`](../OBSERVABILITY.md) for the *why* behind the shape of everything above.
