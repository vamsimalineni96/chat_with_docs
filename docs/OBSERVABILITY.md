# Observability Design

A design doc for the observability layer of this RAG service. It declares the framework being adopted, audits the current state of the repo against that framework, and lays out the work needed to close the gaps.

This is the **north star document** for the observability-related PRs that follow. Every subsequent PR references a gap identified here.

---

## 1. Purpose and scope

This service runs a RAG pipeline (Milvus hybrid retrieval → cross-encoder rerank → NVIDIA NIM chat completion) behind a FastAPI app. It already captures a lot of telemetry through Langfuse. What it does not yet do — and what production GenAI systems must do — is **turn that telemetry into operational tools**: aggregated metrics, alerts, eval signals, regression gates, and reliability guarantees.

This document defines what "operational" means here, and how the repo will get there.

## 2. The framework

We adopt the three-pillar observability framework articulated by Pooja Palod in the *Data Journey* series (links in §8). The framework treats observability as a single system instrumented across three interdependent pillars rather than three separate concerns.

| Pillar | Question it answers | Headline metric |
|---|---|---|
| **Cost** (token economics) | *What does it cost to deliver a correct answer for this task?* | Cost per successful task, by task type |
| **Quality** (evaluation in production) | *Is the system actually producing useful, grounded answers — and is that trending?* | Decomposed judge scores (groundedness, accuracy, completeness) + heuristic failure rate + regression-set pass rate |
| **Latency & reliability** | *Where is time being spent, and which dependencies are failing silently?* | P95/P99 stage-level latency, retry rate, fallback rate, circuit-breaker state |

Two architectural prerequisites underlie all three:

1. **Trace propagation** — a single trace ID, assigned at request intake, carried through every pipeline stage. Without this, no cross-pillar analysis is possible.
2. **Structured logging** — events emitted as queryable key/value records, not free text. Without this, metrics can't be reconstructed after the fact.

### Why these three together

The most interesting production failures are *cross-pillar interactions*, not failures within a single pillar:

- A drop in retrieval relevance precedes a quality decline by 24–48 hours, which precedes a latency increase as the LLM works harder over weaker context.
- A cache-hit-rate decline without a volume change signals a query distribution shift — early warning of a domain mismatch.
- A retry-rate spike precedes an error-rate spike — early warning of upstream reliability problems.

Instrumenting the pillars in isolation hides these.

> "In practice they're not separate. They're instrumented together, they affect each other, and the signals from one pillar frequently explain anomalies in another." — Palod, *Building Observability for a Production GenAI System*

## 3. Current state of the repo

This section is a deliberate audit, not a sales pitch. Each row is grounded in a specific file/line.

### 3.1 Foundation

| Capability | Status | Evidence |
|---|---|---|
| Trace propagation (one trace ID through every stage) | **Implemented** via Langfuse v4.x + OpenTelemetry. Top-level FastAPI route delegates to `@observe`-decorated services; child spans inherit. v4 keeps the v3 OTel-attribute surface we depend on (`LangfuseOtelSpanAttributes`) alongside its newer `propagate_attributes()` idiom; we deliberately kept the OTel-direct path — see §3.1 note below. | [src/utils/observability.py](../src/utils/observability.py) — `update_current_trace` attaches `user_id`/`session_id`/tags/metadata via OTel; metadata is coerced to `dict[str, str]` for v4 validation. |
| Structured logging | **Partial.** A single structured `RAG_PIPELINE_METRICS \| k=v \| k=v` log line is emitted per request. Most other `logger.info(...)` calls are free text. | [src/utils/chat/chat_service.py:161–171](../src/utils/chat/chat_service.py) — the one structured line. Everything else uses `%s`-formatted prose. |
| Per-request task-type tagging | **Implemented** via Langfuse trace tags: `prompt:v2`, `collection:X`, `domain:Y`, `cache-path`/`rag-path`, `normal`/`debug`. | [src/utils/chat/chat_service.py:26–34](../src/utils/chat/chat_service.py), [src/utils/chat/chat_service.py:79–87](../src/utils/chat/chat_service.py). |

> **Note on v3 → v4 SDK migration.** PR #18 bumped `langfuse>=3.0.0` → `langfuse>=4.6.1,<5`. The minimum-viable changes were: (a) `should_export_span=lambda _s: True` on client init to preserve v3 export defaults, (b) proactive `dict[str, str]` coercion of metadata (v4 validates and drops oversized/non-string values). `update_current_trace` continues to set OTel attributes directly because `LangfuseOtelSpanAttributes` remains supported on v4. Migrating to v4's `propagate_attributes()` context-manager idiom would be churn without functional gain right now and is tracked as optional follow-up work.

### 3.2 Cost (token economics)

| Capability | Status | Evidence |
|---|---|---|
| Token counts captured at the LLM stage | **Implemented.** The LangChain `CallbackHandler` is passed to the chat-completion chain; NVIDIA endpoints return usage; Langfuse records it. | [src/utils/services/inference.py:65–75](../src/utils/services/inference.py). |
| Token counts captured at the embedding stage | **Implemented (tiktoken proxy).** `count_tokens` uses tiktoken's `cl100k_base` encoding — a deliberate proxy for NVIDIA's actual tokenizer. Typically within ~10%, sufficient for cost trending. Closed by PR #3. | [src/utils/services/tokenizers.py](../src/utils/services/tokenizers.py), [src/utils/services/embedder.py](../src/utils/services/embedder.py). |
| Reranker cost signal | **Captured as passage count**, not tokens. NVIDIA's rerank endpoint doesn't return token usage, so we log `input_count`/`output_count`. | [src/utils/services/chunk_ranking.py:73–82](../src/utils/services/chunk_ranking.py). |
| Cost-per-successful-task metric | **Implemented.** `scripts/cost_report.py` aggregates Langfuse spans, joins against `scripts/pricing.py`, and emits a markdown report partitioned by task type (`cache-hit` vs `rag-full`) and pipeline stage. Sample output in `docs/reports/cost_sample.md`. Closed by PR #4. | [scripts/cost_report.py](../scripts/cost_report.py), [docs/reports/cost_sample.md](reports/cost_sample.md). |
| Cache hit-rate trending | **Missing aggregation.** `cache-path` vs `rag-path` tags are per-trace; no rolling dashboard, no alert if hit rate falls below threshold. | — |
| Context-length tracking | **Missing.** No measurement of assembled-prompt token count over time. Edge-case prompt growth would be invisible. | — |
| Two-tier model routing (lightweight vs frontier) | **Missing.** Single `LLM_MODEL` from env; no classifier; no Tier-1 short-circuit. | [src/utils/config.py:28](../src/utils/config.py). |
| Cost alerts (20% WoW, cache hit < 25%, prompt-tokens above threshold) | **Missing.** | — |

### 3.3 Quality (evaluation in production)

| Capability | Status | Evidence |
|---|---|---|
| Full inference-time capture (question, retrieved chunks with scores, history, rendered prompt, final answer) | **Implemented** via `debug=true` path on `/chat` and via Langfuse spans on every request. | [src/utils/rag_pipeline.py:55–69, 101–117, 176–193](../src/utils/rag_pipeline.py). |
| LLM-as-judge eval (decomposed dimensions, calibrated, sampled async) | **Implemented** (manual mode). Dataset ([eval/qa_set.jsonl](../eval/qa_set.jsonl), 18 HP4/HP7 pairs) + retrieval metrics ([eval/metrics.py](../eval/metrics.py)) + Llama 3.3 70B judge with YAML-driven rubric ([eval/judge.py](../eval/judge.py), [eval/prompts/judge.yaml](../eval/prompts/judge.yaml)) + markdown reporter ([eval/reporter.py](../eval/reporter.py)) + end-to-end orchestrator ([eval/run_eval.py](../eval/run_eval.py)). Sub-PR #7d adds the scheduled CI workflow + auto-commit of dated reports. Calibration against human labels is the only remaining deferred item. | [eval/](../eval/), [docs/eval-reports/eval_sample.md](eval-reports/eval_sample.md) |
| Heuristic checks (synchronous, deterministic) — refusal detection, citation validation, length bounds, format validation | **Missing.** | — |
| Regression dataset (failures captured from production, used as a CI gate) | **Missing.** | — |
| Weekly human review process (50–100 sampled responses) | **Missing.** Out of scope for a single-engineer project but worth flagging. | — |
| Eval pass/fail blocking deployment | **Missing.** | — |

### 3.4 Latency and reliability

| Capability | Status | Evidence |
|---|---|---|
| Per-stage timings captured | **Implemented.** `db_load`, `milvus`, `llm`, `db_save`, `total` in `chat_service.rag_output`. | [src/utils/chat/chat_service.py:159–180](../src/utils/chat/chat_service.py). |
| P95/P99 rollup by stage and task type | **Missing aggregation.** Raw timings logged; no percentile dashboard. | — |
| Time-to-first-token (TTFT) | **N/A.** Streaming is not yet wired on the `/chat` endpoint. | [app.py:61–170](../app.py). |
| Retry / fallback rate logging | **Missing.** Errors are caught and re-raised; retries done by underlying SDKs are not explicitly counted. | [src/utils/services/inference.py:67–78](../src/utils/services/inference.py). |
| Circuit breakers per dependency (NVIDIA, Milvus, Postgres, Redis) | **Missing.** A single NVIDIA outage would cascade. | — |
| Fallback model / graceful degradation | **Missing.** | — |
| Deadline propagation (end-to-end timeout passed to every external call) | **Missing.** | — |
| Load test producing real percentiles | **Tooling present, results not captured.** `scripts/locust_workload.py` exists; no committed run artifacts. | [scripts/locust_workload.py](../scripts/locust_workload.py). |

### 3.5 Summary

The capture layer is largely in place. The work ahead is in three buckets:

- **Aggregation and surfacing** — turning captured Langfuse data into trended metrics and dashboards. Mostly read-side work.
- **Net-new instrumentation** — eval, heuristics, regression dataset, retry/fallback rate, circuit breakers, fallback model. Mostly write-side work.
- **Refinement of existing** — replacing the embedding token estimate with a real tokenizer, structuring more log lines, defining alert thresholds.

## 4. Roadmap

PRs are sized to be reviewable in one sitting and shippable independently. Each maps to a row in §3.

### Phase 0 — Foundation
- **PR #2 — This document.** (current)

### Phase 1 — Cost pillar

- **PR #3 — Real embedding tokenizer.** Replace `len(text) // 4` with `tiktoken` or NVIDIA's tokenizer. Closes the only inaccurate token count in the system. *Maps to §3.2 row 2.*
- **PR #4 — Cost-per-successful-task aggregation.** A script (or Langfuse dashboard config) that pulls spans, joins token counts × model prices, attributes by task type, and writes a daily `cost_report.md` artifact. *Maps to §3.2 row 4.*
- **PR #5 — Cache hit-rate dashboard + alert.** Trend `cache-path` vs `rag-path` ratio over time; alert if hit rate falls below 25% over a rolling 24h window (Pooja's threshold). *Maps to §3.2 row 5.*

Two-tier routing (§3.2 row 7) is deferred until task-type taxonomy is defined; would otherwise be premature.

### Phase 2 — Quality pillar

- **PR #6 — Eval harness v1.** `eval/qa_set.jsonl` (15–25 HP4/HP7 Q&A pairs), `eval/run_eval.py`, `eval/metrics.py` (recall@k, MRR), `eval/judge.py` (decomposed LLM-as-judge: groundedness, accuracy, completeness — using a *different* model family than the generator to avoid self-preference bias), `eval/reporter.py` (markdown output), `eval/results/latest.md` committed back. *Maps to §3.3 row 2.*
- **PR #7 — Heuristic request-path checks.** Synchronous middleware that scores every response on: refusal-phrase detection ("I think", "I'm not sure", "I cannot"), citation validation (assertions traceable to retrieved chunks), length bounds by task type. Emits `heuristic_failure_rate` as a tag. *Maps to §3.3 row 3.*
- **PR #8 — Regression dataset.** Capture failures (heuristic-flagged + low judge score + user-flagged via a `/feedback` endpoint) into `eval/regressions.jsonl`. Add a CI job that runs the regression set on any PR modifying `src/utils/services/`, `src/utils/rag_pipeline.py`, or `src/prompts/`. Pooja: "Regressions block deployment." *Maps to §3.3 row 4.*

Judge calibration against human labels (§3.3 row 5) is deferred — recommended in the framework, hard to do solo, but called out in the doc as the next step a real team would take.

### Phase 3 — Latency and reliability pillar

- **PR #9 — Stage-level P95/P99 dashboard.** Query Langfuse for the past 7 days of `hybrid_retrieve`, `rerank`, `embed_query`, and chat-completion span durations; compute percentiles by stage + task type; commit `docs/dashboards/latency_profile.md` with the table. Alert thresholds (Pooja's): retrieval P99 > 800ms, end-to-end P95 > 8s. *Maps to §3.4 row 2.*
- **PR #10 — Retry logging + circuit breakers.** Wrap NVIDIA, Milvus, Postgres, Redis calls in `pybreaker`-style circuit breakers. Log every retry with reason and attempt-count. Surface `retry_rate` and circuit state per dependency. *Maps to §3.4 rows 4–5.*
- **PR #11 — Fallback model + chaos test.** Configure a secondary LLM (different NVIDIA endpoint or HF Inference) for graceful degradation when the primary fails the circuit breaker. Chaos test forces the primary down and verifies fallback path; `fallback_rate` becomes a tracked metric. *Maps to §3.4 row 6.*

Streaming + TTFT (§3.4 row 3) is its own design effort and deliberately not in this roadmap.

### Phase 4 — Capstone

- **PR #12 — Four dashboards documented.** `docs/dashboards/{request_health,cost_trend,quality_trend,latency_profile}.md`. Screenshots from Langfuse and any external dashboards. Linked from the README.
- **PR #13 — README rewrite.** Add a *Design decisions* section explicitly referencing this doc and Pooja's framework. This is the section interviewers read.

## 5. Open decisions

Things this doc deliberately defers — they need explicit answers before the relevant PR starts.

1. **Judge model.** Use a different model family from the generator. Concretely: generator is NVIDIA NIM (likely Llama-3 derivative); judge should be GPT-4o, Claude, or Gemini. **Decision needed: which judge model and how to fund the API key.**
2. **Eval cadence.** Default proposed: manual trigger (`workflow_dispatch`) + nightly cron on `main`. PR-label trigger deferred.
3. **Cost dollar-conversion.** Need a `prices.json` mapping model name → $/1K input tokens, $/1K output tokens. Source: NVIDIA pricing page; refresh manually.
4. **Regression CI gate scope.** Default proposed: blocks PRs touching `src/utils/services/`, `src/utils/rag_pipeline.py`, `src/prompts/`. Out-of-scope changes (CI config, docs) don't trigger.
5. **Fallback model identity.** Need to pick a secondary LLM. Cheapest path: a smaller NVIDIA endpoint (e.g. a Llama-3-8B variant). Alternative: HF Inference Endpoint for genuine multi-provider resilience.
6. **Threshold values for alerts.** Pooja gives reasonable defaults (cache hit < 25%, cost WoW +20%, retrieval P99 > 800ms, end-to-end P95 > 8s, refusal rate > 5%). These will be adopted as starting points and tuned after a baseline run.

## 6. What "done" looks like

The framework is fully adopted when an interviewer (or a new teammate) can open this repo and answer the following questions in under five minutes:

- *How much does an average chat cost? What's the variance across task types?* → `docs/dashboards/cost_trend.md`
- *Is the system getting more or less accurate over the last week?* → `docs/dashboards/quality_trend.md`
- *Where is latency being spent? Which dependency is the slowest tail?* → `docs/dashboards/latency_profile.md`
- *What's the most recent regression-set pass rate? Did the last deploy regress?* → eval workflow output on `main`.
- *What happens if NVIDIA is down? Has that path been tested?* → fallback-model PR description and chaos-test artifact.

If any of those questions can't be answered, the framework isn't done.

## 7. Anti-patterns we explicitly avoid

Drawn from Pooja's articles and stated here so future contributions don't drift back into them:

- **Aggregate-only metrics.** Anything reported at "system level" must also be reported by task type. Aggregate is a trap because outliers — where the failures are — get averaged away.
- **Free-text instead of structured logs.** New log statements emit key/value pairs, not "Started processing for user vamsi at 12:34".
- **Single overall quality score.** The judge always produces decomposed scores. Anyone proposing a "0–100 quality number" is bypassing the design.
- **Silent retries.** Every retry is logged, even if it succeeds. A retried request that took 3s and a single-shot request that took 3s are not the same operationally.
- **Untested fallback paths.** Any fallback we add comes with a chaos test that actually exercises it.
- **Goodhart on the judge.** Optimizing prompts directly against judge scores without periodic human spot-check produces systems that please the judge, not users. Spot-check cadence is documented even if not tooled.
- **Eval-set drift.** The regression dataset is rebuilt from real production failures, not curated forever from the initial set.

## 8. References

This framework is sourced from Pooja Palod's *Data Journey* series. The articles are listed in roughly the order they're best read:

- [You Can't Debug What You Can't See](https://datajourney24.substack.com/p/you-cant-debug-what-you-cant-see) — the thesis: GenAI systems fail invisibly without deliberate observability.
- [Token Economics: Why LLM Cost Is an Engineering Problem](https://datajourney24.substack.com/p/token-economics-why-llm-cost-is-an) — the Cost pillar.
- [Evaluation in Production GenAI: Why Tested Systems Still Fail](https://datajourney24.substack.com/p/evaluation-in-production-genai-why) — the Quality pillar.
- [Latency and Reliability in Production](https://datajourney24.substack.com/p/latency-and-reliability-in-production) — the Latency pillar.
- [Building Observability for a Production GenAI System](https://datajourney24.substack.com/p/building-observability-for-a-production) — the unifying framework and deployment checklist.

Quoted phrases throughout this doc are attributed to those articles.
