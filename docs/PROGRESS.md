# Progress log

A running record of what's been built and *why* — written for future-me to pick this work back up cold, and for interviewers who want the narrative behind the diffs.

The roadmap of work lives in [OBSERVABILITY.md §4](OBSERVABILITY.md). This file is the *history*.

## Snapshot

| Pillar | Status |
|---|---|
| **Cost** | ✅ Complete — tokenizer, aggregator, nightly automation all merged. First real report on main: [`docs/reports/cost_2026-05-19.md`](reports/cost_2026-05-19.md). |
| **Quality** | ✅ Complete — harness + dataset CI gate + clean post-ingest-fix baseline. Latest: [`docs/eval-reports/eval_2026-05-21_hp4_clean.md`](eval-reports/eval_2026-05-21_hp4_clean.md). The first baseline ([`eval_2026-05-20.md`](eval-reports/eval_2026-05-20.md)) surfaced an HP7 ingestion gap that was fixed in the ingest-retry/dedup PR. |
| **Latency & reliability** | 🟡 Mostly complete — nightly latency report ([`evals/latency/latency_report.py`](../evals/latency/latency_report.py)), retries wrapped around every NVIDIA call on the read path ([`src/utils/services/retry.py`](../src/utils/services/retry.py)) and write path. Fallback model + circuit breakers deferred (tracked in [OBSERVABILITY.md §3.4](OBSERVABILITY.md)). |
| **Capstone (dashboards, README rewrite)** | ✅ Complete — [`docs/dashboards/index.md`](dashboards/index.md) is the operational snapshot; README reframed around the three-pillar framework. |

10 of 11 roadmap items closed. Remaining: PR #11 (fallback model + chaos test).

---

## PR history

### PR #1 — CI baseline ([`chore/ci-baseline`](https://github.com/vamsimalineni96/chat_with_docs/pulls?q=chore%2Fci-baseline))

**What:** First green CI on the repo. Added:
- `pyproject.toml` with Ruff config (line length 100; rules `E/F/I/W/UP/B`; ignores `E501/B008/B904`).
- `.github/workflows/ci.yml` with three jobs: `lint-and-compile`, `dep-audit` (`pip-audit --strict`), `secret-scan` (`gitleaks`).
- `.github/dependabot.yml` — weekly pip + github-actions updates.
- Ran `ruff check --fix .` — 177 autofixes (type-hint modernization, import sorting).

**Why this shape:** Lint + compile catches ~80% of "oops" PRs cheaply. `pip-audit --strict` was non-negotiable for a portfolio repo because it forces deliberate triage of every CVE rather than silent inheritance. The `--ignore-vuln` flags it produced for the LangChain ecosystem are documented in [ci.yml](../.github/workflows/ci.yml) with the rationale that the only fixes ship in major versions (1.x), and that's a separate migration tracked under [issue #1] / [issue #2].

**Decisions worth remembering:**
- B904 (`raise X from err`) was added to Ruff's `ignore` list because it's not autofixable and 23 instances would have bloated the PR. Filed mentally as tech debt.
- pymupdf bumped 1.26.6 → 1.26.7 to clear `GHSA-cxqh-p2w9-fmr7`.
- starlette CVE deliberately ignored: the fix in 0.49.1 conflicts with fastapi 0.116.1's pin. Documented in `ci.yml` as needing a coordinated fastapi+starlette migration.

### Side quest — branch protection

After PR #1 merged: enabled "Require pull request" + "Require status checks (3)" on `main` via Settings → Rules → Rulesets. CI is now an *enforced* gate, not decoration.

### Side quest — CVE dry-run learning exercise

Deliberately downgraded pymupdf back to 1.26.6 on a throwaway branch (`learn/cve-dry-run`), watched CI fail with the known GHSA ID, opened the advisory, added `--ignore-vuln` with a doc comment, watched CI go green, then closed the PR without merging. Built the muscle for triaging real CVEs.

### PR #2 — Observability design doc ([`docs/observability-design`](https://github.com/vamsimalineni96/chat_with_docs/pulls?q=docs%2Fobservability-design))

**What:** `docs/OBSERVABILITY.md` — adopts [Pooja Palod's three-pillar observability framework](https://datajourney24.substack.com/) (Cost, Quality, Latency), audits the repo against it row-by-row, lays out 11 follow-up PRs.

**Why before any code:** Wanted a north-star doc that subsequent PRs could *cite*, so each gap closed has a named source (§3.X row N). This doc is also a portfolio artifact in its own right — it shows systems-level thinking, not just code.

**Honest finding during the audit:** The repo was *already* well-instrumented (Langfuse `@observe` decorators in chat_service, rag_pipeline, embedder, chunk_ranking, inference). The gaps were in **aggregation, eval, and reliability patterns** — turning captured signals into operational tools. That reshaped the roadmap.

### PR #3 — Real embedding tokenizer ([`cost/embedding-tokenizer`](https://github.com/vamsimalineni96/chat_with_docs/pulls?q=cost%2Fembedding-tokenizer))

**What:** Replaced `_estimate_tokens(text) = max(1, len(text) // 4)` with `count_tokens()` backed by `tiktoken`'s `cl100k_base` encoding. Added the first real unit tests (4 cases pinning behavior contracts, not exact numbers). New `unit-tests` CI job, scoped to minimal deps (no langchain/milvus install — they're not needed by the tokenizer).

**Why tiktoken over `transformers.AutoTokenizer`:** Deliberate proxy. tiktoken is small (~8 MB), fast, no network, no per-model registry. The "true" tokenizer for NVIDIA's embedding model would be more accurate but with a much bigger dep footprint. tiktoken's ~10% accuracy is plenty for week-over-week cost trending; absolute precision matters only if we ever enforce a hard token budget. Documented as a deliberate decision in `OBSERVABILITY.md` §3.2.

### PR #4 — Cost-per-task aggregator ([`cost/cost-per-task-aggregation`](https://github.com/vamsimalineni96/chat_with_docs/pulls?q=cost%2Fcost-per-task-aggregation))

**What:** A pure-Python aggregator (`evals/cost/cost_report.py`) that pulls Langfuse spans, joins them against a model→price table (`evals/cost/pricing.py`), and emits a markdown report partitioning spend by *task type* (`cache-hit` vs `rag-full`) and *pipeline stage* (`embedding`, `rerank`, `llm`). Two modes: `--source live` for real Langfuse, `--source file` for the committed fixture.

**Why both modes:** Self-hosted Langfuse ran on `localhost:3000`, unreachable from GitHub Actions. Fixture mode let CI verify the aggregation logic on every PR without network access; live mode produced real reports locally.

**11 unit tests** target the pure-function pipeline (`classify_stage`, `compute_cost`, `aggregate`, `summarize_by_task`, `summarize_by_stage`). Sample report committed: [`docs/reports/cost_sample.md`](reports/cost_sample.md). The first real fixture run revealed *LLM dominates spend (88.9%), rerank 10.8%, embedding 0.3%* and *cache hits are 463× cheaper than full RAG per task* — exactly the kinds of insight Pooja's framework names.

**Bundled with PR #4:**
- Security cleanup: `.env` was tracked in git from the first commit. `git rm --cached .env`, added to `.gitignore`, rotated all three exposed credentials (NVIDIA API key, Langfuse pair). Created `.env.example` as the canonical template.
- `tests/conftest.py` to stub `NVIDIA_API_KEY` at test-collection time. CI's pytest job imports `src.utils.observability` (which imports `config.py`, which reads `NVIDIA_API_KEY` at module load) and was failing with `KeyError`. The conftest's `os.environ.setdefault(...)` runs before any test module imports, so the chain succeeds.

### PR #5 — Nightly cost-report cron ([`cost/nightly-cron`](https://github.com/vamsimalineni96/chat_with_docs/pulls?q=cost%2Fnightly-cron))

**What:** Migrated Langfuse from self-hosted to **Langfuse Cloud** (free tier), then added `.github/workflows/cost-report-nightly.yml` — cron `0 6 * * *` + `workflow_dispatch`. Reads three GitHub secrets (`LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY`), runs the cost-report script with `--days 7`, commits `docs/reports/cost_<YYYY-MM-DD>.md` back to main with `[skip ci]`.

**Refactor in the same PR:** `fetch_from_langfuse` now reads creds from `os.environ` + `dotenv.load_dotenv` directly, instead of going through `src.utils.config`. That import required `NVIDIA_API_KEY` to be present at module load — fine for the running app but a hassle for any CI job that only needs Langfuse access.

**Why use only Langfuse v1 API endpoints:** The script uses `client.api.trace.list()` and `client.api.trace.get(trace_id)`. The v2 `observations.get_many()` endpoint is Cloud-only (returns 404 with `"v2 APIs are currently in beta and only available on Langfuse Cloud"` on self-hosted). Sticking to v1 means the script works on both deployments, at the cost of one extra HTTP round-trip per trace.

### PR #6 — PAT auth for nightly bot ([`fix/nightly-pat-auth`](https://github.com/vamsimalineni96/chat_with_docs/pulls?q=fix%2Fnightly-pat-auth))

**What:** Single 8-line change to [cost-report-nightly.yml](../.github/workflows/cost-report-nightly.yml): pass `token: ${{ secrets.BOT_PAT }}` to `actions/checkout` so subsequent git operations authenticate as the repo admin (in the bypass list) rather than as `github-actions[bot]` (not in the bypass list).

**Root cause:** Branch protection on main requires PR + 3 status checks. The bot's first auto-commit push was rejected:
```
remote: error: GH013: Repository rule violations found for refs/heads/main.
- Changes must be made through a pull request.
- 3 of 3 required status checks are expected.
```

**Why a PAT and not a blanket GitHub Actions bypass:** The fine-grained PAT is scoped to `Contents: write` on this single repo. Putting `GitHub Actions` itself in the bypass list would let *every* future workflow push to main. Strictly narrower privilege via PAT.

**Setup steps (one-time):**
1. Generated fine-grained PAT, `Contents: write` on `chat_with_docs` only.
2. Stored as repo secret `BOT_PAT`.
3. Added "Repository admin" to ruleset bypass list with mode "Always".

---

## Cross-cutting lessons / incidents

These came up *during* the PRs above and are worth remembering on their own:

1. **`.env` should never be tracked in git.** The repo had `.env` committed from day one — every secret ever in it was in public history. Fixed via `git rm --cached`, added to `.gitignore`, and **rotated all three exposed credentials** because removal-from-tracking doesn't remove from history. Created `.env.example` as the committed template.

2. **`git rm --cached <file>` papercut.** After `--cached` removes a file from the index but leaves it on disk, switching branches can still wipe the local file from the working tree (if the destination branch had it tracked, then the merge that removed it lands). Lost the freshly-edited `.env` once this way; restored from `.env.example` + cloud.langfuse.com.

3. **Don't paste live `.env` contents in any chat tool.** Even in a private conversation, the values can leak via screenshots, sync, audit logs, or future sharing. Treat the chat as untrusted output.

4. **Module-level env-var reads break test imports.** [src/utils/config.py:20](../src/utils/config.py#L20) does `NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]` — correct for app startup but breaks pytest's collection in CI. Fix in `tests/conftest.py` with `os.environ.setdefault(...)`. The cleaner long-term fix is making config lazy / use-site-validated, but that's a wider refactor.

5. **Langfuse v1 vs v2 surface.** v2 endpoints are Cloud-only. Cost report deliberately uses only v1 to stay deployment-portable.

6. **Branch protection blocks bots unless explicitly excluded.** Use a fine-grained PAT scoped to the minimum permission (`Contents: write`), store as a repo secret, pass to `actions/checkout` via `token:`. Add Repository admin (or the bot's identity) to the ruleset bypass list.

---

## What's next

**PR #7 (in the roadmap, the showpiece): eval harness.**

Per [OBSERVABILITY.md §4](OBSERVABILITY.md) — Phase 2, Quality pillar. Concrete shape: `evals/quality/qa_set.jsonl` (15–25 Q&A pairs over HP4/HP7), `evals/quality/run_eval.py` (calls `/chat`, captures retrieved chunks + answer, computes recall@k + MRR), `evals/quality/judge.py` (decomposed LLM-as-judge: groundedness / accuracy / completeness — using a *different* model family than the generator), `evals/quality/reporter.py` (markdown report), `.github/workflows/eval.yml` (`workflow_dispatch` + nightly cron).

This is the single biggest portfolio differentiator — 90% of RAG portfolios skip evals entirely. Budget: ~6 hours of focused work, split across 4–5 sub-PRs (scaffold → metrics → judge → orchestration → workflow).

**Open decision before starting:** which judge model? Pooja's framework strongly recommends a *different family* from the generator to avoid self-preference bias. Options: GPT-4o or Claude via paid API (~$0.05/eval run), or a different NVIDIA model (cheaper but same family).
