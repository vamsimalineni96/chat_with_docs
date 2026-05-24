# Eval harness

The Quality pillar of the observability framework (see [OBSERVABILITY.md §3.3](../../docs/OBSERVABILITY.md)). A hand-crafted Q&A dataset plus a runner that exercises `/chat`, scores retrieval, judges answer quality with an independent LLM, and emits a markdown report.

## Directory layout

| File | Status |
|---|---|
| [`qa_set.jsonl`](qa_set.jsonl) | ✅ 18 Q&A pairs over HP4 + HP7. Edit / extend freely — the file is the eval contract. |
| [`metrics.py`](metrics.py) | ✅ Pure functions: `keyword_recall_at_k`, `reciprocal_rank`, `mrr`, `latency_percentiles`. No LLM, no I/O. |
| [`judge.py`](judge.py) | ✅ LLM-as-judge (Llama 3.3 70B) with decomposed rubric (groundedness / accuracy / completeness). Prompts loaded from [`prompts/judge.yaml`](prompts/judge.yaml). |
| [`reporter.py`](reporter.py) | ✅ Markdown renderer — aggregate + by-book + by-category + per-question + failures sections. |
| [`run_eval.py`](run_eval.py) | ✅ End-to-end orchestrator. Two modes: `--validate-only` (dataset sanity check) and full-run (hits `/chat`, runs judge, writes report). |
| [`prompts/judge.yaml`](prompts/judge.yaml) | ✅ Versioned judge prompts. Tune the rubric here — no code change required. |

The CI workflow that runs this on a schedule lands in **sub-PR #7d**.

## Q&A dataset shape

Each line in [`qa_set.jsonl`](qa_set.jsonl) is one JSON object:

```json
{
  "id": "hp4-001",
  "question": "Who is Cedric Diggory and what happens to him by the end of the Goblet of Fire?",
  "expected_keywords_in_answer": ["Hufflepuff", "champion", "killed", "Voldemort"],
  "expected_keywords_in_top_chunks": ["Cedric", "Hufflepuff"],
  "book": "hp4",
  "category": "character"
}
```

Two keyword sets, used differently:

- `expected_keywords_in_top_chunks` — drives **retrieval** metrics. Did the retriever surface chunks that contained these phrases? Should be short, specific, indexable (proper nouns, distinctive terms).
- `expected_keywords_in_answer` — drives **answer-quality** judgment. A complete answer should touch on these. Used by the judge in sub-PR #7b for the `completeness` score.

`book` is for stratification in reports (hp4 vs hp7 quality, for instance). `category` is for diagnostic slicing (does the system struggle more with `reasoning` than `factual`?). Current categories: `character`, `plot`, `magic`, `factual`, `reasoning`.

## Running

### Validate the dataset only (no LLM, no network)

```bash
python -m evals.quality.run_eval --validate-only
```

### Full end-to-end run

Prerequisites:
- The FastAPI app must be running (default `http://localhost:8000`). Bring it up with `uvicorn app:app --reload`.
- `NVIDIA_API_KEY` set in env so the judge LLM can be called.

```bash
python -m evals.quality.run_eval \
  --output docs/eval-reports/eval_$(date -u +%Y-%m-%d).md
```

Useful flags during iteration:

| Flag | Purpose |
|---|---|
| `--max-questions 3` | Smoke-test with the first N questions instead of all 18 |
| `--api-base http://...` | Point at a non-default app instance |
| `--judge-model meta/llama-3.1-8b-instruct` | Use a cheaper / different judge to compare scoring stability |
| `--latest docs/eval-reports/latest.md` | Stable path written alongside the dated file |

Progress prints to stdout line-by-line as each question completes — useful since a 18-question run takes ~2 minutes.

## Why the dataset matters more than the code

A clever metric pipeline scoring a bad dataset is worse than a simple metric pipeline scoring a thoughtful one. The 18 Q&A pairs here are hand-written to:

- Cover both books roughly evenly so retrieval can't cheat by always preferring one source.
- Mix difficulty levels: `factual` recall, plot synthesis, multi-hop reasoning, magic-system understanding.
- Use **specific proper nouns** as chunk keywords so retrieval recall is a real signal, not a tautology.
- Use **conceptual / fact-based keywords** for answer judgment so the judge isn't just checking exact phrase matches.

Edit `qa_set.jsonl` whenever you notice the dataset has blind spots. The eval will only ever be as good as the questions in it.

## What's deferred

- **Judge calibration against human labels.** Pooja's framework recommends calibrating the LLM judge against 50–200 human-rated examples before trusting it. For a solo portfolio project this is impractical; we'll document the limitation in the eval README and the OBSERVABILITY doc, and plan a periodic spot-check of ~20 judge outputs per quarter.
- **Eval-as-CI-gate.** Once #7d lands, eval will run on a schedule but won't *block* PRs. Wiring it into the regression-dataset CI gate is a separate workstream (Phase 2, row 4 of the roadmap).
