# Eval harness

The Quality pillar of the observability framework (see [OBSERVABILITY.md §3.3](../docs/OBSERVABILITY.md)). A hand-crafted Q&A dataset plus a runner that exercises `/chat`, scores retrieval, judges answer quality with an independent LLM, and emits a markdown report.

## Directory layout

| File | Status |
|---|---|
| [`qa_set.jsonl`](qa_set.jsonl) | ✅ 18 Q&A pairs over HP4 + HP7. Edit / extend freely — the file is the eval contract. |
| [`metrics.py`](metrics.py) | ✅ Pure functions: `keyword_recall_at_k`, `reciprocal_rank`, `mrr`, `latency_percentiles`. No LLM, no I/O. |
| [`run_eval.py`](run_eval.py) | 🟡 *Stub.* Currently only loads and validates `qa_set.jsonl`. Sub-PR #7c wires up the full pipeline. |
| [`judge.py`](judge.py) | ⬜ Empty stub. Sub-PR #7b: LLM-as-judge with decomposed rubric (groundedness / accuracy / completeness). |
| [`reporter.py`](reporter.py) | ⬜ Empty stub. Sub-PR #7b: markdown report renderer. |

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

## Running today

```bash
python -m eval.run_eval --validate-only
```

Expected output:

```
Total Q&A pairs: 18

By book:
  - hp4: 9
  - hp7: 9

By category:
  - plot: 7
  - character: 4
  - magic: 4
  - reasoning: 2
  - factual: 1

Avg expected_keywords_in_answer per entry:     4.1
Avg expected_keywords_in_top_chunks per entry: 1.8
```

Anything else (real eval runs, judge scoring, markdown reports) is being built incrementally in subsequent sub-PRs.

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
