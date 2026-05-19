"""Markdown report renderer for the eval harness — landing in sub-PR #7b.

Will provide a single `render_markdown(eval_rows, ...)` function that
takes a list of per-question result dicts (question, retrieved_chunks,
answer, metrics, judge_scores, latency_ms) and emits a markdown report
matching the shape of docs/reports/cost_sample.md:

    # Eval run — <date>
    ## Aggregate
    | Recall@5 | MRR | Faithfulness | Relevance | Completeness | p50 | p95 |
    ## Per question
    | id | recall@5 | faith | rel | comp | latency |

Output is written to docs/eval-reports/eval_<YYYY-MM-DD>.md plus
docs/eval-reports/latest.md, mirroring the cost-report pattern.
"""
