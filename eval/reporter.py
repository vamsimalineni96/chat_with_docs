"""Markdown report renderer for the eval harness.

Takes a list of per-question result dicts produced by run_eval (#7c
will wire that up) and emits a markdown report following the same
shape as `docs/reports/cost_sample.md`: aggregate stats + per-book +
per-category + per-question + a Failures section that flags rows
where retrieval recall or any judge dimension are below threshold.

Pure function; no Langfuse, no LLM, no I/O of its own. The caller
writes the returned string to disk.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

# Below these thresholds, a row is flagged in the Failures section.
# Pooja's framework recommends watching for refusal-rate >5% and
# groundedness <3.5; these row-level thresholds derive from that.
FAILURE_RECALL_THRESHOLD = 0.5
FAILURE_JUDGE_THRESHOLD = 3  # Any sub-score < this triggers the flag.


def _aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of recall, MRR, and the three judge sub-scores."""
    n = len(rows)
    if n == 0:
        return {
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "groundedness": 0.0,
            "accuracy": 0.0,
            "completeness": 0.0,
        }
    return {
        "recall_at_5": sum(r["recall_at_5"] for r in rows) / n,
        "mrr": sum(r["reciprocal_rank"] for r in rows) / n,
        "groundedness": sum(r["judge"]["groundedness"] for r in rows) / n,
        "accuracy": sum(r["judge"]["accuracy"] for r in rows) / n,
        "completeness": sum(r["judge"]["completeness"] for r in rows) / n,
    }


def _latency_percentiles(latencies_ms: list[float]) -> dict[int, float]:
    """Lightweight p50/p95 — see eval.metrics.latency_percentiles for the
    full version; this duplicates a tiny piece to keep the reporter
    free of any internal eval imports for portability."""
    if not latencies_ms:
        return {50: 0.0, 95: 0.0}
    s = sorted(latencies_ms)
    n = len(s)
    return {
        50: float(s[max(0, n // 2 - 1)]) if n > 0 else 0.0,
        95: float(s[max(0, int(0.95 * n) - 1)]) if n > 0 else 0.0,
    }


def _is_failure(row: dict[str, Any]) -> bool:
    if row["recall_at_5"] < FAILURE_RECALL_THRESHOLD:
        return True
    j = row["judge"]
    return any(
        j[k] < FAILURE_JUDGE_THRESHOLD
        for k in ("groundedness", "accuracy", "completeness")
    )


def render_markdown(
    rows: list[dict[str, Any]],
    *,
    generated_at: str,
    generator_model: str,
    judge_model: str,
) -> str:
    lines: list[str] = []
    agg = _aggregate_scores(rows)
    latencies = [r.get("latency_ms", 0) for r in rows]
    p = _latency_percentiles(latencies)

    lines.append(f"# Eval run — {generated_at}")
    lines.append("")
    lines.append(f"- **Generator:** `{generator_model}`")
    lines.append(
        f"- **Judge:** `{judge_model}` "
        "(different model family — see [docs/PROGRESS.md](../PROGRESS.md))"
    )
    lines.append(f"- **Q&A count:** {len(rows)}")
    lines.append("")

    # Aggregate
    lines.append("## Aggregate")
    lines.append("")
    lines.append(
        "| Recall@5 | MRR | Groundedness | Accuracy | Completeness | p50 latency | p95 latency |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {agg['recall_at_5']:.2f} | {agg['mrr']:.2f} | "
        f"{agg['groundedness']:.2f}/5 | {agg['accuracy']:.2f}/5 | "
        f"{agg['completeness']:.2f}/5 | {p[50]:.0f}ms | {p[95]:.0f}ms |"
    )
    lines.append("")

    # By book
    by_book: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_book[r["book"]].append(r)

    lines.append("## By book")
    lines.append("")
    lines.append(
        "| Book | Count | Recall@5 | MRR | Ground | Acc | Comp |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for book in sorted(by_book.keys()):
        a = _aggregate_scores(by_book[book])
        lines.append(
            f"| `{book}` | {len(by_book[book])} | {a['recall_at_5']:.2f} | "
            f"{a['mrr']:.2f} | {a['groundedness']:.2f} | "
            f"{a['accuracy']:.2f} | {a['completeness']:.2f} |"
        )
    lines.append("")

    # By category
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    lines.append("## By category")
    lines.append("")
    lines.append("| Category | Count | Recall@5 | Ground | Acc | Comp |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cat in sorted(by_cat.keys()):
        a = _aggregate_scores(by_cat[cat])
        lines.append(
            f"| `{cat}` | {len(by_cat[cat])} | {a['recall_at_5']:.2f} | "
            f"{a['groundedness']:.2f} | {a['accuracy']:.2f} | "
            f"{a['completeness']:.2f} |"
        )
    lines.append("")

    # Per question
    lines.append("## Per question")
    lines.append("")
    lines.append(
        "| ID | Book | Category | Recall@5 | Ground | Acc | Comp | Latency |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        j = r["judge"]
        lines.append(
            f"| `{r['id']}` | {r['book']} | {r['category']} | "
            f"{r['recall_at_5']:.2f} | {j['groundedness']} | "
            f"{j['accuracy']} | {j['completeness']} | "
            f"{r.get('latency_ms', 0):.0f}ms |"
        )
    lines.append("")

    # Failures
    failures = [r for r in rows if _is_failure(r)]
    lines.append(f"## Failures ({len(failures)})")
    lines.append("")
    lines.append(
        f"Rows where retrieval recall@5 < {FAILURE_RECALL_THRESHOLD:.0%} "
        f"or any judge sub-score < {FAILURE_JUDGE_THRESHOLD}/5."
    )
    lines.append("")
    if not failures:
        lines.append("_None._")
    else:
        for r in failures:
            j = r["judge"]
            lines.append(f"### `{r['id']}` ({r['book']}, {r['category']})")
            lines.append("")
            lines.append(f"**Question:** {r['question']}")
            lines.append("")
            lines.append(f"**Answer:** {r['answer']}")
            lines.append("")
            lines.append(
                f"**Scores:** recall@5={r['recall_at_5']:.2f}, "
                f"ground={j['groundedness']}, acc={j['accuracy']}, "
                f"comp={j['completeness']}"
            )
            lines.append("")
            lines.append(
                f"**Judge reasoning:** {j.get('reasoning', '(none)')}"
            )
            lines.append("")

    return "\n".join(lines) + "\n"
