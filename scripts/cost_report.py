"""Cost-per-successful-task aggregator.

Produces a markdown cost report from Langfuse spans. Supports two modes:

    python scripts/cost_report.py --source file --input tests/fixtures/observations.json
    python scripts/cost_report.py --source live --days 7

The pure-function aggregation pipeline (classify_stage, task_type_from_tags,
compute_cost, aggregate, summarize_by_task, summarize_by_stage, render_markdown)
is tested in tests/test_cost_report.py without any Langfuse dependency.

See docs/OBSERVABILITY.md §3.2 for the design rationale.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from scripts.pricing import (
    DEFAULT_RERANK_PRICE_PER_CALL_USD,
    MODEL_PRICES,
    RERANK_PRICE_PER_CALL_USD,
)

# ----------------------------------------------------------------------------
# Pure functions — tested directly, no I/O.
# ----------------------------------------------------------------------------

KNOWN_TASK_TAGS = ("cache-path", "rag-path")


@dataclass(frozen=True)
class CostRow:
    trace_id: str
    task_type: str
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def task_type_from_tags(tags: list[str]) -> str:
    """Map trace tags to a coarse task type.

    The chat service emits one of `cache-path` or `rag-path` on every trace
    (see src/utils/chat/chat_service.py). Anything else is `unknown`.
    """
    for tag in tags:
        if tag == "cache-path":
            return "cache-hit"
        if tag == "rag-path":
            return "rag-full"
    return "unknown"


def classify_stage(span_name: str) -> str:
    """Map a Langfuse span name to a pipeline stage label."""
    if span_name in ("embed_query", "embed_passage_batch"):
        return "embedding"
    if span_name == "rerank":
        return "rerank"
    if span_name == "cache_lookup":
        return "cache_lookup"
    # LangChain-emitted LLM generation spans show up under the model class
    # name (e.g. "ChatNVIDIA") or "llm" depending on the integration.
    lowered = span_name.lower()
    if "chatnvidia" in lowered or "llm" in lowered or "chat" in lowered:
        return "llm"
    return "other"


def compute_cost(
    stage: str,
    model: str,
    usage: dict[str, int],
    prices: dict[str, dict[str, float]] | None = None,
    rerank_prices: dict[str, float] | None = None,
    default_rerank_price: float = DEFAULT_RERANK_PRICE_PER_CALL_USD,
) -> float:
    """Compute USD cost for a single observation.

    Rerank is priced per call (NVIDIA's hosted endpoint), so token counts
    are ignored for that stage. Everything else uses per-1K-token pricing.
    Returns 0.0 when the model isn't in the price table — the row is still
    counted so unpriced spans surface as zero rather than crashing the run.
    """
    prices = prices if prices is not None else MODEL_PRICES
    rerank_prices = rerank_prices if rerank_prices is not None else RERANK_PRICE_PER_CALL_USD

    if stage == "rerank":
        return rerank_prices.get(model, default_rerank_price)

    p = prices.get(model)
    if p is None:
        return 0.0
    return (
        usage.get("input", 0) / 1000.0 * p["input_per_1k_usd"]
        + usage.get("output", 0) / 1000.0 * p["output_per_1k_usd"]
    )


def aggregate(
    traces: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    prices: dict[str, dict[str, float]] | None = None,
    rerank_prices: dict[str, float] | None = None,
) -> list[CostRow]:
    """Join observations to traces, classify, and price each row."""
    trace_task = {t["id"]: task_type_from_tags(t.get("tags", [])) for t in traces}
    rows: list[CostRow] = []
    for obs in observations:
        stage = classify_stage(obs.get("name", ""))
        usage = obs.get("usage_details") or {}
        model = obs.get("model") or "unknown"
        cost = compute_cost(stage, model, usage, prices, rerank_prices)
        rows.append(
            CostRow(
                trace_id=obs["trace_id"],
                task_type=trace_task.get(obs["trace_id"], "unknown"),
                stage=stage,
                model=model,
                input_tokens=int(usage.get("input", 0) or 0),
                output_tokens=int(usage.get("output", 0) or 0),
                cost_usd=cost,
            )
        )
    return rows


def summarize_by_task(rows: list[CostRow]) -> list[dict[str, Any]]:
    """Per-task-type aggregates: count of distinct traces, total spend, $/task."""
    by_task: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"trace_ids": set(), "total_cost_usd": 0.0}
    )
    for r in rows:
        by_task[r.task_type]["trace_ids"].add(r.trace_id)
        by_task[r.task_type]["total_cost_usd"] += r.cost_usd

    out = []
    for task, data in by_task.items():
        count = len(data["trace_ids"])
        total = data["total_cost_usd"]
        out.append(
            {
                "task_type": task,
                "count": count,
                "total_cost_usd": round(total, 6),
                "avg_cost_per_task_usd": round(total / count, 6) if count else 0.0,
            }
        )
    out.sort(key=lambda d: d["total_cost_usd"], reverse=True)
    return out


def summarize_by_stage(rows: list[CostRow]) -> list[dict[str, Any]]:
    """Per-pipeline-stage aggregates: total spend, % of grand total, tokens."""
    by_stage: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total_cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
    )
    for r in rows:
        by_stage[r.stage]["total_cost_usd"] += r.cost_usd
        by_stage[r.stage]["input_tokens"] += r.input_tokens
        by_stage[r.stage]["output_tokens"] += r.output_tokens

    grand_total = sum(d["total_cost_usd"] for d in by_stage.values()) or 1.0
    out = []
    for stage, data in by_stage.items():
        out.append(
            {
                "stage": stage,
                "total_cost_usd": round(data["total_cost_usd"], 6),
                "share_of_spend_pct": round(data["total_cost_usd"] / grand_total * 100, 2),
                "input_tokens": data["input_tokens"],
                "output_tokens": data["output_tokens"],
            }
        )
    out.sort(key=lambda d: d["total_cost_usd"], reverse=True)
    return out


def render_markdown(
    rows: list[CostRow],
    *,
    source_label: str,
    window_label: str,
    generated_at: str,
) -> str:
    """Render the aggregated rows as a markdown report."""
    grand_total = sum(r.cost_usd for r in rows)
    trace_count = len({r.trace_id for r in rows})
    avg_per_task = grand_total / trace_count if trace_count else 0.0

    by_task = summarize_by_task(rows)
    by_stage = summarize_by_stage(rows)

    lines: list[str] = []
    lines.append(f"# Cost report — {generated_at}")
    lines.append("")
    lines.append(f"- **Source:** {source_label}")
    lines.append(f"- **Window:** {window_label}")
    lines.append("- **Generator:** [scripts/cost_report.py](../../scripts/cost_report.py)")
    lines.append("")

    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total traces | {trace_count} |")
    lines.append(f"| Total spend (USD) | ${grand_total:.6f} |")
    lines.append(f"| Cost per successful task (avg) | ${avg_per_task:.6f} |")
    lines.append("")

    lines.append("## By task type")
    lines.append("")
    lines.append("| Task | Trace count | Total $ | $/task |")
    lines.append("|---|---:|---:|---:|")
    for row in by_task:
        lines.append(
            f"| `{row['task_type']}` | {row['count']} | "
            f"${row['total_cost_usd']:.6f} | ${row['avg_cost_per_task_usd']:.6f} |"
        )
    lines.append("")

    lines.append("## By pipeline stage")
    lines.append("")
    lines.append("| Stage | Total $ | Share | Input tokens | Output tokens |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in by_stage:
        lines.append(
            f"| `{row['stage']}` | ${row['total_cost_usd']:.6f} | "
            f"{row['share_of_spend_pct']:.1f}% | {row['input_tokens']:,} | "
            f"{row['output_tokens']:,} |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Prices are sourced from [scripts/pricing.py](../../scripts/pricing.py). "
                 "Replace with real contract rates before relying on absolute USD figures.")
    lines.append("- Unpriced models (not in the table) currently contribute $0; "
                 "they still appear in token counts.")
    lines.append("- Rerank is priced per-call (NVIDIA hosted endpoint), not per token.")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# I/O — fixture loader and Langfuse live fetcher.
# ----------------------------------------------------------------------------


def load_fixture(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load (traces, observations) from a JSON fixture."""
    with open(path) as fh:
        data = json.load(fh)
    return data.get("traces", []), data.get("observations", [])


def fetch_from_langfuse(days: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch (traces, observations) from a live Langfuse instance.

    Uses only v1 endpoints (`trace.list`, `trace.get`) so it works on both
    self-hosted Langfuse and Langfuse Cloud. The v2 `observations.get_many`
    endpoint is Cloud-only at the time of writing (returns 404 with
    "v2 APIs are currently in beta and only available on Langfuse Cloud"
    on self-hosted), so we fetch each trace individually — `trace.get` is
    v1 and returns the trace with its observations nested inline.

    If the SDK shape drifts in a future minor release, update the two
    `client.api.trace.*` calls below — the rest of the script is insulated
    from the SDK surface.
    """
    # Imported lazily so `--source file` works in environments without
    # Langfuse installed (e.g. CI runs against the fixture only).
    from langfuse import Langfuse  # type: ignore

    # Read creds from env directly rather than importing src.utils.config —
    # config.py requires NVIDIA_API_KEY at import time, which CI environments
    # running only this script don't (and shouldn't) have. Locally, the .env
    # file is picked up by load_dotenv below.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        raise RuntimeError(
            "Missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY. "
            "Set them in your environment or .env."
        )
    host = os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )

    from_ts = datetime.now(UTC) - timedelta(days=days)

    traces_resp = client.api.trace.list(from_timestamp=from_ts, limit=100)
    traces: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

    for t in traces_resp.data or []:
        traces.append(
            {
                "id": t.id,
                "tags": list(getattr(t, "tags", []) or []),
                "timestamp": (
                    t.timestamp.isoformat() if getattr(t, "timestamp", None) else None
                ),
            }
        )
        try:
            full = client.api.trace.get(t.id)
        except Exception as e:
            print(
                f"warning: failed to fetch observations for trace {t.id}: {e}",
                file=sys.stderr,
            )
            continue
        for o in getattr(full, "observations", []) or []:
            observations.append(
                {
                    "id": o.id,
                    "trace_id": t.id,
                    "name": getattr(o, "name", None) or "",
                    "type": getattr(o, "type", ""),
                    "model": getattr(o, "model", None),
                    "usage_details": dict(getattr(o, "usage_details", {}) or {}),
                }
            )

    return traces, observations


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("live", "file"),
        required=True,
        help="`live` pulls from Langfuse; `file` reads a JSON fixture.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Fixture path (required when --source file).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="How many days back to query Langfuse (live mode only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Markdown file to write.",
    )
    args = parser.parse_args(argv)

    if args.source == "file":
        if not args.input:
            parser.error("--input is required when --source file")
        traces, observations = load_fixture(args.input)
        source_label = f"fixture ({args.input})"
        window_label = "fixture-defined"
    else:
        traces, observations = fetch_from_langfuse(args.days)
        source_label = "live Langfuse"
        window_label = f"last {args.days} day(s)"

    rows = aggregate(traces, observations)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    report = render_markdown(
        rows,
        source_label=source_label,
        window_label=window_label,
        generated_at=generated_at,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"wrote {args.output} ({len(rows)} cost rows over {len(traces)} traces)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
