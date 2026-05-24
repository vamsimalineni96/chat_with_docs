"""Latency aggregator from Langfuse traces.

Produces a markdown latency report with p50/p95/p99 by task and by stage,
plus a "slowest traces" outlier list. Two modes:

    python scripts/latency_report.py --source file --input tests/fixtures/observations_latency.json --output /tmp/latency.md
    python scripts/latency_report.py --source live --days 7 --output docs/reports/latency_$(date -u +%Y-%m-%d).md

The pure-function aggregation pipeline (`percentiles`, `aggregate`,
`summarize_by_task`, `summarize_by_stage`, `slow_traces`,
`render_markdown`) is tested in tests/test_latency_report.py without any
Langfuse dependency.

Stage classification and task-type tag mapping are imported from
`scripts.cost_report` so a span called `rerank` always lands in the same
bucket across both reports.

See docs/OBSERVABILITY.md §3.4 for the design rationale.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from scripts.cost_report import classify_stage, task_type_from_tags

# ----------------------------------------------------------------------------
# Pure functions — tested directly, no I/O.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyRow:
    trace_id: str
    task_type: str
    stage: str
    duration_ms: float


def percentiles(
    values: list[float],
    ps: tuple[int, ...] = (50, 95, 99),
) -> dict[str, float]:
    """Nearest-rank percentiles. Returns {"p50": ..., "p95": ..., ...}.

    No interpolation — for the small samples a portfolio nightly run
    produces, rank-based percentiles are easier to defend and don't
    pretend to precision we don't have. Empty inputs return zeros.
    """
    if not values:
        return {f"p{p}": 0.0 for p in ps}
    sv = sorted(values)
    n = len(sv)
    out: dict[str, float] = {}
    for p in ps:
        k = max(0, min(n - 1, math.ceil(p / 100.0 * n) - 1))
        out[f"p{p}"] = sv[k]
    return out


def aggregate(
    traces: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> tuple[list[LatencyRow], dict[str, float]]:
    """Produce one LatencyRow per observation + a trace_id -> trace_total_ms map.

    Trace-total is taken from the trace itself (`latency_ms`) when present —
    that's the authoritative wall-clock the user actually waited. When the
    trace doesn't carry an explicit latency, we fall back to the sum of
    observation durations. The fallback can over-count if observations
    overlap; the wall-clock from Langfuse is preferred whenever it exists.
    """
    trace_task = {t["id"]: task_type_from_tags(t.get("tags", [])) for t in traces}
    trace_total: dict[str, float] = {
        t["id"]: float(t["latency_ms"]) for t in traces if "latency_ms" in t
    }

    rows: list[LatencyRow] = []
    sum_by_trace: dict[str, float] = defaultdict(float)
    for o in observations:
        duration = float(o.get("duration_ms", 0.0) or 0.0)
        stage = classify_stage(o.get("name", ""))
        tid = o["trace_id"]
        rows.append(
            LatencyRow(
                trace_id=tid,
                task_type=trace_task.get(tid, "unknown"),
                stage=stage,
                duration_ms=duration,
            )
        )
        sum_by_trace[tid] += duration

    for tid, summed in sum_by_trace.items():
        trace_total.setdefault(tid, summed)
    return rows, trace_total


def summarize_by_task(
    rows: list[LatencyRow],
    trace_total: dict[str, float],
) -> list[dict[str, Any]]:
    """Per task type: count of distinct traces + p50/p95/p99 of trace-total ms."""
    by_task: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.trace_id not in seen[r.task_type]:
            seen[r.task_type].add(r.trace_id)
            by_task[r.task_type].append(r.trace_id)

    out: list[dict[str, Any]] = []
    for task, trace_ids in by_task.items():
        totals = [trace_total.get(tid, 0.0) for tid in trace_ids]
        p = percentiles(totals)
        out.append(
            {
                "task_type": task,
                "count": len(trace_ids),
                "p50_ms": round(p["p50"], 1),
                "p95_ms": round(p["p95"], 1),
                "p99_ms": round(p["p99"], 1),
            }
        )
    out.sort(key=lambda d: -d["p95_ms"])
    return out


def summarize_by_stage(rows: list[LatencyRow]) -> list[dict[str, Any]]:
    """Per pipeline stage: p50/p95/p99 of per-observation duration + share of total."""
    by_stage: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_stage[r.stage].append(r.duration_ms)
    grand_total = sum(r.duration_ms for r in rows) or 1.0

    out: list[dict[str, Any]] = []
    for stage, durations in by_stage.items():
        p = percentiles(durations)
        total = sum(durations)
        out.append(
            {
                "stage": stage,
                "count": len(durations),
                "p50_ms": round(p["p50"], 1),
                "p95_ms": round(p["p95"], 1),
                "p99_ms": round(p["p99"], 1),
                "share_of_time_pct": round(total / grand_total * 100, 1),
            }
        )
    out.sort(key=lambda d: -d["share_of_time_pct"])
    return out


def slow_traces(
    rows: list[LatencyRow],
    trace_total: dict[str, float],
    n: int = 10,
) -> list[dict[str, Any]]:
    """Top N traces by total latency. Points your eye at outliers."""
    task_by_trace: dict[str, str] = {}
    for r in rows:
        task_by_trace.setdefault(r.trace_id, r.task_type)
    items = [
        (tid, total, task_by_trace.get(tid, "unknown"))
        for tid, total in trace_total.items()
    ]
    items.sort(key=lambda t: -t[1])
    return [
        {"trace_id": tid, "task_type": task, "total_ms": round(total, 1)}
        for tid, total, task in items[:n]
    ]


def render_markdown(
    rows: list[LatencyRow],
    trace_total: dict[str, float],
    *,
    source_label: str,
    window_label: str,
    generated_at: str,
) -> str:
    """Render the aggregated rows as a markdown report."""
    trace_count = len(trace_total)
    all_totals = list(trace_total.values())
    p_total = percentiles(all_totals)
    by_task = summarize_by_task(rows, trace_total)
    by_stage = summarize_by_stage(rows)
    slow = slow_traces(rows, trace_total, n=10)

    lines: list[str] = []
    lines.append(f"# Latency report — {generated_at}")
    lines.append("")
    lines.append(f"- **Source:** {source_label}")
    lines.append(f"- **Window:** {window_label}")
    lines.append(
        "- **Generator:** [scripts/latency_report.py](../../scripts/latency_report.py)"
    )
    lines.append("")

    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total traces | {trace_count} |")
    lines.append(f"| p50 total latency | {p_total['p50']:.0f} ms |")
    lines.append(f"| p95 total latency | {p_total['p95']:.0f} ms |")
    lines.append(f"| p99 total latency | {p_total['p99']:.0f} ms |")
    lines.append("")

    lines.append("## By task type")
    lines.append("")
    lines.append("| Task | Trace count | p50 | p95 | p99 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in by_task:
        lines.append(
            f"| `{row['task_type']}` | {row['count']} | "
            f"{row['p50_ms']:.0f} ms | {row['p95_ms']:.0f} ms | {row['p99_ms']:.0f} ms |"
        )
    lines.append("")

    lines.append("## By pipeline stage")
    lines.append("")
    lines.append("| Stage | Obs count | p50 | p95 | p99 | Share of time |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in by_stage:
        lines.append(
            f"| `{row['stage']}` | {row['count']} | "
            f"{row['p50_ms']:.0f} ms | {row['p95_ms']:.0f} ms | "
            f"{row['p99_ms']:.0f} ms | {row['share_of_time_pct']:.1f}% |"
        )
    lines.append("")

    lines.append("## Slowest traces (top 10)")
    lines.append("")
    lines.append("| Trace ID | Task | Total |")
    lines.append("|---|---|---:|")
    for row in slow:
        lines.append(
            f"| `{row['trace_id']}` | `{row['task_type']}` | {row['total_ms']:.0f} ms |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Percentiles use nearest-rank with no interpolation. With small N (<100), "
        "p99 is effectively max."
    )
    lines.append(
        "- Trace total is wall-clock from Langfuse when available, else the sum "
        "of observation durations (may over-count if spans overlap)."
    )
    lines.append(
        "- Stage classification is shared with the cost report — see "
        "`scripts.cost_report.classify_stage`."
    )
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# I/O — fixture loader and Langfuse live fetcher.
# ----------------------------------------------------------------------------


def load_fixture(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load (traces, observations) from a JSON fixture."""
    with open(path) as fh:
        data = json.load(fh)
    return data.get("traces", []), data.get("observations", [])


def fetch_from_langfuse(
    days: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch (traces, observations) from a live Langfuse instance.

    Uses only v1 endpoints (`trace.list`, `trace.get`) so it works on both
    self-hosted Langfuse and Langfuse Cloud — same constraint as the cost
    report. Trace-level `latency` (seconds) is converted to milliseconds;
    per-observation `duration_ms` is computed from `end_time - start_time`.
    """
    # Imported lazily so `--source file` works in environments without
    # Langfuse installed (e.g. CI running only against the fixture).
    from langfuse import Langfuse  # type: ignore

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

    client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    from_ts = datetime.now(UTC) - timedelta(days=days)

    traces_resp = client.api.trace.list(from_timestamp=from_ts, limit=100)
    traces: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

    for t in traces_resp.data or []:
        lat_s = getattr(t, "latency", None)
        trace_entry: dict[str, Any] = {
            "id": t.id,
            "tags": list(getattr(t, "tags", []) or []),
            "timestamp": (
                t.timestamp.isoformat() if getattr(t, "timestamp", None) else None
            ),
        }
        if lat_s is not None:
            trace_entry["latency_ms"] = float(lat_s) * 1000.0
        traces.append(trace_entry)

        try:
            full = client.api.trace.get(t.id)
        except Exception as e:
            print(
                f"warning: failed to fetch observations for trace {t.id}: {e}",
                file=sys.stderr,
            )
            continue

        for o in getattr(full, "observations", []) or []:
            start = getattr(o, "start_time", None)
            end = getattr(o, "end_time", None)
            duration_ms = 0.0
            if start is not None and end is not None:
                duration_ms = max(0.0, (end - start).total_seconds() * 1000.0)
            observations.append(
                {
                    "id": o.id,
                    "trace_id": t.id,
                    "name": getattr(o, "name", None) or "",
                    "type": getattr(o, "type", ""),
                    "duration_ms": duration_ms,
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

    rows, trace_total = aggregate(traces, observations)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    report = render_markdown(
        rows,
        trace_total,
        source_label=source_label,
        window_label=window_label,
        generated_at=generated_at,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(
        f"wrote {args.output} ({len(rows)} latency rows over {len(traces)} traces)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
