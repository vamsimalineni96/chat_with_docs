"""Markdown reporter for agentic eval runs.

Outputs a dated report with: overall pass rate, per-case status table,
and detailed failure breakdowns. Same shape as evals/quality/reporter.py
so the docs/eval-reports/ directory stays consistent.
"""

from __future__ import annotations

from .assertions import CaseResult

_DIMENSIONS = ("intent", "tool_calls", "hitl", "http_status")


def _status_cell(result: CaseResult, dimension: str) -> str:
    for o in result.outcomes:
        if o.dimension == dimension:
            return "✅" if o.passed else "❌"
    return "—"  # dimension not asserted


def render_markdown(
    results: list[CaseResult],
    *,
    generated_at: str,
    api_base: str,
) -> str:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    errored = sum(1 for r in results if r.error)
    pass_rate = (passed / total * 100) if total else 0.0

    lines: list[str] = []
    lines.append("# Agentic Eval Report")
    lines.append("")
    lines.append(f"- **Generated:** {generated_at}")
    lines.append(f"- **API:** `{api_base}`")
    lines.append(f"- **Cases:** {total}")
    lines.append(f"- **Passed:** {passed} ({pass_rate:.1f}%)")
    lines.append(f"- **Errored:** {errored}")
    lines.append("")

    # Per-case table
    lines.append("## Cases")
    lines.append("")
    header_cells = ["Case", *(_DIMENSIONS), "Overall"]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for r in results:
        overall = "✅" if r.passed else ("⚠️" if r.error else "❌")
        cells = [r.case_id, *(_status_cell(r, d) for d in _DIMENSIONS), overall]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Failure details
    failures = [r for r in results if not r.passed]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for r in failures:
            lines.append(f"### {r.case_id}")
            if r.description:
                lines.append(f"*{r.description}*")
            lines.append("")
            lines.append(f"- **Question:** {r.question}")
            lines.append(f"- **HTTP status:** {r.http_status}")
            if r.error:
                lines.append(f"- **Error:** `{r.error}`")
            for o in r.failures():
                lines.append(f"- **{o.dimension} failed:** {o.reason}")
            # Show the actual agentic decisions for context
            if r.raw_response:
                debug = r.raw_response.get("debug") or {}
                intent = debug.get("intent")
                tool_calls = debug.get("tool_calls") or []
                pending = r.raw_response.get("pending_approval")
                lines.append("")
                lines.append("**Actual:**")
                lines.append(f"  - intent: `{intent}`")
                lines.append(
                    f"  - tool_calls: `{[tc.get('name') for tc in tool_calls]}`"
                )
                if pending:
                    lines.append(
                        f"  - pending_approval: kind=`{pending.get('kind')}`, "
                        f"candidates={len(pending.get('candidates') or [])}"
                    )
                else:
                    lines.append("  - pending_approval: `None`")
            lines.append("")
    return "\n".join(lines) + "\n"
