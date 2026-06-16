"""Load and validate agentic eval cases from YAML.

Each case is one YAML file under evals/agentic/cases/. The shape:

  id: refund-bob-webcam-disambig
  description: Refund with multiple matching payments must pause for disambig
  question: "refund bob wilson for the webcam"
  expected:
    intent: tool_call            # research | tool_call | both | out_of_scope
    tool_calls:                  # ordered; subset match on args
      - name: get_customer_payments
        args_contain:
          name_or_email: bob
          product_filter: webcam
    hitl:
      paused: true
      kind: disambig             # approval | disambig
      min_candidates: 2
    http_status: 200

Any expected.* key is optional — missing = "don't assert this dimension".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CASES_DIR = Path(__file__).parent / "cases"


@dataclass
class ExpectedToolCall:
    name: str
    args_contain: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpectedHITL:
    paused: bool | None = None
    kind: str | None = None              # "approval" | "disambig"
    min_candidates: int | None = None


@dataclass
class Expected:
    intent: str | None = None
    tool_calls: list[ExpectedToolCall] | None = None
    hitl: ExpectedHITL | None = None
    http_status: int | None = None


@dataclass
class Case:
    id: str
    question: str
    description: str
    expected: Expected
    source_path: Path


def _parse_expected(raw: dict[str, Any]) -> Expected:
    tool_calls_raw = raw.get("tool_calls")
    tool_calls: list[ExpectedToolCall] | None = None
    if tool_calls_raw is not None:
        tool_calls = [
            ExpectedToolCall(
                name=tc["name"],
                args_contain=tc.get("args_contain") or {},
            )
            for tc in tool_calls_raw
        ]

    hitl_raw = raw.get("hitl")
    hitl: ExpectedHITL | None = None
    if hitl_raw is not None:
        hitl = ExpectedHITL(
            paused=hitl_raw.get("paused"),
            kind=hitl_raw.get("kind"),
            min_candidates=hitl_raw.get("min_candidates"),
        )

    return Expected(
        intent=raw.get("intent"),
        tool_calls=tool_calls,
        hitl=hitl,
        http_status=raw.get("http_status"),
    )


def load_case(path: Path) -> Case:
    """Parse one case file. Raises ValueError on missing required fields."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    for required in ("id", "question"):
        if required not in raw:
            raise ValueError(f"{path.name}: missing required field {required!r}")

    return Case(
        id=raw["id"],
        question=raw["question"],
        description=raw.get("description", ""),
        expected=_parse_expected(raw.get("expected") or {}),
        source_path=path,
    )


def load_all_cases(cases_dir: Path = CASES_DIR) -> list[Case]:
    """Load every .yaml file under cases_dir, sorted by id for stable ordering."""
    files = sorted(cases_dir.glob("*.yaml"))
    cases = [load_case(p) for p in files]
    ids_seen: set[str] = set()
    for c in cases:
        if c.id in ids_seen:
            raise ValueError(f"duplicate case id {c.id!r} (in {c.source_path.name})")
        ids_seen.add(c.id)
    return cases
