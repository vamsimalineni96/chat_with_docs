"""LLM supervisor that decides which agent(s) handle a user question.

Replaces the simpler `intent_classifier` in the multi-agent graph.
The key difference: the supervisor can output "both", which triggers
parallel execution of research + action agents and downstream synthesis.

Plans:
  research     → RAG pipeline over the uploaded document corpus
  action       → ReAct MCP agent (Stripe, inventory, orders)
  both         → both agents run concurrently; aggregator synthesises
  out_of_scope → canned response, no agent invoked

Same failure policy as intent_classifier: any LLM/JSON error falls
back to "research" — it's the safest default (worst case is a RAG
answer instead of a refusal).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils.observability import observe

DEFAULT_SUPERVISOR_MODEL = os.environ.get(
    "SUPERVISOR_MODEL", "meta/llama-3.1-8b-instruct"
)
DEFAULT_MAX_RETRIES = 2

VALID_PLANS = ("research", "action", "both", "out_of_scope")
FALLBACK_PLAN = "research"
FALLBACK_REASONING = "supervisor unavailable; defaulting to research"

_PROMPT_PATH = Path(__file__).parent / "prompts" / "supervisor.yaml"
_PROMPTS: dict[str, str] | None = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupervisorResult:
    plan: str       # one of VALID_PLANS
    reasoning: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"plan": self.plan, "reasoning": self.reasoning, "error": self.error}


def _load_prompts() -> dict[str, str]:
    global _PROMPTS
    if _PROMPTS is None:
        with open(_PROMPT_PATH) as fh:
            loaded = yaml.safe_load(fh)
        for required in ("system_prompt", "user_prompt", "retry_prompt"):
            if required not in loaded:
                raise RuntimeError(f"supervisor prompts missing key: {required}")
        _PROMPTS = loaded
    return _PROMPTS


def _build_user_prompt(question: str, history: list[dict] | None = None) -> str:
    last_turns = (history or [])[-2:]
    if last_turns:
        lines = ["Recent conversation:"]
        for turn in last_turns:
            role = turn.get("role", "").capitalize()
            content = (turn.get("content") or "")[:200]
            lines.append(f"  {role}: {content}")
        lines.append("")
        history_context = "\n".join(lines) + "\n"
    else:
        history_context = ""
    return _load_prompts()["user_prompt"].format(
        question=question, history_context=history_context
    )


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ValueError("no JSON object in response")
    return json.loads(match.group())


def _validate(parsed: dict[str, Any]) -> tuple[str, str]:
    plan = parsed.get("plan")
    if plan not in VALID_PLANS:
        raise ValueError(f"invalid plan: {plan!r} (expected one of {VALID_PLANS})")
    return plan, str(parsed.get("reasoning", ""))


def _call_supervisor_llm(system: str, user: str, model: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415
    from langchain_nvidia_ai_endpoints import ChatNVIDIA  # noqa: PLC0415

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")

    llm = ChatNVIDIA(
        model=model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=100,
    )
    response = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return response.content


@observe(name="supervisor")
def run_supervisor(
    question: str,
    *,
    history: list[dict] | None = None,
    model: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> SupervisorResult:
    """Decide which agent(s) should handle this question."""
    model = model or DEFAULT_SUPERVISOR_MODEL
    prompts = _load_prompts()
    system = prompts["system_prompt"]
    user = _build_user_prompt(question, history=history)

    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = _call_supervisor_llm(system, user, model)
            parsed = _extract_json(raw)
            plan, reasoning = _validate(parsed)
            return SupervisorResult(plan=plan, reasoning=reasoning)
        except Exception as e:
            last_error = f"attempt {attempt + 1}: {e}"
            logger.warning("supervisor attempt %d failed: %s", attempt + 1, e)
            user = prompts["retry_prompt"]

    return SupervisorResult(
        plan=FALLBACK_PLAN,
        reasoning=FALLBACK_REASONING,
        error=last_error,
    )
