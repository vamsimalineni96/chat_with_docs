"""LLM-as-judge for the eval harness.

Scores a candidate answer against retrieved context + expected keywords
using a separate LLM (default Llama 3.3 70B via NVIDIA NIM). The rubric
is decomposed — groundedness / accuracy / completeness, each 1-5 — to
avoid the Goodhart trap of optimizing against a single quality scalar.

Generator (Gemma) and judge (Llama) are deliberately from different
model families to minimize self-preference bias; see docs/PROGRESS.md
and docs/OBSERVABILITY.md §3.3 for the discussion.

All prompts live in `eval/prompts/judge.yaml`. Nothing is hardcoded
here. To tune the rubric, edit the YAML and bump its `version` field —
this module loads them once at first call and caches.

The `_call_judge_llm` function is intentionally a module-level shim so
tests can monkeypatch it without hitting the network.
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

DEFAULT_JUDGE_MODEL = os.environ.get(
    "EVAL_JUDGE_MODEL", "meta/llama-3.3-70b-instruct"
)
DEFAULT_MAX_RETRIES = 2

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.yaml"
_PROMPTS: dict[str, str] | None = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeResult:
    """Decomposed quality scores for a single (question, answer) pair.

    `error` is None on success. On failure (the LLM never returned valid
    JSON after retries), all three scores are 0 and `error` carries the
    last exception message. Downstream reporters should special-case
    rows with `error` rather than treating zeros as legitimate scores.
    """

    groundedness: int  # 1-5
    accuracy: int  # 1-5
    completeness: int  # 1-5
    reasoning: str
    error: str | None = None

    @property
    def overall(self) -> float:
        if self.error:
            return 0.0
        return (self.groundedness + self.accuracy + self.completeness) / 3.0


def _load_prompts() -> dict[str, str]:
    global _PROMPTS
    if _PROMPTS is None:
        with open(_PROMPT_PATH) as fh:
            loaded = yaml.safe_load(fh)
        for required in ("system_prompt", "user_prompt", "retry_prompt"):
            if required not in loaded:
                raise RuntimeError(
                    f"judge prompts file missing required key: {required}"
                )
        _PROMPTS = loaded
    return _PROMPTS


def _build_user_prompt(
    question: str,
    answer: str,
    retrieved_context: str,
    expected_keywords: list[str],
) -> str:
    template = _load_prompts()["user_prompt"]
    return template.format(
        question=question,
        answer=answer,
        retrieved_context=retrieved_context,
        expected_keywords=", ".join(expected_keywords),
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Find the first {...} block in text and parse it.

    Models occasionally emit leading prose despite instructions ("Here is
    my evaluation:\\n{...}\\nLet me know..."). The greedy regex picks up
    the outermost braces, which is enough for a flat schema.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ValueError("no JSON object in response")
    return json.loads(match.group())


def _validate_scores(parsed: dict[str, Any]) -> tuple[int, int, int, str]:
    for k in ("groundedness", "accuracy", "completeness"):
        v = parsed.get(k)
        if not isinstance(v, int) or not (1 <= v <= 5):
            raise ValueError(f"invalid {k}: {v!r} (expected int 1-5)")
    return (
        int(parsed["groundedness"]),
        int(parsed["accuracy"]),
        int(parsed["completeness"]),
        str(parsed.get("reasoning", "")),
    )


def _call_judge_llm(system: str, user: str, model: str) -> str:
    """Call NVIDIA NIM. Module-level so tests can monkeypatch it.

    Imports are deferred to inside the function so `import eval.judge`
    doesn't pull langchain — tests that mock this function never touch it.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")

    llm = ChatNVIDIA(
        model=model,
        api_key=api_key,
        temperature=0.1,
        max_tokens=400,
    )
    response = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return response.content


def judge(
    question: str,
    answer: str,
    retrieved_context: str,
    expected_keywords: list[str],
    *,
    model: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> JudgeResult:
    """Score `answer` on the three rubric dimensions.

    Retries up to `max_retries` times on JSON parse / range failures with
    an explicit "respond with only JSON" reminder. Returns a JudgeResult
    with all-zero scores and an `error` field if every attempt fails.
    """
    model = model or DEFAULT_JUDGE_MODEL
    prompts = _load_prompts()
    system = prompts["system_prompt"]
    user = _build_user_prompt(question, answer, retrieved_context, expected_keywords)

    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = _call_judge_llm(system, user, model)
            parsed = _extract_json(raw)
            g, a, c, r = _validate_scores(parsed)
            return JudgeResult(
                groundedness=g, accuracy=a, completeness=c, reasoning=r
            )
        except Exception as e:
            last_error = f"attempt {attempt + 1}: {e}"
            logger.warning("judge attempt %d failed: %s", attempt + 1, e)
            user = prompts["retry_prompt"]

    return JudgeResult(0, 0, 0, "", error=last_error)
