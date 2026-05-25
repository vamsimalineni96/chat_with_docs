"""LLM-driven intent classifier for the chat graph.

Sits at the head of the graph in `src/agents/graph.py` and decides
which downstream path a user question should take:

  - `in_corpus`    — RAG over the uploaded documents (the default).
  - `out_of_scope` — short-circuit to a canned response without
                     burning a Milvus + rerank + LLM call.

Designed to be cheap and fast — uses an 8B model by default
(`meta/llama-3.1-8b-instruct`) since this is a structured decision
task, not a creative one. All prompts live in
`src/agents/prompts/classify_intent.yaml` — bump the file's `version`
when the rubric changes.

Failure policy: any error (LLM down, bad JSON after retries,
unrecognised intent value) falls back to `in_corpus`. A false
positive on `out_of_scope` is far worse than missing one
short-circuit opportunity — it would refuse a legitimate question.

Mirrors the structure of `evals/quality/judge.py`: prompts in YAML,
retries on bad JSON, the `_call_classifier_llm` module-level shim
gets monkeypatched in tests so no real NVIDIA call ever fires from
the unit suite.
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

DEFAULT_CLASSIFIER_MODEL = os.environ.get(
    "INTENT_CLASSIFIER_MODEL", "meta/llama-3.1-8b-instruct"
)
DEFAULT_MAX_RETRIES = 2

VALID_INTENTS = ("in_corpus", "out_of_scope")
FALLBACK_INTENT = "in_corpus"
FALLBACK_REASONING = "classifier unavailable; defaulting to in_corpus"

_PROMPT_PATH = Path(__file__).parent / "prompts" / "classify_intent.yaml"
_PROMPTS: dict[str, str] | None = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentResult:
    """Classifier output for one question.

    `error` is None on success. On failure (LLM unreachable, JSON
    malformed after retries, unrecognised intent value), `intent`
    holds the safe fallback (`in_corpus`) and `error` carries the
    last exception message so a caller can surface it in logs /
    debug payload.
    """

    intent: str  # one of VALID_INTENTS
    reasoning: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "reasoning": self.reasoning,
            "error": self.error,
        }


def _load_prompts() -> dict[str, str]:
    global _PROMPTS
    if _PROMPTS is None:
        with open(_PROMPT_PATH) as fh:
            loaded = yaml.safe_load(fh)
        for required in ("system_prompt", "user_prompt", "retry_prompt"):
            if required not in loaded:
                raise RuntimeError(
                    f"classify_intent prompts file missing required key: {required}"
                )
        _PROMPTS = loaded
    return _PROMPTS


def _build_user_prompt(question: str) -> str:
    return _load_prompts()["user_prompt"].format(question=question)


def _extract_json(text: str) -> dict[str, Any]:
    """Find the first {...} block in text and parse it.

    Models occasionally emit leading prose despite instructions; the
    greedy regex picks up the outermost braces, which is enough for
    a flat schema with two scalar fields.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ValueError("no JSON object in response")
    return json.loads(match.group())


def _validate(parsed: dict[str, Any]) -> tuple[str, str]:
    intent = parsed.get("intent")
    if intent not in VALID_INTENTS:
        raise ValueError(f"invalid intent: {intent!r} (expected one of {VALID_INTENTS})")
    reasoning = str(parsed.get("reasoning", ""))
    return intent, reasoning


def _call_classifier_llm(system: str, user: str, model: str) -> str:
    """Call NVIDIA NIM. Module-level so tests can monkeypatch it.

    Imports deferred so `import src.agents.intent_classifier` doesn't
    pull langchain — tests that mock this function never touch the
    real network or the langchain stack.
    """
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415
    from langchain_nvidia_ai_endpoints import ChatNVIDIA  # noqa: PLC0415

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")

    llm = ChatNVIDIA(
        model=model,
        api_key=api_key,
        temperature=0.0,  # deterministic — this is a routing decision
        max_tokens=100,
    )
    response = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return response.content


def classify_intent(
    question: str,
    *,
    model: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> IntentResult:
    """Classify a question and return the routing decision.

    Retries up to `max_retries` times on JSON parse / value failures
    using the `retry_prompt` (which reminds the model to return only
    JSON). On total failure, falls back to `in_corpus` so a
    legitimate question is never refused due to classifier downtime.
    """
    model = model or DEFAULT_CLASSIFIER_MODEL
    prompts = _load_prompts()
    system = prompts["system_prompt"]
    user = _build_user_prompt(question)

    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = _call_classifier_llm(system, user, model)
            parsed = _extract_json(raw)
            intent, reasoning = _validate(parsed)
            return IntentResult(intent=intent, reasoning=reasoning)
        except Exception as e:
            last_error = f"attempt {attempt + 1}: {e}"
            logger.warning(
                "classify_intent attempt %d failed: %s", attempt + 1, e
            )
            user = prompts["retry_prompt"]

    return IntentResult(
        intent=FALLBACK_INTENT,
        reasoning=FALLBACK_REASONING,
        error=last_error,
    )
