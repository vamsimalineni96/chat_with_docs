"""Behavior contract for the intent classifier.

`_call_classifier_llm` is monkeypatched in every test so no real
NVIDIA HTTP requests fire. Tests target the same surface as
test_eval_judge.py: happy-path JSON parse, recovery from surrounding
prose, retry-then-succeed, graceful failure after max retries,
invalid intent value, and the in_corpus fallback policy.
"""

from __future__ import annotations

from src.agents import intent_classifier as ic


def _constant(response_text: str):
    def _mock(system, user, model):
        return response_text

    return _mock


def test_classify_happy_path_in_corpus(monkeypatch):
    monkeypatch.setattr(
        ic,
        "_call_classifier_llm",
        _constant(
            '{"intent": "in_corpus", "reasoning": "asks about Cedric Diggory"}'
        ),
    )
    result = ic.classify_intent("Who is Cedric Diggory?")
    assert result.intent == "in_corpus"
    assert result.reasoning == "asks about Cedric Diggory"
    assert result.error is None


def test_classify_happy_path_out_of_scope(monkeypatch):
    monkeypatch.setattr(
        ic,
        "_call_classifier_llm",
        _constant(
            '{"intent": "out_of_scope", "reasoning": "general chemistry"}'
        ),
    )
    result = ic.classify_intent("What is the chemical formula for caffeine?")
    assert result.intent == "out_of_scope"
    assert result.reasoning == "general chemistry"
    assert result.error is None


def test_classify_strips_leading_prose(monkeypatch):
    """Some 8B models emit prose before the JSON despite instructions —
    the regex extractor should still find the JSON object."""
    monkeypatch.setattr(
        ic,
        "_call_classifier_llm",
        _constant(
            "Sure, here's my classification:\n"
            '{"intent": "in_corpus", "reasoning": "about HP4"}\n'
            "Let me know if you need anything else!"
        ),
    )
    result = ic.classify_intent("Who is Snape?")
    assert result.intent == "in_corpus"


def test_classify_retries_then_succeeds(monkeypatch):
    """Bad JSON on attempt 1, valid on attempt 2 — the retry_prompt
    swap inside classify_intent should let the second call succeed.
    """
    responses = iter(
        [
            "no json here at all",
            '{"intent": "in_corpus", "reasoning": "recovered"}',
        ]
    )

    def _alternating(system, user, model):
        return next(responses)

    monkeypatch.setattr(ic, "_call_classifier_llm", _alternating)
    result = ic.classify_intent("Anything", max_retries=2)
    assert result.intent == "in_corpus"
    assert result.error is None


def test_classify_falls_back_to_in_corpus_after_max_retries(monkeypatch):
    """The whole point of the fallback policy: classifier downtime
    must never refuse a legitimate question.
    """
    monkeypatch.setattr(
        ic, "_call_classifier_llm", _constant("not json no way")
    )
    result = ic.classify_intent("Anything", max_retries=1)
    assert result.intent == ic.FALLBACK_INTENT  # in_corpus
    assert result.error is not None
    assert "attempt" in result.error


def test_classify_falls_back_on_invalid_intent_value(monkeypatch):
    """If the LLM returns valid JSON but with a junk intent string,
    we don't accept it — same fallback as a parse failure.
    """
    monkeypatch.setattr(
        ic,
        "_call_classifier_llm",
        _constant('{"intent": "made_up_category", "reasoning": "..."}'),
    )
    result = ic.classify_intent("Where is my order?", max_retries=0)
    assert result.intent == ic.FALLBACK_INTENT
    assert result.error is not None


def test_classify_happy_path_tool_call(monkeypatch):
    """tool_call is the third valid intent (added in PR #5) — routes to
    the MCP ReAct sub-agent. Mirrors the in_corpus / out_of_scope shape.
    """
    monkeypatch.setattr(
        ic,
        "_call_classifier_llm",
        _constant(
            '{"intent": "tool_call", "reasoning": "asks for order status by ID"}'
        ),
    )
    result = ic.classify_intent("Where is order ORD-1001?")
    assert result.intent == "tool_call"
    assert result.reasoning == "asks for order status by ID"
    assert result.error is None


def test_classify_falls_back_when_llm_raises(monkeypatch):
    def _raises(system, user, model):
        raise RuntimeError("NVIDIA 503")

    monkeypatch.setattr(ic, "_call_classifier_llm", _raises)
    result = ic.classify_intent("Anything", max_retries=1)
    assert result.intent == ic.FALLBACK_INTENT
    assert result.error is not None
    assert "503" in result.error


def test_intent_result_to_dict_shape():
    r = ic.IntentResult(intent="in_corpus", reasoning="x", error=None)
    d = r.to_dict()
    assert d == {"intent": "in_corpus", "reasoning": "x", "error": None}
