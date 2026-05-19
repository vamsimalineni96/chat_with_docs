"""Behavior contract for the LLM-as-judge.

`_call_judge_llm` is monkeypatched in every test so no real NVIDIA HTTP
requests fire. Tests target: happy-path JSON parse, recovery from
surrounding prose, retry on bad JSON then success, graceful failure
after max retries, score-range validation, and the overall property.
"""

import json

from eval import judge as judge_mod


def _make_constant_mock(response_text: str):
    def _mock(system, user, model):
        return response_text
    return _mock


def test_judge_happy_path(monkeypatch):
    response = json.dumps(
        {
            "groundedness": 5,
            "accuracy": 4,
            "completeness": 3,
            "reasoning": "Solid grounding, accurate, missing one keyword.",
        }
    )
    monkeypatch.setattr(judge_mod, "_call_judge_llm", _make_constant_mock(response))

    result = judge_mod.judge(
        question="Who is Cedric?",
        answer="Cedric is a Hufflepuff champion.",
        retrieved_context="Cedric Diggory was a Hufflepuff student selected for the Triwizard tournament.",
        expected_keywords=["Hufflepuff", "champion"],
    )

    assert result.error is None
    assert result.groundedness == 5
    assert result.accuracy == 4
    assert result.completeness == 3
    assert "Solid" in result.reasoning


def test_judge_parses_json_surrounded_by_prose(monkeypatch):
    response = (
        "Here is my evaluation:\n"
        '{"groundedness": 4, "accuracy": 4, "completeness": 4, "reasoning": "ok"}\n'
        "Let me know if you have questions."
    )
    monkeypatch.setattr(judge_mod, "_call_judge_llm", _make_constant_mock(response))

    result = judge_mod.judge("q", "a", "c", ["kw"])
    assert result.error is None
    assert result.groundedness == 4


def test_judge_retries_on_bad_json_then_succeeds(monkeypatch):
    call_count = {"n": 0}

    def alternating(system, user, model):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "I cannot evaluate this answer."  # no JSON block
        return json.dumps(
            {"groundedness": 5, "accuracy": 5, "completeness": 5, "reasoning": "ok"}
        )

    monkeypatch.setattr(judge_mod, "_call_judge_llm", alternating)

    result = judge_mod.judge("q", "a", "c", ["kw"])
    assert result.error is None
    assert result.groundedness == 5
    assert call_count["n"] == 2


def test_judge_fails_gracefully_after_max_retries(monkeypatch):
    monkeypatch.setattr(
        judge_mod, "_call_judge_llm", _make_constant_mock("no json at all")
    )

    result = judge_mod.judge("q", "a", "c", ["kw"], max_retries=1)
    assert result.error is not None
    assert result.groundedness == 0
    assert result.accuracy == 0
    assert result.completeness == 0


def test_judge_rejects_out_of_range_scores(monkeypatch):
    bad = json.dumps(
        {"groundedness": 7, "accuracy": 4, "completeness": 4, "reasoning": "n/a"}
    )
    monkeypatch.setattr(judge_mod, "_call_judge_llm", _make_constant_mock(bad))

    result = judge_mod.judge("q", "a", "c", ["kw"], max_retries=0)
    assert result.error is not None
    assert "groundedness" in result.error


def test_judge_result_overall_averages_subscores():
    r = judge_mod.JudgeResult(
        groundedness=5, accuracy=4, completeness=3, reasoning=""
    )
    assert abs(r.overall - 4.0) < 1e-9


def test_judge_result_overall_zero_on_error():
    r = judge_mod.JudgeResult(
        groundedness=0, accuracy=0, completeness=0, reasoning="", error="boom"
    )
    assert r.overall == 0.0
