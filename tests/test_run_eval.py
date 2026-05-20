"""Behavior contract for the eval orchestrator.

These tests exercise `evaluate_one` and `run_eval` with stub callers —
no real HTTP, no real LLM. The stubs let us pin the per-row error
handling and the assembled-row shape independently of the chat service
and the judge implementation.
"""

from eval.judge import JudgeResult
from eval.run_eval import evaluate_one, run_eval

# --- chat caller stubs ------------------------------------------------------


def _ok_chat_response(question: str) -> dict:
    """Return a plausible /chat response with one matching retrieved chunk."""
    return {
        "answer": "Cedric is a Hufflepuff Triwizard champion who is killed.",
        "debug": {
            "retrieved_chunks": [
                {"text": "Cedric Diggory was a Hufflepuff student selected for the Triwizard tournament.", "source": "hp4.pdf"},
                {"text": "Filler chunk about Quidditch.", "source": "hp4.pdf"},
            ],
            "reranked_top_k": [
                {"text": "Cedric Diggory was a Hufflepuff student selected for the Triwizard tournament.", "source": "hp4.pdf"},
            ],
            "timings_ms": {"total": 950.0},
        },
    }


def _failing_chat_response(question: str) -> dict:
    raise RuntimeError("simulated /chat 503")


def _empty_chat_response(question: str) -> dict:
    """A /chat response that returned nothing useful — empty retrieval."""
    return {
        "answer": "I don't know.",
        "debug": {
            "retrieved_chunks": [],
            "reranked_top_k": [],
            "timings_ms": {"total": 800.0},
        },
    }


# --- judge stubs ------------------------------------------------------------


def _ok_judge(question, answer, context, kws) -> JudgeResult:
    return JudgeResult(
        groundedness=5, accuracy=4, completeness=3, reasoning="Solid."
    )


def _failing_judge(question, answer, context, kws) -> JudgeResult:
    raise RuntimeError("simulated judge HTTP error")


def _errored_judge(question, answer, context, kws) -> JudgeResult:
    return JudgeResult(0, 0, 0, "", error="bad json after 2 retries")


# --- evaluate_one -----------------------------------------------------------


def _qa() -> dict:
    return {
        "id": "hp4-001",
        "question": "Who is Cedric Diggory?",
        "expected_keywords_in_answer": ["Hufflepuff", "champion"],
        "expected_keywords_in_top_chunks": ["Cedric", "Hufflepuff"],
        "book": "hp4",
        "category": "character",
    }


def test_evaluate_one_happy_path():
    row = evaluate_one(_qa(), chat_caller=_ok_chat_response, judge_caller=_ok_judge)
    assert row["error"] is None
    assert row["id"] == "hp4-001"
    assert row["recall_at_5"] == 1.0  # Both kws appear in chunk text
    assert row["reciprocal_rank"] == 1.0
    assert row["judge"]["groundedness"] == 5
    assert row["judge"]["accuracy"] == 4
    assert row["latency_ms"] == 950.0


def test_evaluate_one_records_chat_failure_as_row_error():
    row = evaluate_one(_qa(), chat_caller=_failing_chat_response, judge_caller=_ok_judge)
    assert row["error"] is not None
    assert "chat call failed" in row["error"]
    # Scores stay at the zero defaults so the reporter doesn't crash.
    assert row["recall_at_5"] == 0.0
    assert row["judge"]["groundedness"] == 0


def test_evaluate_one_records_judge_failure_as_row_error():
    row = evaluate_one(_qa(), chat_caller=_ok_chat_response, judge_caller=_failing_judge)
    assert row["error"] is not None
    assert "judge call failed" in row["error"]
    # Retrieval metrics WERE computed before the judge ran — preserve them.
    assert row["recall_at_5"] == 1.0
    # But judge scores are zero.
    assert row["judge"]["groundedness"] == 0


def test_evaluate_one_propagates_judge_returned_error():
    """A JudgeResult with error != None should surface in row['error']."""
    row = evaluate_one(_qa(), chat_caller=_ok_chat_response, judge_caller=_errored_judge)
    assert row["error"] is not None
    assert "judge returned error" in row["error"]


def test_evaluate_one_handles_empty_retrieval():
    row = evaluate_one(_qa(), chat_caller=_empty_chat_response, judge_caller=_ok_judge)
    assert row["error"] is None
    assert row["recall_at_5"] == 0.0  # No chunks → no recall
    assert row["reciprocal_rank"] == 0.0
    # But the judge still ran and returned scores.
    assert row["judge"]["groundedness"] == 5


# --- run_eval (sequential loop) --------------------------------------------


def test_run_eval_runs_each_qa_once():
    qas = [_qa(), {**_qa(), "id": "hp4-002"}, {**_qa(), "id": "hp4-003"}]
    rows = run_eval(qas, chat_caller=_ok_chat_response, judge_caller=_ok_judge)
    assert [r["id"] for r in rows] == ["hp4-001", "hp4-002", "hp4-003"]
    assert all(r["error"] is None for r in rows)


def test_run_eval_continues_after_per_row_failure():
    qas = [_qa(), {**_qa(), "id": "hp4-002"}]
    # Fail only the first call to chat.
    call_count = {"n": 0}

    def flaky(question):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated 500")
        return _ok_chat_response(question)

    rows = run_eval(qas, chat_caller=flaky, judge_caller=_ok_judge)
    assert len(rows) == 2
    assert rows[0]["error"] is not None
    assert rows[1]["error"] is None


def test_run_eval_invokes_on_row_callback():
    qas = [_qa(), {**_qa(), "id": "hp4-002"}]
    seen = []
    run_eval(
        qas,
        chat_caller=_ok_chat_response,
        judge_caller=_ok_judge,
        on_row=lambda r: seen.append(r["id"]),
    )
    assert seen == ["hp4-001", "hp4-002"]


def test_run_eval_empty_input():
    rows = run_eval([], chat_caller=_ok_chat_response, judge_caller=_ok_judge)
    assert rows == []
