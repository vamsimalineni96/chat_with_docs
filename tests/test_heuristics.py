"""Behavior contract for the inline heuristic checks.

Pure-function tests: no mocking, no fixtures, no I/O. Each check is
exercised independently, then `evaluate_heuristics` is exercised to
verify the combined report.
"""

from src.utils.services.heuristics import (
    HeuristicReport,
    check_citation,
    check_length,
    check_refusal,
    evaluate_heuristics,
)

# A reusable "good" answer / chunk pair that all three checks pass on.
GOOD_CHUNKS = [
    {
        "text": (
            "Cedric Diggory was a Hufflepuff student selected for the "
            "Triwizard tournament alongside Harry Potter."
        )
    },
]
GOOD_ANSWER = (
    "Cedric Diggory was a Hufflepuff student selected for the Triwizard "
    "tournament, where he was killed by Voldemort at the end of the maze."
)


# --- check_refusal -----------------------------------------------------------


def test_refusal_passes_on_substantive_answer():
    assert check_refusal(GOOD_ANSWER) is True


def test_refusal_flags_dont_know():
    assert check_refusal("I don't know the answer.") is False


def test_refusal_flags_im_not_sure():
    assert check_refusal("I'm not sure based on the context.") is False


def test_refusal_flags_cannot():
    assert check_refusal("I cannot answer that from the given context.") is False


def test_refusal_flags_couldnt_find():
    # The canned no-retrieval response should be flagged.
    canned = (
        "I couldn't find anything in the indexed document that touches on that."
    )
    assert check_refusal(canned) is False


def test_refusal_flags_context_doesnt_specify():
    # gemma-4-31b-it's most common hedge in the eval runs.
    assert check_refusal("The context doesn't specify the three tasks.") is False


def test_refusal_is_case_insensitive():
    assert check_refusal("I DON'T KNOW.") is False


def test_refusal_fails_on_empty():
    assert check_refusal("") is False


# --- check_citation ----------------------------------------------------------


def test_citation_passes_when_answer_overlaps_chunks():
    assert check_citation(GOOD_ANSWER, GOOD_CHUNKS) is True


def test_citation_fails_when_answer_is_ungrounded():
    # Plausible-looking answer but ZERO 12-char overlap with the chunk.
    ungrounded = "Severus Snape brewed potions in the dungeons every Tuesday."
    assert check_citation(ungrounded, GOOD_CHUNKS) is False


def test_citation_passes_vacuously_when_no_chunks():
    # Empty retrieval is the early-return path; refusal check catches
    # that case, citation has nothing to compare against.
    assert check_citation(GOOD_ANSWER, []) is True
    assert check_citation(GOOD_ANSWER, None) is True


def test_citation_fails_on_empty_answer():
    assert check_citation("", GOOD_CHUNKS) is False


def test_citation_fails_when_chunks_have_no_text():
    # Defensive: chunks without text == nothing to compare against.
    # We treat this as a vacuous pass since we can't meaningfully evaluate.
    assert check_citation(GOOD_ANSWER, [{"text": ""}]) is True


# --- check_length ------------------------------------------------------------


def test_length_passes_on_normal_answer():
    assert check_length(GOOD_ANSWER) is True


def test_length_fails_on_empty():
    assert check_length("") is False


def test_length_fails_on_truncated():
    assert check_length("Yes.") is False


def test_length_fails_on_runaway():
    assert check_length("x" * 5000) is False


def test_length_passes_at_boundaries():
    # min boundary: exactly 30 chars
    assert check_length("a" * 30) is True
    # max boundary: exactly 4000 chars
    assert check_length("a" * 4000) is True


# --- evaluate_heuristics (combined) ------------------------------------------


def test_evaluate_all_pass_on_good_answer():
    report = evaluate_heuristics(GOOD_ANSWER, GOOD_CHUNKS)
    assert isinstance(report, HeuristicReport)
    assert report.overall_passed is True
    assert report.failed_check_names == []


def test_evaluate_flags_multiple_failures():
    # Empty answer fails refusal (empty), citation (too short), length (empty).
    report = evaluate_heuristics("", GOOD_CHUNKS)
    assert report.overall_passed is False
    assert set(report.failed_check_names) == {"refusal", "citation", "length"}


def test_evaluate_flags_just_refusal():
    # 30+ chars, substring of chunk, but contains a hedge phrase.
    hedged = "I don't know — Cedric Diggory was a Hufflepuff student."
    report = evaluate_heuristics(hedged, GOOD_CHUNKS)
    assert report.refusal_check_passed is False
    assert report.citation_check_passed is True
    assert report.length_check_passed is True
    assert report.failed_check_names == ["refusal"]


def test_report_to_dict_shape():
    report = evaluate_heuristics(GOOD_ANSWER, GOOD_CHUNKS)
    d = report.to_dict()
    assert d["overall_passed"] is True
    assert d["refusal_check_passed"] is True
    assert d["citation_check_passed"] is True
    assert d["length_check_passed"] is True
    assert d["failed_checks"] == []
