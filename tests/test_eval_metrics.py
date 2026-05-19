"""Behavior contracts for the eval-harness metrics.

All assertions target qualitative invariants (>=0, <=1, monotonic, correct
boundaries) rather than exact floating-point equalities where possible —
so the suite survives minor implementation tweaks.
"""

from eval.metrics import (
    keyword_recall_at_k,
    latency_percentiles,
    mrr,
    reciprocal_rank,
)


def _chunks(*texts: str) -> list[dict]:
    return [{"text": t} for t in texts]


# ----------------------------------------------------------------------------
# keyword_recall_at_k
# ----------------------------------------------------------------------------


def test_keyword_recall_full_match_returns_one():
    chunks = _chunks("Cedric Diggory was a Hufflepuff champion.")
    assert keyword_recall_at_k(["Cedric", "Hufflepuff"], chunks, k=5) == 1.0


def test_keyword_recall_no_match_returns_zero():
    chunks = _chunks("Some unrelated text about Quidditch.")
    assert keyword_recall_at_k(["Voldemort", "Horcrux"], chunks, k=5) == 0.0


def test_keyword_recall_case_insensitive():
    chunks = _chunks("CEDRIC DIGGORY WAS A HUFFLEPUFF CHAMPION.")
    assert keyword_recall_at_k(["cedric", "hufflepuff"], chunks, k=5) == 1.0


def test_keyword_recall_respects_k():
    # Keyword only appears at position 6, not in top-5
    chunks = _chunks(
        "filler 1", "filler 2", "filler 3", "filler 4", "filler 5",
        "Voldemort returns to power.",
    )
    assert keyword_recall_at_k(["Voldemort"], chunks, k=5) == 0.0
    assert keyword_recall_at_k(["Voldemort"], chunks, k=6) == 1.0


def test_keyword_recall_partial_match():
    chunks = _chunks("Cedric Diggory was a Hufflepuff champion.")
    # 1 of 2 keywords present (Cedric yes, Voldemort no) -> 0.5
    assert keyword_recall_at_k(["Cedric", "Voldemort"], chunks, k=5) == 0.5


# ----------------------------------------------------------------------------
# reciprocal_rank
# ----------------------------------------------------------------------------


def test_reciprocal_rank_first_position():
    chunks = _chunks("Cedric is mentioned here.", "Filler chunk.")
    assert reciprocal_rank(["Cedric"], chunks) == 1.0


def test_reciprocal_rank_third_position():
    chunks = _chunks("filler", "filler", "Cedric appears here.")
    assert abs(reciprocal_rank(["Cedric"], chunks) - 1.0 / 3.0) < 1e-9


def test_reciprocal_rank_no_match_returns_zero():
    chunks = _chunks("filler", "filler", "filler")
    assert reciprocal_rank(["Voldemort"], chunks) == 0.0


# ----------------------------------------------------------------------------
# mrr
# ----------------------------------------------------------------------------


def test_mrr_averages_correctly():
    assert abs(mrr([1.0, 0.5, 1.0 / 3.0]) - (1.0 + 0.5 + 1.0 / 3.0) / 3.0) < 1e-9


def test_mrr_empty_returns_zero():
    assert mrr([]) == 0.0


# ----------------------------------------------------------------------------
# latency_percentiles
# ----------------------------------------------------------------------------


def test_latency_percentiles_basic():
    # 100 evenly-spaced latencies: p50 ~ 50, p95 ~ 95, p99 ~ 99
    latencies = [float(i) for i in range(1, 101)]
    pcts = latency_percentiles(latencies, percentiles=(50, 95, 99))
    assert pcts[50] == 50.0
    assert pcts[95] == 95.0
    assert pcts[99] == 99.0


def test_latency_percentiles_empty_input():
    assert latency_percentiles([], percentiles=(50, 95, 99)) == {50: 0.0, 95: 0.0, 99: 0.0}
