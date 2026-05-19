"""Pure-function retrieval and latency metrics for the eval harness.

These don't import langfuse, langchain, or any model SDK — they take plain
dicts/lists in and return plain numbers out. Easy to unit-test against a
fixture. The end-to-end eval driver (run_eval.py, landing in a later
sub-PR) is what wires these up against `/chat` output.
"""

from __future__ import annotations


def _matches_any_keyword(text: str, keywords: list[str]) -> bool:
    """True iff any keyword appears (case-insensitive substring) in text."""
    if not text:
        return False
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in keywords)


def keyword_recall_at_k(
    expected_keywords: list[str],
    retrieved_chunks: list[dict],
    k: int,
) -> float:
    """Fraction of `expected_keywords` that appear in at least one of the
    top-`k` chunks' text fields.

    Returns 0.0 if `expected_keywords` is empty (rather than dividing by
    zero); callers should treat that as a malformed Q&A entry, not a
    perfect score.
    """
    if not expected_keywords:
        return 0.0
    if k <= 0 or not retrieved_chunks:
        return 0.0

    top = retrieved_chunks[:k]
    blob = " ".join((c.get("text") or "") for c in top).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in blob)
    return hits / len(expected_keywords)


def reciprocal_rank(
    expected_keywords: list[str],
    retrieved_chunks: list[dict],
) -> float:
    """1 / (rank of the first chunk that contains ANY expected keyword).

    Ranks are 1-indexed. Returns 0.0 if no chunk matches — same convention
    used in standard MRR papers (BEIR, MS MARCO).
    """
    if not expected_keywords or not retrieved_chunks:
        return 0.0

    for i, chunk in enumerate(retrieved_chunks, start=1):
        if _matches_any_keyword(chunk.get("text") or "", expected_keywords):
            return 1.0 / i
    return 0.0


def mrr(reciprocal_ranks: list[float]) -> float:
    """Mean of per-query reciprocal ranks."""
    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def latency_percentiles(
    latencies_ms: list[float],
    percentiles: tuple[int, ...] = (50, 95, 99),
) -> dict[int, float]:
    """Compute requested percentiles from a list of latency samples.

    Uses the same nearest-rank method as `numpy.percentile(..., method=
    "lower")` so results are reproducible without pulling in numpy. For
    small N (~20 samples in our eval set) this is good enough; for
    statistical rigor you'd want at least a few hundred.
    """
    if not latencies_ms:
        return {p: 0.0 for p in percentiles}

    sorted_l = sorted(latencies_ms)
    n = len(sorted_l)
    out: dict[int, float] = {}
    for p in percentiles:
        # nearest-rank: index = ceil(p/100 * n) - 1, clamped to [0, n-1]
        idx = max(0, min(n - 1, -(-p * n // 100) - 1))
        out[p] = float(sorted_l[idx])
    return out
