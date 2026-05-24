"""Synchronous heuristic checks for chat responses.

The Quality pillar of the observability framework
(see docs/OBSERVABILITY.md §3.3 row 3) needs two complementary signals:

  1. LLM-as-judge — high-fidelity, slow, expensive, runs offline on the
     curated eval dataset. Tells us "how good is the system on questions
     we picked?"
  2. Heuristic checks — deterministic, fast, free, runs inline on every
     real request. Tells us "are the answers going out the door
     suspicious right now?"

This module is #2. Three pure functions, no LLM, no I/O. Each returns
a `HeuristicReport` with per-check booleans and an `overall_passed`
field. The caller is responsible for tagging the Langfuse trace and
including the report in any debug payload.

Design choice — heuristics flag, they do not block. An ungrounded
answer is *worth investigating*; it is not automatically *wrong*. The
"failure" terminology refers to the answer failing a sanity check, not
the request failing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# Common refusal / hedge phrases. Calibrated against actual model
# outputs we've seen in eval and live runs — gemma-4-31b-it tends
# toward "the context doesn't specify" / "There's no mention of X"
# rather than "I cannot answer". When a new phrasing slips through,
# extend the list and add a test case.
REFUSAL_PATTERNS = (
    r"\bi (?:don'?t|do not) know\b",
    r"\bi'?m not (?:sure|certain)\b",
    r"\bi (?:can'?t|cannot|couldn'?t)\b",
    r"\bcontext (?:doesn'?t|does not) (?:specify|provide|mention|contain)\b",
    r"\bno (?:information|details) (?:about|on|regarding)\b",
    r"\bunable to (?:determine|find|locate)\b",
    r"\bnot (?:provided|specified|mentioned) in the (?:context|text)\b",
    # Caught live: "There's no mention of the chemical formula for caffeine
    # in the provided context" — the syntactic inversion of "context doesn't
    # mention X" that the earlier patterns missed.
    r"\bno mention of\b",
    # Caught live: "none of them mention caffeine or its chemical formula"
    # — paired with "no mention of" for the same response.
    r"\bnone of (?:them|these|those)\b",
)

# Below this many characters, an answer is almost certainly evasive
# or truncated. Tuned against eval runs where the shortest legitimate
# answer was ~60 characters.
MIN_ANSWER_LEN = 30

# Above this, the LLM is either rambling or echoing context. RAG
# answers in this codebase are configured for max_tokens=400; 4000
# characters comfortably contains a verbose 400-token answer.
MAX_ANSWER_LEN = 4000

# Citation check uses character-n-grams: an answer "cites" a chunk if
# a contiguous run of CITATION_NGRAM_SIZE characters from the answer
# appears verbatim in any retrieved chunk. Word-level shingles miss
# inflection / casing; pure substring search would flag any common
# stopword string ("the"). 12 chars is roughly two content words —
# specific enough to be meaningful, loose enough to catch paraphrase.
CITATION_NGRAM_SIZE = 12
CITATION_MIN_MATCHES = 1


@dataclass(frozen=True)
class HeuristicReport:
    """Per-check pass/fail + overall pass for one (answer, chunks) pair.

    `overall_passed` is True only if every check passed. The individual
    booleans are exposed so downstream consumers (Langfuse tags, eval
    report, cost aggregator) can break out *which* check fired.
    """

    refusal_check_passed: bool
    citation_check_passed: bool
    length_check_passed: bool
    overall_passed: bool

    @property
    def failed_check_names(self) -> list[str]:
        """Names of checks that failed, for tagging / logging."""
        out: list[str] = []
        if not self.refusal_check_passed:
            out.append("refusal")
        if not self.citation_check_passed:
            out.append("citation")
        if not self.length_check_passed:
            out.append("length")
        return out

    def to_dict(self) -> dict[str, bool | list[str]]:
        d = asdict(self)
        d["failed_checks"] = self.failed_check_names
        return d


def check_refusal(answer: str) -> bool:
    """Pass = answer does NOT look like a hedge / refusal.

    Returns False if any refusal pattern matches. Case-insensitive.
    """
    if not answer:
        return False
    lowered = answer.lower()
    return not any(re.search(p, lowered) for p in REFUSAL_PATTERNS)


def check_citation(answer: str, retrieved_chunks: list[dict] | None) -> bool:
    """Pass = the answer shares at least one phrase with the retrieved chunks.

    This is a grounding sanity check, not a strict citation requirement.
    If the answer's char-n-grams don't appear in any retrieved chunk, the
    model is almost certainly drawing on training data instead of the
    retrieved context — worth flagging.

    Edge case: empty retrieval ⇒ this check passes vacuously. The
    pipeline already short-circuits with a canned "I couldn't find
    anything" response in that case, and that response will be caught
    by the refusal check instead.
    """
    if not retrieved_chunks:
        return True
    if not answer or len(answer) < CITATION_NGRAM_SIZE:
        return False

    haystack = " ".join((c.get("text") or "") for c in retrieved_chunks).lower()
    if not haystack:
        return True  # no chunk text to compare against — vacuous pass

    needle = answer.lower()
    matches = 0
    for i in range(len(needle) - CITATION_NGRAM_SIZE + 1):
        ngram = needle[i : i + CITATION_NGRAM_SIZE]
        if ngram in haystack:
            matches += 1
            if matches >= CITATION_MIN_MATCHES:
                return True
    return False


def check_length(answer: str) -> bool:
    """Pass = answer length is within sane bounds.

    Catches two failure modes: empty/truncated outputs (very short) and
    runaway / context-echo outputs (very long).
    """
    if not answer:
        return False
    n = len(answer)
    return MIN_ANSWER_LEN <= n <= MAX_ANSWER_LEN


def evaluate_heuristics(
    answer: str,
    retrieved_chunks: list[dict] | None,
) -> HeuristicReport:
    """Run all three checks. The combined report goes to Langfuse / debug."""
    refusal_ok = check_refusal(answer)
    citation_ok = check_citation(answer, retrieved_chunks)
    length_ok = check_length(answer)
    return HeuristicReport(
        refusal_check_passed=refusal_ok,
        citation_check_passed=citation_ok,
        length_check_passed=length_ok,
        overall_passed=refusal_ok and citation_ok and length_ok,
    )
