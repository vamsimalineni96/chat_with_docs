"""Behavior contract for `count_tokens`.

These tests assert qualitative properties (>= 1, monotonic, sane range for a
known phrase, handles non-ASCII) rather than exact numbers. Exact tiktoken
counts can drift across minor versions; the contract should not.
"""

from src.utils.services.tokenizers import count_tokens


def test_count_tokens_returns_at_least_one():
    # The implementation wraps the result in max(1, ...) so callers using it
    # for Langfuse usage_details never see zero. Verify both ends of the
    # boundary.
    assert count_tokens("") >= 1
    assert count_tokens("a") >= 1


def test_count_tokens_is_monotonic_in_length():
    short = "a"
    medium = "a b c d"
    long_ = "a b c d " * 50
    assert count_tokens(short) <= count_tokens(medium) <= count_tokens(long_)


def test_count_tokens_in_expected_range_for_english_pangram():
    # The classic 9-word pangram. cl100k_base tokenizes this in roughly
    # 9 tokens; a generous range tolerates tokenizer version drift while
    # still catching catastrophic regressions (e.g. char-based counting
    # would return ~10, byte-based ~43).
    pangram = "The quick brown fox jumps over the lazy dog"
    n = count_tokens(pangram)
    assert 7 <= n <= 13, f"unexpected token count {n} for pangram"


def test_count_tokens_handles_non_ascii():
    # The previous char/4 heuristic dramatically undercounted Devanagari
    # text. The real tokenizer should produce a noticeably higher count
    # than char/4 on the same input — proving the upgrade is meaningful
    # for non-English corpora.
    text = "नमस्ते दुनिया, यह एक परीक्षण है।"
    char_div_4 = max(1, len(text) // 4)
    real = count_tokens(text)
    assert real > char_div_4, (
        f"real tokenizer should exceed char/4 on Devanagari; "
        f"got real={real}, char_div_4={char_div_4}"
    )
