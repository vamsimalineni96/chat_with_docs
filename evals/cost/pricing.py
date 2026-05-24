"""Model -> USD pricing table used by the cost-report aggregator.

These are synthetic placeholders modeled on public NVIDIA NIM and reference
pricing for similar OSS models. Replace with the per-token rates from your
actual NVIDIA contract / DGX Cloud invoice before relying on the report's
absolute numbers. Relative attribution (which task type / stage drives spend)
remains useful even when the absolute prices are stale.

Rerank is priced per *call* on NVIDIA's hosted endpoint, not per token.
We approximate by attributing 1 cent per rerank invocation; substitute the
real per-invocation price when known.
"""

# Price-per-1K-tokens, separated by direction (input vs output).
MODEL_PRICES: dict[str, dict[str, float]] = {
    # LLMs
    "meta/llama-3.1-70b-instruct": {
        "input_per_1k_usd": 0.00060,
        "output_per_1k_usd": 0.00060,
    },
    "meta/llama-3.1-8b-instruct": {
        "input_per_1k_usd": 0.00020,
        "output_per_1k_usd": 0.00020,
    },
    "meta/llama-3.3-70b-instruct": {
        "input_per_1k_usd": 0.00060,
        "output_per_1k_usd": 0.00060,
    },
    "google/gemma-4-31b-it": {
        # Placeholder — Gemma pricing on NVIDIA NIM not published. Approximated
        # against similarly-sized Llama variants. Replace when contract rates
        # are available.
        "input_per_1k_usd": 0.00020,
        "output_per_1k_usd": 0.00020,
    },
    # Embeddings
    "nvidia/llama-3.2-nv-embedqa-1b-v2": {
        "input_per_1k_usd": 0.00010,
        "output_per_1k_usd": 0.00000,
    },
    "nvidia/nv-embedqa-mistral-7b-v2": {
        "input_per_1k_usd": 0.00012,
        "output_per_1k_usd": 0.00000,
    },
}

# Rerank pricing is per-call (the "usage" we record is passage count, not
# tokens), so it's modeled separately. NVIDIA's hosted rerank is cheap
# per-call relative to LLM completions; treat these as ~order-of-magnitude
# estimates until contract rates are plugged in.
RERANK_PRICE_PER_CALL_USD: dict[str, float] = {
    "nvidia/nv-rerankqa-mistral-4b-v3": 0.00010,
    "nvidia/llama-3.2-nv-rerankqa-1b-v2": 0.00005,
}

DEFAULT_RERANK_PRICE_PER_CALL_USD: float = 0.00010
