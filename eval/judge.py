"""LLM-as-judge for the eval harness — landing in sub-PR #7b.

Will provide a single `judge(question, answer, retrieved_context,
expected_keywords)` function that calls the eval judge model
(`EVAL_JUDGE_MODEL`, default `meta/llama-3.3-70b-instruct`) via the
NVIDIA NIM endpoint with a decomposed rubric:

    - groundedness (1–5): is the answer supported by the retrieved context?
    - accuracy    (1–5): is the information correct vs the expected facts?
    - completeness (1–5): does the answer cover the expected keywords?

Returns a dict with the three scores and a one-sentence rationale. Includes
retry-on-parse-failure since LLM JSON output isn't always clean.

Generator (Gemma) and judge (Llama) are deliberately from different model
families to avoid self-preference bias — see docs/OBSERVABILITY.md §3.3
and docs/PROGRESS.md.
"""
