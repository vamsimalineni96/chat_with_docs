"""Research agent — wraps the full RAG pipeline into one callable.

Used by the multi-agent graph so the "both" parallel node can run
research + action concurrently with asyncio.gather. Mirrors the shape
of run_tool_agent so both agents return the same dict structure.

Why a wrapper instead of re-using the existing graph nodes directly:
the existing nodes are designed to be called sequentially inside the
LangGraph state machine. For parallel execution we need a plain async
function we can hand to asyncio.gather.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.utils.observability import observe

logger = logging.getLogger(__name__)

RESEARCH_FAILURE_ANSWER = (
    "I tried to look that up in the documents, but something went wrong. "
    "Please try again in a moment."
)

RESEARCH_NO_RETRIEVAL_ANSWER = (
    "I couldn't find anything in the indexed documents that covers that. "
    "Try rephrasing, or upload a document that covers the topic."
)


@observe(name="research_agent")
async def run_research_agent(
    question: str,
    query_vec: list[float],
    collection_name: str,
    history: list[dict[str, Any]],
    debug_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full RAG pipeline and return a node-shaped result dict.

    Returns:
        {
          "answer": str,
          "retrieved": list[dict],
          "t_milvus_start": float,
          "t_milvus_end": float,
          "t_llm_start": float,
          "t_llm_end": float,
          "error": str | None,
        }

    Never raises — errors are caught and returned as RESEARCH_FAILURE_ANSWER.
    """
    import asyncio  # noqa: PLC0415

    from src.utils.rag_pipeline import (  # noqa: PLC0415
        generate_answer,
        retrieve_chunks,
        rerank_chunks,
    )

    t_start = time.perf_counter()
    try:
        retrieved_result = await asyncio.to_thread(
            retrieve_chunks,
            question=question,
            collection_name=collection_name,
            debug_info=debug_info,
        )

        retrieved = retrieved_result.get("retrieved", [])
        t_milvus_start = retrieved_result.get("t_milvus_start", t_start)
        t_milvus_end = retrieved_result.get("t_milvus_end", t_start)

        if not retrieved:
            return {
                "answer": RESEARCH_NO_RETRIEVAL_ANSWER,
                "retrieved": [],
                "t_milvus_start": t_milvus_start,
                "t_milvus_end": t_milvus_end,
                "t_llm_start": t_milvus_end,
                "t_llm_end": t_milvus_end,
                "error": None,
            }

        rerank_result = await asyncio.to_thread(
            rerank_chunks,
            question=question,
            retrieved=retrieved,
            debug_info=debug_info,
        )
        top_chunks = rerank_result.get("top_chunks", retrieved)

        t_llm_start = time.perf_counter()
        generate_result = await asyncio.to_thread(
            generate_answer,
            question=question,
            retrieved=retrieved,
            top_chunks=top_chunks,
            history=history,
            query_vec=query_vec,
            debug_info=debug_info,
        )
        t_llm_end = time.perf_counter()

        return {
            "answer": generate_result.get("answer", RESEARCH_FAILURE_ANSWER),
            "retrieved": retrieved,
            "t_milvus_start": t_milvus_start,
            "t_milvus_end": t_milvus_end,
            "t_llm_start": generate_result.get("t_llm_start", t_llm_start),
            "t_llm_end": generate_result.get("t_llm_end", t_llm_end),
            "error": None,
        }

    except Exception as e:
        logger.warning("Research agent failed: %s", e, exc_info=True)
        t_end = time.perf_counter()
        return {
            "answer": RESEARCH_FAILURE_ANSWER,
            "retrieved": [],
            "t_milvus_start": t_start,
            "t_milvus_end": t_start,
            "t_llm_start": t_start,
            "t_llm_end": t_end,
            "error": str(e),
        }
