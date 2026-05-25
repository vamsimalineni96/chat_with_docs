"""RAG pipeline building blocks.

Each major stage (retrieve, rerank, generate) is exposed as a
top-level `@observe`-decorated function so the LangGraph nodes in
`src/agents/graph.py` can wire them together. Each one creates the
clients it needs (NIM, reranker, Milvus) per call — matching the
previous behavior — and writes its diagnostic data into the optional
`debug_info` dict when the request was made with `debug=True`.

The previous `answer_question` orchestrator and the LCEL
`build_generation_chain` are gone; their work now lives in the
LangGraph state machine. The pure helpers (`build_context`,
`format_history_for_prompt`, the inner hybrid-retrieve call) stay
here because they belong to the RAG pipeline regardless of how it's
orchestrated.
"""

import time
from typing import Any

from src.utils import config
from src.utils.errors import InferenceError
from src.utils.observability import observe, update_current_observation
from src.utils.services.chunk_ranking import NVidiaReranker
from src.utils.services.heuristics import evaluate_heuristics
from src.utils.services.inference import NIMClient
from src.utils.services.logger_config import logger
from src.utils.services.milvus_store import MilvusStoreHandler, get_cache_store

# Surfaced to the caller (and the graph) when retrieval returns
# nothing. Module constant rather than inline string so tests and
# downstream tooling can pin against it.
CANNED_NO_RETRIEVAL_ANSWER = (
    "I couldn't find anything in the indexed document that touches on that. "
    "Could you try rephrasing, or asking about a different topic from the book?"
)


def build_context(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        score = c.get("score")
        score_str = f"{score:.4f}" if isinstance(score, int | float) else "n/a"
        parts.append(
            f"[Chunk {i} | score={score_str} | source={c.get('source')}]\n{c['text']}\n"
        )
    return "\n\n".join(parts)


def format_history_for_prompt(history: list[dict], max_turns: int = config.HISTORY_MAX_TURNS) -> str:
    if not history:
        return "None"

    trimmed = history[-max_turns:]
    lines = []
    for msg in trimmed:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "Assistant"
        elif role == "system":
            prefix = "System"
        else:
            prefix = role or "Unknown"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


@observe(name="hybrid_retrieve", as_type="span")
def _retrieve_for_query(
    milvus_store: MilvusStoreHandler,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Hybrid (dense + BM25) retrieval for a single query string."""
    results = milvus_store.search_similar_chunks(query=query, top_k=top_k)
    update_current_observation(
        input={"query": query, "top_k": top_k},
        output={
            "count": len(results),
            "top_chunks": [
                {
                    "score": r.get("score"),
                    "source": r.get("source"),
                    "chunk_order": r.get("chunk_order"),
                    "text_preview": (r.get("text") or "")[:200],
                }
                for r in results[:5]
            ],
        },
    )
    return results


@observe(name="retrieve")
def retrieve_chunks(
    question: str,
    collection_name: str,
    *,
    debug_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 1 — hybrid retrieval, plus dense/sparse diagnostics in debug mode.

    Returns a dict the caller (graph node) merges into state:
      retrieved          : list of chunk dicts (may be empty)
      t_milvus_start/end : wall-clock timings for the LLM latency dashboard
    """
    milvus_store = MilvusStoreHandler(collection_name=collection_name)

    logger.info("Retrieving context from Milvus DB (hybrid dense + BM25)")
    try:
        t_start = time.perf_counter()
        retrieved = _retrieve_for_query(
            milvus_store, question, top_k=config.RETRIEVE_K
        )
        t_end = time.perf_counter()
    except Exception as e:
        logger.exception("Failed to retrieve context from Milvus: %s", e)
        raise InferenceError("Failed to retrieve context from Milvus.") from e

    if debug_info is not None:
        debug_info["retrieved_chunks"] = retrieved
        # Hybrid-retrieval diagnostic: re-run dense-only and BM25-only searches
        # so the UI can show what each component contributes vs. the fused list.
        try:
            debug_info["dense_only_chunks"] = milvus_store.search_dense_only(
                query=question, top_k=config.RETRIEVE_K
            )
        except Exception as e:
            logger.warning("Dense-only diagnostic search failed (non-fatal): %s", e)
            debug_info["dense_only_chunks"] = []
        try:
            debug_info["sparse_only_chunks"] = milvus_store.search_sparse_only(
                query=question, top_k=config.RETRIEVE_K
            )
        except Exception as e:
            logger.warning("Sparse-only diagnostic search failed (non-fatal): %s", e)
            debug_info["sparse_only_chunks"] = []

    return {
        "retrieved": retrieved,
        "t_milvus_start": t_start,
        "t_milvus_end": t_end,
    }


@observe(name="rerank")
def rerank_chunks(
    question: str,
    retrieved: list[dict[str, Any]],
    *,
    debug_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 2 — rerank against the question, slice down to TOP_K.

    Returns:
      top_chunks       : the TOP_K-sliced reranked chunks (LLM context)
      t_rerank_start/end
    """
    reranker = NVidiaReranker()
    try:
        t_start = time.perf_counter()
        reranked = reranker.execute(question=question, retrieved_chunks=retrieved)
        t_end = time.perf_counter()
    except InferenceError:
        raise
    except Exception as e:
        logger.exception("Unexpected error during rerank: %s", e)
        raise InferenceError("Unexpected error during rerank.") from e

    sliced = reranked[: config.TOP_K]
    if debug_info is not None:
        debug_info["reranked_chunks"] = reranked
        debug_info["reranked_top_k"] = sliced

    return {
        "top_chunks": sliced,
        "t_rerank_start": t_start,
        "t_rerank_end": t_end,
    }


@observe(name="generate")
def generate_answer(
    question: str,
    retrieved: list[dict[str, Any]],
    top_chunks: list[dict[str, Any]],
    history: list[dict[str, Any]],
    query_vec: list[float],
    *,
    debug_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 3 — assemble prompt, invoke LLM, write cache.

    Cache write is skipped when `debug_info is not None` (i.e. in
    debug mode) to avoid polluting the cache with intermediate
    inspection runs. Uses the pre-rerank `retrieved` chunk ids for
    cache metadata (matching pre-LangGraph behavior).
    """
    llm = NIMClient()
    history_text = format_history_for_prompt(history, max_turns=config.HISTORY_MAX_TURNS)
    context = build_context(top_chunks)

    if debug_info is not None:
        debug_info["history_text"] = history_text
        try:
            rendered = llm._prompt.format_messages(
                history_text=history_text,
                question=question,
                context=context,
            )
            debug_info["rendered_prompt"] = [
                {"role": getattr(m, "type", "unknown"), "content": m.content}
                for m in rendered
            ]
        except Exception as e:
            logger.warning("Failed to capture rendered prompt for debug: %s", e)
            debug_info["rendered_prompt"] = []

    try:
        t_start = time.perf_counter()
        answer = llm.chat_completion(
            history_text=history_text,
            question=question,
            context=context,
        )
        t_end = time.perf_counter()
    except InferenceError:
        raise
    except Exception as e:
        logger.exception("Unexpected error during generation: %s", e)
        raise InferenceError("Unexpected error during generation.") from e

    if config.TOGGLE_CACHE and debug_info is None:
        try:
            context_chunk_ids = [item.get("id") for item in retrieved if item.get("id")]
            get_cache_store().put_entry(
                question_text=question,
                query_vec=query_vec,
                answer_text=answer,
                context_chunk_ids=context_chunk_ids,
                model_name=config.LLM_MODEL,
                prompt_version=config.PROMPT_VERSION,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )
            logger.info("Stored Q/A pair in semantic cache.")
        except Exception as e:
            logger.exception("Cache write failed (non-fatal): %s", e)

    return {
        "answer": answer,
        "t_llm_start": t_start,
        "t_llm_end": t_end,
    }


def compute_heuristics_for_answer(
    answer: str,
    *,
    retrieved_chunks: list[dict[str, Any]],
    debug_info: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Evaluate heuristics on `answer` and return the report as a dict.

    The graph's postprocess node calls this. Tagging the Langfuse
    trace from a child observation tags the child instead of the
    trace root — so this function returns the report and the trace
    root (`chat_service.rag_output`) does the actual tagging.

    Failures never propagate — heuristics are observability, not
    enforcement. A broken regex must not break the request path.
    """
    try:
        report = evaluate_heuristics(answer, retrieved_chunks)
    except Exception as e:
        logger.warning("Heuristic evaluation failed (non-fatal): %s", e)
        return None

    report_dict = report.to_dict()
    if debug_info is not None:
        debug_info["heuristics"] = report_dict
    return report_dict
