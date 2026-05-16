import time
from typing import Any

from langchain_core.runnables import RunnableLambda

from src.utils import config
from src.utils.errors import InferenceError
from src.utils.observability import observe, update_current_observation
from src.utils.services.chunk_ranking import NVidiaReranker
from src.utils.services.inference import NIMClient
from src.utils.services.logger_config import logger
from src.utils.services.milvus_store import MilvusStoreHandler, get_cache_store


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


def build_generation_chain(reranker: NVidiaReranker, llm: NIMClient):
    """
    LCEL sub-chain that runs *after* retrieval:
      input: {question, retrieved, history, _timings, _debug?}
      rerank -> slice top_k -> assemble (context + history_text) -> chat completion
    """

    def rerank(payload: dict[str, Any]) -> dict[str, Any]:
        retrieved = payload["retrieved"]
        reranked = reranker.execute(question=payload["question"], retrieved_chunks=retrieved)
        # Slice down to TOP_K for the LLM context.
        sliced = reranked[: config.TOP_K]
        debug = payload.get("_debug")
        if debug is not None:
            debug["reranked_chunks"] = reranked
            debug["reranked_top_k"] = sliced
        return {**payload, "reranked": sliced}

    def assemble(payload: dict[str, Any]) -> dict[str, Any]:
        context = build_context(payload["reranked"])
        history_text = payload.get("history_text") or format_history_for_prompt(
            payload["history"], max_turns=config.HISTORY_MAX_TURNS
        )
        debug = payload.get("_debug")
        if debug is not None:
            debug["history_text"] = history_text
        return {**payload, "context": context, "history_text": history_text}

    def generate(payload: dict[str, Any]) -> str:
        timings = payload["_timings"]
        debug = payload.get("_debug")
        if debug is not None:
            try:
                rendered = llm._prompt.format_messages(
                    history_text=payload["history_text"],
                    question=payload["question"],
                    context=payload["context"],
                )
                debug["rendered_prompt"] = [
                    {"role": getattr(m, "type", "unknown"), "content": m.content}
                    for m in rendered
                ]
            except Exception as e:
                logger.warning("Failed to capture rendered prompt for debug: %s", e)
                debug["rendered_prompt"] = []

        timings["t_llm_start"] = time.perf_counter()
        answer = llm.chat_completion(
            history_text=payload["history_text"],
            question=payload["question"],
            context=payload["context"],
        )
        timings["t_llm_end"] = time.perf_counter()
        return answer

    return (
        RunnableLambda(rerank)
        | RunnableLambda(assemble)
        | RunnableLambda(generate)
    )


@observe(name="answer_question")
def answer_question(
    question: str,
    query_vec: list[float],
    collection_name: str,
    history: list[dict],
    debug: bool = False,
) -> dict[str, Any]:
    """
    Main RAG orchestration.

    Pipeline:
      1. Hybrid (dense + BM25) retrieval for the original question, RETRIEVE_K chunks.
      2. Rerank chunks against the question.
      3. Slice to TOP_K and assemble prompt.
      4. Call the LLM.

    Returns a dict with answer, stage timings, and (when debug=True) intermediate
    artifacts.

    Raises:
        InferenceError: for retrieval/rerank/LLM failures.
    """
    milvus_store = MilvusStoreHandler(collection_name=collection_name)
    nim_client = NIMClient()
    nim_reranker = NVidiaReranker()

    debug_info: dict[str, Any] | None = {} if debug else None
    history_text = format_history_for_prompt(history, max_turns=config.HISTORY_MAX_TURNS)

    logger.info("Retrieving context from Milvus DB (hybrid dense + BM25)")
    try:
        t_milvus_start = time.perf_counter()
        retrieved = _retrieve_for_query(
            milvus_store, question, top_k=config.RETRIEVE_K
        )
        t_milvus_end = time.perf_counter()
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

    if not retrieved:
        logger.info("No relevant context found in the vector store.")
        return {
            "answer": "I couldn't find anything in the indexed document that touches on that. "
                      "Could you try rephrasing, or asking about a different topic from the book?",
            "t_milvus_start": t_milvus_start,
            "t_milvus_end": t_milvus_end,
            "t_llm_start": t_milvus_start,
            "t_llm_end": t_milvus_end,
            "debug": debug_info,
        }

    chain = build_generation_chain(nim_reranker, nim_client)
    timings: dict[str, float] = {}
    chain_input: dict[str, Any] = {
        "question": question,
        "retrieved": retrieved,
        "history": history,
        "history_text": history_text,
        "_timings": timings,
    }
    if debug_info is not None:
        chain_input["_debug"] = debug_info

    try:
        logger.info("Invoking generation chain (rerank -> prompt -> LLM)")
        answer = chain.invoke(chain_input)
    except InferenceError:
        raise
    except Exception as e:
        logger.exception("Unexpected error from generation chain: %s", e)
        raise InferenceError("Unexpected error from generation chain.") from e

    t_llm_start = timings.get("t_llm_start", t_milvus_end)
    t_llm_end = timings.get("t_llm_end", t_llm_start)

    if config.TOGGLE_CACHE and not debug:
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
        "t_milvus_start": t_milvus_start,
        "t_milvus_end": t_milvus_end,
        "t_llm_start": t_llm_start,
        "t_llm_end": t_llm_end,
        "debug": debug_info,
    }
