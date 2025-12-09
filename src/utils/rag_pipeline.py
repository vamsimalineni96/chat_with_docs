import time
import json
from typing import List, Dict, Any

from pydantic import ValidationError

from src.utils import config
from src.utils.errors import InferenceError
from src.utils.services.milvus_store import MilvusStoreHandler, get_cache_store
from src.utils.services.inference import NIMClient
from src.utils.services.logger_config import logger
from src.utils.services.chunk_ranking import NVidiaReranker


def build_context(chunks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Chunk {i} | score={c['score']:.4f} | source={c['source']}]\n{c['text']}\n"
        )
    return "\n\n".join(parts)


def format_history_for_prompt(history: List[Dict], max_turns: int = 6) -> str:
    """
    Turn last `max_turns` messages into a readable dialogue block.
    """
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


def answer_question(
    question: str,
    query_vec: List[float],
    collection_name: str,
    history: List[Dict],
):
    """
    Main RAG orchestration for answering a question.

    Raises:
        InferenceError: for any Milvus/LLM issues that should surface to app layer.
    """
    milvus_store = MilvusStoreHandler(collection_name=collection_name)
    cache_store = get_cache_store()
    nim_client = NIMClient()
    nim_reranker = NVidiaReranker()

    logger.info("Retrieving context from Milvus DB")
    try:
        t_milvus_start = time.perf_counter()
        retrieved = milvus_store.search_similar_chunks(
            query_vec=query_vec, top_k=config.TOP_K
        )
        t_milvus_end = time.perf_counter()
    except Exception as e:
        logger.exception("Failed to retrieve context from Milvus: %s", e)
        raise InferenceError("Failed to retrieve context from Milvus.") from e

    if not retrieved:
        logger.info("No relevant context found in the vector store.")
        return (
            "No relevant context found in the vector store.",
            t_milvus_start,
            t_milvus_end,
            t_milvus_start,
            t_milvus_end,
        )
    
    try:
        reranked= nim_reranker.execute(question= question, retrieved_chunks= retrieved)
    except Exception as e:
        logger.exception(f"Failed to rerank the chunks: {e}")
        raise InferenceError("Failed to rerank the chunks") from e 

    logger.info("Building the context from the reranked chunks")
    context = build_context(reranked)
    history_text = format_history_for_prompt(history, max_turns=6)

    try:
        t_llm_start = time.perf_counter()
        answer = nim_client.chat_completion(
            history_text=history_text, question=question, context=context
        )
        t_llm_end = time.perf_counter()
    except InferenceError:
        # Already wrapped correctly
        raise
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error("Invalid model output: %s", e)
        raise InferenceError("LLM output validation failed.") from e
    except Exception as e:
        logger.exception("Unexpected error while generating answer: %s", e)
        raise InferenceError("Unexpected error during answer generation.") from e

    context_chunk_ids = [item.get("id") for item in retrieved]

    # If you want to re-enable cache writes, they should be wrapped similarly:
    # try:
    # logger.info("Storing the conversation in the rag cache")
    # cache_store.put_entry(
    #     question_text=question,
    #     query_vec=query_vec,
    #     answer_text=answer,
    #     context_chunk_ids=context_chunk_ids,
    #     model_name=config.LLM_MODEL,
    #     prompt_version=config.PROMPT_VERSION,
    # )
    # except Exception as e:
    #     logger.exception("Failed to write to cache: %s", e)
    #     # non-fatal; don't raise

    return answer, t_milvus_start, t_milvus_end, t_llm_start, t_llm_end
