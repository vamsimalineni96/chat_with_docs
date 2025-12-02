# rag_pipeline.py
from typing import List, Dict, Any

from src.utils import config
from src.utils.services.milvus_store import MilvusStoreHandler, get_cache_store
from src.utils.services.inference import NIMClient
from src.utils.services.logger_config import logger


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
    history: list of {"role": "user"|"assistant"|"system", "content": "..."}
    """
    if not history:
        return "None"

    # Keep only last `max_turns` messages
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
) -> str:
    milvus_store = MilvusStoreHandler(collection_name=collection_name)
    cache_store = get_cache_store()
    nim_client = NIMClient()

    logger.info("Retrieving context from Milvus Db")
    retrieved = milvus_store.search_similar_chunks(
        query_vec=query_vec, top_k=config.TOP_K
    )
    if not retrieved:
        return "No relevant context found in the vector store."

    context = build_context(retrieved)

    # Format history into a text block
    history_text = format_history_for_prompt(history, max_turns=6)

    answer = nim_client.chat_completion(
        history_text=history_text, question=question, context=context
    )

    context_chunk_ids = [item.get("id") for item in retrieved]

    logger.info("Storing the conversation in the rag cache")
    cache_store.put_entry(
        question_text=question,
        query_vec=query_vec,
        answer_text=answer,
        context_chunk_ids=context_chunk_ids,
        model_name=config.LLM_MODEL,
        prompt_version=config.PROMPT_VERSION,
    )
    return answer
