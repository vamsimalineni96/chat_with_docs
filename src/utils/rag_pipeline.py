# rag_pipeline.py
from typing import List, Dict, Any

from src.utils import config
from src.utils.services.milvus_store import MilvusStoreHandler
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
    collection_name: str,
    history: List[Dict],
) -> str:
    milvus_store = MilvusStoreHandler(collection_name=collection_name)
    nim_client = NIMClient()
    logger.info("Retrieving context from Milvus Db")
    retrieved = milvus_store.search_similar_chunks(question, top_k=config.TOP_K)
    if not retrieved:
        return "No relevant context found in the vector store."

    context = build_context(retrieved)
    # 2. Format history into a text block
    history_text = format_history_for_prompt(history, max_turns=6)
    
    return nim_client.chat_completion(
        history_text=history_text, question=question, context=context
    )
