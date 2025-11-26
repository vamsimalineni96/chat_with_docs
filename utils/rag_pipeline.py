# rag_pipeline.py
from typing import List, Dict, Any

import utils.config as config
from utils.milvus_store import MilvusStoreHandler
from utils.nim_client import nim_chat_completion



def build_context(chunks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Chunk {i} | score={c['score']:.4f} | source={c['source']}]\n{c['text']}\n"
        )
    return "\n\n".join(parts)


def answer_question(question: str, collection_name: str) -> str:
    milvus_store= MilvusStoreHandler(collection_name= collection_name)
    retrieved = milvus_store.search_similar_chunks(question, top_k=config.TOP_K)
    if not retrieved:
        return "No relevant context found in the vector store."

    context = build_context(retrieved)

    system_prompt = (
        "You are a helpful assistant that answers questions **only** using the provided context. "
        "If the answer is not clearly supported, say you don't know.\n"
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context from Milvus (NIM embeddings):\n{context}\n\n"
        "Now give a concise answer."
    )

    return nim_chat_completion(system_prompt, user_prompt)
