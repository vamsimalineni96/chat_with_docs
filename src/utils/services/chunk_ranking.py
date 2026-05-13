from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIARerank

from src.utils.config import RERANK_MODEL, NVIDIA_API_KEY
from src.utils.errors import RerankError
from src.utils.observability import observe, update_current_generation
from src.utils.services.logger_config import logger


class NVidiaReranker:
    """LangChain-backed wrapper around `NVIDIARerank`.

    Returns the same `[{text, score, source}, ...]` shape as before so
    `rag_pipeline.build_context` continues to work unchanged. The original
    Milvus similarity score is preserved as `score`; the rerank logit is
    stored under `rerank_score` for callers that want it.
    """

    def __init__(self):
        self.model = RERANK_MODEL
        self.reranker = NVIDIARerank(
            model=self.model,
            api_key=NVIDIA_API_KEY,
        )

    @observe(name="rerank", as_type="generation")
    def execute(
        self, question: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not question or not isinstance(question, str):
            raise RerankError("Query must be a non-empty string.")
        if not retrieved_chunks:
            raise RerankError("Retrieved chunks list is empty — cannot rerank.")

        documents = [
            Document(
                page_content=item["text"],
                metadata={
                    "score": item["score"],
                    "source": item["source"],
                    "_orig_index": idx,
                },
            )
            for idx, item in enumerate(retrieved_chunks)
        ]

        try:
            logger.info("Reranking the retrieved chunks via NVIDIARerank")
            reranked_docs = self.reranker.compress_documents(
                documents=documents, query=question
            )
        except Exception as e:
            logger.exception("Error calling NVIDIA rerank via LangChain: %s", e)
            raise RerankError(f"NVIDIA rerank call failed: {e}") from e

        if not reranked_docs:
            raise RerankError("Rerank API returned an empty ranking list.")

        reranked_chunks: List[Dict[str, Any]] = []
        for doc in reranked_docs:
            meta = doc.metadata or {}
            reranked_chunks.append(
                {
                    "text": doc.page_content,
                    "score": meta.get("score"),
                    "source": meta.get("source"),
                    "rerank_score": meta.get("relevance_score"),
                }
            )

        update_current_generation(
            model=self.model,
            input={"question": question, "input_count": len(retrieved_chunks)},
            output={
                "output_count": len(reranked_chunks),
                "top_rerank_scores": [c.get("rerank_score") for c in reranked_chunks[:5]],
            },
            usage_details={"input": len(retrieved_chunks), "total": len(retrieved_chunks)},
            metadata={"unit": "passages_reranked"},
        )
        return reranked_chunks
