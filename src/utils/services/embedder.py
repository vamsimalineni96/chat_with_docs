from typing import List, Tuple

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import NVIDIA_API_KEY, EMBED_MODEL
from src.utils.errors import EmbeddingError
from src.utils.observability import observe, update_current_generation
from src.utils.services.logger_config import logger


def _estimate_tokens(text: str) -> int:
    """Rough char-based token estimate. NVIDIA doesn't return usage on embed."""
    return max(1, len(text) // 4)


class EmbeddingHandler:
    """LangChain-backed wrapper around NVIDIA embeddings.

    Public API (`get_embedding`, `get_document_embeddings`) is preserved so
    callers in milvus_store / app.py / chat_service do not change. The
    underlying `NVIDIAEmbeddings` instance is exposed via `.embeddings` for
    use by `langchain-milvus.Milvus` as its `embedding_function`.

    `embed_query` is used when `input_type == "query"`; `embed_documents`
    otherwise. The NVIDIA endpoint distinguishes the two internally.
    """

    def __init__(self):
        logger.info("Initializing EmbeddingHandler (LangChain NVIDIAEmbeddings)...")
        self.embeddings = NVIDIAEmbeddings(
            model=EMBED_MODEL,
            api_key=NVIDIA_API_KEY,
            truncate="NONE",
        )

    @observe(as_type="embedding", name="embed_query")
    def get_embedding(self, text: str, input_type: str = "query") -> List[float]:
        try:
            if input_type == "passage":
                vector = self.embeddings.embed_documents([text])[0]
            else:
                vector = self.embeddings.embed_query(text)
            logger.info("Successfully retrieved embedding of length %d", len(vector))

            tokens = _estimate_tokens(text)
            update_current_generation(
                model=EMBED_MODEL,
                input=text,
                output={"embedding_dim": len(vector)},
                usage_details={"input": tokens, "total": tokens},
                metadata={"input_type": input_type},
            )
            return vector
        except Exception as e:
            logger.exception("Error while creating embedding via NVIDIAEmbeddings: %s", e)
            raise EmbeddingError("Failed to create embedding via NVIDIA endpoint.") from e

    @observe(as_type="embedding", name="embed_passage_batch")
    def get_document_embeddings(
        self, chunk_size: int, chunk_overlap: int, long_text: str
    ) -> Tuple[List[List[float]], List[str]]:
        logger.info(
            "Splitting text into chunks with size %d and overlap %d",
            chunk_size,
            chunk_overlap,
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ".", "!", "?", " "],
        )
        text_chunks = splitter.split_text(long_text)
        logger.info("Generated %d text chunks", len(text_chunks))

        try:
            vectors = self.embeddings.embed_documents(text_chunks)
        except Exception as e:
            logger.exception("Failed batch document embedding: %s", e)
            raise EmbeddingError("Failed to embed document chunks.") from e

        total_tokens = sum(_estimate_tokens(c) for c in text_chunks)
        update_current_generation(
            model=EMBED_MODEL,
            input={
                "chunk_count": len(text_chunks),
                "first_chunk_preview": (text_chunks[0] if text_chunks else "")[:200],
            },
            output={
                "vector_count": len(vectors),
                "embedding_dim": len(vectors[0]) if vectors else 0,
            },
            usage_details={"input": total_tokens, "total": total_tokens},
            metadata={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "input_type": "passage",
            },
        )

        logger.info("Completed generating embeddings for document.")
        return vectors, text_chunks
