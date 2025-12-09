# errors.py

class AppError(Exception):
    """Base class for all custom application exceptions."""
    pass


class InferenceError(AppError):
    """Raised when the LLM/NIM inference fails in a way the client should know about."""
    pass


class DatabaseError(AppError):
    """Generic database-related failure."""
    pass


class ConversationServiceError(DatabaseError):
    """Failures in conversation/message management."""
    pass


class ConversationOwnershipError(ConversationServiceError):
    """User tried to access/modify a conversation they don't own."""
    pass


class MilvusError(AppError):
    """Vector store / Milvus-related failures."""
    pass


class CacheError(AppError):
    """Semantic cache-related failures (Milvus cache collection)."""
    pass


class EmbeddingError(AppError):
    """Embedding creation or document-embedding failures."""
    pass


class PDFParseError(AppError):
    """PDF parsing / text extraction failures."""
    pass


class RedisLockError(AppError):
    """Redis connection / locking failures."""
    pass


class RAGPipelineError(AppError):
    """High-level RAG pipeline failures."""
    pass

class ConversationLockError(RedisLockError):
    """Raised when lock acquisition/release fails in a conversation-safe way."""
    pass

class RerankError(AppError):
    """Raised when there is an error in reranking"""
    pass