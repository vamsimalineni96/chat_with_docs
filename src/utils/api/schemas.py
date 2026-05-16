# app/api/schemas.py (or wherever you keep schemas)

from typing import Any
from uuid import UUID

from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_external_id: str          # some stable ID: email, auth ID, etc.
    question: str                   # the user's new message
    collection_name: str           # which Milvus collection to use
    conversation_id: UUID | None = None  # null => start a new convo
    debug: bool = False            # when True, bypass cache and echo retrieval/prompt back


class ChatDebug(BaseModel):
    retrieved_chunks: list[dict[str, Any]] = []
    dense_only_chunks: list[dict[str, Any]] = []
    sparse_only_chunks: list[dict[str, Any]] = []
    reranked_chunks: list[dict[str, Any]] = []
    reranked_top_k: list[dict[str, Any]] = []
    rendered_prompt: list[dict[str, str]] = []
    history_text: str | None = None
    timings_ms: dict[str, float] = {}


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    debug: ChatDebug | None = None


class InferenceResponse(BaseModel):
    response: str
