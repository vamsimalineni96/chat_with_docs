# app/api/schemas.py (or wherever you keep schemas)

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import UUID

class ChatRequest(BaseModel):
    user_external_id: str          # some stable ID: email, auth ID, etc.
    question: str                   # the user's new message
    collection_name: str           # which Milvus collection to use
    conversation_id: Optional[UUID] = None  # null => start a new convo
    debug: bool = False            # when True, bypass cache and echo retrieval/prompt back


class ChatDebug(BaseModel):
    retrieved_chunks: List[Dict[str, Any]] = []
    dense_only_chunks: List[Dict[str, Any]] = []
    sparse_only_chunks: List[Dict[str, Any]] = []
    reranked_chunks: List[Dict[str, Any]] = []
    reranked_top_k: List[Dict[str, Any]] = []
    rendered_prompt: List[Dict[str, str]] = []
    history_text: Optional[str] = None
    timings_ms: Dict[str, float] = {}


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    debug: Optional[ChatDebug] = None


class InferenceResponse(BaseModel):
    response: str