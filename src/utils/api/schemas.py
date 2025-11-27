# app/api/schemas.py (or wherever you keep schemas)

from pydantic import BaseModel
from typing import Optional
from uuid import UUID

class ChatRequest(BaseModel):
    user_external_id: str          # some stable ID: email, auth ID, etc.
    question: str                   # the user's new message
    collection_name: str           # which Milvus collection to use
    conversation_id: Optional[UUID] = None  # null => start a new convo


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str

class InferenceResponse(BaseModel):
    response: str