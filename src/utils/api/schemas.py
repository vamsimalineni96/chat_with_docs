from typing import Any
from uuid import UUID

from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_external_id: str
    question: str
    collection_name: str
    conversation_id: UUID | None = None
    debug: bool = False


class ChatDebug(BaseModel):
    retrieved_chunks: list[dict[str, Any]] = []
    dense_only_chunks: list[dict[str, Any]] = []
    sparse_only_chunks: list[dict[str, Any]] = []
    reranked_chunks: list[dict[str, Any]] = []
    reranked_top_k: list[dict[str, Any]] = []
    rendered_prompt: list[dict[str, str]] = []
    history_text: str | None = None
    timings_ms: dict[str, float] = {}


class PendingApproval(BaseModel):
    """Embedded in ChatResponse when the agent needs human input.

    Two kinds:
      - "approval": user clicks Approve/Reject for a single pre-identified payment.
                    `args` contains payment_intent_id + amount.
      - "disambig": user picks one payment from `candidates`. Selecting IS the
                    approval — the chosen pi_id flows into /approve directly.
    """
    kind: str = "approval"              # "approval" | "disambig"
    tool: str
    display: str
    approval_token: str
    args: dict[str, Any] | None = None  # set for kind="approval"
    candidates: list[dict[str, Any]] | None = None  # set for kind="disambig"


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    debug: ChatDebug | None = None
    pending_approval: PendingApproval | None = None


class InferenceResponse(BaseModel):
    response: str


class ApprovalRequest(BaseModel):
    """Sent by the UI when user clicks Approve, Reject, or picks a candidate."""
    approval_token: str
    decision: str                # "approved" | "rejected"
    user_external_id: str
    conversation_id: str
    # For disambig flow: the pi_id the user picked from the candidate list.
    # When set, this is treated as informed consent and the refund executes
    # against this id (overrides whatever args were stored at pause time).
    selected_payment_intent_id: str | None = None


class ApprovalResponse(BaseModel):
    conversation_id: str
    answer: str
