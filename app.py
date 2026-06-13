import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace as otel_trace
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.utils import config
from src.utils.api.background_tasks import upload_pdf
from src.utils.api.schemas import (
    ApprovalRequest,
    ApprovalResponse,
    ChatRequest,
    ChatResponse,
    PendingApproval,
)
from src.utils.api.task_registry import create_task, get_status
from src.utils.chat.chat_service import cache_output, rag_output
from src.utils.db.database import Base, engine, get_db
from src.utils.db.database_debug import DBInspector
from src.utils.errors import (
    CacheError,
    ConversationOwnershipError,
    ConversationServiceError,
    EmbeddingError,
    InferenceError,
    MilvusError,
)
from src.utils.observability import observe, update_current_trace
from src.utils.services.approval_store import consume_token, create_token
from src.utils.services.conversation_store import get_conversation_service
from src.utils.services.embedder import EmbeddingHandler
from src.utils.services.logger_config import logger
from src.utils.services.milvus_store import MilvusStoreHandler, get_cache_store
from src.utils.services.redis_lock import ConversationLockError, get_redis_lock

_tracer = otel_trace.get_tracer(__name__)

conversation_service = get_conversation_service()
redis_service = get_redis_lock()
embedder = EmbeddingHandler()
app = FastAPI()


@observe(name="embed_question")
def embed_question(text: str) -> list[float]:
    return embedder.get_embedding(text=text, input_type="query")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=config.THREAD_POOL_MAX_WORKERS)

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
def on_startup():
    logger.info("Creating tables (if not exist)")
    Base.metadata.create_all(bind=engine)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Multi-user, multi-turn chat endpoint with per-conversation locking.

    A root OTel span ("chat_request") is started manually here so that
    embed_question and rag_output/cache_output all appear as child spans
    under a single Langfuse trace instead of as separate root traces.
    The @observe decorator is not used on the route itself because it
    interferes with FastAPI's response serialization.
    """
    conv_id: str | None = None
    lock_token: str | None = None
    _root_span_ctx = _tracer.start_as_current_span("chat_request")
    _root_span_ctx.__enter__()
    try:
        # 1) User + conversation setup
        logger.info("Fetching/Creating the user in the database")
        user = conversation_service.get_or_create_user(
            db, external_id=payload.user_external_id
        )

        if payload.conversation_id:
            logger.info("Fetching the conversation by conversation id")
            conversation = conversation_service.get_conversation_by_id(
                db,
                conversation_id=payload.conversation_id,
                user=user,
            )
            if conversation is None:
                logger.info("Conversation not found; creating a new one")
                conversation = conversation_service.create_conversation(
                    db,
                    user=user,
                    title=config.DEFAULT_CONVERSATION_TITLE,
                )
        else:
            logger.info(
                "No conversation id provided, creating a new conversation for user_id=%s",
                user.id,
            )
            conversation = conversation_service.create_conversation(
                db,
                user=user,
                title=config.DEFAULT_CONVERSATION_TITLE,
            )

        conv_id = str(conversation.id)
        update_current_trace(user_id=payload.user_external_id, session_id=conv_id)

        # 2) Locking
        try:
            logger.info(
                "Locking the conversation using Redis for conv_id=%s", conv_id
            )
            lock_token = redis_service.acquire_conversation_lock(
                conv_id, wait=False
            )
        except ConversationLockError as e:
            logger.warning(
                "Conversation lock error for conv_id=%s: %s", conv_id, e
            )
            raise HTTPException(
                status_code=409,
                detail="Another message is being processed for this conversation.",
            )

        # 3) Embeddings + Cache + RAG
        logger.info("Generating the embedding for question")
        try:
            q_embed = await asyncio.to_thread(
                embed_question,
                text=payload.question,
            )
        except EmbeddingError as e:
            logger.error("Embedding error in /chat for conv_id=%s: %s", conv_id, e)
            raise HTTPException(
                status_code=502,
                detail={
                    "error_type": "EMBEDDING_ERROR",
                    "message": str(e),
                },
            )

        if config.TOGGLE_CACHE and not payload.debug:
            logger.info("Searching the cache store for similar answer")
            try:
                cached_answer = await asyncio.to_thread(cache_output, payload, q_embed)
            except CacheError as e:
                logger.error("Cache error in /chat for conv_id=%s: %s", conv_id, e)
                # Treat cache failure as non-fatal: just fall back to RAG
                cached_answer = None

            if cached_answer:
                return ChatResponse(conversation_id=conv_id, answer=cached_answer)
            logger.info("Cache miss, routing to RAG for answering")

        if payload.debug:
            logger.info("Debug request — bypassing cache and forcing full RAG")
        else:
            logger.info("Using RAG to answer the question")

        # Multi-agent "both" path runs two LLM pipelines + aggregator —
        # inherently slower than single-agent paths. Default raised to 180s.
        timeout_s = int(os.environ.get("CHAT_TIMEOUT_SECONDS", "180"))
        try:
            rag_result = await asyncio.wait_for(
                rag_output(payload, db, conversation, user, query_vec=q_embed),
                timeout=timeout_s,
            )
        except TimeoutError:
            logger.error(
                "RAG pipeline timed out after %ds for conv_id=%s", timeout_s, conv_id
            )
            raise HTTPException(
                status_code=504,
                detail={
                    "error_type": "TIMEOUT_ERROR",
                    "message": "The request took too long to process. Please try again.",
                },
            )
        # HITL: if the action agent detected a destructive tool, pause and
        # return a pending_approval token the UI renders as Approve/Reject
        # (kind="approval") or as a candidate picker (kind="disambig").
        raw_approval = rag_result.get("pending_approval")
        pending: PendingApproval | None = None
        if raw_approval:
            kind = raw_approval.get("kind", "approval")
            # Store the specific tool and args/candidates so /approve can act
            # without re-running the ReAct loop (which is fragile and may not
            # rediscover the same payment intent on retry).
            paused_state = {
                "conversation_id": conv_id,
                "user_external_id": payload.user_external_id,
                "kind": kind,
                "tool": raw_approval["tool"],
                "display": raw_approval["display"],
                "args": raw_approval.get("args"),
                "candidates": raw_approval.get("candidates"),
            }
            token = create_token(paused_state)
            pending = PendingApproval(
                kind=kind,
                tool=raw_approval["tool"],
                display=raw_approval["display"],
                approval_token=token,
                args=raw_approval.get("args"),
                candidates=raw_approval.get("candidates"),
            )

        return ChatResponse(
            conversation_id=conv_id,
            answer=rag_result["answer"],
            debug=rag_result.get("debug"),
            pending_approval=pending,
        )

    except ConversationOwnershipError as e:
        logger.error(
            "Ownership error in /chat for conv_id=%s, user_external_id=%s: %s",
            conv_id,
            payload.user_external_id,
            e,
        )
        raise HTTPException(
            status_code=403,
            detail={
                "error_type": "CONVERSATION_OWNERSHIP_ERROR",
                "message": str(e),
            },
        )

    except ConversationServiceError as e:
        logger.exception(
            "Conversation service error in /chat for conv_id=%s: %s", conv_id, e
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "CONVERSATION_SERVICE_ERROR",
                "message": str(e),
            },
        )

    except InferenceError as e:
        logger.error(
            "Inference error in /chat for conv_id=%s: %s", conv_id, e, exc_info=e
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error_type": "INFERENCE_ERROR",
                "message": str(e),
            },
        )

    except (MilvusError, CacheError) as e:
        logger.exception("Vector store/cache error in /chat: %s", e)
        raise HTTPException(
            status_code=502,
            detail={
                "error_type": "VECTOR_STORE_ERROR",
                "message": str(e),
            },
        )

    except SQLAlchemyError as e:
        logger.exception("Raw SQLAlchemy error in /chat: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "DATABASE_ERROR",
                "message": "Database operation failed.",
            },
        )

    except Exception as e:
        logger.exception("Unexpected error while handling /chat: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while processing the request.",
        )

    finally:
        # Always try to release the lock if we got one
        if conv_id and lock_token:
            try:
                logger.info(
                    "Releasing the conversation lock for conv_id=%s", conv_id
                )
                redis_service.release_conversation_lock(conv_id, lock_token)
            except ConversationLockError as e:
                logger.warning(
                    "Failed to release conversation lock for conv_id=%s: %s",
                    conv_id,
                    e,
                )
        _root_span_ctx.__exit__(None, None, None)


@app.post("/approve", response_model=ApprovalResponse)
async def approve(
    payload: ApprovalRequest,
    db: Session = Depends(get_db),
):
    """Execute (or cancel) a pending destructive tool action.

    On approval: calls Stripe SDK directly (bypassing the agent's MCP tools).
    On rejection: stores a cancellation message and returns.

    Wrapped in a manual OTel root span so the whole approval flow shows up
    in Langfuse as its own trace, tagged so you can correlate with the
    original chat trace via session_id (conversation_id).
    """
    _approval_ctx = _tracer.start_as_current_span("hitl_approval")
    _approval_ctx.__enter__()
    try:
        paused = consume_token(payload.approval_token)
        if paused is None:
            raise HTTPException(
                status_code=410,
                detail={
                    "error_type": "APPROVAL_EXPIRED",
                    "message": "Approval request expired or already used.",
                },
            )

        if payload.decision not in ("approved", "rejected"):
            raise HTTPException(
                status_code=422,
                detail="decision must be 'approved' or 'rejected'",
            )

        kind = paused.get("kind", "approval")
        tool_name = paused["tool"]
        args = paused.get("args") or {}
        candidates = paused.get("candidates") or []

        # Resolve which payment_intent we're acting on:
        #   - approval flow: stored in args at pause time
        #   - disambig flow: user picked from candidates; must match one of them
        if kind == "disambig" and payload.decision == "approved":
            picked = payload.selected_payment_intent_id or ""
            valid_ids = {c.get("id") for c in candidates}
            if not picked or picked not in valid_ids:
                raise HTTPException(
                    status_code=422,
                    detail="selected_payment_intent_id must match one of the candidates",
                )
            pi_id = picked
            amount_cents = next(
                (c.get("amount_cents", 0) for c in candidates if c.get("id") == picked),
                0,
            )
        else:
            pi_id = args.get("payment_intent_id") or ""
            amount_cents = args.get("amount", 0)

        # Tag the trace so it's discoverable in Langfuse alongside the
        # original chat trace (same session_id = conversation_id).
        try:
            update_current_trace(
                name="hitl_approval",
                user_id=payload.user_external_id,
                session_id=str(payload.conversation_id),
                tags=[
                    f"hitl_decision:{payload.decision}",
                    f"hitl_tool:{tool_name}",
                    f"hitl_kind:{kind}",
                ],
                metadata={
                    "payment_intent_id": pi_id,
                    "amount_cents": amount_cents,
                    "approval_token_prefix": payload.approval_token[:8],
                    "candidate_count": len(candidates),
                },
            )
        except Exception as e:
            logger.debug("HITL approval trace tagging failed (non-fatal): %s", e)

        user = conversation_service.get_or_create_user(db, external_id=payload.user_external_id)
        conversation = conversation_service.get_conversation_by_id(
            db, conversation_id=payload.conversation_id, user=user
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        if payload.decision == "rejected":
            answer = "Understood — the refund has been cancelled."
            conversation_service.add_message(
                db, conversation=conversation, user=user, role="assistant", content=answer
            )
            return ApprovalResponse(
                conversation_id=str(payload.conversation_id),
                answer=answer,
            )

        # Approved — execute the refund directly via Stripe SDK in a traced span.
        reason = args.get("reason") if args else None
        if not pi_id:
            answer = "❌ Refund failed: no payment intent ID stored."
        else:
            answer = await asyncio.to_thread(
                _execute_approved_refund, pi_id, reason
            )

        conversation_service.add_message(
            db, conversation=conversation, user=user, role="assistant", content=answer
        )
        return ApprovalResponse(
            conversation_id=str(payload.conversation_id),
            answer=answer,
        )
    finally:
        _approval_ctx.__exit__(None, None, None)


@observe(name="stripe_create_refund")
def _execute_approved_refund(payment_intent_id: str, reason: str | None) -> str:
    """Issue the Stripe refund and format the user-facing answer.

    Separated into its own @observe'd function so the Stripe API call
    shows up as a nested span under hitl_approval in Langfuse.
    """
    import stripe  # noqa: PLC0415
    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")

    if not stripe.api_key:
        return "❌ Refund failed: Stripe key not configured on the app."

    try:
        params: dict[str, Any] = {"payment_intent": payment_intent_id}
        if reason:
            params["reason"] = reason
        refund = stripe.Refund.create(**params)
        logger.info("HITL approved refund: %s → %s", payment_intent_id, refund.id)

        # Attach the result to the span so it's visible in Langfuse
        try:
            from src.utils.observability import update_current_observation  # noqa: PLC0415
            update_current_observation(
                output={
                    "refund_id": refund.id,
                    "status": refund.status,
                    "amount_cents": refund.amount,
                },
            )
        except Exception:
            pass

        amount_str = f"${refund.amount / 100:.2f}"
        return (
            f"✅ Refund processed successfully.\n\n"
            f"- **Refund ID:** `{refund.id}`\n"
            f"- **Amount:** {amount_str}\n"
            f"- **Status:** {refund.status}"
        )
    except stripe.StripeError as e:
        logger.exception("HITL refund failed for %s: %s", payment_intent_id, e)
        return f"❌ Refund failed: {e.user_message or str(e)}"


@app.get("/list_conversations")
def list_conversations(user_external_id: str, db: Session = Depends(get_db)):
    """List all conversations for a given user (most-recently-updated first)."""
    try:
        user = conversation_service.get_or_create_user(db, external_id=user_external_id)
        convs = conversation_service.list_conversations(db, user=user)
        return {
            "user_external_id": user_external_id,
            "conversations": [
                {
                    "conversation_id": str(c.id),
                    "title": c.title,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                }
                for c in convs
            ],
        }
    except ConversationServiceError as e:
        logger.exception("Error listing conversations for user=%s: %s", user_external_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list_messages")
def list_messages(
    user_external_id: str,
    conversation_id: UUID,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    """List messages in a conversation (chronological), gated by user ownership."""
    try:
        user = conversation_service.get_or_create_user(db, external_id=user_external_id)
        conv = conversation_service.get_conversation_by_id(
            db, conversation_id=conversation_id, user=user
        )
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        msgs = conversation_service.get_recent_messages(
            db, conversation=conv, user=user, limit=limit
        )
        return {
            "conversation_id": str(conv.id),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "sequence_no": m.sequence_no,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in msgs
            ],
        }
    except ConversationOwnershipError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ConversationServiceError as e:
        logger.exception("Error listing messages: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_pdf_async")
async def upload_pdf_async(
    pdf_name: str,
    collection_name: str,
):
    """Kick off an async PDF ingest. All pages are indexed."""
    task_id = create_task()
    executor.submit(
        asyncio.run,
        upload_pdf(pdf_name, collection_name, task_id),
    )
    return {"message": "Processing started", "task_id": task_id}


@app.get("/task_status/{task_id}")
def check_task_status(task_id: str):
    return {"status": get_status(task_id)}

@app.post("/clear_post_gres")
async def clear_db():
    """
    Clear the postgres database.
    """
    try:
        with Session(engine) as session:
            for table in reversed(Base.metadata.sorted_tables):
                logger.info("Deleting from %s...", table.name)
                session.execute(table.delete())
            session.commit()
            logger.info("All tables cleared.")
    except SQLAlchemyError as e:
        logger.exception("Database error while clearing Postgres: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Database error while clearing Postgres.",
        )
    except Exception as e:
        logger.exception("Unexpected error while clearing Postgres: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while clearing Postgres.",
        )

    return {"message": "All tables are cleared"}


@app.post("/clear_cache")
async def clear_cache():
    try:
        cache_store = get_cache_store()
        cache_store.delete_collection()
    except CacheError as e:
        logger.exception("Error while clearing cache collection: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "CACHE_ERROR",
                "message": str(e),
            },
        )
    except Exception as e:
        logger.exception("Unexpected error while clearing cache: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while clearing cache.",
        )
    return {"message": "cleared the cache_store"}


@app.post("/debug_database")
async def debug_db(user_id: str = None, conv_id: UUID = None):
    """
    Print the database texts for debugging.
    """
    try:
        db_debugger = DBInspector()
        print("Printing users")
        db_debugger.print_users()
        print("Printing conversations")
        db_debugger.print_conversations(user_external_id=user_id)
        print(f"Printing Messages from conversation: {conv_id}")
        db_debugger.print_messages(conversation_id=conv_id)
    except Exception as e:
        logger.exception("Error while debugging database: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while debugging database.",
        )
    return {"message": "Details are printed"}


@app.post("/view_milvus_store")
async def view_store():
    try:
        milvus_store = MilvusStoreHandler()
        milvus_store.view_collection(collection_name=config.COLLECTION_NAME)
    except MilvusError as e:
        logger.exception("Error while viewing Milvus store: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "VECTOR_STORE_ERROR",
                "message": str(e),
            },
        )
    except Exception as e:
        logger.exception("Unexpected error while viewing Milvus store: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while viewing Milvus store.",
        )

    return {"message": "the details are printed"}

@app.post("/clear_milvus")
async def clear_milvus(name:str):
    """Enter the name of the collection you want to delete"""
    try:
        milvus_store = MilvusStoreHandler(collection_name= name)
        milvus_store.delete_collection()
    except MilvusError as e:
        logger.exception(f"Error while deleting Milvus collection: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "VECTOR_STORE_ERROR",
                "message": str(e),
            },
        )
    except Exception as e:
        logger.exception("Unexpected error while deleting Milvus collection: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while deleting Milvus collection.",
        )
    return {"message": f"Collection: {name} is dropped"}
