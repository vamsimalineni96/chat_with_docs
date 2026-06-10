import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace as otel_trace
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.utils import config
from src.utils.api.background_tasks import upload_pdf
from src.utils.api.schemas import ChatRequest, ChatResponse
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
        return ChatResponse(
            conversation_id=conv_id,
            answer=rag_result["answer"],
            debug=rag_result.get("debug"),
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
