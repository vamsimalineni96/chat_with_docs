from uuid import UUID
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.utils import config
from src.utils.services.logger_config import logger
from src.utils.chat.chat_service import cache_output, rag_output
from src.utils.services.pdf_parser import PDFParser
from src.utils.services.milvus_store import MilvusStoreHandler, get_cache_store
from src.utils.services.conversation_store import get_conversation_service
from src.utils.services.redis_lock import get_redis_lock, ConversationLockError
from src.utils.db.database_debug import DBInspector
from src.utils.db.database import get_db, Base, engine
from src.utils.api.schemas import ChatRequest, ChatResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    logger.info("Created the tables")
    Base.metadata.create_all(bind=engine)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Multi-user, multi-turn chat endpoint with per-conversation locking.
    """
    conversation_service = get_conversation_service()
    redis_service = get_redis_lock()

    # Obtaining the conversation and user id from the database
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
            logger.info("Conversation is not found")
            conversation = conversation_service.create_conversation(
                db,
                user=user,
                title="Harry Potter chat",  # optional
            )
            logger.info("Created the new conversation in the database")

    conv_id = str(conversation.id)

    # Locking the converstation to allow for only one user per conv_id at a time
    try:
        logger.info(f"Locking the conversation using redis for conv_id: {conv_id}")
        lock_token = redis_service.acquire_conversation_lock(conv_id, wait=False)
    except ConversationLockError:
        raise HTTPException(
            status_code=409,
            detail="Another message is being processed for this conversation.",
        )

    # Searching the cache store before jumping into rag
    try:
        if config.TOGGLE_CACHE:
            logger.info("Searching the cache store for similar answer")
            cached_answer = cache_output(payload)
            if cached_answer:
                return ChatResponse(conversation_id=conv_id, answer=cached_answer)

            else:
                logger.info("Cache is missed, routing to RAG for answering")
                rag_answer = rag_output(payload, db, conversation, user)
                return ChatResponse(conversation_id=conv_id, answer=rag_answer)
        else:
            logger.info("Using RAG to answer the question")
            rag_answer = rag_output(payload, db, conversation, user)
            return ChatResponse(conversation_id=conv_id, answer=rag_answer)
        
    finally:
        logger.info(f"Releasing the conversation lock for conv_id: {conv_id}")
        redis_service.release_conversation_lock(conv_id, lock_token)


@app.post("/upload_pdf")
async def upload_pdf(pdf_name: str, collection_name: str):
    """
    Upload the pdf for chatting.
    """
    vector_store = MilvusStoreHandler(collection_name=collection_name)
    pdf_path = f"pdfs/{pdf_name}"

    parser = PDFParser(pdf_path)
    pages = parser.parse_pdf()
    print(f"Total pages parsed: {len(pages)}")

    # Continuous full text of the book
    for i in range(1, len(pages)):
        long_text = pages[i].get("text")
        vector_store.store_in_milvus(text=long_text)
        print(f"Uploaded page: {i} to the vectordb")


@app.post("/clear_post_gres")
async def clear_db():
    """
    Clear the post gres database
    """
    with Session(engine) as session:
        for table in reversed(Base.metadata.sorted_tables):
            logger.info(f"Deleting from {table.name}...")
            session.execute(table.delete())
        session.commit()
        logger.info("All tables cleared.")
    return {"message": "All tables are cleared"}


@app.post("/clear_cache")
async def clear_cache():
    cache_store = get_cache_store()
    cache_store.delete_collection()
    return {"message": "cleared the cache_store"}


@app.post("/debug_database")
async def debug_db(user_id: str = None, conv_id: UUID = None):
    """
    Print the database texts for debugging
    """
    db_debugger = DBInspector()
    print("Printing users")
    db_debugger.print_users()
    print("Printing conversations")
    db_debugger.print_conversations(user_external_id=user_id)
    print(f"Printing Messages from conversation: {conv_id}")
    db_debugger.print_messages(conversation_id=conv_id)
    return {"message": "Details are printed"}
