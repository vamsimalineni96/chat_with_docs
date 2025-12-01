from uuid import UUID
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.utils.rag_pipeline import answer_question
from src.utils.services.logger_config import logger
from src.utils.services.pdf_parser import PDFParser
from src.utils.services.milvus_store import MilvusStoreHandler
from src.utils.services.conversation_store import ConversationService
from src.utils.services.redis_lock import ConversationLock, ConversationLockError
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
    converstion_service = ConversationService()
    redis_service = ConversationLock()

    logger.info("Fetching/Creating the user in the database")
    user = converstion_service.get_or_create_user(
        db, external_id=payload.user_external_id
    )

    if payload.conversation_id:
        logger.info("Fetching the conversation by conversation id")
        conversation = converstion_service.get_conversation_by_id(
            db,
            conversation_id=payload.conversation_id,
            user=user,
        )
        if conversation is None:
            logger.info("Conversation is not found")
            conversation = converstion_service.create_conversation(
                db,
                user=user,
                title="Harry Potter chat",  # optional
            )
            logger.info("Created the new conversation in the database")

    conv_id = str(conversation.id)

    try:
        logger.info(f"Locking the conversation using redis for conv_id: {conv_id}")
        lock_token = redis_service.acquire_conversation_lock(conv_id, wait=False)
    except ConversationLockError:
        raise HTTPException(
            status_code=409,
            detail="Another message is being processed for this conversation.",
        )

    try:
        logger.info("Accessing recent messages from the database")
        recent_msgs = converstion_service.get_recent_messages(
            db, conversation, limit=20, user= user
        )
        history_for_llm = [{"role": m.role, "content": m.content} for m in recent_msgs]

        logger.info("Storing the new message into the database")
        converstion_service.add_message(
            db,
            conversation=conversation,
            user=user,
            role="user",
            content=payload.question,
        )

        answer = answer_question(
            question=payload.question,
            collection_name=payload.collection_name,
            history=history_for_llm,
        )

        logger.info("Storing the chatbot's reply to the user query")
        converstion_service.add_message(
            db,
            conversation=conversation,
            user=user,
            role="assistant",
            content=answer,
        )

        return ChatResponse(
            conversation_id=conv_id,
            answer=answer,
        )
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
