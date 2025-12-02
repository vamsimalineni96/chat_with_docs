from src.utils import config
from src.utils.services.logger_config import logger
from src.utils.services.milvus_store import get_cache_store
from src.utils.services.conversation_store import get_conversation_service
from src.utils.rag_pipeline import answer_question

cache_service = get_cache_store()
converstion_service = get_conversation_service()


def cache_output(payload, q_embed):
    cached = cache_service.search_similar(
        query=payload.question,
        q_vec=q_embed,
        model_name=config.LLM_MODEL,
        prompt_version=config.PROMPT_VERSION,
        min_similarity=0.9,
    )

    if cached:
        logger.info("Returning the cached answer")
        return cached["answer_text"]


def rag_output(
    payload,
    db,
    conversation,
    user,
    query_vec
):
    # Generating the answer based on the recent messages,and retrieved context.
    logger.info("Accessing recent messages from the database")
    recent_msgs = converstion_service.get_recent_messages(
        db, conversation, limit=20, user=user
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
        query_vec= query_vec,
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

    return answer
