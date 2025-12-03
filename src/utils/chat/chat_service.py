import time
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
    t0=time.perf_counter()
    logger.info("Accessing recent messages from the database")
    t_db_start = time.perf_counter()
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
    t_db_end = time.perf_counter()

    answer, t_milvus_start, t_milvus_end, t_llm_start, t_llm_end = answer_question(
        question=payload.question,
        query_vec= query_vec,
        collection_name=payload.collection_name,
        history=history_for_llm,
    )
    logger.info("Storing the chatbot's reply to the user query")
    t_save_start = time.perf_counter()
    converstion_service.add_message(
        db,
        conversation=conversation,
        user=user,
        role="assistant",
        content=answer,
    )
    t_save_end = time.perf_counter()
    t1=time.perf_counter()

    logger.info(
        "RAG_PIPELINE_METRICS | conv_id=%s | domain=%s | "
        "db_load_ms=%.2f | milvus_ms=%.2f | llm_ms=%.2f | db_save_ms=%.2f | total_ms=%.2f",
        conversation.id,
        "harry_potter",
        (t_db_end - t_db_start) * 1000,
        (t_milvus_end - t_milvus_start) * 1000,
        (t_llm_end - t_llm_start) * 1000,
        (t_save_end - t_save_start) * 1000,
        (t1 - t0) * 1000,
    )
    return answer
