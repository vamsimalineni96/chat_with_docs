import time

from src.utils import config
from src.utils.errors import CacheError, ConversationServiceError, InferenceError
from src.utils.observability import observe, update_current_trace
from src.utils.rag_pipeline import answer_question
from src.utils.services.conversation_store import get_conversation_service
from src.utils.services.logger_config import logger
from src.utils.services.milvus_store import get_cache_store

cache_service = get_cache_store()
converstion_service = get_conversation_service()


@observe(name="cache_lookup")
def cache_output(payload, q_embed):
    """
    Try to fetch a cached answer.

    Raises:
        CacheError: if cache lookup fails unexpectedly.
    """
    update_current_trace(
        user_id=payload.user_external_id,
        session_id=str(payload.conversation_id) if payload.conversation_id else None,
        tags=[
            f"prompt:{config.PROMPT_VERSION}",
            f"collection:{payload.collection_name}",
            f"domain:{config.METRICS_DOMAIN}",
            "cache-path",
            "debug" if getattr(payload, "debug", False) else "normal",
        ],
        metadata={"question": payload.question, "cache_enabled": config.TOGGLE_CACHE},
    )
    try:
        cached = cache_service.search_similar(
            query=payload.question,
            q_vec=q_embed,
            model_name=config.LLM_MODEL,
            prompt_version=config.PROMPT_VERSION,
            min_similarity=config.CACHE_MIN_SIMILARITY,
        )
    except CacheError:
        # Already wrapped at Milvus layer
        raise
    except Exception as e:
        logger.exception("Unexpected error while searching cache: %s", e)
        raise CacheError("Unexpected error while searching cache.") from e

    if cached:
        logger.info("Returning the cached answer")
        return cached["answer_text"]
    return None


@observe(name="rag_output")
def rag_output(
    payload,
    db,
    conversation,
    user,
    query_vec,
):
    """
    Run full RAG pipeline:
      - Load recent messages
      - Save user message
      - Run RAG (Milvus + NIM)
      - Save assistant message
      - Log timing metrics

    Raises:
        ConversationServiceError: for DB issues.
        InferenceError: for RAG/LLM/Milvus issues.
    """
    update_current_trace(
        user_id=user.external_id,
        session_id=str(conversation.id),
        tags=[
            f"prompt:{config.PROMPT_VERSION}",
            f"collection:{payload.collection_name}",
            f"domain:{config.METRICS_DOMAIN}",
            "rag-path",
            "debug" if getattr(payload, "debug", False) else "normal",
        ],
        metadata={"question": payload.question, "cache_enabled": config.TOGGLE_CACHE},
    )
    t0 = time.perf_counter()

    logger.info("Accessing recent messages from the database")
    try:
        t_db_start = time.perf_counter()
        recent_msgs = converstion_service.get_recent_messages(
            db, conversation, limit=config.HISTORY_LIMIT, user=user
        )
        history_for_llm = [{"role": m.role, "content": m.content} for m in recent_msgs]
    except ConversationServiceError:
        raise
    except Exception as e:
        logger.exception("Unexpected error while loading recent messages: %s", e)
        raise ConversationServiceError("Failed to load recent messages.") from e

    logger.info("Storing the new message into the database")
    try:
        converstion_service.add_message(
            db,
            conversation=conversation,
            user=user,
            role="user",
            content=payload.question,
        )
        t_db_end = time.perf_counter()
    except ConversationServiceError:
        raise
    except Exception as e:
        logger.exception("Unexpected error while storing user message: %s", e)
        raise ConversationServiceError("Failed to store user message.") from e

    logger.info("Running RAG pipeline to generate answer")
    try:
        result = answer_question(
            question=payload.question,
            query_vec=query_vec,
            collection_name=payload.collection_name,
            history=history_for_llm,
            debug=getattr(payload, "debug", False),
        )
    except InferenceError:
        # Already wrapped appropriately
        raise
    except Exception as e:
        logger.exception("Unexpected error from RAG pipeline: %s", e)
        raise InferenceError("Unexpected error from RAG pipeline.") from e

    answer = result["answer"]
    t_milvus_start = result["t_milvus_start"]
    t_milvus_end = result["t_milvus_end"]
    t_llm_start = result["t_llm_start"]
    t_llm_end = result["t_llm_end"]
    debug_info = result.get("debug")

    # Heuristics evaluation is computed in the RAG pipeline; tagging
    # happens *here* because this is the trace-root span. Tagging from
    # inside answer_question (a child observation) attaches the tag to
    # the child instead of bubbling up to the trace, making it
    # invisible in the Langfuse trace-level tag view.
    heuristics_report = result.get("heuristics")
    if heuristics_report is not None:
        tags = [f"heuristic_pass:{str(heuristics_report['overall_passed']).lower()}"]
        if not heuristics_report["overall_passed"]:
            tags.append(
                "heuristic_failed:" + ",".join(heuristics_report["failed_checks"])
            )
        try:
            update_current_trace(tags=tags)
        except Exception as e:
            logger.debug(
                "update_current_trace from heuristics tagging failed (non-fatal): %s",
                e,
            )

    logger.info("Storing the chatbot's reply to the user query")
    try:
        t_save_start = time.perf_counter()
        converstion_service.add_message(
            db,
            conversation=conversation,
            user=user,
            role="assistant",
            content=answer,
        )
        t_save_end = time.perf_counter()
    except ConversationServiceError:
        raise
    except Exception as e:
        logger.exception("Unexpected error while storing assistant message: %s", e)
        raise ConversationServiceError("Failed to store assistant message.") from e

    t1 = time.perf_counter()

    logger.info(
        "RAG_PIPELINE_METRICS | conv_id=%s | domain=%s | "
        "db_load_ms=%.2f | milvus_ms=%.2f | llm_ms=%.2f | db_save_ms=%.2f | total_ms=%.2f",
        conversation.id,
        config.METRICS_DOMAIN,
        (t_db_end - t_db_start) * 1000,
        (t_milvus_end - t_milvus_start) * 1000,
        (t_llm_end - t_llm_start) * 1000,
        (t_save_end - t_save_start) * 1000,
        (t1 - t0) * 1000,
    )

    if debug_info is not None:
        debug_info["timings_ms"] = {
            "db_load": (t_db_end - t_db_start) * 1000,
            "milvus": (t_milvus_end - t_milvus_start) * 1000,
            "llm": (t_llm_end - t_llm_start) * 1000,
            "db_save": (t_save_end - t_save_start) * 1000,
            "total": (t1 - t0) * 1000,
        }

    return {"answer": answer, "debug": debug_info}
