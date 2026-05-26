import time

from src.agents.graph import get_chat_graph
from src.utils import config
from src.utils.errors import CacheError, ConversationServiceError, InferenceError
from src.utils.observability import observe, update_current_trace
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
async def rag_output(
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

    logger.info("Invoking chat graph to generate answer")
    try:
        # The graph orchestrates: retrieve → (rerank → generate | canned_no_retrieval)
        # → postprocess (heuristics). Each stage is its own @observe'd span in
        # Langfuse. See src/agents/graph.py for the topology.
        result = await get_chat_graph().ainvoke(
            {
                "question": payload.question,
                "query_vec": query_vec,
                "collection_name": payload.collection_name,
                "history": history_for_llm,
                "debug_flag": getattr(payload, "debug", False),
            }
        )
    except InferenceError:
        # Already wrapped appropriately
        raise
    except Exception as e:
        logger.exception("Unexpected error from chat graph: %s", e)
        raise InferenceError("Unexpected error from chat graph.") from e

    answer = result["answer"]
    t_milvus_start = result["t_milvus_start"]
    t_milvus_end = result["t_milvus_end"]
    t_llm_start = result["t_llm_start"]
    t_llm_end = result["t_llm_end"]
    debug_info = result.get("debug_info")

    # Trace-root tagging: the @observe(name="rag_output") span is the
    # trace root, so update_current_trace from here lands at the trace
    # level (where Langfuse's tag filter looks). Tagging from inside a
    # child observation tags the child instead — see PR #44 history.
    # We batch the heuristic + intent tags into one call.
    trace_tags: list[str] = []

    intent = result.get("intent")
    if intent:
        trace_tags.append(f"intent:{intent}")

    # tool_call branch only — one tag per MCP tool the ReAct sub-agent
    # invoked. Tagging by name (not by name+args) keeps Langfuse's tag
    # filter usable for "show me all sessions that hit get_order_status".
    tool_calls = result.get("tool_calls") or []
    for tc in tool_calls:
        name = tc.get("name")
        if name:
            trace_tags.append(f"tool:{name}")

    tool_failure_reason = result.get("tool_failure_reason")
    if tool_failure_reason:
        trace_tags.append(f"mcp_failure:{tool_failure_reason}")

    heuristics_report = result.get("heuristics")
    if heuristics_report is not None:
        trace_tags.append(
            f"heuristic_pass:{str(heuristics_report['overall_passed']).lower()}"
        )
        if not heuristics_report["overall_passed"]:
            trace_tags.append(
                "heuristic_failed:" + ",".join(heuristics_report["failed_checks"])
            )

    if trace_tags:
        try:
            update_current_trace(tags=trace_tags)
        except Exception as e:
            logger.debug(
                "update_current_trace from chat_service tagging failed (non-fatal): %s",
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
        if tool_calls:
            debug_info["tool_calls"] = tool_calls

    return {"answer": answer, "debug": debug_info}
