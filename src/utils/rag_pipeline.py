import time
from typing import List, Dict, Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from src.utils import config
from src.utils.errors import InferenceError
from src.utils.observability import (
    observe,
    update_current_observation,
    langfuse_callback,
)
from src.utils.services.milvus_store import MilvusStoreHandler, get_cache_store
from src.utils.services.inference import NIMClient
from src.utils.services.logger_config import logger
from src.utils.services.chunk_ranking import NVidiaReranker


def build_context(chunks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        score = c.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        parts.append(
            f"[Chunk {i} | score={score_str} | source={c.get('source')}]\n{c['text']}\n"
        )
    return "\n\n".join(parts)


def format_history_for_prompt(history: List[Dict], max_turns: int = config.HISTORY_MAX_TURNS) -> str:
    if not history:
        return "None"

    trimmed = history[-max_turns:]
    lines = []
    for msg in trimmed:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "Assistant"
        elif role == "system":
            prefix = "System"
        else:
            prefix = role or "Unknown"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


@observe(name="hybrid_retrieve", as_type="span")
def _retrieve_for_query(
    milvus_store: MilvusStoreHandler,
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Hybrid (dense + BM25) retrieval for a single query string."""
    results = milvus_store.search_similar_chunks(query=query, top_k=top_k)
    update_current_observation(
        input={"query": query, "top_k": top_k},
        output={
            "count": len(results),
            "top_chunks": [
                {
                    "score": r.get("score"),
                    "source": r.get("source"),
                    "chunk_order": r.get("chunk_order"),
                    "text_preview": (r.get("text") or "")[:200],
                }
                for r in results[:5]
            ],
        },
    )
    return results


class RAGState(TypedDict, total=False):
    """State that flows through the LangGraph RAG pipeline.

    `timings` and `debug` are mutable dicts owned by the caller — nodes mutate
    them in place so `answer_question` can read the final values after
    `graph.invoke` returns. Top-level fields (`retrieved`, `reranked`,
    `context`, `answer`, `history_text`) are returned by nodes and merged into
    state by LangGraph's default channel reducer (replace).
    """
    question: str
    history: List[Dict]
    history_text: str
    retrieved: List[Dict[str, Any]]
    reranked: List[Dict[str, Any]]
    context: str
    answer: str
    timings: Dict[str, float]
    debug: Optional[Dict[str, Any]]


def build_rag_graph(
    milvus_store: MilvusStoreHandler,
    reranker: NVidiaReranker,
    llm: NIMClient,
):
    """
    Compile the RAG graph:

        retrieve ──► [retrieved empty?] ──► empty_response ──► END
                              │
                              └► rerank ──► assemble ──► generate ──► END
    """

    def retrieve_node(state: RAGState) -> Dict[str, Any]:
        question = state["question"]
        timings = state["timings"]
        debug = state.get("debug")

        logger.info("Retrieving context from Milvus DB (hybrid dense + BM25)")
        try:
            timings["t_milvus_start"] = time.perf_counter()
            retrieved = _retrieve_for_query(
                milvus_store, question, top_k=config.RETRIEVE_K
            )
            timings["t_milvus_end"] = time.perf_counter()
        except Exception as e:
            logger.exception("Failed to retrieve context from Milvus: %s", e)
            raise InferenceError("Failed to retrieve context from Milvus.") from e

        if debug is not None:
            debug["retrieved_chunks"] = retrieved
            # Hybrid-retrieval diagnostic: re-run dense-only and BM25-only
            # searches so the UI can show what each component contributes vs.
            # the fused list.
            try:
                debug["dense_only_chunks"] = milvus_store.search_dense_only(
                    query=question, top_k=config.RETRIEVE_K
                )
            except Exception as e:
                logger.warning("Dense-only diagnostic search failed (non-fatal): %s", e)
                debug["dense_only_chunks"] = []
            try:
                debug["sparse_only_chunks"] = milvus_store.search_sparse_only(
                    query=question, top_k=config.RETRIEVE_K
                )
            except Exception as e:
                logger.warning("Sparse-only diagnostic search failed (non-fatal): %s", e)
                debug["sparse_only_chunks"] = []

        return {"retrieved": retrieved}

    def rerank_node(state: RAGState) -> Dict[str, Any]:
        reranked = reranker.execute(
            question=state["question"], retrieved_chunks=state["retrieved"]
        )
        sliced = reranked[: config.TOP_K]
        debug = state.get("debug")
        if debug is not None:
            debug["reranked_chunks"] = reranked
            debug["reranked_top_k"] = sliced
        return {"reranked": sliced}

    def assemble_node(state: RAGState) -> Dict[str, Any]:
        context = build_context(state["reranked"])
        history_text = state.get("history_text") or format_history_for_prompt(
            state["history"], max_turns=config.HISTORY_MAX_TURNS
        )
        debug = state.get("debug")
        if debug is not None:
            debug["history_text"] = history_text
        return {"context": context, "history_text": history_text}

    def generate_node(state: RAGState) -> Dict[str, Any]:
        timings = state["timings"]
        debug = state.get("debug")
        if debug is not None:
            try:
                rendered = llm._prompt.format_messages(
                    history_text=state["history_text"],
                    question=state["question"],
                    context=state["context"],
                )
                debug["rendered_prompt"] = [
                    {"role": getattr(m, "type", "unknown"), "content": m.content}
                    for m in rendered
                ]
            except Exception as e:
                logger.warning("Failed to capture rendered prompt for debug: %s", e)
                debug["rendered_prompt"] = []

        timings["t_llm_start"] = time.perf_counter()
        answer = llm.chat_completion(
            history_text=state["history_text"],
            question=state["question"],
            context=state["context"],
        )
        timings["t_llm_end"] = time.perf_counter()
        return {"answer": answer}

    def empty_response_node(state: RAGState) -> Dict[str, Any]:
        # No retrieval hits — short-circuit with an apology. Still populate
        # LLM timings so the upstream metrics logger has all four stamps.
        logger.info("No relevant context found in the vector store.")
        timings = state["timings"]
        t_end = timings.get("t_milvus_end", time.perf_counter())
        timings["t_llm_start"] = t_end
        timings["t_llm_end"] = t_end
        return {
            "answer": (
                "I couldn't find anything in the indexed document that touches on that. "
                "Could you try rephrasing, or asking about a different topic from the book?"
            ),
        }

    def route_after_retrieve(state: RAGState) -> str:
        return "empty_response" if not state.get("retrieved") else "rerank"

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("assemble", assemble_node)
    graph.add_node("generate", generate_node)
    graph.add_node("empty_response", empty_response_node)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"rerank": "rerank", "empty_response": "empty_response"},
    )
    graph.add_edge("rerank", "assemble")
    graph.add_edge("assemble", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("empty_response", END)

    return graph.compile()


@observe(name="answer_question")
def answer_question(
    question: str,
    query_vec: List[float],
    collection_name: str,
    history: List[Dict],
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Main RAG orchestration (LangGraph).

    Pipeline:
      1. Hybrid (dense + BM25) retrieval for the original question, RETRIEVE_K chunks.
      2. If retrieval is empty → apology branch and short-circuit.
      3. Otherwise: rerank → slice to TOP_K → assemble prompt → call the LLM.

    Returns a dict with answer, stage timings, and (when debug=True) intermediate
    artifacts.

    Raises:
        InferenceError: for retrieval/rerank/LLM failures.
    """
    milvus_store = MilvusStoreHandler(collection_name=collection_name)
    nim_client = NIMClient()
    nim_reranker = NVidiaReranker()

    debug_info: Optional[Dict[str, Any]] = {} if debug else None
    history_text = format_history_for_prompt(history, max_turns=config.HISTORY_MAX_TURNS)

    graph = build_rag_graph(milvus_store, nim_reranker, nim_client)
    timings: Dict[str, float] = {}
    initial_state: RAGState = {
        "question": question,
        "history": history,
        "history_text": history_text,
        "timings": timings,
        "debug": debug_info,
    }

    cb = langfuse_callback()
    invoke_config = {"callbacks": [cb]} if cb else {}

    try:
        logger.info("Invoking RAG graph (retrieve -> [rerank -> assemble -> generate])")
        final_state = graph.invoke(initial_state, config=invoke_config)
    except InferenceError:
        raise
    except Exception as e:
        logger.exception("Unexpected error from RAG graph: %s", e)
        raise InferenceError("Unexpected error from RAG graph.") from e

    answer = final_state["answer"]
    retrieved = final_state.get("retrieved") or []

    t_milvus_start = timings.get("t_milvus_start", time.perf_counter())
    t_milvus_end = timings.get("t_milvus_end", t_milvus_start)
    t_llm_start = timings.get("t_llm_start", t_milvus_end)
    t_llm_end = timings.get("t_llm_end", t_llm_start)

    if config.TOGGLE_CACHE and not debug and retrieved:
        try:
            context_chunk_ids = [item.get("id") for item in retrieved if item.get("id")]
            get_cache_store().put_entry(
                question_text=question,
                query_vec=query_vec,
                answer_text=answer,
                context_chunk_ids=context_chunk_ids,
                model_name=config.LLM_MODEL,
                prompt_version=config.PROMPT_VERSION,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )
            logger.info("Stored Q/A pair in semantic cache.")
        except Exception as e:
            logger.exception("Cache write failed (non-fatal): %s", e)

    return {
        "answer": answer,
        "t_milvus_start": t_milvus_start,
        "t_milvus_end": t_milvus_end,
        "t_llm_start": t_llm_start,
        "t_llm_end": t_llm_end,
        "debug": debug_info,
    }
