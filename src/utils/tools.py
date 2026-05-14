"""LangChain tools exposed to the LLM for agentic RAG.

These are the functions the LLM can choose to call. Each is registered via
`@tool` so LangChain can derive the JSON schema (name, args, types, docstring)
the model needs to issue a tool_call.

Currently exposed:
  - `search_chunks(query, top_k)`: hybrid Milvus retrieval + NVIDIA rerank,
    returns the top passages formatted as text.

Build via `build_search_chunks_tool(milvus_store, reranker)` so the heavy
service instances are captured in a closure (the LLM only ever sees the
declared args).
"""

from typing import List, Dict, Any, Callable

from langchain_core.tools import tool

from src.utils import config
from src.utils.errors import RerankError
from src.utils.observability import observe, update_current_observation
from src.utils.services.chunk_ranking import NVidiaReranker
from src.utils.services.logger_config import logger
from src.utils.services.milvus_store import MilvusStoreHandler


def _format_chunks_for_llm(chunks: List[Dict[str, Any]]) -> str:
    """Format reranked passages as a single string for the LLM to read."""
    if not chunks:
        return "No matching passages found in the indexed document."
    parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        score = c.get("rerank_score") if c.get("rerank_score") is not None else c.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        parts.append(
            f"[Passage {i} | source={c.get('source')} | score={score_str}]\n{c['text']}"
        )
    return "\n\n".join(parts)


def build_search_chunks_tool(
    milvus_store: MilvusStoreHandler,
    reranker: NVidiaReranker,
) -> Callable:
    """Construct the `search_chunks` tool with services captured in a closure.

    The returned object is a `BaseTool` (created by LangChain's `@tool`) ready
    to be passed to `llm.bind_tools([...])` and `ToolNode([...])`.
    """

    @tool
    @observe(name="tool:search_chunks", as_type="span")
    def search_chunks(query: str, top_k: int = 5) -> str:
        """Search the indexed document for passages relevant to a query.

        Use this whenever the user's question requires looking up content
        from the document — characters, plot events, settings, quotations,
        names, dates, themes. Never answer document questions from your
        own training data.

        You can call this multiple times in one turn for multi-part
        questions (e.g., comparing two characters: call once per character).

        Args:
            query: A specific retrieval query. Include character names,
                event names, locations, or other concrete terms when
                possible.
            top_k: Number of passages to return (1-10, default 5).

        Returns:
            A formatted string with each passage labelled by source and
            rerank score, or a "no matches" notice if nothing was found.
        """
        # Clamp top_k to a sane range — the LLM sometimes sends huge values.
        top_k = max(1, min(int(top_k), 10))

        logger.info("Tool search_chunks invoked: query=%r top_k=%d", query, top_k)
        retrieved = milvus_store.search_similar_chunks(
            query=query, top_k=config.RETRIEVE_K
        )
        if not retrieved:
            update_current_observation(
                input={"query": query, "top_k": top_k},
                output={"count": 0, "note": "no hits"},
            )
            return "No matching passages found in the indexed document."

        try:
            reranked = reranker.execute(
                question=query, retrieved_chunks=retrieved
            )
        except RerankError as e:
            logger.warning(
                "Rerank failed inside search_chunks tool (falling back to raw retrieval): %s",
                e,
            )
            reranked = retrieved

        sliced = reranked[:top_k]

        update_current_observation(
            input={"query": query, "top_k": top_k},
            output={
                "count": len(sliced),
                "top_passages": [
                    {
                        "source": c.get("source"),
                        "rerank_score": c.get("rerank_score"),
                        "text_preview": (c.get("text") or "")[:200],
                    }
                    for c in sliced[:3]
                ],
            },
        )

        return _format_chunks_for_llm(sliced)

    return search_chunks
