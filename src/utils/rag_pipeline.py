"""Agentic RAG pipeline (LangGraph) with query rewriting + self-correction.

The LLM drives the loop: it sees a document search tool and decides when
(and how many times) to retrieve. Before the agent runs, a `rewriter` node
resolves pronouns / context references from chat history into a standalone
search query. After the agent emits a final answer, a `verifier` node grades
it against retrieved passages and can loop back to the agent with a critique.

Graph shape:

    rewriter ──► agent ◄────────────────────────┐
                  │                             │
                  │ tool_calls? ──► tools ──────┤
                  │ no                          │
                  ▼                             │
              verifier ── grounded? ── yes / cap ──► END
                  │                             │
                  │ no, with critique injected  │
                  └─────────────────────────────┘

`rewriter_node`     : LLM call WITHOUT tools — produces a self-contained
                      version of the user's question for the agent to use
                      when picking tool args. No-ops when there's no
                      conversation history.
`agent_node`        : LLM call with tools bound
`tools` (tool_node) : executes tool_calls, returns ToolMessages
`verifier_node`     : LLM call WITHOUT tools — grades the answer for grounding

Caps:
  - `MAX_AGENT_ITERATIONS` : total agent LLM calls (across initial + retries)
  - `MAX_CORRECTIONS`      : how many times the verifier may trigger a retry
"""

import time
from typing import List, Dict, Any, Optional, TypedDict, Annotated

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.utils import config
from src.utils.errors import InferenceError
from src.utils.observability import observe, langfuse_callback
from src.utils.services.chunk_ranking import NVidiaReranker
from src.utils.services.inference import NIMClient
from src.utils.services.logger_config import logger
from src.utils.services.milvus_store import MilvusStoreHandler, get_cache_store
from src.utils.tools import build_search_chunks_tool


# Total agent LLM calls allowed. Bumped from 5 to 8 to give a couple of
# verifier-triggered retries some headroom before the iteration safety valve
# forces termination.
MAX_AGENT_ITERATIONS = 8

# How many times the verifier may say "retry, this isn't grounded" before
# we accept the last attempt and stop. 2 is plenty in practice.
MAX_CORRECTIONS = 2


REWRITER_SYSTEM_PROMPT = """You are a query rewriter for a document search tool.

You will be given a conversation between a user and an assistant, plus the
user's latest question. Your job: rewrite the latest question into a single
SELF-CONTAINED retrieval query that:

  - Resolves pronouns and context references ("him" → "Harry", "that scene"
    → "the Yule Ball scene", "the second one" → "the second task")
  - Expands abbreviations and clarifies ambiguous terms when context makes
    them obvious
  - Preserves the original intent — do NOT add facts the user didn't imply

Output ONLY the rewritten query as a single line of text. No preamble, no
quotes, no explanation. If the question is already fully self-contained
(no pronouns to resolve, no missing context), output it UNCHANGED.
"""


VERIFIER_SYSTEM_PROMPT = """You are a strict grounding reviewer.

You will be shown:
1. A user's question.
2. The passages that were retrieved to answer it (may be empty).
3. The candidate answer that was generated.

Your job: judge whether every factual claim in the answer is supported by
the passages. The answer is GROUNDED if:
  - All claims trace back to specific passages, OR
  - The answer honestly says it can't find the information (passages were empty or off-topic), OR
  - The question was a meta-question about the conversation, not the document.

The answer is NOT GROUNDED if it adds facts, names, dates, or specifics
that the passages do not support.

Respond in EXACTLY this format, no preamble, no markdown:

GROUNDED: yes
CRITIQUE: none

OR:

GROUNDED: no
CRITIQUE: <one short sentence pointing to the specific unsupported claim>
"""


class AgentState(TypedDict, total=False):
    """State for the agentic loop + verifier.

    `messages` uses LangGraph's `add_messages` reducer so node returns of
    `{"messages": [new_msg]}` are *appended* to the running list rather than
    replacing it. Other fields use the default replace reducer.

    `timings` and `debug` are mutable dicts owned by the caller — nodes
    mutate them in place so the caller can read final values after invoke.

    `iterations`        : how many times `agent_node` has run.
    `correction_count`  : how many times `verifier_node` has said "retry".
    `verifier_verdict`  : "ok" | "retry" | "exhausted" — set by `verifier_node`,
                          consulted by the conditional edge after it.
    `rewritten_query`   : the user question after pronoun/context resolution
                          by `rewriter_node`. None if rewriting was skipped
                          (no history) or the question was already standalone.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    timings: Dict[str, float]
    debug: Optional[Dict[str, Any]]
    iterations: int
    correction_count: int
    verifier_verdict: Optional[str]
    rewritten_query: Optional[str]


def make_tool_node(tools: List):
    """Build a tool-executor node.

    Looks at the last AIMessage in `state["messages"]`, executes every
    tool_call it carries, and returns a list of ToolMessages that
    `add_messages` will append to the running message trail.

    Equivalent to `langgraph.prebuilt.ToolNode([...])` but written inline so
    we don't depend on the prebuilt package (its version pinning is fussy).
    """
    tool_map = {t.name: t for t in tools}

    def tool_node(state: "AgentState") -> Dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None) or []
        outputs: List[ToolMessage] = []
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = (tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})) or {}
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)

            tool = tool_map.get(name)
            if tool is None:
                content = f"Error: unknown tool {name!r}"
                logger.warning("LLM requested unknown tool %r", name)
            else:
                try:
                    result = tool.invoke(args)
                    content = result if isinstance(result, str) else str(result)
                except Exception as e:
                    logger.exception("Tool %s raised — returning error message to LLM", name)
                    content = f"Error calling {name}: {e}"
            outputs.append(ToolMessage(content=content, tool_call_id=tc_id, name=name))
        return {"messages": outputs}

    return tool_node


def format_history_for_prompt(history: List[Dict], max_turns: int = config.HISTORY_MAX_TURNS) -> str:
    """Render history as plain text. Retained for debug capture and any
    external callers that still want a single string."""
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


def history_to_messages(
    history: List[Dict],
    max_turns: int = config.HISTORY_MAX_TURNS,
) -> List[BaseMessage]:
    """Convert stored chat history rows to LangChain BaseMessages."""
    if not history:
        return []
    # max_turns is "exchanges"; multiply by 2 to cover user+assistant pairs.
    trimmed = history[-(max_turns * 2):]
    out: List[BaseMessage] = []
    for m in trimmed:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


def _parse_verifier_verdict(text: str) -> Dict[str, Any]:
    """Pull `grounded` and `critique` out of the verifier's response.

    Failure-open: if we can't parse the verdict, treat it as grounded=True
    so the graph still terminates with the last answer rather than spinning.
    """
    if not text:
        return {"grounded": True, "critique": ""}

    grounded = True
    critique = ""
    for raw in text.splitlines():
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("GROUNDED:"):
            value = line.split(":", 1)[1].strip().lower()
            grounded = not value.startswith("no")
        elif upper.startswith("CRITIQUE:"):
            critique = line.split(":", 1)[1].strip()
    return {"grounded": grounded, "critique": critique}


def build_agent_graph(llm_with_tools, raw_llm, tool_node):
    """Compile the rewriter → agent ↔ tools → verifier graph.

    `llm_with_tools` is the ChatNVIDIA with tools bound (used by `agent_node`).
    `raw_llm`         is the same ChatNVIDIA without tools (used by
                       `rewriter_node` and `verifier_node` so they can't be
                       tempted to call tools themselves).
    """

    @observe(name="rewriter", as_type="generation")
    def rewriter_node(state: AgentState) -> Dict[str, Any]:
        """Resolve the user's question against chat history.

        No-ops when there is no prior turn (the question can't depend on
        history that doesn't exist). Otherwise asks the LLM to produce a
        standalone version and injects it as a SystemMessage hint that the
        agent will see when picking tool args.
        """
        messages = state["messages"]

        # Locate the latest HumanMessage (the question we're rewriting).
        latest_human_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                latest_human_idx = i
                break
        if latest_human_idx is None:
            return {}

        # If nothing precedes the latest human turn besides the System
        # prompt, there's no history to resolve against — skip.
        prior = [m for m in messages[:latest_human_idx]
                 if isinstance(m, (HumanMessage, AIMessage))]
        if not prior:
            logger.info("rewriter: no history — skipping rewrite.")
            return {}

        question = messages[latest_human_idx].content or ""
        history_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {(m.content or '')[:500]}"
            for m in prior
        )
        payload = f"CONVERSATION SO FAR:\n{history_text}\n\nUSER'S LATEST QUESTION:\n{question}"

        try:
            response = raw_llm.invoke([
                SystemMessage(content=REWRITER_SYSTEM_PROMPT),
                HumanMessage(content=payload),
            ])
            rewritten = (response.content or "").strip().strip('"').strip()
        except Exception as e:
            logger.warning("rewriter LLM call failed — skipping rewrite: %s", e)
            return {}

        # If the rewriter returned something identical (or empty/garbage),
        # don't bother injecting a redundant hint.
        if not rewritten or rewritten.lower() == question.strip().lower():
            logger.info("rewriter: no change.")
            return {"rewritten_query": rewritten or None}

        logger.info("rewriter: %r → %r", question[:80], rewritten[:80])
        if state.get("debug") is not None:
            state["debug"]["rewritten_query"] = rewritten

        hint = SystemMessage(content=(
            f"NOTE FROM THE QUERY REWRITER: The user's question, resolved "
            f"against the conversation history, is: {rewritten!r}. "
            f"Use this as your primary search query if you decide to retrieve."
        ))
        return {"messages": [hint], "rewritten_query": rewritten}

    def agent_node(state: AgentState) -> Dict[str, Any]:
        timings = state["timings"]
        iter_n = state.get("iterations", 0)

        timings.setdefault("t_llm_start", time.perf_counter())
        response = llm_with_tools.invoke(state["messages"])
        timings["t_llm_end"] = time.perf_counter()

        if state.get("debug") is not None:
            dbg = state["debug"]
            steps = dbg.setdefault("agent_steps", [])
            steps.append(
                {
                    "iteration": iter_n,
                    "correction_round": state.get("correction_count", 0),
                    "tool_calls": getattr(response, "tool_calls", []) or [],
                    "content_preview": (getattr(response, "content", "") or "")[:300],
                }
            )

        return {"messages": [response], "iterations": iter_n + 1}

    def should_continue(state: AgentState) -> str:
        """After agent runs: route to tools (if tool_calls) or verifier (if final answer).
        Also force END if the iteration safety valve trips."""
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        if state.get("iterations", 0) >= MAX_AGENT_ITERATIONS:
            logger.warning(
                "Agent hit iteration cap (%d) — forcing END.", MAX_AGENT_ITERATIONS
            )
            return "end"
        if tool_calls:
            return "tools"
        return "verify"

    @observe(name="verifier", as_type="generation")
    def verifier_node(state: AgentState) -> Dict[str, Any]:
        """Grade the agent's final answer against the retrieved passages.

        Returns a state update setting `verifier_verdict` to one of:
          - "ok"        : the answer is grounded → END
          - "retry"     : ungrounded, inject critique, loop back to agent
          - "exhausted" : ungrounded but hit MAX_CORRECTIONS → END anyway
        """
        correction_count = state.get("correction_count", 0)

        # The final answer is the most recent AIMessage with no tool_calls.
        final_answer = ""
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage) and not (getattr(m, "tool_calls", None) or []):
                final_answer = m.content or ""
                break

        # The user's question is the first HumanMessage in the trail.
        question = ""
        for m in state["messages"]:
            if isinstance(m, HumanMessage):
                question = m.content or ""
                break

        # Concatenate every tool result so the grader sees what the agent had.
        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        passages = "\n\n".join((tm.content or "") for tm in tool_msgs).strip() \
            or "(no passages were retrieved)"

        grading_payload = (
            f"USER QUESTION:\n{question}\n\n"
            f"RETRIEVED PASSAGES:\n{passages}\n\n"
            f"CANDIDATE ANSWER:\n{final_answer}"
        )

        try:
            verdict_msg = raw_llm.invoke([
                SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
                HumanMessage(content=grading_payload),
            ])
            parsed = _parse_verifier_verdict(verdict_msg.content or "")
        except Exception as e:
            logger.warning("Verifier LLM call failed — failing open (treating as grounded): %s", e)
            parsed = {"grounded": True, "critique": ""}

        if state.get("debug") is not None:
            state["debug"].setdefault("verifier_checks", []).append(
                {
                    "round": correction_count,
                    "grounded": parsed["grounded"],
                    "critique": parsed["critique"],
                    "answer_under_review": final_answer[:300],
                }
            )

        if parsed["grounded"]:
            return {"verifier_verdict": "ok", "correction_count": correction_count}

        # Not grounded — would we retry, or are we out of budget?
        if correction_count + 1 >= MAX_CORRECTIONS:
            logger.warning(
                "Verifier wanted a retry but MAX_CORRECTIONS=%d hit — ending with last answer.",
                MAX_CORRECTIONS,
            )
            return {
                "verifier_verdict": "exhausted",
                "correction_count": correction_count + 1,
            }

        # Inject a critique and loop back to the agent.
        hint = SystemMessage(
            content=(
                "REVIEWER FEEDBACK on your previous answer: "
                f"{parsed['critique'] or 'unspecified grounding issue'}. "
                "Please revise. You may call the search tool again with a "
                "refined query if you need more passages, then write a new "
                "answer that sticks strictly to what the passages support."
            )
        )
        logger.info(
            "Verifier rejected answer (round %d/%d) — looping back to agent.",
            correction_count + 1,
            MAX_CORRECTIONS,
        )
        return {
            "messages": [hint],
            "verifier_verdict": "retry",
            "correction_count": correction_count + 1,
        }

    def should_finish_after_verify(state: AgentState) -> str:
        return "agent" if state.get("verifier_verdict") == "retry" else "end"

    graph = StateGraph(AgentState)
    graph.add_node("rewriter", rewriter_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("verifier", verifier_node)

    graph.set_entry_point("rewriter")
    graph.add_edge("rewriter", "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "verify": "verifier", "end": END},
    )
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "verifier",
        should_finish_after_verify,
        {"agent": "agent", "end": END},
    )
    return graph.compile()


@observe(name="answer_question")
def answer_question(
    question: str,
    query_vec: List[float],
    collection_name: str,
    history: List[Dict],
    debug: bool = False,
) -> Dict[str, Any]:
    """Agentic RAG orchestration with self-correction.

    Pipeline:
      1. agent_node — LLM with tools bound. Decides to retrieve or answer.
      2. tools      — executes any tool_calls the LLM emitted.
      3. verifier_node — after a final answer, grades it for grounding.
      4. If ungrounded, inject critique and re-enter agent (≤ MAX_CORRECTIONS).

    Returns a dict with answer, stage timings, and (when debug=True) the
    full message trail, agent steps, and verifier checks.

    Raises:
        InferenceError: for unrecoverable LLM / graph failures.
    """
    milvus_store = MilvusStoreHandler(collection_name=collection_name)
    reranker = NVidiaReranker()
    nim_client = NIMClient()

    debug_info: Optional[Dict[str, Any]] = {} if debug else None

    # Build the tool, bind it to the LLM, wrap in a tool node.
    search_chunks = build_search_chunks_tool(milvus_store, reranker)
    tools = [search_chunks]
    llm_with_tools = nim_client.llm.bind_tools(tools)
    tool_node = make_tool_node(tools)

    # The verifier uses the SAME ChatNVIDIA *without* tool binding so it
    # can't be tempted to issue tool_calls instead of grading.
    graph = build_agent_graph(llm_with_tools, nim_client.llm, tool_node)

    history_msgs = history_to_messages(history)
    initial_messages: List[BaseMessage] = [
        SystemMessage(content=nim_client.system_prompt),
        *history_msgs,
        HumanMessage(content=question),
    ]

    timings: Dict[str, float] = {}
    initial_state: AgentState = {
        "messages": initial_messages,
        "timings": timings,
        "debug": debug_info,
        "iterations": 0,
        "correction_count": 0,
    }

    cb = langfuse_callback()
    invoke_config = {"callbacks": [cb]} if cb else {}

    t_graph_start = time.perf_counter()
    try:
        logger.info(
            "Invoking agentic RAG graph (max_iters=%d, max_corrections=%d)",
            MAX_AGENT_ITERATIONS,
            MAX_CORRECTIONS,
        )
        final_state = graph.invoke(initial_state, config=invoke_config)
    except InferenceError:
        raise
    except Exception as e:
        logger.exception("Unexpected error from agentic RAG graph: %s", e)
        raise InferenceError("Unexpected error from agentic RAG graph.") from e
    t_graph_end = time.perf_counter()

    # Final answer: the most recent AIMessage with no tool_calls.
    answer = ""
    for m in reversed(final_state["messages"]):
        if isinstance(m, AIMessage) and not (getattr(m, "tool_calls", None) or []):
            answer = (m.content or "").strip()
            break

    retrieved_any = any(isinstance(m, ToolMessage) for m in final_state["messages"])
    iterations_used = final_state.get("iterations", 0)
    corrections_used = final_state.get("correction_count", 0)
    verdict = final_state.get("verifier_verdict")

    if debug_info is not None:
        debug_info["total_iterations"] = iterations_used
        debug_info["corrections_used"] = corrections_used
        debug_info["verifier_verdict"] = verdict
        debug_info["retrieved_any"] = retrieved_any
        debug_info["final_messages"] = [
            {
                "type": getattr(m, "type", "unknown"),
                "content": (getattr(m, "content", "") or "")[:500],
                "tool_calls": getattr(m, "tool_calls", None) or [],
                "tool_call_id": getattr(m, "tool_call_id", None),
                "name": getattr(m, "name", None),
            }
            for m in final_state["messages"]
        ]

    if config.TOGGLE_CACHE and not debug and retrieved_any and answer:
        try:
            get_cache_store().put_entry(
                question_text=question,
                query_vec=query_vec,
                answer_text=answer,
                context_chunk_ids=[],
                model_name=config.LLM_MODEL,
                prompt_version=config.PROMPT_VERSION,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )
            logger.info("Stored Q/A pair in semantic cache.")
        except Exception as e:
            logger.exception("Cache write failed (non-fatal): %s", e)

    t_llm_start = timings.get("t_llm_start", t_graph_start)
    t_llm_end = timings.get("t_llm_end", t_graph_end)
    t_milvus_start = t_llm_start
    t_milvus_end = t_llm_start

    return {
        "answer": answer,
        "t_milvus_start": t_milvus_start,
        "t_milvus_end": t_milvus_end,
        "t_llm_start": t_llm_start,
        "t_llm_end": t_llm_end,
        "debug": debug_info,
    }
