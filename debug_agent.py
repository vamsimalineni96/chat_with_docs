"""
Debug the agentic RAG loop end-to-end.

Bypasses FastAPI / Streamlit / Postgres entirely. Builds the graph, streams it,
and prints exactly what each node added to `state["messages"]` at every step.
Useful for seeing the LLM's tool-call decisions in their raw form.

Usage:
    python debug_agent.py "your question here"
    python debug_agent.py "your question here" docs    # second arg = collection

Example:
    python debug_agent.py "Compare how Hermione behaved at the Yule Ball vs. the second task"
"""

import sys
import json

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.utils import config
from src.utils.observability import langfuse_callback
from src.utils.rag_pipeline import build_agent_graph, make_tool_node
from src.utils.services.chunk_ranking import NVidiaReranker
from src.utils.services.inference import NIMClient
from src.utils.services.milvus_store import MilvusStoreHandler
from src.utils.tools import build_search_chunks_tool


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printers
# ─────────────────────────────────────────────────────────────────────────────

CLS_COLORS = {
    "SystemMessage": "\033[36m",   # cyan
    "HumanMessage":  "\033[32m",   # green
    "AIMessage":     "\033[35m",   # magenta
    "ToolMessage":   "\033[33m",   # yellow
}
RESET = "\033[0m"


def format_message(m: BaseMessage, idx: int = None) -> str:
    cls = type(m).__name__
    colour = CLS_COLORS.get(cls, "")
    header = f"{colour}[{idx}] {cls}{RESET}" if idx is not None else f"{colour}{cls}{RESET}"

    parts = [header]

    name = getattr(m, "name", None)
    if name:
        parts.append(f"    name           = {name}")

    tc_id = getattr(m, "tool_call_id", None)
    if tc_id:
        parts.append(f"    tool_call_id   = {tc_id}")

    content = m.content or ""
    if content:
        snippet = content if len(content) < 400 else content[:400] + " …(truncated)"
        parts.append(f"    content        = {snippet!r}")

    tool_calls = getattr(m, "tool_calls", None) or []
    if tool_calls:
        parts.append(f"    tool_calls     = (list, {len(tool_calls)} entries)")
        for tc_idx, tc in enumerate(tool_calls):
            tc_dict = tc if isinstance(tc, dict) else {
                "name": getattr(tc, "name", None),
                "args": getattr(tc, "args", {}),
                "id":   getattr(tc, "id", None),
            }
            parts.append(
                f"      [{tc_idx}] id={tc_dict.get('id')!r}  "
                f"name={tc_dict.get('name')!r}  "
                f"args={json.dumps(tc_dict.get('args', {}), ensure_ascii=False)}"
            )

    return "\n".join(parts)


def banner(text: str, char: str = "─") -> str:
    line = char * 78
    return f"\n{line}\n  {text}\n{line}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} "your question" [collection_name]')
        sys.exit(1)

    question = sys.argv[1]
    collection = sys.argv[2] if len(sys.argv) > 2 else config.COLLECTION_NAME

    print(banner(f"Q: {question}", "═"))
    print(f"  collection: {collection}")
    print(f"  llm model:  {config.LLM_MODEL}")

    # Build everything fresh — no history, no cache, no FastAPI layer.
    milvus_store = MilvusStoreHandler(collection_name=collection)
    reranker = NVidiaReranker()
    nim_client = NIMClient()

    search_chunks = build_search_chunks_tool(milvus_store, reranker)
    tools = [search_chunks]
    llm_with_tools = nim_client.llm.bind_tools(tools)
    tool_node = make_tool_node(tools)
    graph = build_agent_graph(llm_with_tools, tool_node)

    initial_messages = [
        SystemMessage(content=nim_client.system_prompt),
        HumanMessage(content=question),
    ]
    initial_state = {
        "messages": initial_messages,
        "timings": {},
        "debug": None,
        "iterations": 0,
    }

    print(banner("Initial state.messages"))
    for i, m in enumerate(initial_messages):
        print(format_message(m, i))

    # Stream node updates so we can print each node's effect on state.
    cb = langfuse_callback()
    invoke_config = {"callbacks": [cb]} if cb else {}

    # Mirror the state ourselves so we can show the running message count.
    running_messages = list(initial_messages)
    iterations = 0
    correction_count = 0
    verifier_verdict = None
    step = 0

    for event in graph.stream(initial_state, config=invoke_config, stream_mode="updates"):
        for node_name, update in event.items():
            # A node that returns `{}` produces `update=None` from LangGraph.
            update = update or {}
            step += 1
            new_msgs = update.get("messages", []) or []
            running_messages.extend(new_msgs)
            if "iterations" in update:
                iterations = update["iterations"]
            if "correction_count" in update:
                correction_count = update["correction_count"]
            if "verifier_verdict" in update:
                verifier_verdict = update["verifier_verdict"]

            print(banner(
                f"Step {step} — node `{node_name}`  "
                f"(iter={iterations}, corrections={correction_count}, "
                f"verdict={verifier_verdict!r}, "
                f"len(messages)={len(running_messages)})"
            ))

            # Full state.messages after this node fired. Mark which entries
            # this node just appended with a ► so the delta is still obvious.
            new_count = len(new_msgs)
            new_start_idx = len(running_messages) - new_count
            print("  state.messages =")
            for i, m in enumerate(running_messages):
                marker = "►" if i >= new_start_idx and new_count > 0 else " "
                lines = format_message(m, i).splitlines()
                # Prefix the header line with the delta marker.
                print(f"  {marker} {lines[0]}")
                for line in lines[1:]:
                    print(f"    {line}")
            if not new_msgs:
                print("  (no message updates from this node)")

    # Final full trail
    print(banner("Final state.messages — the full conversation trail", "═"))
    for i, m in enumerate(running_messages):
        print(format_message(m, i))
        print()

    print(banner("Summary", "═"))
    print(f"  total iterations:     {iterations}")
    tool_msgs = [m for m in running_messages if isinstance(m, ToolMessage)]
    print(f"  total tool calls:     {len(tool_msgs)}")
    print(f"  corrections used:     {correction_count}")
    print(f"  verifier final verdict: {verifier_verdict!r}")
    # Pull the final-answer AIMessage (skipping any trailing critique SystemMessage).
    final_answer_msg = next(
        (m for m in reversed(running_messages)
         if type(m).__name__ == "AIMessage" and not getattr(m, "tool_calls", None)),
        running_messages[-1],
    )
    print(f"  final answer length:  {len(final_answer_msg.content or '')} chars")
    print()
    print("FINAL ANSWER:")
    print(final_answer_msg.content or "(empty)")


if __name__ == "__main__":
    main()
