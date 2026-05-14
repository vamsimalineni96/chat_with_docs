"""Visualize the agentic RAG graph.

Builds the same graph that `answer_question` uses, then renders it three ways:

  1. Mermaid source code (always works) → printed to stdout AND saved to
     `graph.mmd`. Paste into https://mermaid.live to view.
  2. PNG via mermaid.ink (needs internet)            → saved to `graph.png`.
  3. ASCII art (needs `pip install grandalf`)        → printed to stdout.

Usage:
    python visualize_graph.py
"""

from langchain_core.messages import AIMessage
from src.root import PROJECT_ROOT
from src.utils.rag_pipeline import build_agent_graph, make_tool_node
from src.utils.tools import build_search_chunks_tool


# Minimal stubs — just enough for the graph to compile. No real LLM / Milvus
# calls happen during visualization.
class _Stub:
    def search_similar_chunks(self, **kw): return []
    def execute(self, **kw): return []


class _DummyLLM:
    def bind_tools(self, tools):
        class _Bound:
            def invoke(self, _): return AIMessage(content="dummy")
        return _Bound()

    def invoke(self, _):
        return AIMessage(content="dummy")


def main():
    tool = build_search_chunks_tool(_Stub(), _Stub())
    llm = _DummyLLM()
    graph = build_agent_graph(llm.bind_tools([tool]), llm, make_tool_node([tool]))

    g = graph.get_graph()

    mmd_path = PROJECT_ROOT / "graph.mmd"
    png_path = PROJECT_ROOT / "graph.png"

    # 1. Mermaid source — always works, no extra deps.
    mermaid_src = g.draw_mermaid()
    print("─── Mermaid source ─── (paste into https://mermaid.live)\n")
    print(mermaid_src)
    with open(mmd_path, "w") as f:
        f.write(mermaid_src)
    print(f"→ saved to {mmd_path}\n")

    # 2. PNG via mermaid.ink (needs internet but no local deps).
    try:
        png_bytes = g.draw_mermaid_png()
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        print(f"→ saved rendered PNG to {png_path}\n")
    except Exception as e:
        print(f"(skip PNG: {e})\n")

    # 3. ASCII art — needs `pip install grandalf`.
    try:
        print("─── ASCII rendering ───\n")
        g.print_ascii()
    except Exception as e:
        print(f"(skip ASCII: install `grandalf` with `pip install grandalf`)")
        print(f"  reason: {e}")


if __name__ == "__main__":
    main()
