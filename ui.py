"""
Streamlit UI for testing multi-user / multi-session chat against the FastAPI app.

Run:
    uvicorn app:app --reload   (in one terminal)
    streamlit run ui.py        (in another)

Open multiple browser tabs to simulate concurrent users.
"""

import os
from typing import List, Dict, Any, Optional

import requests
import streamlit as st

API_BASE = os.getenv("CHAT_API_BASE", "http://localhost:8000")
DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION_NAME", "docs")
TIMEOUT = int(os.getenv("UI_REQUEST_TIMEOUT", "120"))


# --- API helpers --------------------------------------------------------------

def list_conversations(user_external_id: str) -> List[Dict[str, Any]]:
    r = requests.get(
        f"{API_BASE}/list_conversations",
        params={"user_external_id": user_external_id},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json().get("conversations", [])


def list_messages(user_external_id: str, conversation_id: str) -> List[Dict[str, Any]]:
    r = requests.get(
        f"{API_BASE}/list_messages",
        params={
            "user_external_id": user_external_id,
            "conversation_id": conversation_id,
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json().get("messages", [])


def post_chat(
    user_external_id: str,
    question: str,
    collection_name: str,
    conversation_id: Optional[str],
    debug: bool = False,
) -> Dict[str, Any]:
    payload = {
        "user_external_id": user_external_id,
        "question": question,
        "collection_name": collection_name,
        "debug": debug,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    r = requests.post(f"{API_BASE}/chat", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def render_debug_panel(debug: Dict[str, Any]) -> None:
    """Render an expander showing retrieval / rerank / prompt / timings."""
    with st.expander("🐛 Debug — retrieval, rerank, prompt, timings", expanded=False):
        timings = debug.get("timings_ms") or {}
        if timings:
            cols = st.columns(len(timings))
            for col, (k, v) in zip(cols, timings.items()):
                col.metric(k, f"{v:.0f} ms")

        tab_retrieved, tab_hybrid, tab_reranked, tab_prompt, tab_history = st.tabs(
            ["Retrieved", "Hybrid Diagnostic", "Reranked", "Prompt", "History"]
        )

        with tab_retrieved:
            chunks = debug.get("retrieved_chunks") or []
            st.caption(f"{len(chunks)} chunks retrieved from Milvus (hybrid dense + BM25)")
            for i, c in enumerate(chunks, 1):
                score = c.get("score")
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
                st.markdown(
                    f"**#{i}** — score `{score_str}` · source `{c.get('source')}` · "
                    f"chunk_order `{c.get('chunk_order')}`"
                )
                st.text((c.get("text") or "")[:1000])
                st.divider()

        with tab_hybrid:
            dense = debug.get("dense_only_chunks") or []
            sparse = debug.get("sparse_only_chunks") or []
            hybrid = debug.get("retrieved_chunks") or []

            dense_ids = {str(c.get("id")) for c in dense if c.get("id") is not None}
            sparse_ids = {str(c.get("id")) for c in sparse if c.get("id") is not None}
            both_ids = dense_ids & sparse_ids
            sparse_only_ids = sparse_ids - dense_ids

            st.caption(
                "Side-by-side: pure dense (cosine on the embedding) vs. pure BM25 "
                "(keyword on the text). The hybrid result is RRF-fused from both."
            )
            mcols = st.columns(4)
            mcols[0].metric("Dense top-N", len(dense))
            mcols[1].metric("BM25 top-N", len(sparse))
            mcols[2].metric("Overlap", len(both_ids))
            mcols[3].metric("BM25-only", len(sparse_only_ids))

            st.markdown(
                "_BM25-only_ chunks are the ones that pure dense embedding would **miss** — "
                "the value-add of hybrid retrieval. _Dense-only_ chunks are matches by meaning "
                "without exact-term overlap."
            )

            col_dense, col_sparse = st.columns(2)
            with col_dense:
                st.markdown("### 🔹 Dense (cosine)")
                if not dense:
                    st.caption("No dense results.")
                for i, c in enumerate(dense, 1):
                    cid = str(c.get("id"))
                    in_sparse = cid in sparse_ids
                    badge = "🤝 also BM25" if in_sparse else "🔹 dense-only"
                    score = c.get("score")
                    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
                    st.markdown(f"**#{i}** `{score_str}` · {badge}")
                    st.text((c.get("text") or "")[:400])
                    st.divider()

            with col_sparse:
                st.markdown("### 🔸 BM25 (keyword)")
                if not sparse:
                    st.caption("No BM25 results.")
                for i, c in enumerate(sparse, 1):
                    cid = str(c.get("id"))
                    in_dense = cid in dense_ids
                    badge = "🤝 also dense" if in_dense else "🔸 BM25-only"
                    score = c.get("score")
                    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
                    st.markdown(f"**#{i}** `{score_str}` · {badge}")
                    st.text((c.get("text") or "")[:400])
                    st.divider()

            if hybrid:
                st.markdown("---")
                st.markdown("### 🟣 Hybrid (RRF-fused, what the reranker actually saw)")
                hyb_only_dense = sum(1 for c in hybrid if str(c.get("id")) in dense_ids and str(c.get("id")) not in sparse_ids)
                hyb_only_sparse = sum(1 for c in hybrid if str(c.get("id")) in sparse_ids and str(c.get("id")) not in dense_ids)
                hyb_both = sum(1 for c in hybrid if str(c.get("id")) in (dense_ids & sparse_ids))
                st.caption(
                    f"Of the {len(hybrid)} hybrid hits: "
                    f"{hyb_both} from both, {hyb_only_dense} dense-only, {hyb_only_sparse} BM25-only."
                )

        with tab_reranked:
            top_k = debug.get("reranked_top_k") or []
            full = debug.get("reranked_chunks") or []
            st.caption(
                f"{len(top_k)} chunks sent to the LLM (top of {len(full)} reranked). "
                "Rerank order shown."
            )
            for i, c in enumerate(top_k, 1):
                rscore = c.get("rerank_score")
                ms = c.get("score")
                rscore_str = f"{rscore:.4f}" if isinstance(rscore, (int, float)) else "n/a"
                ms_str = f"{ms:.4f}" if isinstance(ms, (int, float)) else "n/a"
                st.markdown(
                    f"**#{i}** — rerank `{rscore_str}` · milvus `{ms_str}` · "
                    f"source `{c.get('source')}`"
                )
                st.text((c.get("text") or "")[:1000])
                st.divider()

        with tab_prompt:
            messages = debug.get("rendered_prompt") or []
            for m in messages:
                role = m.get("role", "?")
                content = m.get("content", "")
                st.markdown(f"**{role.upper()}**")
                st.text(content)
                st.divider()

        with tab_history:
            st.text(debug.get("history_text") or "(empty)")


# --- UI -----------------------------------------------------------------------

st.set_page_config(page_title="Talk with Harry", layout="wide")

if "user_external_id" not in st.session_state:
    st.session_state.user_external_id = ""
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection_name" not in st.session_state:
    st.session_state.collection_name = DEFAULT_COLLECTION
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
# Debug payloads keyed by index in messages list; only present for assistant turns.
if "debug_by_idx" not in st.session_state:
    st.session_state.debug_by_idx = {}


with st.sidebar:
    st.header("Session")
    user_input = st.text_input(
        "User external ID",
        value=st.session_state.user_external_id,
        placeholder="e.g. alice, bob, vamsi@example.com",
    )
    collection = st.text_input(
        "Milvus collection",
        value=st.session_state.collection_name,
    )

    if user_input != st.session_state.user_external_id:
        st.session_state.user_external_id = user_input
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.session_state.debug_by_idx = {}
    st.session_state.collection_name = collection

    st.session_state.debug_mode = st.toggle(
        "🐛 Debug mode",
        value=st.session_state.debug_mode,
        help="Bypasses cache and shows retrieved chunks, reranked order, and the rendered prompt.",
    )

    st.divider()
    st.subheader("Conversations")

    if not st.session_state.user_external_id:
        st.caption("Enter a user ID above to load conversations.")
    else:
        if st.button("➕ New conversation", use_container_width=True):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()

        try:
            convs = list_conversations(st.session_state.user_external_id)
        except requests.RequestException as e:
            st.error(f"Failed to load conversations: {e}")
            convs = []

        if not convs:
            st.caption("No conversations yet — send a message to start one.")
        else:
            for c in convs:
                cid = c["conversation_id"]
                title = c.get("title") or "(untitled)"
                label = f"💬 {title}\n\n`{cid[:8]}…` · {c.get('updated_at','')[:19]}"
                is_active = cid == st.session_state.conversation_id
                if st.button(
                    label,
                    key=f"conv_{cid}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.conversation_id = cid
                    st.session_state.debug_by_idx = {}
                    try:
                        st.session_state.messages = list_messages(
                            st.session_state.user_external_id, cid
                        )
                    except requests.RequestException as e:
                        st.error(f"Failed to load messages: {e}")
                        st.session_state.messages = []
                    st.rerun()

    st.divider()
    st.caption(f"API: `{API_BASE}`")
    if st.session_state.conversation_id:
        st.caption(f"Active conv: `{st.session_state.conversation_id[:8]}…`")
    else:
        st.caption("Active conv: _new_")


# Main pane
st.title("Bodycam Search Chat")

if not st.session_state.user_external_id:
    st.info("👈 Enter a user ID in the sidebar to start chatting.")
    st.stop()

# Render history
for idx, msg in enumerate(st.session_state.messages):
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant":
            dbg = st.session_state.debug_by_idx.get(idx)
            if dbg:
                render_debug_panel(dbg)

# Input
prompt = st.chat_input("Ask a question…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        debug_payload: Optional[Dict[str, Any]] = None
        with st.spinner("Thinking…"):
            try:
                resp = post_chat(
                    user_external_id=st.session_state.user_external_id,
                    question=prompt,
                    collection_name=st.session_state.collection_name,
                    conversation_id=st.session_state.conversation_id,
                    debug=st.session_state.debug_mode,
                )
                answer = resp.get("answer", "(no answer)")
                st.session_state.conversation_id = resp.get("conversation_id")
                debug_payload = resp.get("debug")
            except requests.HTTPError as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    detail = e.response.text if e.response is not None else ""
                answer = f"⚠️ HTTP {e.response.status_code if e.response else '?'}: {detail}"
            except requests.RequestException as e:
                answer = f"⚠️ Request failed: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        if debug_payload:
            assistant_idx = len(st.session_state.messages) - 1
            st.session_state.debug_by_idx[assistant_idx] = debug_payload
            render_debug_panel(debug_payload)

    st.rerun()
