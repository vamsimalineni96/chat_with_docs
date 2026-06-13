"""
Streamlit UI for testing multi-user / multi-session chat against the FastAPI app.

Run:
    uvicorn app:app --reload   (in one terminal)
    streamlit run ui.py        (in another)

Open multiple browser tabs to simulate concurrent users.
"""

import os
from typing import Any

import requests
import streamlit as st

API_BASE = os.getenv("CHAT_API_BASE", "http://localhost:8000")
DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION_NAME", "shopco_docs")
TIMEOUT = int(os.getenv("UI_REQUEST_TIMEOUT", "200"))


# --- API helpers --------------------------------------------------------------

def list_conversations(user_external_id: str) -> list[dict[str, Any]]:
    r = requests.get(
        f"{API_BASE}/list_conversations",
        params={"user_external_id": user_external_id},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json().get("conversations", [])


def list_messages(user_external_id: str, conversation_id: str) -> list[dict[str, Any]]:
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


def post_approve(
    approval_token: str,
    decision: str,
    user_external_id: str,
    conversation_id: str,
    selected_payment_intent_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "approval_token": approval_token,
        "decision": decision,
        "user_external_id": user_external_id,
        "conversation_id": conversation_id,
    }
    if selected_payment_intent_id:
        payload["selected_payment_intent_id"] = selected_payment_intent_id
    r = requests.post(f"{API_BASE}/approve", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def post_chat(
    user_external_id: str,
    question: str,
    collection_name: str,
    conversation_id: str | None,
    debug: bool = False,
) -> dict[str, Any]:
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


def render_debug_panel(debug: dict[str, Any]) -> None:
    """Render an expander showing retrieval / rerank / prompt / timings."""
    with st.expander("🐛 Debug — retrieval, rerank, prompt, timings", expanded=False):
        timings = debug.get("timings_ms") or {}
        if timings:
            cols = st.columns(len(timings))
            for col, (k, v) in zip(cols, timings.items(), strict=False):
                col.metric(k, f"{v:.0f} ms")

        tab_retrieved, tab_hybrid, tab_reranked, tab_prompt, tab_history = st.tabs(
            ["Retrieved", "Hybrid Diagnostic", "Reranked", "Prompt", "History"]
        )

        with tab_retrieved:
            chunks = debug.get("retrieved_chunks") or []
            st.caption(f"{len(chunks)} chunks retrieved from Milvus (hybrid dense + BM25)")
            for i, c in enumerate(chunks, 1):
                score = c.get("score")
                score_str = f"{score:.4f}" if isinstance(score, int | float) else "n/a"
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
                    score_str = f"{score:.4f}" if isinstance(score, int | float) else "n/a"
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
                    score_str = f"{score:.4f}" if isinstance(score, int | float) else "n/a"
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
                rscore_str = f"{rscore:.4f}" if isinstance(rscore, int | float) else "n/a"
                ms_str = f"{ms:.4f}" if isinstance(ms, int | float) else "n/a"
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
if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None


with st.sidebar:
    st.header("Session")
    user_input = st.text_input(
        "User external ID",
        value=st.session_state.user_external_id,
        placeholder="e.g. alice, bob, vamsi@example.com",
    )
    if user_input != st.session_state.user_external_id:
        st.session_state.user_external_id = user_input
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.session_state.debug_by_idx = {}
    st.session_state.collection_name = DEFAULT_COLLECTION

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
st.title("Chat")

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
        debug_payload: dict[str, Any] | None = None
        pending_approval: dict[str, Any] | None = None
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
                pending_approval = resp.get("pending_approval")
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status == 504:
                    answer = (
                        "⏱️ This request is taking longer than usual — the AI is working "
                        "on a complex multi-step answer. Please try again in a moment."
                    )
                elif status == 409:
                    answer = "⏳ Another message is being processed. Please wait a moment before sending again."
                elif status == 502:
                    answer = "⚠️ The AI service is temporarily unavailable. Please try again."
                else:
                    try:
                        detail = e.response.json().get("detail", "")
                        if isinstance(detail, dict):
                            detail = detail.get("message", str(detail))
                    except Exception:
                        detail = "An unexpected error occurred."
                    answer = f"⚠️ Something went wrong: {detail}"
            except requests.Timeout:
                answer = (
                    "⏱️ The request timed out — the AI may still be processing. "
                    "Please try again in a moment."
                )
            except requests.RequestException as e:
                answer = f"⚠️ Could not reach the server: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        if pending_approval:
            st.session_state.pending_approval = pending_approval
        if debug_payload:
            assistant_idx = len(st.session_state.messages) - 1
            st.session_state.debug_by_idx[assistant_idx] = debug_payload
            render_debug_panel(debug_payload)

    st.rerun()


# ---------------------------------------------------------------------------
# HITL card — rendered at main screen level so it survives rerun.
# Two shapes:
#   - kind="approval": Approve/Reject for a single pre-identified payment
#   - kind="disambig": one button per candidate; clicking IS the approval
# ---------------------------------------------------------------------------
if st.session_state.pending_approval:
    pa = st.session_state.pending_approval
    kind = pa.get("kind", "approval")
    st.markdown("---")

    if kind == "disambig":
        st.markdown("### 🔀 Multiple matching payments")
        st.info(pa.get("display", "Please pick which payment to refund."))
        candidates = pa.get("candidates") or []
        for cand in candidates:
            cid = cand.get("id", "")
            amt_cents = cand.get("amount_cents", 0) or 0
            currency = (cand.get("currency") or "usd").upper()
            desc = cand.get("description") or "(no description)"
            label = (
                f"💸 Refund ${amt_cents/100:.2f} {currency} — {desc}\n\n`{cid}`"
            )
            if st.button(label, key=f"disambig_{cid}", use_container_width=True):
                with st.spinner("Processing refund…"):
                    try:
                        resp = post_approve(
                            approval_token=pa["approval_token"],
                            decision="approved",
                            user_external_id=st.session_state.user_external_id,
                            conversation_id=st.session_state.conversation_id,
                            selected_payment_intent_id=cid,
                        )
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": resp.get("answer", "Refund processed successfully.")
                        })
                        st.session_state.pending_approval = None
                        st.rerun()
                    except requests.RequestException as ex:
                        st.error(f"Failed: {ex}")
        if st.button("❌ Cancel — none of the above", use_container_width=True, key="disambig_cancel"):
            with st.spinner("Cancelling…"):
                try:
                    resp = post_approve(
                        approval_token=pa["approval_token"],
                        decision="rejected",
                        user_external_id=st.session_state.user_external_id,
                        conversation_id=st.session_state.conversation_id,
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": resp.get("answer", "Refund cancelled.")
                    })
                    st.session_state.pending_approval = None
                    st.rerun()
                except requests.RequestException as ex:
                    st.error(f"Failed: {ex}")
    else:
        st.markdown("### ⚠️ Approval Required")
        st.warning(f"**{pa.get('display', 'Action requires your confirmation')}**")

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("✅ Approve", type="primary", use_container_width=True):
                with st.spinner("Processing refund…"):
                    try:
                        resp = post_approve(
                            approval_token=pa["approval_token"],
                            decision="approved",
                            user_external_id=st.session_state.user_external_id,
                            conversation_id=st.session_state.conversation_id,
                        )
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": resp.get("answer", "Refund processed successfully.")
                        })
                        st.session_state.pending_approval = None
                        st.rerun()
                    except requests.RequestException as ex:
                        st.error(f"Failed: {ex}")
        with col2:
            if st.button("❌ Reject", use_container_width=True):
                with st.spinner("Cancelling…"):
                    try:
                        resp = post_approve(
                            approval_token=pa["approval_token"],
                            decision="rejected",
                            user_external_id=st.session_state.user_external_id,
                            conversation_id=st.session_state.conversation_id,
                        )
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": resp.get("answer", "Refund cancelled.")
                        })
                        st.session_state.pending_approval = None
                        st.rerun()
                    except requests.RequestException as ex:
                        st.error(f"Failed: {ex}")
