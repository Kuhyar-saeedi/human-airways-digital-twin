"""
pages/9_Ask_AI.py
==================
Fully offline RAG Q&A — works without any API keys.

Retrieval: TF-IDF (always) or sentence-transformers (if installed, better quality).
Answer:    Local extractive composition from 29 knowledge base documents.
Upgrade:   If ANTHROPIC_API_KEY is set, Claude claude-haiku generates a polished answer.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rag import (
    get_knowledge_base, compose_local_answer,
    generate_answer, _get_api_key, _HAS_SEMANTIC,
)

st.set_page_config(page_title="Ask AI", page_icon="🤖", layout="wide")

st.title("🤖 Ask AI — Human Airways Digital Twin Q&A")
st.caption("Ask anything about the project. Works fully offline — no API key needed.")

# ── Status banner ─────────────────────────────────────────────────────────────
if _get_api_key():
    st.success("Claude API connected — AI-generated answers enabled.", icon="🟢")
elif _HAS_SEMANTIC:
    st.info("Offline mode — semantic search (sentence-transformers) + local answers.", icon="🔵")
else:
    st.info("Offline mode — TF-IDF search + local answers. All 29 KB docs indexed.", icon="📚")

# ── Load KB ───────────────────────────────────────────────────────────────────
if "kb" not in st.session_state:
    with st.spinner("Loading knowledge base…"):
        st.session_state.kb = get_knowledge_base()
kb = st.session_state.kb

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Try these questions")
    examples = [
        "What is POD and how is it used here?",
        "Why does pressure need only 3 modes?",
        "Which parameter affects pressure the most?",
        "How accurate is the RBF surrogate?",
        "What is airway resistance and why does it matter?",
        "How do Sobol indices differ from Pearson correlation?",
        "How many nodes are in the mesh?",
        "What does the design landscape show?",
        "How do I open VTK files in ParaView?",
        "How do I run the PyQt desktop app?",
        "What is K-fold cross-validation?",
        "How do I deploy the app online for free?",
        "What is a Digital Twin?",
        "How to use the Geometry Explorer colour modes?",
        "What does precompute.py do?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:25]}", use_container_width=True):
            st.session_state["prefill"] = ex

    st.divider()
    st.markdown(f"**Knowledge base: {len(kb._docs)} documents**")
    with st.expander("View all topics"):
        for doc in kb._docs:
            st.caption(f"• {doc['title']}")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for score, doc in msg["sources"]:
                    st.markdown(f"**{doc['title']}** — relevance: {score:.3f}")

# ── Input ─────────────────────────────────────────────────────────────────────
prefill   = st.session_state.pop("prefill", "")
question  = st.chat_input("Ask anything about the project…") or prefill

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Retrieve
    results = kb.retrieve(question, top_k=3)

    # Generate answer
    has_key = bool(_get_api_key())
    if has_key:
        context_chunks = [
            f"### {doc['title']}\n{doc['content']}" for _, doc in results
        ]
        with st.spinner("Generating answer…"):
            answer = generate_answer(question, context_chunks)
        if not answer:
            answer = compose_local_answer(question, results, kb)
    else:
        answer = compose_local_answer(question, results, kb)

    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("📄 Sources", expanded=False):
            for score, doc in results:
                st.markdown(f"**{doc['title']}** — relevance: {score:.3f}")
                st.caption(doc["content"][:300] + "…")

    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": answer,
        "sources": results,
    })
