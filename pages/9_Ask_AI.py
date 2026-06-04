"""
pages/9_Ask_AI.py
==================
RAG-powered Q&A about the Human Airways Digital Twin project.

Retrieval: TF-IDF cosine similarity over the built-in knowledge base.
Generation: Claude claude-haiku via Anthropic API (optional).
           Without an API key the page shows retrieved context only.

To enable full AI answers:
  - Set ANTHROPIC_API_KEY environment variable, or
  - Add it to .streamlit/secrets.toml:  ANTHROPIC_API_KEY = "sk-ant-..."
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rag import get_knowledge_base, generate_answer, _get_api_key

st.set_page_config(page_title="Ask AI", page_icon="🤖", layout="wide")

st.title("🤖 Ask AI — Human Airways Digital Twin Q&A")
st.caption(
    "Ask anything about the project: methodology, parameters, results, or how to use the app. "
    "Powered by TF-IDF retrieval + Claude claude-haiku."
)

# ── API key status ────────────────────────────────────────────────────────────
has_key = bool(_get_api_key())

if has_key:
    st.success("Claude API connected — full AI answers enabled.", icon="🟢")
else:
    st.warning(
        "No `ANTHROPIC_API_KEY` found. "
        "The page will show relevant documentation excerpts without AI generation.  \n"
        "Add your key to `.streamlit/secrets.toml` or set the environment variable "
        "to unlock full AI answers.",
        icon="⚠️",
    )

st.divider()

# ── Initialise session state ──────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "kb" not in st.session_state:
    with st.spinner("Loading knowledge base…"):
        st.session_state.kb = get_knowledge_base()

kb = st.session_state.kb

# ── Sidebar: example questions ────────────────────────────────────────────────
with st.sidebar:
    st.header("Example Questions")
    examples = [
        "What is POD and how is it used here?",
        "How many DOE snapshots are in the dataset?",
        "What does the glottis area parameter affect?",
        "How accurate is the RBF surrogate?",
        "What is Latin Hypercube Sampling?",
        "How do I run the precomputation script?",
        "What anatomical regions are modelled?",
        "Which parameter has the most effect on pressure?",
        "How fast is the RBF inference?",
        "How do I deploy the app online for free?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}", width='stretch'):
            st.session_state["prefill_question"] = ex

    st.divider()
    if st.button("Clear chat history", width='stretch'):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("**Knowledge base**")
    st.caption(f"{len(kb._docs)} documents indexed")
    for doc in kb._docs:
        st.caption(f"• {doc['title']}")

# ── Chat display ──────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("context"):
            with st.expander("📄 Retrieved context", expanded=False):
                for score, doc in msg["context"]:
                    st.markdown(f"**{doc['title']}** (relevance: {score:.2f})")
                    st.caption(doc["content"][:400] + "…")

# ── Input ─────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_question", "")
question = st.chat_input("Ask a question about the project…") or prefill

if question:
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Retrieve
    with st.spinner("Searching knowledge base…"):
        results = kb.retrieve(question, top_k=3)

    context_chunks = [f"### {doc['title']}\n{doc['content']}" for _, doc in results]

    # Generate or show context
    with st.chat_message("assistant"):
        if has_key:
            with st.spinner("Generating answer…"):
                answer = generate_answer(question, context_chunks)
            if answer:
                st.markdown(answer)
            else:
                st.warning("Generation failed. Showing retrieved context below.")
                for score, doc in results:
                    st.markdown(f"**{doc['title']}** (relevance: {score:.2f})")
                    st.markdown(doc["content"])
                answer = "\n\n".join(f"**{d['title']}**\n{d['content']}" for _, d in results)
        else:
            # No API key: show top retrieved document as the answer
            top_doc = results[0][1]
            answer = (
                f"**{top_doc['title']}**\n\n"
                + top_doc["content"].strip()
                + "\n\n---\n*Enable AI answers by setting your `ANTHROPIC_API_KEY`.*"
            )
            st.markdown(answer)

        with st.expander("📄 Retrieved context", expanded=False):
            for score, doc in results:
                st.markdown(f"**{doc['title']}** (relevance: {score:.2f})")
                st.caption(doc["content"][:400] + "…")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "context": results,
    })
