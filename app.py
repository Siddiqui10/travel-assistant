"""
app.py
-------
Streamlit front-end for the AI Smart Travel Planning Assistant.

Run with:
    streamlit run app.py

Features covered here (mapped to the functional requirements):
- Upload multiple travel guide PDFs           -> sidebar uploader
- Search / ask destination questions          -> chat box (Agentic RAG)
- Generate itinerary / compare / budget       -> same chat box, router decides
- Show which agent workflow handled the query -> shown as a small tag above the answer
- Show source documents used                  -> RAG tool already prints "SOURCES USED"
"""

import os
import tempfile

import streamlit as st

import config
from rag_engine import TravelRAGEngine
import router

st.set_page_config(page_title="AI Smart Travel Planning Assistant", page_icon="✈️", layout="wide")

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = TravelRAGEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = st.session_state.rag_engine.collection.count() > 0


# ------------------------------------------------------------------
# Sidebar - PDF upload + knowledge base status
# ------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Travel Guide Library")
    st.caption("Upload destination PDFs to build the knowledge base used for RAG.")

    uploaded_files = st.file_uploader(
        "Upload travel guide PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Ingest uploaded PDFs", disabled=not uploaded_files):
        with st.spinner("Reading PDFs, creating embeddings, storing in ChromaDB..."):
            total_chunks = 0
            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                n = st.session_state.rag_engine.ingest_pdf(tmp_path)
                total_chunks += n
                os.unlink(tmp_path)
            st.session_state.docs_loaded = True
        st.success(f"Ingested {len(uploaded_files)} file(s), {total_chunks} chunks added.")

    st.divider()
    n_docs = st.session_state.rag_engine.collection.count()
    st.metric("Chunks in knowledge base", n_docs)

    if st.button("Load sample data/ folder"):
        with st.spinner("Ingesting sample PDFs from data/ ..."):
            summary = st.session_state.rag_engine.ingest_folder()
        st.session_state.docs_loaded = True
        st.write(summary)

    st.divider()
    st.caption("Model: " + config.HF_MODEL_REPO)


# ------------------------------------------------------------------
# Main area - chat interface
# ------------------------------------------------------------------
st.title("✈️ AI Smart Travel Planning Assistant")
st.caption("Agentic AI + Agentic RAG travel planner — ask about destinations, itineraries, budgets, weather, or comparisons.")

if not st.session_state.docs_loaded:
    st.info("Upload at least one travel guide PDF (or click 'Load sample data/ folder') to get grounded answers.")

example_cols = st.columns(4)
examples = [
    "Plan a 3 day itinerary for Goa",
    "Compare Manali vs Kerala for a honeymoon",
    "What's the budget for 4 days in Manali for 2 people?",
    "What's the weather like in Goa right now?",
]
for col, ex in zip(example_cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state.pending_query = ex

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "intent" in msg:
            st.caption(f"🧭 Routed to workflow: **{msg['intent']}**")
        st.write(msg["content"])

user_query = st.chat_input("Ask about a destination, request an itinerary, compare places, or check budget...")
if "pending_query" in st.session_state:
    user_query = st.session_state.pop("pending_query")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Agents are working on your request..."):
            result = router.handle_query(user_query)
        st.caption(f"🧭 Routed to workflow: **{result['intent']}**")
        st.write(result["answer"])

    st.session_state.chat_history.append({
        "role": "assistant", "content": result["answer"], "intent": result["intent"]
    })
