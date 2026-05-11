"""
Streamlit UI for the local RAG chatbot.

Run with:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.chatbot import RAGChatbot
from src.document_loader import ingest_path
from src.logging_config import setup_logging
from src.vector_store import VectorStore


load_dotenv()
setup_logging()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:0.8b")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "10"))


st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_vector_store() -> VectorStore:
    return VectorStore(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name="documents",
        embedding_model_name=EMBEDDING_MODEL,
    )


def get_chatbot(vector_store: VectorStore) -> RAGChatbot:
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot(
            vector_store=vector_store,
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            top_k=TOP_K,
            history_turns=HISTORY_TURNS,
        )
    return st.session_state.chatbot


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    seen: set = set()
    chips: list[str] = []
    for s in sources:
        key = (s.get("filename"), s.get("page"))
        if key in seen:
            continue
        seen.add(key)
        label = s.get("filename") or "unknown"
        if s.get("page"):
            label += f" (p.{s['page']})"
        chips.append(label)
    if chips:
        st.caption("📎 Sources: " + " · ".join(chips))


def main() -> None:
    st.title("📚 RAG Chatbot")
    st.caption(
        f"Local RAG over your documents · Ollama `{OLLAMA_MODEL}` · ChromaDB · "
        f"multilingual embeddings"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    vs = get_vector_store()
    chatbot = get_chatbot(vs)

    with st.sidebar:
        st.header("Knowledge Base")
        st.metric("Indexed chunks", vs.count())

        uploaded = st.file_uploader(
            "Upload PDFs / TXT / MD",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if st.button("Ingest", type="primary", disabled=not uploaded):
            with st.spinner("Loading, splitting, embedding..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    total = 0
                    failures: list[str] = []
                    for up in uploaded:
                        tmp_path = Path(tmpdir) / up.name
                        tmp_path.write_bytes(up.getbuffer())
                        try:
                            chunks = ingest_path(
                                tmp_path,
                                chunk_size=CHUNK_SIZE,
                                chunk_overlap=CHUNK_OVERLAP,
                            )
                            total += vs.add_chunks(chunks)
                        except Exception as exc:
                            failures.append(f"{up.name}: {exc}")
            if failures:
                for msg in failures:
                    st.error(f"Failed — {msg}")
            if total:
                st.success(f"Ingested {total} chunks from {len(uploaded) - len(failures)} file(s).")
                st.rerun()

        st.divider()

        if st.button("Reset conversation"):
            chatbot.reset_history()
            st.session_state.messages = []
            st.rerun()

        if st.button("Clear knowledge base", type="secondary"):
            vs.clear()
            chatbot.reset_history()
            st.session_state.messages = []
            st.success("Knowledge base cleared.")
            st.rerun()

        st.divider()
        with st.expander("Settings"):
            st.write(f"**Model:** `{OLLAMA_MODEL}`")
            st.write(f"**Ollama URL:** `{OLLAMA_BASE_URL}`")
            st.write(f"**Embeddings:** `{EMBEDDING_MODEL}`")
            st.write(f"**Chunk size / overlap:** {CHUNK_SIZE} / {CHUNK_OVERLAP}")
            st.write(f"**Top-K retrieved:** {TOP_K}")
            st.write(f"**History turns:** {HISTORY_TURNS}")

    if vs.count() == 0:
        st.info("👋 Upload one or more documents in the sidebar to start chatting.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    if user_input := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                response = st.write_stream(chatbot.chat(user_input))
            except Exception as exc:
                response = (
                    f"⚠️ Error talking to Ollama: {exc}\n\n"
                    f"Make sure Ollama is running and the model `{OLLAMA_MODEL}` is pulled "
                    f"(`ollama pull {OLLAMA_MODEL}`)."
                )
                st.error(response)
            sources = chatbot.last_sources
            render_sources(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "sources": sources}
        )


if __name__ == "__main__":
    main()
