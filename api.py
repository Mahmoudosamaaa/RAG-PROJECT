"""
FastAPI service exposing the RAG pipeline.

Run with either:
    uvicorn api:app --reload --port 8001
    python api.py                          (reads API_PORT from .env, default 8001)

Then open:
    http://localhost:8001/docs    (Swagger UI)
    http://localhost:8001/redoc   (alternative API docs)

The Streamlit UI (app.py) and this API can run side-by-side — both read/write
the same on-disk ChromaDB collection.
"""

import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.chatbot import RAGChatbot
from src.document_loader import SUPPORTED_EXTENSIONS, ingest_path
from src.logging_config import setup_logging
from src.vector_store import VectorStore


load_dotenv()
setup_logging()
logger = logging.getLogger("api")


# --- Config (same defaults as app.py) ---
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
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8001"))


# --- Shared state (process-local) ---
_vector_store: Optional[VectorStore] = None
_conversations: dict[str, RAGChatbot] = {}


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name="documents",
            embedding_model_name=EMBEDDING_MODEL,
        )
    return _vector_store


def get_or_create_conversation(conversation_id: Optional[str]) -> tuple[str, RAGChatbot]:
    if conversation_id and conversation_id in _conversations:
        return conversation_id, _conversations[conversation_id]
    new_id = conversation_id or str(uuid.uuid4())
    bot = RAGChatbot(
        vector_store=get_vector_store(),
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        top_k=TOP_K,
        history_turns=HISTORY_TURNS,
    )
    _conversations[new_id] = bot
    logger.info("Created conversation %s", new_id)
    return new_id, bot


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting up — warming up vector store...")
    get_vector_store()
    logger.info("API ready")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="RAG Chatbot API",
    description=(
        "REST API for the local RAG chatbot. "
        "Upload documents, retrieve relevant chunks, chat with citations. "
        "Backed by ChromaDB + sentence-transformers + Ollama."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --- Pydantic schemas ---

class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    embedding_model: str
    indexed_chunks: int
    active_conversations: int


class CountResponse(BaseModel):
    count: int = Field(..., description="Number of chunks currently indexed")


class IngestFileResult(BaseModel):
    filename: str
    chunks_added: int
    error: Optional[str] = None


class IngestResponse(BaseModel):
    results: list[IngestFileResult]
    total_chunks_added: int
    total_chunks_in_store: int


class Source(BaseModel):
    filename: str
    page: Optional[int] = None
    distance: Optional[float] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's question")
    conversation_id: Optional[str] = Field(
        None,
        description="Pass an existing conversation_id to keep memory across turns. "
        "Omit to start a new conversation.",
    )


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[Source]


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["status"])
def health():
    """Service liveness + a snapshot of current state."""
    return HealthResponse(
        status="ok",
        model=OLLAMA_MODEL,
        embedding_model=EMBEDDING_MODEL,
        indexed_chunks=get_vector_store().count(),
        active_conversations=len(_conversations),
    )


@app.get("/collection", response_model=CountResponse, tags=["collection"])
def collection_count():
    """How many chunks are currently in the vector store."""
    return CountResponse(count=get_vector_store().count())


@app.delete("/collection", tags=["collection"])
def collection_clear():
    """Wipe every chunk from the vector store. Conversations keep their history."""
    get_vector_store().clear()
    return {"message": "Collection cleared"}


@app.post("/ingest", response_model=IngestResponse, tags=["collection"])
async def ingest(files: list[UploadFile] = File(..., description="PDF / TXT / MD files")):
    """Upload one or more documents. They're chunked, embedded, and persisted."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    vs = get_vector_store()
    results: list[IngestFileResult] = []
    total_added = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for up in files:
            tmp_path = Path(tmpdir) / (up.filename or "upload.bin")
            suffix = tmp_path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                results.append(
                    IngestFileResult(
                        filename=up.filename or "(unknown)",
                        chunks_added=0,
                        error=f"Unsupported file type: {suffix}",
                    )
                )
                continue

            try:
                content = await up.read()
                tmp_path.write_bytes(content)
                chunks = ingest_path(
                    tmp_path,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )
                added = vs.add_chunks(chunks)
                total_added += added
                results.append(
                    IngestFileResult(
                        filename=up.filename or tmp_path.name,
                        chunks_added=added,
                    )
                )
            except Exception as exc:
                logger.exception("Ingest failed for %s", up.filename)
                results.append(
                    IngestFileResult(
                        filename=up.filename or "(unknown)",
                        chunks_added=0,
                        error=str(exc),
                    )
                )

    return IngestResponse(
        results=results,
        total_chunks_added=total_added,
        total_chunks_in_store=vs.count(),
    )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest):
    """Ask a question. Returns the full answer and the sources used."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    conv_id, bot = get_or_create_conversation(req.conversation_id)
    try:
        answer = "".join(bot.chat(req.message))
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

    sources = [
        Source(
            filename=s.get("filename", "unknown"),
            page=s.get("page"),
        )
        for s in bot.last_sources
    ]
    return ChatResponse(conversation_id=conv_id, answer=answer, sources=sources)


@app.post("/chat/stream", tags=["chat"])
def chat_stream(req: ChatRequest):
    """Same as /chat, but streams the answer as plain text.

    Tip: Swagger's "Try it out" waits for the full response — for a true
    streaming experience use `curl -N` or a JS fetch client.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    conv_id, bot = get_or_create_conversation(req.conversation_id)

    def generate():
        try:
            for token in bot.chat(req.message):
                yield token
        except Exception as exc:
            logger.exception("Streaming chat failed")
            yield f"\n[ERROR] {exc}"

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"X-Conversation-Id": conv_id},
    )


@app.post("/conversations/{conversation_id}/reset", tags=["chat"])
def reset_conversation(conversation_id: str):
    """Clear the conversation memory for a given conversation_id."""
    bot = _conversations.get(conversation_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Conversation not found")
    bot.reset_history()
    return {"message": "Conversation reset", "conversation_id": conversation_id}


@app.delete("/conversations/{conversation_id}", tags=["chat"])
def delete_conversation(conversation_id: str):
    """Forget a conversation entirely."""
    if conversation_id not in _conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    del _conversations[conversation_id]
    return {"message": "Conversation deleted", "conversation_id": conversation_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
