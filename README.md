# RAG Chatbot

Chat with your own documents. PDFs / TXT / MD in, grounded answers with citations out.

Inspired by [Reenu90/rag-chatbot](https://github.com/Reenu90/rag-chatbot), with two changes for a fully free / local setup:

- **LLM:** Ollama (`qwen3.5:0.8b`) instead of Anthropic Claude
- **Embeddings:** multilingual MiniLM-L12-v2 instead of L6-v2 (handles Arabic & 50+ other languages)

ChromaDB, Streamlit, sentence-transformers, chunking, citations, conversation memory — all match the upstream design.

## Architecture

```
app.py                     Streamlit UI       (port 8501)
api.py                     FastAPI service    (port 8001, Swagger at /docs)
src/document_loader.py     PDF / TXT / MD load + chunk
src/vector_store.py        ChromaDB persistent vector store
src/chatbot.py             Retrieval + prompt + Ollama streaming
src/logging_config.py      Centralized terminal logging
chroma_db/                 Auto-created on first run
```

The Streamlit UI and the FastAPI service share the **same on-disk ChromaDB**.
Run them side-by-side or pick whichever you prefer.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- The model pulled locally:
  ```powershell
  ollama pull qwen3.5:0.8b
  ```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## Run

### Option A — Streamlit UI

```powershell
streamlit run app.py
```

Open http://localhost:8501. Upload some documents from the sidebar, click **Ingest**, then ask questions.

### Option B — FastAPI service with Swagger UI

```powershell
python api.py                              # reads API_PORT from .env (default 8001)
# or equivalently:
uvicorn api:app --reload --port 8001
```

Open http://localhost:8001/docs for the interactive Swagger UI, or http://localhost:8001/redoc for ReDoc.

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service status + model + indexed-chunk count |
| `GET` | `/collection` | Number of indexed chunks |
| `DELETE` | `/collection` | Wipe the vector store |
| `POST` | `/ingest` | Upload PDF/TXT/MD files (multipart) |
| `POST` | `/chat` | Ask a question — returns full answer + sources (JSON) |
| `POST` | `/chat/stream` | Ask a question — streams the answer as plain text |
| `POST` | `/conversations/{id}/reset` | Clear memory for a conversation |
| `DELETE` | `/conversations/{id}` | Forget a conversation entirely |

Pass `conversation_id` in the chat body to keep multi-turn memory. Omit it to start a new conversation — the response's `conversation_id` is the id to reuse on the next call.

**curl example (streaming):**

```powershell
curl -N -X POST http://localhost:8001/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"message": "What is the Transformer architecture?"}'
```

### Run both at once

Open two terminals — both processes share the same `./chroma_db/`:

```powershell
# terminal 1
streamlit run app.py

# terminal 2
python api.py
```

## Terminal logs

Every step of the RAG pipeline is logged to the terminal where you launched the process — document loads, chunk counts, embedding times, retrieval matches and distances, LLM streaming durations, conversation resets. Set `LOG_LEVEL=DEBUG` in `.env` for per-match debug detail (filename + page + distance for every retrieved chunk).

## Configuration

All settings live in `.env`. Highlights:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3.5:0.8b` | Any model you've pulled in Ollama |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Any sentence-transformers model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Splitter window |
| `TOP_K` | `5` | Chunks retrieved per query |
| `HISTORY_TURNS` | `10` | Conversation memory window |

## Cost

Zero. Everything runs on your machine — Ollama serves the LLM, sentence-transformers does embeddings on CPU, ChromaDB persists vectors to disk.

## Upgrading answer quality

`qwen3.5:0.8b` is small (~1 GB) — fast, with the new Qwen 3.5 architecture. For tougher / multi-hop questions, pull a bigger model and update `.env`:

```powershell
ollama pull qwen3.5:2b      # solid step up
ollama pull qwen3.5:4b      # noticeably stronger
ollama pull qwen3.5:9b      # comparable to old 13B-class models
```
