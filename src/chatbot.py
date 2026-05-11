"""
RAG chatbot orchestrating retrieval, prompt construction, and Ollama streaming.
"""

import logging
import time
from collections import deque
from typing import Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .vector_store import VectorStore


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly from the provided document context.

Rules:
- Use only the information in the CONTEXT below to answer the user's question.
- If the context does not contain the answer, say so plainly. Do not invent facts.
- When citing, refer to documents by filename (and page number if shown).
- Keep answers concise and grounded in the cited material.
- Reply in the same language as the user's question."""


class RAGChatbot:
    def __init__(
        self,
        vector_store: VectorStore,
        model: str = "qwen3.5:0.8b",
        base_url: str = "http://localhost:11434",
        top_k: int = 5,
        history_turns: int = 10,
    ) -> None:
        self._vs = vector_store
        self._llm = ChatOllama(
            model=model,
            base_url=base_url,
            # Disable "thinking" mode (Qwen 3+, DeepSeek-R1, etc.). Without this,
            # thinking-trained models emit tokens into a separate `thinking` field
            # that langchain ignores, so `.content` stays empty and the UI looks stuck.
            reasoning=False,
            # Cap output length so a runaway chain can't stall the response.
            num_predict=1024,
            # Lower temperature → more grounded, less prone to long tangents.
            temperature=0.2,
        )
        self._top_k = top_k
        self._history: deque = deque(maxlen=history_turns * 2)
        self._last_sources: list[dict] = []
        self._model_name = model
        logger.info(
            "Chatbot ready (model=%s, base_url=%s, top_k=%d, history_turns=%d)",
            model,
            base_url,
            top_k,
            history_turns,
        )

    @property
    def last_sources(self) -> list[dict]:
        return self._last_sources

    def chat(self, user_message: str) -> Iterator[str]:
        msg_preview = user_message if len(user_message) <= 80 else user_message[:77] + "..."
        logger.info("Turn: '%s' (history=%d msgs)", msg_preview, len(self._history))

        retrieved = self._vs.query(user_message, n_results=self._top_k)
        self._last_sources = [
            {
                "filename": r["metadata"].get("filename", "unknown"),
                "page": r["metadata"].get("page"),
            }
            for r in retrieved
        ]

        messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(self._history)
        messages.append(
            HumanMessage(
                content=(
                    f"CONTEXT:\n{self._format_context(retrieved)}\n\n"
                    f"QUESTION: {user_message}"
                )
            )
        )

        logger.info("Streaming from %s (prompt msgs=%d)...", self._model_name, len(messages))
        t0 = time.perf_counter()
        chunks: list[str] = []
        for piece in self._llm.stream(messages):
            token = piece.content if hasattr(piece, "content") else str(piece)
            if token:
                chunks.append(token)
                yield token

        assistant_text = "".join(chunks)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Stream done (%.2fs, %d chunks, %d chars)",
            elapsed,
            len(chunks),
            len(assistant_text),
        )
        self._history.append(HumanMessage(content=user_message))
        self._history.append(AIMessage(content=assistant_text))

    def reset_history(self) -> None:
        logger.info("Conversation history reset")
        self._history.clear()
        self._last_sources = []

    def _format_context(self, retrieved: list[dict]) -> str:
        if not retrieved:
            return "(no documents in knowledge base or no relevant matches found)"

        blocks: list[str] = []
        for i, r in enumerate(retrieved, start=1):
            md = r["metadata"]
            label = f"[Source {i}: {md.get('filename', '?')}"
            if md.get("page"):
                label += f", page {md['page']}"
            label += "]"
            blocks.append(f"{label}\n{r['text']}")
        return "\n\n".join(blocks)
