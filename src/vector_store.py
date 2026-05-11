"""
ChromaDB-backed vector store for the RAG pipeline.

Wraps a persistent Chroma collection and a sentence-transformers embedding
function so callers only deal with chunks-in / matches-out.
"""

import logging
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "documents",
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model_name

        logger.info(
            "Initializing ChromaDB (persist_dir=%s, collection=%s, embed_model=%s)",
            persist_dir,
            collection_name,
            embedding_model_name,
        )
        t0 = time.perf_counter()
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name,
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )
        logger.info(
            "Vector store ready in %.2fs — %d existing chunks",
            time.perf_counter() - t0,
            self._collection.count(),
        )

    def add_chunks(self, chunks: list[Document]) -> int:
        if not chunks:
            logger.info("add_chunks called with 0 chunks — nothing to do")
            return 0

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []

        for i, chunk in enumerate(chunks):
            md = chunk.metadata or {}
            filename = md.get("filename", "doc")
            page = md.get("page", "x")
            chunk_index = md.get("chunk_index", i)
            ids.append(f"{filename}-p{page}-c{chunk_index}")
            texts.append(chunk.page_content)
            metadatas.append(_clean_metadata(md))

        t0 = time.perf_counter()
        self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Embedded + upserted %d chunks in %.2fs (collection total: %d)",
            len(ids),
            elapsed,
            self._collection.count(),
        )
        return len(ids)

    def query(self, text: str, n_results: int = 5) -> list[dict]:
        if self.count() == 0:
            logger.info("Query against empty collection — returning no matches")
            return []

        preview = text if len(text) <= 80 else text[:77] + "..."
        t0 = time.perf_counter()
        result = self._collection.query(
            query_texts=[text],
            n_results=n_results,
        )
        elapsed = time.perf_counter() - t0

        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]

        if not documents or not documents[0]:
            logger.info("Query '%s' → 0 matches (%.2fs)", preview, elapsed)
            return []

        dists = distances[0]
        logger.info(
            "Query '%s' → %d matches (distance %.3f-%.3f, %.2fs)",
            preview,
            len(documents[0]),
            min(dists),
            max(dists),
            elapsed,
        )
        out: list[dict] = []
        for doc, md, dist in zip(documents[0], metadatas[0], dists):
            out.append({"text": doc, "metadata": md or {}, "distance": dist})
            logger.debug(
                "  match: file=%s page=%s distance=%.3f",
                (md or {}).get("filename"),
                (md or {}).get("page"),
                dist,
            )
        return out

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        logger.info("Clearing collection '%s'", self._collection_name)
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception as exc:
            logger.debug("delete_collection raised (ok if collection was missing): %s", exc)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_fn,
        )
        logger.info("Collection cleared")


def _clean_metadata(md: dict) -> dict:
    """Chroma metadata values must be str / int / float / bool / None."""
    out: dict = {}
    for k, v in md.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out
