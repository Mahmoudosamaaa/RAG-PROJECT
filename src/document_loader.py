"""
Document loading and chunking for the RAG pipeline.

Loads PDF / TXT / MD files, attaches per-page metadata for PDFs, and splits
the resulting documents into overlapping chunks ready for embedding.
"""

import logging
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


def load_document(file_path: str | Path) -> list[Document]:
    """Load one supported file into LangChain Document objects.

    PDFs become one Document per page. Text and Markdown files become one
    Document for the full file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {supported}")

    logger.info("Loading %s (type=%s)", path.name, suffix)
    if suffix == ".pdf":
        docs = _load_pdf(path)
        total_chars = sum(len(d.page_content) for d in docs)
        logger.info(
            "Loaded %s — %d page(s), %d chars total", path.name, len(docs), total_chars
        )
        return docs

    doc = _load_text(path)
    logger.info("Loaded %s — %d chars", path.name, len(doc.page_content))
    return [doc]


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks for embedding and retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = splitter.split_documents(list(documents))
    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = chunk_index

    logger.info(
        "Split into %d chunks (chunk_size=%d, overlap=%d)",
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def ingest_path(
    file_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Load one file and return its text chunks."""
    return split_documents(load_document(file_path), chunk_size, chunk_overlap)


def ingest_paths(
    file_paths: Iterable[str | Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Load many files and return one combined list of chunks."""
    documents: list[Document] = []
    for file_path in file_paths:
        documents.extend(load_document(file_path))
    return split_documents(documents, chunk_size, chunk_overlap)


def _load_text(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    return Document(page_content=text, metadata=_base_metadata(path))


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        metadata = _base_metadata(path)
        metadata["page"] = page_number
        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def _base_metadata(path: Path) -> dict[str, object]:
    return {
        "source": str(path.resolve()),
        "filename": path.name,
        "file_type": path.suffix.lower().lstrip("."),
    }
