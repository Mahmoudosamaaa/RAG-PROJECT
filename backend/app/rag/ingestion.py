"""
Document ingestion utilities for the RAG pipeline.

In RAG, ingestion is the first step:
1. Load a document from disk.
2. Turn it into text with useful metadata.
3. Split the text into smaller chunks for embedding and retrieval.
"""

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from backend.app.config import settings


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def ensure_data_directories() -> None:
    """Create runtime folders used by uploads and vector indexes."""
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.faiss_index_path.mkdir(parents=True, exist_ok=True)


def load_document(file_path: str | Path) -> list[Document]:
    """
    Load one supported file into LangChain Document objects.

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

    if suffix == ".pdf":
        return _load_pdf(path)

    return [_load_text(path)]


def split_documents(
    documents: Iterable[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split documents into chunks small enough for embedding and retrieval.

    The chunk settings default to values from backend/app/config.py, so later
    changes can be made in one place.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        add_start_index=True,
    )

    chunks = splitter.split_documents(list(documents))
    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = chunk_index

    return chunks


def ingest_file(file_path: str | Path) -> list[Document]:
    """Load one file and return its text chunks."""
    ensure_data_directories()
    return split_documents(load_document(file_path))


def ingest_files(file_paths: Iterable[str | Path]) -> list[Document]:
    """Load many files and return one combined list of chunks."""
    ensure_data_directories()

    documents: list[Document] = []
    for file_path in file_paths:
        documents.extend(load_document(file_path))

    return split_documents(documents)


def _load_text(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    return Document(page_content=text, metadata=_base_metadata(path))


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
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
