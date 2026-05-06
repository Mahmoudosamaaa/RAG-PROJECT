"""
config.py - Application settings loaded from .env file

WHY THIS FILE?
--------------
Instead of hardcoding values like model names and paths throughout the code,
we keep them in ONE place (.env file) and load them here. This makes it easy
to change settings without touching any code.

HOW IT WORKS:
- pydantic-settings reads the .env file automatically
- Each variable becomes a typed Python attribute
- You import `settings` from this file anywhere you need a config value
"""

from pydantic_settings import BaseSettings
from pathlib import Path


# Get the project root directory (2 levels up from this file)
# This file is at: backend/app/config.py
# Project root is at: RAG_Project/
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """
    All configuration for the RAG system.
    Values are loaded from the .env file in the project root.
    """

    # --- Embedding Model ---
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # --- LLM (Ollama) ---
    ollama_model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434"

    # --- FAISS ---
    faiss_index_dir: str = "backend/data/faiss_index"

    # --- Document Upload ---
    upload_dir: str = "backend/data/uploads"

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Retriever ---
    retriever_top_k: int = 4

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"

    @property
    def faiss_index_path(self) -> Path:
        """Absolute path to FAISS index storage."""
        return PROJECT_ROOT / self.faiss_index_dir

    @property
    def upload_path(self) -> Path:
        """Absolute path to uploads folder."""
        return PROJECT_ROOT / self.upload_dir


# Create a single settings instance to import everywhere
# Usage: from backend.app.config import settings
settings = Settings()
