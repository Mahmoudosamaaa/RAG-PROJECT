"""
Centralized logging configuration.

Call `setup_logging()` once at the entry point of each process (app.py, api.py).
Module code just uses `logger = logging.getLogger(__name__)` and emits normally.
"""

import logging
import os
import sys


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger with a console handler that writes UTF-8 to stderr.

    Idempotent — safe to call multiple times. Level can be overridden via the
    LOG_LEVEL env var (DEBUG / INFO / WARNING / ERROR).
    """
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    # Force UTF-8 on the stderr stream so non-ASCII characters in logs
    # (e.g. Arabic snippets, ∗ symbols) don't crash on Windows cp1252.
    try:
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove any previously attached console handler we own so re-calls don't duplicate.
    for h in list(root.handlers):
        if getattr(h, "_rag_console", False):
            root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    handler._rag_console = True  # type: ignore[attr-defined]
    handler.setLevel(log_level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)

    # Quiet down a few chatty third-party loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
