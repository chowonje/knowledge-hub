"""Backward-compatible core import surface."""

from __future__ import annotations

__all__ = ["Config", "Document", "PaperInfo", "SearchResult", "SQLiteDatabase", "VectorDatabase"]


def __getattr__(name: str):
    if name == "Config":
        from knowledge_hub.core.config import Config

        return Config
    if name in {"Document", "PaperInfo", "SearchResult"}:
        from knowledge_hub.core.models import Document, PaperInfo, SearchResult

        return {"Document": Document, "PaperInfo": PaperInfo, "SearchResult": SearchResult}[name]
    if name in {"SQLiteDatabase", "VectorDatabase"}:
        from knowledge_hub.core.database import SQLiteDatabase, VectorDatabase

        return {"SQLiteDatabase": SQLiteDatabase, "VectorDatabase": VectorDatabase}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
