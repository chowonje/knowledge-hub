"""Compatibility import for the SQLite persistence runtime."""

from knowledge_hub.infrastructure.persistence.sqlite import SQLITE_BUSY_TIMEOUT_MS, SQLiteDatabase

__all__ = ["SQLiteDatabase", "SQLITE_BUSY_TIMEOUT_MS"]
