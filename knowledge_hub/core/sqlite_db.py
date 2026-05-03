"""Legacy SQLite facade re-export.

Keep this module as a compatibility surface while the canonical implementation
resides under ``knowledge_hub.infrastructure.persistence``.
"""

from knowledge_hub.infrastructure.persistence.sqlite import (
    SQLITE_BUSY_TIMEOUT_MS,
    SQLiteDatabase,
    _DELEGATED_METHODS,
)

__all__ = ["SQLiteDatabase", "SQLITE_BUSY_TIMEOUT_MS", "_DELEGATED_METHODS"]
