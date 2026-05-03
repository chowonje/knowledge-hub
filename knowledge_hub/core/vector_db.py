"""Backward-compatible vector database import surface."""

from knowledge_hub.infrastructure.persistence.vector import SQLITE_BUSY_TIMEOUT_MS, VectorDatabase

__all__ = ["SQLITE_BUSY_TIMEOUT_MS", "VectorDatabase"]
