"""Compatibility shim for legacy core imports."""

from knowledge_hub.infrastructure.persistence.stores.event_store import EventStore

__all__ = ["EventStore"]
