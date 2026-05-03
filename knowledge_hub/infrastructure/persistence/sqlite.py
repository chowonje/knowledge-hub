"""SQLite compatibility facade over the infrastructure store registry."""

from __future__ import annotations

from typing import Any

from knowledge_hub.infrastructure.persistence.store_registry import (
    DELEGATED_METHODS as _DELEGATED_METHODS,
    SQLITE_BUSY_TIMEOUT_MS,
    StoreRegistry,
)


class SQLiteDatabase:
    """Backward-compatible SQLite facade.

    The canonical implementation now lives under ``knowledge_hub.infrastructure``.
    Legacy imports from ``knowledge_hub.core.sqlite_db`` re-export this class.
    """

    def __init__(
        self,
        db_path: str,
        enable_event_store: bool = True,
        *,
        bootstrap: bool = True,
        read_only: bool = False,
    ):
        self._registry = StoreRegistry(
            db_path,
            enable_event_store=enable_event_store,
            bootstrap=bootstrap,
            read_only=read_only,
        )
        self.db_path = self._registry.db_path
        self.read_only = self._registry.read_only
        self.bootstrap = self._registry.bootstrap
        self.conn = self._registry.conn
        self.migration_manager = self._registry.migration_manager
        self.event_store = self._registry.event_store
        for attr in (
            "note_store",
            "paper_store",
            "document_memory_store",
            "paper_card_v2_store",
            "ontology_store",
            "ontology_profile_store",
            "epistemic_store",
            "web_card_v2_store",
            "vault_card_v2_store",
            "learning_store",
            "learning_graph_store",
            "mcp_job_store",
            "ko_note_store",
            "sync_conflict_store",
            "feature_store",
            "claim_store",
            "claim_card_store",
            "crawl_pipeline_store",
            "quality_mode_store",
            "rag_answer_log_store",
            "ops_action_receipt_store",
            "entity_resolution_store",
        ):
            setattr(self, attr, getattr(self._registry, attr))

    @property
    def registry(self) -> StoreRegistry:
        return self._registry

    def transaction(self):
        return self._registry.transaction()

    def _set_system_meta(self, key: str, value: str) -> None:
        self._registry.set_system_meta(key, value)

    def _get_system_meta(self, key: str, default: str = "") -> str:
        return self._registry.get_system_meta(key, default)

    def apply_ontology_pending(self, pending_id: int):
        return self._registry.apply_ontology_pending(pending_id)

    def apply_pending_ontology(self, pending_id: int):
        return self._registry.apply_pending_ontology(pending_id)

    def reject_ontology_pending(self, pending_id: int):
        return self._registry.reject_ontology_pending(pending_id)

    def close(self) -> None:
        self._registry.close()

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dir(self._registry)) | set(_DELEGATED_METHODS))

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._registry, name)
        except AttributeError as exc:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}") from exc


__all__ = ["SQLiteDatabase", "SQLITE_BUSY_TIMEOUT_MS", "_DELEGATED_METHODS"]
