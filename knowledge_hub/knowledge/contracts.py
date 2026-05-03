"""Knowledge-domain repository contracts."""

from __future__ import annotations

from typing import Any, Optional, Protocol

from knowledge_hub.core.models import FeatureSnapshot


class OntologyRepository(Protocol):
    def get_ontology_entity(self, entity_id: str) -> Optional[dict[str, Any]]: ...
    def list_ontology_entities(
        self, entity_type: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]: ...
    def get_relations(self, entity_type: str, entity_id: str) -> list[dict[str, Any]]: ...
    def list_relations(
        self,
        limit: int = 2000,
        updated_after: str | None = None,
        relation: str | None = None,
        source_type: str | None = None,
        target_type: str | None = None,
        predicate_id: str | None = None,
    ) -> list[dict[str, Any]]: ...


class NoteRepository(Protocol):
    def get_note(self, note_id: str) -> Optional[dict[str, Any]]: ...
    def list_notes(
        self,
        source_type: str | None = None,
        para_category: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]: ...
    def merge_note_metadata(self, note_id: str, patch: dict[str, Any]) -> bool: ...


class ClaimRepository(Protocol):
    def get_claim(self, claim_id: str) -> Optional[dict[str, Any]]: ...
    def list_claims(
        self,
        subject_id: str | None = None,
        predicate: str | None = None,
        object_id: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]: ...
    def list_claims_by_note(self, note_id: str, limit: int = 200) -> list[dict[str, Any]]: ...
    def list_claims_by_record(self, record_id: str, limit: int = 200) -> list[dict[str, Any]]: ...
    def list_claims_by_entity(self, entity_id: str, limit: int = 200) -> list[dict[str, Any]]: ...


class FeatureRepository(Protocol):
    def upsert_snapshot(self, *, snapshot: FeatureSnapshot | None = None, **kwargs: Any) -> None: ...
    def get_snapshot(
        self,
        *,
        topic_slug: str,
        feature_kind: str,
        feature_key: str,
    ) -> Optional[dict[str, Any]]: ...
    def list_snapshots(
        self,
        *,
        topic_slug: str | None = None,
        feature_kind: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]: ...


class FeatureComputationRepository(NoteRepository, ClaimRepository, OntologyRepository, Protocol):
    """Repository contract for feature snapshot computation."""


__all__ = [
    "ClaimRepository",
    "FeatureComputationRepository",
    "FeatureRepository",
    "NoteRepository",
    "OntologyRepository",
]
