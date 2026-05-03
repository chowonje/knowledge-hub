from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from knowledge_hub.application.query_frame import NormalizedQueryFrame


@runtime_checkable
class DomainPack(Protocol):
    def normalize(
        self,
        query: str,
        *,
        source_type: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        sqlite_db: Any | None = None,
        query_plan: dict[str, Any] | None = None,
    ) -> NormalizedQueryFrame: ...

    def classify_family(
        self,
        query: str,
        *,
        source_type: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> str: ...

    def build_query_plan(
        self,
        query: str,
        *,
        source_type: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        sqlite_db: Any | None = None,
    ) -> dict[str, Any]: ...

    def resolve_lookup(
        self,
        entities: list[str],
        *,
        sqlite_db: Any | None = None,
    ) -> tuple[list[str], list[str]]: ...

    def representative_hint(
        self,
        entities: list[str],
        *,
        sqlite_db: Any | None = None,
    ) -> list[dict[str, Any]]: ...

    def select_evidence_policy(
        self,
        frame: NormalizedQueryFrame | dict[str, Any],
    ) -> dict[str, Any]: ...

    def claim_alignment(
        self,
        cards: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...


PaperDomainPack = DomainPack


__all__ = ["DomainPack", "PaperDomainPack"]
