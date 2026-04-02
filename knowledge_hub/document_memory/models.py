"""Typed document-memory unit models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_list(values: Any, *, limit: int | None = None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        candidates = [values]
    else:
        try:
            candidates = list(values)
        except Exception:
            candidates = [values]
    result: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        token = _clean_text(raw)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
        if limit is not None and len(result) >= limit:
            break
    return result


def _clean_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


@dataclass
class DocumentMemoryUnit:
    unit_id: str
    document_id: str
    document_title: str = ""
    source_type: str = ""
    source_ref: str = ""
    unit_type: str = "section"
    title: str = ""
    section_path: str = ""
    contextual_summary: str = ""
    source_excerpt: str = ""
    context_header: str = ""
    document_thesis: str = ""
    parent_unit_id: str = ""
    scope_id: str = ""
    confidence: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)
    order_index: int = 0
    content_type: str = "plain"
    links: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    document_date: str = ""
    event_date: str = ""
    observed_at: str = ""
    search_text: str = ""
    version: str = "document-memory-v1"
    created_at: str = ""
    updated_at: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "unit_id": _clean_text(self.unit_id),
            "document_id": _clean_text(self.document_id),
            "document_title": _clean_text(self.document_title),
            "source_type": _clean_text(self.source_type),
            "source_ref": _clean_text(self.source_ref),
            "unit_type": _clean_text(self.unit_type) or "section",
            "title": _clean_text(self.title),
            "section_path": _clean_text(self.section_path),
            "contextual_summary": _clean_text(self.contextual_summary),
            "source_excerpt": _clean_text(self.source_excerpt),
            "context_header": _clean_text(self.context_header),
            "document_thesis": _clean_text(self.document_thesis),
            "parent_unit_id": _clean_text(self.parent_unit_id),
            "scope_id": _clean_text(self.scope_id),
            "confidence": float(self.confidence or 0.0),
            "provenance": _clean_dict(self.provenance),
            "order_index": int(self.order_index or 0),
            "content_type": _clean_text(self.content_type) or "plain",
            "links": _clean_list(self.links, limit=32),
            "tags": _clean_list(self.tags, limit=32),
            "claims": _clean_list(self.claims, limit=32),
            "concepts": _clean_list(self.concepts, limit=32),
            "document_date": _clean_text(self.document_date),
            "event_date": _clean_text(self.event_date),
            "observed_at": _clean_text(self.observed_at),
            "search_text": _clean_text(self.search_text),
            "version": _clean_text(self.version) or "document-memory-v1",
            "created_at": _clean_text(self.created_at),
            "updated_at": _clean_text(self.updated_at),
        }

    def to_payload(self) -> dict[str, Any]:
        record = self.to_record()
        return {
            "unitId": record["unit_id"],
            "documentId": record["document_id"],
            "documentTitle": record["document_title"],
            "sourceType": record["source_type"],
            "sourceRef": record["source_ref"],
            "unitType": record["unit_type"],
            "title": record["title"],
            "sectionPath": record["section_path"],
            "contextualSummary": record["contextual_summary"],
            "sourceExcerpt": record["source_excerpt"],
            "contextHeader": record["context_header"],
            "documentThesis": record["document_thesis"],
            "parentUnitId": record["parent_unit_id"],
            "scopeId": record["scope_id"],
            "confidence": record["confidence"],
            "provenance": dict(record["provenance"]),
            "orderIndex": record["order_index"],
            "contentType": record["content_type"],
            "links": list(record["links"]),
            "tags": list(record["tags"]),
            "claims": list(record["claims"]),
            "concepts": list(record["concepts"]),
            "documentDate": record["document_date"],
            "eventDate": record["event_date"],
            "observedAt": record["observed_at"],
            "searchText": record["search_text"],
            "version": record["version"],
            "createdAt": record["created_at"],
            "updatedAt": record["updated_at"],
        }

    @classmethod
    def from_row(cls, row: dict[str, Any] | None) -> "DocumentMemoryUnit | None":
        if not row:
            return None
        return cls(
            unit_id=_clean_text(row.get("unit_id") or row.get("unitId")),
            document_id=_clean_text(row.get("document_id") or row.get("documentId")),
            document_title=_clean_text(row.get("document_title") or row.get("documentTitle")),
            source_type=_clean_text(row.get("source_type") or row.get("sourceType")),
            source_ref=_clean_text(row.get("source_ref") or row.get("sourceRef")),
            unit_type=_clean_text(row.get("unit_type") or row.get("unitType")) or "section",
            title=_clean_text(row.get("title")),
            section_path=_clean_text(row.get("section_path") or row.get("sectionPath")),
            contextual_summary=_clean_text(row.get("contextual_summary") or row.get("contextualSummary")),
            source_excerpt=_clean_text(row.get("source_excerpt") or row.get("sourceExcerpt")),
            context_header=_clean_text(row.get("context_header") or row.get("contextHeader")),
            document_thesis=_clean_text(row.get("document_thesis") or row.get("documentThesis")),
            parent_unit_id=_clean_text(row.get("parent_unit_id") or row.get("parentUnitId")),
            scope_id=_clean_text(row.get("scope_id") or row.get("scopeId")),
            confidence=float(row.get("confidence") or 0.0),
            provenance=_clean_dict(row.get("provenance")),
            order_index=int(row.get("order_index") or row.get("orderIndex") or 0),
            content_type=_clean_text(row.get("content_type") or row.get("contentType")) or "plain",
            links=_clean_list(row.get("links"), limit=32),
            tags=_clean_list(row.get("tags"), limit=32),
            claims=_clean_list(row.get("claims"), limit=32),
            concepts=_clean_list(row.get("concepts"), limit=32),
            document_date=_clean_text(row.get("document_date") or row.get("documentDate")),
            event_date=_clean_text(row.get("event_date") or row.get("eventDate")),
            observed_at=_clean_text(row.get("observed_at") or row.get("observedAt")),
            search_text=_clean_text(row.get("search_text") or row.get("searchText")),
            version=_clean_text(row.get("version")) or "document-memory-v1",
            created_at=_clean_text(row.get("created_at") or row.get("createdAt")),
            updated_at=_clean_text(row.get("updated_at") or row.get("updatedAt")),
        )
