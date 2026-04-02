"""Canonical semantic-unit contracts layered above document memory and claims.

This module keeps the initial KnowledgeOS vNext contracts additive. Existing
document-memory rows, claim rows, and paper-memory cards remain the source of
truth; these helpers only expose a stable, inspectable view for labs surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _clean_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_heading_path(value: Any) -> list[str]:
    out: list[str] = []
    for item in _clean_list(value):
        token = _clean_text(item)
        if token:
            out.append(token)
    return out


def _clean_bbox(value: Any) -> list[float]:
    raw = _clean_list(value)
    if len(raw) != 4:
        return []
    bbox: list[float] = []
    for item in raw:
        try:
            bbox.append(float(item))
        except Exception:
            return []
    return bbox


def _clean_page(value: Any) -> int | None:
    try:
        page = int(value)
    except Exception:
        return None
    return page if page > 0 else None


def _stable_id(*parts: Any) -> str:
    base = "||".join(_clean_text(part) for part in parts if _clean_text(part))
    if not base:
        base = "semantic-unit"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class SemanticDocument:
    id: str
    title: str = ""
    source_type: str = ""
    source_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": _clean_text(self.id),
            "title": _clean_text(self.title),
            "sourceType": _clean_text(self.source_type),
            "sourceRef": _clean_text(self.source_ref),
            "metadata": _clean_dict(self.metadata),
        }


@dataclass(slots=True)
class Element:
    id: str
    document_id: str
    element_type: str
    text: str
    page: int | None = None
    bbox: list[float] = field(default_factory=list)
    heading_path: list[str] = field(default_factory=list)
    source_span: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": _clean_text(self.id),
            "documentId": _clean_text(self.document_id),
            "elementType": _clean_text(self.element_type),
            "text": _clean_text(self.text),
            "page": _clean_page(self.page),
            "bbox": _clean_bbox(self.bbox),
            "headingPath": _clean_heading_path(self.heading_path),
            "sourceSpan": _clean_text(self.source_span),
            "metadata": _clean_dict(self.metadata),
        }

    @classmethod
    def from_memory_unit(cls, value: dict[str, Any] | Any | None) -> "Element | None":
        if value is None:
            return None
        if isinstance(value, dict):
            row = dict(value)
        else:
            row = {
                "unit_id": getattr(value, "unit_id", ""),
                "document_id": getattr(value, "document_id", ""),
                "source_type": getattr(value, "source_type", ""),
                "source_ref": getattr(value, "source_ref", ""),
                "unit_type": getattr(value, "unit_type", ""),
                "title": getattr(value, "title", ""),
                "section_path": getattr(value, "section_path", ""),
                "contextual_summary": getattr(value, "contextual_summary", ""),
                "source_excerpt": getattr(value, "source_excerpt", ""),
                "context_header": getattr(value, "context_header", ""),
                "document_thesis": getattr(value, "document_thesis", ""),
                "confidence": getattr(value, "confidence", 0.0),
                "order_index": getattr(value, "order_index", 0),
                "provenance": getattr(value, "provenance", {}),
            }
        provenance = dict(row.get("provenance") or {})
        heading_path = _clean_heading_path(provenance.get("heading_path"))
        return cls(
            id=_clean_text(row.get("unit_id") or row.get("unitId")) or _stable_id(row.get("document_id"), row.get("title"), row.get("order_index")),
            document_id=_clean_text(row.get("document_id") or row.get("documentId")),
            element_type=_clean_text(provenance.get("element_type") or row.get("unit_type") or row.get("unitType") or "section"),
            text=_clean_text(row.get("source_excerpt") or row.get("sourceExcerpt") or row.get("contextual_summary") or row.get("contextualSummary")),
            page=_clean_page(provenance.get("page")),
            bbox=_clean_bbox(provenance.get("bbox")),
            heading_path=heading_path,
            source_span=_clean_text(provenance.get("source_span") or row.get("section_path") or row.get("sectionPath") or row.get("title")),
            metadata={
                "unitId": _clean_text(row.get("unit_id") or row.get("unitId")),
                "unitType": _clean_text(row.get("unit_type") or row.get("unitType")),
                "title": _clean_text(row.get("title")),
                "sectionPath": _clean_text(row.get("section_path") or row.get("sectionPath")),
                "sourceType": _clean_text(row.get("source_type") or row.get("sourceType")),
                "sourceRef": _clean_text(row.get("source_ref") or row.get("sourceRef")),
                "contextHeader": _clean_text(row.get("context_header") or row.get("contextHeader")),
                "documentThesis": _clean_text(row.get("document_thesis") or row.get("documentThesis")),
                "confidence": float(row.get("confidence") or 0.0),
            },
        )


@dataclass(slots=True)
class EvidenceLink:
    claim_id: str
    element_id: str
    role: str = "evidence"
    excerpt: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "claimId": _clean_text(self.claim_id),
            "elementId": _clean_text(self.element_id),
            "role": _clean_text(self.role) or "evidence",
            "excerpt": _clean_text(self.excerpt),
            "provenance": _clean_dict(self.provenance),
        }


@dataclass(slots=True)
class MemoryCard:
    id: str
    document_id: str
    title: str = ""
    summary: str = ""
    refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": _clean_text(self.id),
            "documentId": _clean_text(self.document_id),
            "title": _clean_text(self.title),
            "summary": _clean_text(self.summary),
            "refs": [_clean_text(item) for item in self.refs if _clean_text(item)],
            "metadata": _clean_dict(self.metadata),
        }


def document_semantic_units_payload(document: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(document, dict) or not document:
        return {
            "document": {},
            "elements": [],
            "memoryCard": {},
            "counts": {"elements": 0},
        }
    summary = dict(document.get("summary") or {})
    units = list(document.get("units") or [])
    document_id = _clean_text(document.get("documentId"))
    document_payload = SemanticDocument(
        id=document_id,
        title=_clean_text(document.get("documentTitle")),
        source_type=_clean_text(document.get("sourceType")),
        source_ref=_clean_text(summary.get("sourceRef") or summary.get("source_ref") or document_id),
        metadata={
            "summaryUnitId": _clean_text(summary.get("unitId") or summary.get("unit_id")),
            "documentThesis": _clean_text(summary.get("documentThesis") or summary.get("document_thesis")),
        },
    ).to_payload()
    elements = [
        element.to_payload()
        for element in (Element.from_memory_unit(unit) for unit in units)
        if element is not None
    ]
    memory_card = MemoryCard(
        id=_clean_text(summary.get("unitId") or summary.get("unit_id") or f"{document_id}:memory-card"),
        document_id=document_id,
        title=_clean_text(summary.get("title") or document.get("documentTitle")),
        summary=_clean_text(summary.get("contextualSummary") or summary.get("contextual_summary")),
        refs=[
            _clean_text(summary.get("unitId") or summary.get("unit_id")),
            *[_clean_text(unit.get("unitId") or unit.get("unit_id")) for unit in units[:6]],
        ],
        metadata={
            "type": "document_summary",
            "documentThesis": _clean_text(summary.get("documentThesis") or summary.get("document_thesis")),
            "sourceType": _clean_text(document.get("sourceType")),
        },
    ).to_payload()
    return {
        "document": document_payload,
        "elements": elements,
        "memoryCard": memory_card,
        "counts": {"elements": len(elements)},
    }


def evidence_links_from_claim(claim: dict[str, Any] | None, *, paper_id: str = "", paper_title: str = "") -> list[dict[str, Any]]:
    if not isinstance(claim, dict):
        return []
    claim_id = _clean_text(claim.get("claim_id") or claim.get("claimId"))
    claim_text = _clean_text(claim.get("claim_text") or claim.get("claimText"))
    evidence_ptrs = list(claim.get("evidence_ptrs") or claim.get("evidencePtrs") or [])
    links: list[dict[str, Any]] = []
    for index, ptr in enumerate(evidence_ptrs):
        if not isinstance(ptr, dict):
            continue
        note_id = _clean_text(ptr.get("note_id") or ptr.get("noteId"))
        path = _clean_text(ptr.get("path"))
        anchor = _clean_text(ptr.get("snippet_hash") or ptr.get("snippetHash") or note_id or path or index)
        element_id = f"evidence:{_stable_id(claim_id, anchor)}"
        provenance = {
            "noteId": note_id,
            "path": path,
            "paperId": _clean_text(ptr.get("paper_id") or ptr.get("paperId") or ptr.get("arxiv_id") or paper_id),
            "paperTitle": _clean_text(paper_title),
            "claimDecision": _clean_text(ptr.get("claim_decision") or ptr.get("claimDecision")),
            "type": _clean_text(ptr.get("type") or "note"),
            "snippetHash": _clean_text(ptr.get("snippet_hash") or ptr.get("snippetHash")),
        }
        links.append(
            EvidenceLink(
                claim_id=claim_id,
                element_id=element_id,
                role="evidence",
                excerpt=claim_text,
                provenance=provenance,
            ).to_payload()
        )
    if links:
        return links
    return [
        EvidenceLink(
            claim_id=claim_id,
            element_id=f"evidence:{_stable_id(claim_id, claim_text)}",
            role="claim_text",
            excerpt=claim_text,
            provenance={"paperId": _clean_text(paper_id), "paperTitle": _clean_text(paper_title)},
        ).to_payload()
    ]


__all__ = [
    "Element",
    "EvidenceLink",
    "MemoryCard",
    "SemanticDocument",
    "document_semantic_units_payload",
    "evidence_links_from_claim",
]
