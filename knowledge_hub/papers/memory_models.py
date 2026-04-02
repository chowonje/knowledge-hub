"""Typed paper memory card models."""

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


@dataclass
class PaperMemoryCard:
    memory_id: str
    paper_id: str
    source_note_id: str = ""
    title: str = ""
    paper_core: str = ""
    problem_context: str = ""
    method_core: str = ""
    evidence_core: str = ""
    limitations: str = ""
    concept_links: list[str] = field(default_factory=list)
    claim_refs: list[str] = field(default_factory=list)
    published_at: str = ""
    evidence_window: str = ""
    search_text: str = ""
    quality_flag: str = "unscored"
    version: str = "paper-memory-v1"
    created_at: str = ""
    updated_at: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "memory_id": _clean_text(self.memory_id),
            "paper_id": _clean_text(self.paper_id),
            "source_note_id": _clean_text(self.source_note_id),
            "title": _clean_text(self.title),
            "paper_core": _clean_text(self.paper_core),
            "problem_context": _clean_text(self.problem_context),
            "method_core": _clean_text(self.method_core),
            "evidence_core": _clean_text(self.evidence_core),
            "limitations": _clean_text(self.limitations),
            "concept_links": _clean_list(self.concept_links, limit=12),
            "claim_refs": _clean_list(self.claim_refs, limit=12),
            "published_at": _clean_text(self.published_at),
            "evidence_window": _clean_text(self.evidence_window),
            "search_text": _clean_text(self.search_text),
            "quality_flag": _clean_text(self.quality_flag) or "unscored",
            "version": _clean_text(self.version) or "paper-memory-v1",
            "created_at": _clean_text(self.created_at),
            "updated_at": _clean_text(self.updated_at),
        }

    def to_payload(self) -> dict[str, Any]:
        record = self.to_record()
        return {
            "memoryId": record["memory_id"],
            "paperId": record["paper_id"],
            "sourceNoteId": record["source_note_id"],
            "title": record["title"],
            "paperCore": record["paper_core"],
            "problemContext": record["problem_context"],
            "methodCore": record["method_core"],
            "evidenceCore": record["evidence_core"],
            "limitations": record["limitations"],
            "conceptLinks": list(record["concept_links"]),
            "claimRefs": list(record["claim_refs"]),
            "publishedAt": record["published_at"],
            "evidenceWindow": record["evidence_window"],
            "searchText": record["search_text"],
            "qualityFlag": record["quality_flag"],
            "version": record["version"],
            "createdAt": record["created_at"],
            "updatedAt": record["updated_at"],
        }

    @classmethod
    def from_row(cls, row: dict[str, Any] | None) -> "PaperMemoryCard | None":
        if not row:
            return None
        return cls(
            memory_id=_clean_text(row.get("memory_id") or row.get("memoryId")),
            paper_id=_clean_text(row.get("paper_id") or row.get("paperId")),
            source_note_id=_clean_text(row.get("source_note_id") or row.get("sourceNoteId")),
            title=_clean_text(row.get("title")),
            paper_core=_clean_text(row.get("paper_core") or row.get("paperCore")),
            problem_context=_clean_text(row.get("problem_context") or row.get("problemContext")),
            method_core=_clean_text(row.get("method_core") or row.get("methodCore")),
            evidence_core=_clean_text(row.get("evidence_core") or row.get("evidenceCore")),
            limitations=_clean_text(row.get("limitations")),
            concept_links=_clean_list(row.get("concept_links") or row.get("conceptLinks"), limit=12),
            claim_refs=_clean_list(row.get("claim_refs") or row.get("claimRefs"), limit=12),
            published_at=_clean_text(row.get("published_at") or row.get("publishedAt")),
            evidence_window=_clean_text(row.get("evidence_window") or row.get("evidenceWindow")),
            search_text=_clean_text(row.get("search_text") or row.get("searchText")),
            quality_flag=_clean_text(row.get("quality_flag") or row.get("qualityFlag")) or "unscored",
            version=_clean_text(row.get("version")) or "paper-memory-v1",
            created_at=_clean_text(row.get("created_at") or row.get("createdAt")),
            updated_at=_clean_text(row.get("updated_at") or row.get("updatedAt")),
        )
