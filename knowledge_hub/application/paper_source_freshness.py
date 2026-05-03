"""Audit paper source freshness against stored document-memory hashes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from knowledge_hub.infrastructure.persistence.stores.derivative_lifecycle import mark_derivatives_stale_for_document
from knowledge_hub.papers.source_text import resolve_paper_source_snapshot


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _document_memory_source_hash(sqlite_db: Any, document_id: str) -> str:
    rows = list(sqlite_db.list_document_memory_units(document_id, limit=20) or [])
    for row in rows:
        token = _clean_text(row.get("source_content_hash"))
        if token:
            return token
    return ""


def _paper_targets(
    sqlite_db: Any,
    *,
    paper_ids: list[str] | tuple[str, ...] | None,
    limit: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for raw in list(paper_ids or []):
        paper_id = _clean_text(raw)
        if not paper_id or paper_id.casefold() in seen:
            continue
        seen.add(paper_id.casefold())
        paper = sqlite_db.get_paper(paper_id)
        if paper:
            rows.append(dict(paper))
        else:
            missing.append(paper_id)
    if rows or missing or paper_ids:
        return rows, missing
    return [dict(row or {}) for row in sqlite_db.list_papers(limit=max(1, int(limit))) or []], []


def audit_paper_source_freshness(
    sqlite_db: Any,
    *,
    paper_ids: list[str] | tuple[str, ...] | None = None,
    limit: int = 100,
    sample_limit: int = 10,
    apply: bool = False,
) -> dict[str, Any]:
    """Compare current paper source hashes with document-memory source hashes.

    The audit is dry-run by default. With ``apply=True`` it marks stale
    document-memory rows and dependent paper derivatives; it does not rebuild.
    """

    papers, missing_ids = _paper_targets(sqlite_db, paper_ids=paper_ids, limit=limit)
    items: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    counts = {
        "fresh": 0,
        "staleCandidate": 0,
        "missingDocumentMemory": 0,
        "unableToHash": 0,
        "missingPaper": len(missing_ids),
    }
    marked_stale = 0

    for paper in papers:
        paper_id = _clean_text(paper.get("arxiv_id") or paper.get("paper_id"))
        document_id = f"paper:{paper_id}" if paper_id else ""
        snapshot = resolve_paper_source_snapshot(paper)
        source_counts[snapshot.source_key or "unknown"] = source_counts.get(snapshot.source_key or "unknown", 0) + 1
        current_hash = _clean_text(snapshot.source_content_hash)
        stored_hash = _document_memory_source_hash(sqlite_db, document_id) if document_id else ""
        status = "fresh"
        if not current_hash:
            status = "unable_to_hash"
            counts["unableToHash"] += 1
        elif not stored_hash:
            status = "missing_document_memory"
            counts["missingDocumentMemory"] += 1
        elif current_hash != stored_hash:
            status = "stale_candidate"
            counts["staleCandidate"] += 1
        else:
            counts["fresh"] += 1

        changed = 0
        if apply and current_hash and document_id:
            changed = mark_derivatives_stale_for_document(
                sqlite_db.conn,
                document_id=document_id,
                source_content_hash=current_hash,
                source_type="paper",
            )
            marked_stale += changed
            sqlite_db.conn.commit()

        item = {
            "paperId": paper_id,
            "title": _clean_text(paper.get("title")),
            "status": status,
            "sourceKey": snapshot.source_key,
            "sourcePath": snapshot.path,
            "currentSourceContentHash": current_hash,
            "storedDocumentMemorySourceHash": stored_hash,
            "markedStaleCount": changed,
            "warnings": list(snapshot.warnings),
        }
        items.append(item)
        if status != "fresh" and len(samples) < max(0, int(sample_limit)):
            samples.append(item)

    for missing_id in missing_ids:
        item = {
            "paperId": missing_id,
            "title": "",
            "status": "missing_paper",
            "sourceKey": "",
            "sourcePath": "",
            "currentSourceContentHash": "",
            "storedDocumentMemorySourceHash": "",
            "markedStaleCount": 0,
            "warnings": ["paper_not_found"],
        }
        items.append(item)
        if len(samples) < max(0, int(sample_limit)):
            samples.append(item)

    return {
        "schema": "knowledge-hub.paper-source-freshness.result.v1",
        "status": "ok",
        "dryRun": not bool(apply),
        "applied": bool(apply),
        "scannedCount": len(papers) + len(missing_ids),
        "counts": counts,
        "markedStaleCount": marked_stale,
        "sourceTypeCounts": source_counts,
        "sampleItems": samples,
        "items": items,
        "checkedAt": datetime.now(timezone.utc).isoformat(),
    }


__all__ = ["audit_paper_source_freshness"]
